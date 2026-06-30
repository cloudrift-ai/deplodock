"""Trace + compile + per-sequence execution of an embedding-model trunk.

``EmmyForwardRunner`` owns the emmy side of the vLLM plugin: at
server start it traces the HuggingFace ``AutoModel`` trunk (hidden states out,
no lm_head) with a dynamic seq_len, compiles it through the CUDA backend
(greedy fork picks from the global prior — no GPU tuning), and builds ONE
``CompiledProgram`` over a single ``max_seq_len``-sized buffer set — one program
serves every request at any seq_len ≤ ``max_seq_len``.

Per request it captures (once per distinct seq_len, then replays) a whole-program
CUDA graph over that buffer set — one host-side launch instead of ~hundreds. Each
graph is captured at its EXACT seq_len, so every kernel runs at its exact grid (no
oversized-grid masking); the buffers are allocated once at ``max_seq_len`` and each
request's inputs upload into their contiguous prefix.

No vllm imports here — the class is driven by ``vllm_model.EmmyEmbedModel``
but is independently testable with torch + cupy alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emmy.compiler.backend.cuda.program import CompiledProgram

logger = logging.getLogger(__name__)

# Example-tensor size handed to torch.export (the CLI's --seq-len default).
# The traced dim is symbolic, so the value never reaches the kernels.
_TRACE_SEQ = 32
# Per-seq_len causal-mask host cache (FIFO). At S=4096 one fp16 mask is
# ~32 MB, so the cap keeps the cache bounded while serving mixed lengths.
_MASK_CACHE_MAX = 16


def _causal_mask_np(s: int, np_dtype) -> np.ndarray:
    """``(1, 1, s, s)`` additive causal mask (0 on/below diagonal, -inf above),
    the numpy twin of ``trace.huggingface.build_causal_mask``."""
    mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)
    return mask.astype(np_dtype)[None, None, :, :]


class EmmyForwardRunner:
    """One compiled dynamic-seq_len trunk + its per-sequence execution."""

    def __init__(
        self,
        program: CompiledProgram,
        input_names: tuple[str, str, str],
        output_name: str,
        np_dtype,
        max_seq_len: int,
        batch_cap: int = 1,
    ):
        self._program = program
        self._ids_name, self._mask_name, self._pos_name = input_names
        self._output_name = output_name
        self._np_dtype = np_dtype
        # Shared buffer set is sized for max_seq_len; every accepted request
        # (S ≤ max_seq_len) uses the captured-graph path.
        self.max_seq_len = max_seq_len
        # batch_cap == 1: symbolic-seq, one sequence per forward (the default).
        # batch_cap > 1: ONE static (batch_cap, max_seq_len) program; each
        # scheduler step runs as a padded batched forward (`forward_hidden_states_batched`).
        self.batch_cap = batch_cap
        # int seq_len -> device-built (1,1,S,S) cupy causal mask (built on first
        # sight of each S, reused across same-S requests via a D2D prefix copy).
        self._mask_cache: dict = {}

    @classmethod
    def create(cls, model_id: str, max_seq_len: int, dtype_str: str = "float16", batch: int = 1) -> EmmyForwardRunner:
        import torch
        from transformers import AutoModel

        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.cuda.program import CompiledProgram
        from emmy.compiler.backend.gpu_lock import gpu_lock
        from emmy.compiler.loader.binder import bind_constants
        from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX, build_torch_dynamic_shapes, parse_position_specs
        from emmy.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
        from emmy.compiler.trace.torch import trace_module

        if max_seq_len > DYNAMIC_DIM_MAX:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds emmy's dynamic-dim max ({DYNAMIC_DIM_MAX}); "
                f"serve with --max-model-len {DYNAMIC_DIM_MAX} or lower"
            )
        dtype = getattr(torch, dtype_str)
        np_dtype = np.dtype(dtype_str)

        logger.info("[serving] loading %s trunk (dtype=%s)...", model_id, dtype_str)
        # vLLM instantiates models inside a CUDA device context; the HF trunk
        # is only traced + read for constants here (then freed), so force CPU —
        # this also sidesteps transformers' accelerate requirement for
        # non-default device contexts. The wrapper builds buffers and the trace
        # runs the forward on example tensors — all of it must sit on one device.
        with torch.device("cpu"):
            model = AutoModel.from_pretrained(model_id, dtype=dtype)
            model.eval()
            if batch > 1:
                # Static batched path: ONE fully-static (batch, max_seq_len)
                # program (the symbolic-seq kernels miscompute batch>1; a static
                # trace bakes correct batch strides). Each step pads to this shape.
                logger.info("[serving] tracing STATIC batched (batch=%d, S=%d)...", batch, max_seq_len)
                wrapper = build_full_model_wrapper(model, max_seq_len, dtype, dynamic=True)
                example = (
                    torch.zeros((batch, max_seq_len), dtype=torch.long),
                    build_causal_mask(max_seq_len, dtype),
                    torch.arange(max_seq_len).unsqueeze(0).expand(batch, max_seq_len).contiguous(),
                )
                graph = trace_module(wrapper, example)  # no dynamic_shapes → fully static
            else:
                logger.info("[serving] tracing (dynamic seq_len, example S=%d)...", _TRACE_SEQ)
                specs = parse_position_specs(
                    ["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"]
                )
                wrapper = build_full_model_wrapper(model, _TRACE_SEQ, dtype, dynamic=True)
                example = (
                    torch.zeros((1, _TRACE_SEQ), dtype=torch.long),
                    build_causal_mask(_TRACE_SEQ, dtype),
                    torch.arange(_TRACE_SEQ).unsqueeze(0),
                )
                graph = trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs))

        logger.info("[serving] compiling...")
        compiled = CudaBackend(tune_db="auto").compile(graph)
        if len(compiled.inputs) != 3:
            raise RuntimeError(f"expected 3 graph inputs (input_ids, attention_mask, position_ids), got {compiled.inputs}")
        if len(compiled.outputs) != 1:
            raise RuntimeError(f"expected 1 graph output (hidden states), got {compiled.outputs}")

        # Weight sources in the traced dtype: named_buffers (NOT state_dict)
        # so non-persistent buffers — the wrapper's precomputed rotary
        # cos/sin — bind too; remove_duplicate=False so tied weights surface
        # under every traced alias.
        sources: dict[str, np.ndarray] = {}
        for path, t in wrapper.named_parameters(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        for path, t in wrapper.named_buffers(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        const_feed = bind_constants(compiled, sources)

        ids_name, mask_name, pos_name = compiled.inputs
        if batch > 1:
            # Static (batch, max_seq_len) buffers. The causal mask + position_ids
            # are request-independent (full causal at max_seq_len, arange positions),
            # so they're fed once here and never updated — only ids change per step.
            feed = {
                ids_name: np.zeros((batch, max_seq_len), dtype=np.int64),
                mask_name: _causal_mask_np(max_seq_len, np_dtype),
                pos_name: np.tile(np.arange(max_seq_len, dtype=np.int64), (batch, 1)),
            }
        else:
            # Shared buffer set sized at max_seq_len so each captured graph (one per
            # seq_len) replays over the same prefix-occupied buffers; every accepted
            # request (S ≤ max_seq_len) fits, so all use the captured path.
            feed = {
                ids_name: np.zeros((1, max_seq_len), dtype=np.int64),
                mask_name: _causal_mask_np(max_seq_len, np_dtype),
                pos_name: np.arange(max_seq_len, dtype=np.int64).reshape(1, max_seq_len),
            }
        with gpu_lock():
            program = CompiledProgram.build(compiled, {**const_feed, **feed})
        del model, wrapper, sources, const_feed
        logger.info(
            "[serving] ready: %d launches, max_seq_len=%d, batch_cap=%d",
            len(program.compiled.launches),
            max_seq_len,
            batch,
        )
        return cls(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=compiled.outputs[0],
            np_dtype=np_dtype,
            max_seq_len=max_seq_len,
            batch_cap=batch,
        )

    @property
    def hidden_size(self) -> int:
        out = self._program.compiled.buf_by_name[self._output_name]
        return int(out.shape[-1].as_static())

    def _mask(self, s: int):
        """``(1, 1, s, s)`` additive causal mask as a cached cupy device array —
        the device twin of :func:`_causal_mask_np`, built once per S on the GPU so
        the hot path never builds/uploads it from host."""
        import cupy as cp  # noqa: PLC0415

        cached = self._mask_cache.get(s)
        if cached is not None:
            return cached
        mask = cp.triu(cp.full((s, s), float("-inf"), dtype=cp.float32), k=1).astype(self._np_dtype)[None, None, :, :]
        if len(self._mask_cache) >= _MASK_CACHE_MAX:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[s] = mask
        return mask

    def forward_hidden_states(self, token_ids):
        """Run one sequence: ``token_ids`` a 1-D int torch CUDA tensor of length
        ``S <= max_seq_len``. Returns an ``(S, hidden)`` torch CUDA tensor in the
        trunk dtype.

        Zero-copy device path: bridge the torch input to cupy (``cp.from_dlpack``,
        no host copy), size the launch grids to S, copy ids / device-built causal
        mask / position_ids into the shared buffers' prefix (device-to-device),
        capture-or-reuse the whole-program graph for this S, replay it, and wrap
        the output buffer's prefix back as a torch tensor (``torch.from_dlpack``)
        — no GPU↔host round-trip. All cupy work runs on torch's current stream so
        the prefix copy, the graph replay, and the output read stay ordered; the
        result is cloned because the shared buffer is reused by the next request."""
        import cupy as cp  # noqa: PLC0415
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        s = int(token_ids.shape[0])
        if s > self.max_seq_len:
            raise ValueError(f"seq_len {s} exceeds max_seq_len {self.max_seq_len}")
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            feed = {
                self._ids_name: cp.from_dlpack(token_ids.detach().reshape(1, s)),
                self._mask_name: self._mask(s),
                self._pos_name: cp.arange(s, dtype=cp.int64).reshape(1, s),
            }
            self._program.set_sym_values({"seq_len": s})
            self._program.upload_prefix_device(feed)
            self._program.capture_program_graph()
            self._program.replay_program_graph()
            out = self._program.output_prefix_device({"seq_len": s})[self._output_name][0]
            return torch.from_dlpack(out).clone()

    def forward_hidden_states_batched(self, token_ids_list):
        """Run up to ``batch_cap`` sequences (each a 1-D int torch CUDA tensor of
        length ``S <= max_seq_len``) in ONE static batched forward. Returns a list
        of ``(S_i, hidden)`` torch CUDA tensors in the same order.

        The static program is fixed at ``(batch_cap, max_seq_len)``, so each step
        is padded to that shape: short rows pad on the right and short batches pad
        with dummy rows. Causal masking makes every row's real prefix independent
        of its right-padding (a token only attends to earlier positions), and dummy
        rows are simply not read out — so padding never corrupts a real result.
        Inputs longer than ``batch_cap`` are processed in successive batched groups."""
        import cupy as cp  # noqa: PLC0415
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        B, S = self.batch_cap, self.max_seq_len
        results: list = [None] * len(token_ids_list)
        for start in range(0, len(token_ids_list), B):
            group = token_ids_list[start : start + B]
            lens = [int(t.shape[0]) for t in group]
            if any(s > S for s in lens):
                raise ValueError(f"seq_len {max(lens)} exceeds max_seq_len {S}")
            with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                ids = torch.zeros((B, S), dtype=torch.int64, device=group[0].device)
                for i, t in enumerate(group):
                    ids[i, : lens[i]] = t.detach().to(torch.int64)
                self._program.upload_prefix_device({self._ids_name: cp.from_dlpack(ids)})
                self._program.capture_program_graph()  # static graph, one cached entry
                self._program.replay_program_graph()
                out = torch.from_dlpack(self._program.output_prefix_device()[self._output_name])
                for i, si in enumerate(lens):
                    results[start + i] = out[i, :si].clone()
        return results
