"""Trace + compile + per-sequence execution of an embedding-model trunk.

``DeplodockForwardRunner`` owns the deplodock side of the vLLM plugin: at
server start it traces the HuggingFace ``AutoModel`` trunk (hidden states out,
no lm_head) with a dynamic seq_len, compiles it through the CUDA backend
(greedy fork picks from the global prior — no GPU tuning), and builds ONE
``CompiledProgram``. Per sequence it re-binds fresh inputs (``rebind``) and
launches (``run_once``) — one compiled program serves every request at any
seq_len ≤ ``DYNAMIC_DIM_MAX``.

No vllm imports here — the class is driven by ``vllm_model.DeplodockEmbedModel``
but is independently testable with torch + cupy alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deplodock.compiler.backend.cuda.program import CompiledProgram

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


class DeplodockForwardRunner:
    """One compiled dynamic-seq_len trunk + its per-sequence execution."""

    def __init__(self, program: CompiledProgram, input_names: tuple[str, str, str], output_name: str, np_dtype, max_seq_len: int):
        self._program = program
        self._ids_name, self._mask_name, self._pos_name = input_names
        self._output_name = output_name
        self._np_dtype = np_dtype
        self.max_seq_len = max_seq_len
        self._mask_cache: dict[int, np.ndarray] = {}

    @classmethod
    def create(cls, model_id: str, max_seq_len: int, dtype_str: str = "float16") -> DeplodockForwardRunner:
        import torch
        from transformers import AutoModel

        from deplodock.compiler.backend.cuda.backend import CudaBackend
        from deplodock.compiler.backend.cuda.program import CompiledProgram
        from deplodock.compiler.backend.gpu_lock import gpu_lock
        from deplodock.compiler.loader.binder import bind_constants
        from deplodock.compiler.trace.dynamic import DYNAMIC_DIM_MAX, build_torch_dynamic_shapes, parse_position_specs
        from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
        from deplodock.compiler.trace.torch import trace_module

        if max_seq_len > DYNAMIC_DIM_MAX:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds deplodock's dynamic-dim max ({DYNAMIC_DIM_MAX}); "
                f"serve with --max-model-len {DYNAMIC_DIM_MAX} or lower"
            )
        dtype = getattr(torch, dtype_str)

        logger.info("[serving] loading %s trunk (dtype=%s)...", model_id, dtype_str)
        # vLLM instantiates models inside a CUDA device context; the HF trunk
        # is only traced + read for constants here (then freed), so force CPU —
        # this also sidesteps transformers' accelerate requirement for
        # non-default device contexts.
        with torch.device("cpu"):
            model = AutoModel.from_pretrained(model_id, dtype=dtype)
        model.eval()

        logger.info("[serving] tracing (dynamic seq_len, example S=%d)...", _TRACE_SEQ)
        specs = parse_position_specs(
            ["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"]
        )
        # Same CPU pin as the load above: the wrapper builds buffers and the
        # trace runs the forward on example tensors — all of it must sit on
        # one device, regardless of vLLM's CUDA default-device context.
        with torch.device("cpu"):
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
        np_dtype = np.dtype(dtype_str)
        sources: dict[str, np.ndarray] = {}
        for path, t in wrapper.named_parameters(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        for path, t in wrapper.named_buffers(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        const_feed = bind_constants(compiled, sources)

        ids_name, mask_name, pos_name = compiled.inputs
        feed = {
            ids_name: np.zeros((1, _TRACE_SEQ), dtype=np.int64),
            mask_name: _causal_mask_np(_TRACE_SEQ, np_dtype),
            pos_name: np.arange(_TRACE_SEQ, dtype=np.int64).reshape(1, _TRACE_SEQ),
        }
        with gpu_lock():
            program = CompiledProgram.build(compiled, {**const_feed, **feed})
        del model, wrapper, sources, const_feed
        logger.info("[serving] ready: %d launches, max_seq_len=%d", len(program.compiled.launches), max_seq_len)
        return cls(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=compiled.outputs[0],
            np_dtype=np_dtype,
            max_seq_len=max_seq_len,
        )

    @property
    def hidden_size(self) -> int:
        out = self._program.compiled.buf_by_name[self._output_name]
        return int(out.shape[-1].as_static())

    def _mask(self, s: int) -> np.ndarray:
        cached = self._mask_cache.get(s)
        if cached is not None:
            return cached
        mask = _causal_mask_np(s, self._np_dtype)
        if len(self._mask_cache) >= _MASK_CACHE_MAX:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[s] = mask
        return mask

    def forward_hidden_states(self, token_ids: np.ndarray) -> np.ndarray:
        """Run one sequence: ``token_ids`` shape ``(S,)`` int64, ``S <=
        max_seq_len``. Returns ``(S, hidden)`` numpy in the traced dtype."""
        from deplodock.compiler.backend.gpu_lock import gpu_lock

        s = int(token_ids.shape[0])
        feed = {
            self._ids_name: token_ids.reshape(1, s),
            self._mask_name: self._mask(s),
            self._pos_name: np.arange(s, dtype=np.int64).reshape(1, s),
        }
        with gpu_lock():
            self._program.rebind(feed)
            self._program.run_once()
            return self._program.outputs()[self._output_name][0]
