"""``DeplodockGenRunner`` — per-layer attention-split runner (Phase 2 of
``plans/generative-inference-support.md``).

Sibling to ``DeplodockForwardRunner`` (the embedding runner). Carves SDPA out of every
decoder layer (``build_attention_split_wrapper``), compiles **two programs per layer**
(``pre`` + ``post``) over the flattened ``[num_tokens, H]`` layout with ``num_tokens``
symbolic, and exposes the per-token, everything-but-attention compute:

- ``embed(input_ids) -> hidden[T, H]`` — token embedding lookup (the runner owns embedding).
- ``forward_layer_pre(L, hidden, positions) -> (q, k, v)`` — un-rotated 2-D seam q[T,Hq·D],
  k/v[T,Hkv·D] (RoPE is applied downstream by the caller / vLLM, A2 — ``positions`` unused here).
- ``forward_layer_post(L, attn_out, residual) -> hidden`` — o_proj + residual + post-norm + MLP.
- ``final_norm(hidden) -> hidden``.

The caller stitches attention between ``pre`` and ``post`` (a reference torch SDPA in the
Phase-2 host stitch; vLLM's paged ``Attention`` in Phase 3). I/O is host numpy (the
serving ``rebind`` path) — correctness-first; the device zero-copy + captured-replay path
is the Phase-3/4 optimization. NOTE: two capacity-sized ``CompiledProgram``s per layer is
the memory budget the plan flags (Phase 2 "Memory budget" / Top risk #9).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Program:
    """One compiled dynamic-``num_tokens`` split subgraph, run via the host ``rebind`` path."""

    def __init__(self, program, input_names, output_names):
        self.program = program
        self.input_names = input_names
        self.output_names = output_names

    def run(self, arrays):
        """arrays: numpy arrays aligned to ``input_names`` (each ``[T, …]``). Returns the
        outputs in ``output_names`` order, sliced to the runtime ``T``."""
        from deplodock.compiler.backend.gpu_lock import gpu_lock

        t = arrays[0].shape[0]
        feed = dict(zip(self.input_names, arrays, strict=True))
        with gpu_lock():
            self.program.rebind(feed)  # resolves num_tokens from the input shapes
            self.program.run_once()
            out = self.program.outputs({"num_tokens": t})
        return [out[n] for n in self.output_names]

    def run_device(self, arrays):
        """Device-resident captured-replay twin of :meth:`run` for the **static M=bucket** decode
        programs. ``arrays``: torch CUDA tensors aligned to ``input_names`` (each ``[T, …]``,
        ``T <= bucket``). Uploads the ``T`` real rows into the buffer prefix (device-to-device),
        captures-or-replays the whole-program graph, and returns the outputs as torch CUDA tensors
        sliced to ``T`` — no host round-trip. Stale prefix padding rows are safe (pre/post are
        per-token-independent; only ``[:T]`` is read out). All cupy work runs on torch's current
        stream so the upload, replay and output read stay ordered."""
        import cupy as cp
        import torch

        from deplodock.compiler.backend.gpu_lock import gpu_lock

        t = arrays[0].shape[0]
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            feed = {n: cp.from_dlpack(a.detach().contiguous()) for n, a in zip(self.input_names, arrays, strict=True)}
            self.program.upload_prefix_device(feed)
            self.program.capture_program_graph()  # static graph → one cached entry (empty sym_values)
            self.program.replay_program_graph()
            outs = self.program.output_prefix_device()
            return [torch.from_dlpack(outs[n])[:t].clone() for n in self.output_names]


def _pad_rows(arr, bucket):
    """Pad axis 0 from ``t`` up to ``bucket`` with zeros. The decode programs are static at
    M=bucket; padding rows are computed then sliced away — safe because pre/post are
    per-token-independent (pointwise + matmul over the hidden axis)."""
    import numpy as np

    t = arr.shape[0]
    if t == bucket:
        return arr
    out = np.zeros((bucket, *arr.shape[1:]), dtype=arr.dtype)
    out[:t] = arr
    return out


def _compile_split(wrapper, example_args, argnames, np_dtype):
    """Trace ``wrapper`` and build a :class:`_Program`. ``argnames`` (a list) ties each named
    arg's axis-0 to a shared symbolic ``num_tokens`` Dim — the **prefill** program (one program,
    any width). ``argnames=None`` traces a **fully static** graph at the example shapes — the
    **decode-bucket** program (efficient at small M; the symbolic program's hint-sized M-tile is
    pathological at decode — see ``plans/generative-decode-perf-findings.md``)."""
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.torch import trace_module

    dynamic_shapes = None
    if argnames:
        from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{n}:0" for n in argnames]))
    graph = trace_module(wrapper, tuple(example_args), dynamic_shapes=dynamic_shapes)
    compiled = CudaBackend(tune_db="auto").compile(graph)

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
    const_feed = bind_constants(compiled, sources)

    feed = {n: a.detach().cpu().to(torch.float32).numpy().astype(np_dtype) for n, a in zip(compiled.inputs, example_args, strict=True)}
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
    return _Program(program, list(compiled.inputs), list(compiled.outputs))


class DeplodockGenRunner:
    def __init__(self, *, embed_weight, norm, pre, post, attn_meta, np_dtype, pre_decode=None, post_decode=None, decode_bucket=16):
        self._embed_weight = embed_weight  # numpy [vocab, H]
        self._norm = norm  # torch module
        self._pre = pre  # list[_Program] — symbolic (prefill / any width)
        self._post = post
        self._pre_decode = pre_decode  # list[_Program] — static M=decode_bucket (or None → no bucket)
        self._post_decode = post_decode
        self._decode_bucket = decode_bucket
        self.head_dim, self.num_heads, self.num_kv_heads, self.scaling = attn_meta
        self._np_dtype = np_dtype

    @property
    def num_layers(self) -> int:
        return len(self._pre)

    @property
    def decode_bucket(self) -> int:
        return self._decode_bucket

    @property
    def has_device_decode(self) -> bool:
        """True when the static decode-bucket programs exist → the device-resident decode path
        (``embed_device`` / ``forward_layer_*_device`` / ``final_norm_device``) is available."""
        return self._pre_decode is not None

    @classmethod
    def create(cls, model_id, *, dtype_str="float16", decode_bucket=16):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("[gen_runner] loading %s (%s, CPU trace)...", model_id, dtype_str)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_str)).eval()
            return cls.from_model(model, dtype_str=dtype_str, decode_bucket=decode_bucket)

    @classmethod
    def from_model(cls, model, *, dtype_str="float16", decode_bucket=16):
        """Build from an already-loaded CausalLM module (the network-free path). ``model``
        must be on CPU for the trace."""
        import numpy as np
        import torch

        from deplodock.compiler.trace.huggingface import build_attention_split_wrapper

        dtype = getattr(torch, dtype_str)
        np_dtype = np.dtype(dtype_str)
        trunk = getattr(model, "model", model)
        layers = trunk.layers
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        num_heads = attn0.q_proj.out_features // head_dim
        num_kv = attn0.k_proj.out_features // head_dim
        scaling = attn0.scaling
        hidden = model.config.hidden_size
        attn_width = num_heads * head_dim

        pre_programs, post_programs = [], []
        pre_decode, post_decode = [], []
        decode_ok = decode_bucket and decode_bucket > 0
        for i, block in enumerate(layers):
            logger.info("[gen_runner] compiling layer %d/%d (pre + post%s)...", i + 1, len(layers), " + decode" if decode_ok else "")
            pre_w, post_w = build_attention_split_wrapper(block)
            with torch.device("cpu"):
                pre_programs.append(_compile_split(pre_w, [torch.zeros(8, hidden, dtype=dtype)], ["hidden"], np_dtype))
                post_programs.append(
                    _compile_split(
                        post_w,
                        [torch.zeros(8, attn_width, dtype=dtype), torch.zeros(8, hidden, dtype=dtype)],
                        ["attn_out", "residual"],
                        np_dtype,
                    )
                )
                # Static M=decode_bucket twins — fast at decode (small M). If a layer's static
                # compile fails (e.g. a demoted-matmul lowering gap at this bucket), drop the
                # decode path entirely and fall back to the symbolic programs (slow but correct).
                if decode_ok:
                    try:
                        pre_decode.append(_compile_split(pre_w, [torch.zeros(decode_bucket, hidden, dtype=dtype)], None, np_dtype))
                        post_decode.append(
                            _compile_split(
                                post_w,
                                [torch.zeros(decode_bucket, attn_width, dtype=dtype), torch.zeros(decode_bucket, hidden, dtype=dtype)],
                                None,
                                np_dtype,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001 — any lowering/compile failure → disable the bucket
                        logger.warning("[gen_runner] decode-bucket compile failed at layer %d (%s); decode falls back to symbolic", i, ex)
                        decode_ok = False

        embed_weight = trunk.embed_tokens.weight.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        use_decode = decode_ok and len(pre_decode) == len(layers)
        return cls(
            embed_weight=embed_weight,
            norm=trunk.norm,
            pre=pre_programs,
            post=post_programs,
            attn_meta=(head_dim, num_heads, num_kv, scaling),
            np_dtype=np_dtype,
            pre_decode=pre_decode if use_decode else None,
            post_decode=post_decode if use_decode else None,
            decode_bucket=decode_bucket,
        )

    def embed(self, input_ids):
        """``input_ids``: list/1-D of ints → ``[T, H]`` numpy in the runner dtype."""
        import numpy as np

        return self._embed_weight[np.asarray(input_ids, dtype=np.int64)]

    def forward_layer_pre(self, layer, hidden, positions=None):
        """``hidden[T, H]`` numpy → un-rotated ``(q[T,Hq·D], k[T,Hkv·D], v[T,Hkv·D])``.
        ``positions`` is unused under A2 (RoPE applied downstream); kept for signature parity.
        Uses the static decode-bucket program when ``T <= decode_bucket`` (pad → run → slice)."""
        del positions
        h = hidden.astype(self._np_dtype, copy=False)
        t = h.shape[0]
        if self._pre_decode is not None and t <= self._decode_bucket:
            q, k, v = self._pre_decode[layer].run([_pad_rows(h, self._decode_bucket)])
            return q[:t], k[:t], v[:t]
        return tuple(self._pre[layer].run([h]))

    def forward_layer_post(self, layer, attn_out, residual):
        """``(attn_out[T,Hq·D], residual[T,H])`` numpy → ``layer_out[T, H]`` numpy. Decode-bucketed
        like ``forward_layer_pre``."""
        a = attn_out.astype(self._np_dtype, copy=False)
        r = residual.astype(self._np_dtype, copy=False)
        t = a.shape[0]
        if self._post_decode is not None and t <= self._decode_bucket:
            out = self._post_decode[layer].run([_pad_rows(a, self._decode_bucket), _pad_rows(r, self._decode_bucket)])[0]
            return out[:t]
        return self._post[layer].run([a, r])[0]

    def final_norm(self, hidden):
        """Apply the model's final norm (held as a torch module) to ``hidden[T, H]`` numpy."""
        import numpy as np
        import torch

        with torch.no_grad():
            out = self._norm(torch.from_numpy(np.ascontiguousarray(hidden)))
        return out.numpy()

    # --- Device-resident decode path (Phase A of plans/generative-device-resident-decode.md) ---
    # Used by the vLLM plugin for the decode hot path (T <= decode_bucket); the numpy methods
    # above stay for prefill / the standalone ``deplodock generate`` oracle.

    def _ensure_device(self):
        """Lazily build CUDA copies of the embed table + final norm (once) for the device path.
        A **deep copy** of the norm — `.to()` is in-place, and the host `final_norm` (oracle /
        prefill) still feeds it CPU tensors, so the shared module must stay on CPU."""
        if getattr(self, "_dev_ready", False):
            return
        import copy

        import torch

        self._embed_weight_dev = torch.from_numpy(self._embed_weight).cuda()
        self._norm_dev = copy.deepcopy(self._norm).to("cuda")
        self._dev_ready = True

    def embed_device(self, input_ids):
        """``input_ids``: 1-D int torch CUDA tensor → ``[T, H]`` CUDA tensor (on-device gather)."""
        self._ensure_device()
        return self._embed_weight_dev[input_ids.long()]

    def forward_layer_pre_device(self, layer, hidden):
        """Device twin of :meth:`forward_layer_pre` (decode path, ``T <= decode_bucket``):
        ``hidden[T,H]`` CUDA → un-rotated ``(q, k, v)`` CUDA tensors."""
        return tuple(self._pre_decode[layer].run_device([hidden]))

    def forward_layer_post_device(self, layer, attn_out, residual):
        """Device twin of :meth:`forward_layer_post`: ``(attn_out, residual)`` CUDA → ``[T,H]`` CUDA."""
        return self._post_decode[layer].run_device([attn_out, residual])[0]

    def final_norm_device(self, hidden):
        """Apply the final norm on CUDA to a ``hidden[T,H]`` CUDA tensor."""
        import torch

        self._ensure_device()
        with torch.no_grad():
            return self._norm_dev(hidden)
