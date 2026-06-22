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


def _compile_split(wrapper, example_args, argnames, np_dtype):
    """Trace ``wrapper`` with axis 0 of every arg bound to a shared ``num_tokens`` Dim,
    compile, bind constants, build a :class:`_Program`."""
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from deplodock.compiler.trace.torch import trace_module

    specs = [f"num_tokens@{name}:0" for name in argnames]  # shared NAME ties all axes
    graph = trace_module(wrapper, tuple(example_args), dynamic_shapes=build_torch_dynamic_shapes(parse_position_specs(specs)))
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
    def __init__(self, *, embed_weight, norm, pre, post, attn_meta, np_dtype):
        self._embed_weight = embed_weight  # numpy [vocab, H]
        self._norm = norm  # torch module
        self._pre = pre  # list[_Program]
        self._post = post  # list[_Program]
        self.head_dim, self.num_heads, self.num_kv_heads, self.scaling = attn_meta
        self._np_dtype = np_dtype

    @property
    def num_layers(self) -> int:
        return len(self._pre)

    @classmethod
    def create(cls, model_id, *, dtype_str="float16"):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("[gen_runner] loading %s (%s, CPU trace)...", model_id, dtype_str)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_str)).eval()
            return cls.from_model(model, dtype_str=dtype_str)

    @classmethod
    def from_model(cls, model, *, dtype_str="float16"):
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
        for i, block in enumerate(layers):
            logger.info("[gen_runner] compiling layer %d/%d (pre + post)...", i + 1, len(layers))
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

        embed_weight = trunk.embed_tokens.weight.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        return cls(
            embed_weight=embed_weight,
            norm=trunk.norm,
            pre=pre_programs,
            post=post_programs,
            attn_meta=(head_dim, num_heads, num_kv, scaling),
            np_dtype=np_dtype,
        )

    def embed(self, input_ids):
        """``input_ids``: list/1-D of ints → ``[T, H]`` numpy in the runner dtype."""
        import numpy as np

        return self._embed_weight[np.asarray(input_ids, dtype=np.int64)]

    def forward_layer_pre(self, layer, hidden, positions=None):
        """``hidden[T, H]`` numpy → un-rotated ``(q[T,Hq·D], k[T,Hkv·D], v[T,Hkv·D])``.
        ``positions`` is unused under A2 (RoPE applied downstream); kept for signature parity."""
        del positions
        return tuple(self._pre[layer].run([hidden.astype(self._np_dtype, copy=False)]))

    def forward_layer_post(self, layer, attn_out, residual):
        """``(attn_out[T,Hq·D], residual[T,H])`` numpy → ``layer_out[T, H]`` numpy."""
        return self._post[layer].run([attn_out.astype(self._np_dtype, copy=False), residual.astype(self._np_dtype, copy=False)])[0]

    def final_norm(self, hidden):
        """Apply the model's final norm (held as a torch module) to ``hidden[T, H]`` numpy."""
        import numpy as np
        import torch

        with torch.no_grad():
            out = self._norm(torch.from_numpy(np.ascontiguousarray(hidden)))
        return out.numpy()
