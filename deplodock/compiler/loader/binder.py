"""Apply ``ConstantOp.load_ops`` to a source ndarray via the reference
NumPy backend.

The loader produces raw source tensors (from safetensors or
``module.named_parameters()``); this binder is the small adapter that
runs each constant's ``load_ops`` chain on its source array. Reusing
the existing ``Backend.run`` interpreter (``backend/base.py``) means
every op already has a numpy ``forward()`` — no new code path.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp


def apply_load_ops(source: np.ndarray, load_ops: tuple) -> np.ndarray:
    """Run ``load_ops`` over ``source`` using the NumPy backend.

    Builds a tiny single-input graph and dispatches it through the
    default ``Backend.run`` interpreter. Each load op must already have
    a working ``Op.forward`` (true for ``TransposeOp`` / ``ReshapeOp``
    by construction — those are the only ops the fold pass produces).
    """
    if not load_ops:
        return np.ascontiguousarray(source)

    g = Graph()
    src_dtype = str(source.dtype)
    in_id = g.add_node(op=InputOp(), inputs=[], output=Tensor("src", tuple(source.shape), src_dtype))
    g.inputs.append(in_id)

    cur = in_id
    cur_shape = tuple(source.shape)
    for i, op in enumerate(load_ops):
        cur_shape = op.infer_output_shape([cur_shape])
        nid = g.add_node(op=op, inputs=[cur], output=Tensor(f"step_{i}", cur_shape, src_dtype))
        cur = nid
    g.outputs.append(cur)

    from deplodock.compiler.backend.base import Backend

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result, _ = _NumpyInterp().run(g, input_data={in_id: np.ascontiguousarray(source)})
    return np.ascontiguousarray(result.outputs[cur])


def bind_constants(graph: Graph, sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the per-node ``input_data`` dict for every ``ConstantOp``.

    ``sources`` maps each ``ConstantOp.source_path`` to its raw source
    ndarray (typically read from safetensors or pulled from a live
    ``nn.Module``). For each constant, this runs ``load_ops`` over the
    source and stores the result keyed by the constant's node id.
    Scalar constants (``value is not None``) are skipped — the backend
    materializes them on its own. Constants without a source_path are
    skipped too (synthetic constants emitted by passes carry their
    ``value`` directly).
    """
    out: dict[str, np.ndarray] = {}
    for nid, op in graph.loadable_constants():
        if op.source_path not in sources:
            continue
        out[nid] = apply_load_ops(sources[op.source_path], op.load_ops)
    return out


def bind_constants_from_module(graph: Graph, module) -> dict[str, np.ndarray]:  # noqa: ANN001 — torch.nn.Module, duck-typed
    """Bind every parameter/buffer ``ConstantOp`` from a *live* ``nn.Module``.

    The whole-model trace wrapper carries computed buffers — the precomputed
    rotary ``cos``/``sin`` and the causal mask — that aren't in the checkpoint's
    safetensors, so :func:`load_constants_from_safetensors` can't supply them.
    Binding from the traced module's own ``state_dict`` covers those *and* the
    weights uniformly: each ``ConstantOp.source_path`` is the module-attribute
    path captured at trace time, so it matches a ``state_dict`` key verbatim.

    ``state_dict`` (not ``named_parameters``) is the right source because it
    lists *tied* weights under every name they're registered as — e.g. a model
    with ``tie_word_embeddings=True`` traces an ``lm_head.weight`` constant, but
    ``named_parameters`` dedups it down to the shared ``embed_tokens.weight``
    only, leaving the final projection unbound (→ zero logits). Tensors are cast
    to float32 numpy (the backend dtype; also sidesteps reading bf16 checkpoints
    through numpy)."""
    sources: dict[str, np.ndarray] = {}
    for name, t in module.state_dict().items():
        sources[name] = t.detach().cpu().float().numpy()
    return bind_constants(graph, sources)
