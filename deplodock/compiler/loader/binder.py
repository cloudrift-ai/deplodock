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
from deplodock.compiler.ir.base import ConstantOp, InputOp


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
    in_id = g.add_node(op=InputOp(), inputs=[], output=Tensor("src", tuple(source.shape), str(source.dtype)))
    g.inputs.append(in_id)

    cur = in_id
    cur_shape = tuple(source.shape)
    for i, op in enumerate(load_ops):
        cur_shape = op.infer_output_shape([cur_shape])
        nid = g.add_node(op=op, inputs=[cur], output=Tensor(f"step_{i}", cur_shape, "f32"))
        cur = nid
    g.outputs.append(cur)

    from deplodock.compiler.backend.base import Backend

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result = _NumpyInterp().run(g, input_data={in_id: source.astype(np.float32, copy=False)})
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
    for nid, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp):
            continue
        if node.op.value is not None:
            continue
        path = node.op.source_path
        if path is None or path not in sources:
            continue
        out[nid] = apply_load_ops(sources[path], node.op.load_ops)
    return out
