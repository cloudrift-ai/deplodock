"""Numpy graph interpreter shared by the numpy and loop backends.

Walks a ``Graph`` in topological order, seeds input / constant buffers,
and calls ``node.op.forward(*args)`` at every compute node. ``LoopOp``
nodes dispatch through the same mechanism — their ``forward`` method is
defined in :mod:`deplodock.compiler.ir.loop.interpret` — so this walker
works identically on pre-fusion and post-fusion graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.ir.base import ConstantOp, InputOp

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


def interpret_graph(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Walk ``graph`` in topo order and return ``{output_id: ndarray}``.

    ``input_data`` maps graph input ids (and optionally ``ConstantOp`` ids
    whose value should be overridden) to ndarrays. ConstantOps with a
    scalar ``value`` that are not overridden are materialized as
    single-element float32 arrays.
    """
    input_data = input_data or {}
    input_set = set(graph.inputs)
    values: dict[str, np.ndarray] = {}

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        shape = tuple(int(d) for d in node.output.shape if isinstance(d, int))

        if nid in input_set:
            if nid not in input_data:
                raise KeyError(f"Missing input for node {nid!r}")
            values[nid] = _coerce(input_data[nid], shape)
            continue

        if isinstance(node.op, ConstantOp):
            if nid in input_data:
                values[nid] = _coerce(input_data[nid], shape)
            elif node.op.value is not None:
                values[nid] = np.array([node.op.value], dtype=np.float32)
            else:
                raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in input_data")
            continue

        if isinstance(node.op, InputOp):
            # Sentinel with no consumer data (graph.inputs already handled above).
            continue

        args = [values[inp] for inp in node.inputs]
        result = node.op.forward(*args)
        arr = np.asarray(result, dtype=np.float32)
        if shape and arr.shape != shape:
            arr = arr.reshape(shape)
        values[nid] = arr

    return {name: values[name] for name in graph.outputs}


def _coerce(data, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if shape and arr.shape != shape:
        arr = arr.reshape(shape)
    return arr
