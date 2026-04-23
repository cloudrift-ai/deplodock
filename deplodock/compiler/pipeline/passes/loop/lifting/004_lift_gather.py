"""Wrap a standalone GatherOp as a LoopOp copy kernel.

Gather is data-dependent: output element ``(o_0, ..., o_{n-1})`` reads
``data`` at coordinates equal to the output coords except position
``axis``, which is replaced by the integer value of ``idx[o_0, ..., o_{n-1}]``.

The resulting LoopOp has two Ports — the idx port loads the gather
index; the data port's ``index`` Expr at the gather axis references
``$0`` (cast to int). The emit threads previously-loaded port values
into the axis env so the data port can reference the idx port's value.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Cast, Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Stmt, Write
from deplodock.compiler.ir.tensor.ir import GatherOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", GatherOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, GatherOp):
        return None

    data_id, idx_id = node.inputs[0], node.inputs[1]
    out_shape = tuple(node.output.shape)
    ndim = len(out_shape)
    axis = int(node.op.axis) if int(node.op.axis) >= 0 else ndim + int(node.op.axis)

    axes = tuple(Axis(name=f"a{i}", extent=int(d)) for i, d in enumerate(out_shape))

    # Load 0: idx — identity load over all output axes.
    idx_index = tuple(Var(a.name) for a in axes)
    # Load 1: data — every dim is a free axis except ``axis``, which reads the
    # loaded idx value cast to int (referenced by the idx Load's SSA name).
    data_index = tuple(Cast("int", Var("idx")) if i == axis else Var(axes[i].name) for i in range(ndim))

    inner: tuple[Stmt, ...] = (
        Load(name="idx", source=0, index=idx_index),
        Load(name="data", source=1, index=data_index),
        Write(output=0, index=tuple(Var(a.name) for a in axes), value="data"),
    )
    body: tuple[Stmt, ...] = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    for buf_id in (idx_id, data_id):
        if buf_id not in frag.nodes:
            ext = graph.nodes.get(buf_id)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(buf_id, shape, dtype), node_id=buf_id)

    out_id = frag.add_node(kernel, [idx_id, data_id], Tensor(f"kernel_{nid}", out_shape, node.output.dtype))
    frag.outputs = [out_id]
    return frag
