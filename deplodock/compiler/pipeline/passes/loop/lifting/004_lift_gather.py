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

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import CastExpr, Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tensor.ir import GatherOp
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", GatherOp)]


def rewrite(root: Node, inp_data: Node, inp_idx: Node, out: Tensor) -> Graph | None:
    out_shape = tuple(out.shape)
    ndim = len(out_shape)
    axis = int(root.op.axis) if int(root.op.axis) >= 0 else ndim + int(root.op.axis)

    axes = tuple(Axis(name=f"a{i}", extent=int(d)) for i, d in enumerate(out_shape))

    # Load 0: idx — identity load over all output axes.
    idx_index = tuple(Var(a.name) for a in axes)
    # Load 1: data — every dim is a free axis except ``axis``, which reads the
    # loaded idx value cast to int (referenced by the idx Load's SSA name).
    data_index = tuple(CastExpr("int", Var("idx")) if i == axis else Var(axes[i].name) for i in range(ndim))

    inner: Body = (
        Load(name="idx", input=inp_idx.id, index=idx_index),
        Load(name="data", input=inp_data.id, index=data_index),
        Write(output=f"kernel_{root.id}", index=tuple(Var(a.name) for a in axes), value="data"),
    )
    body: Body = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    for buf in (inp_idx, inp_data):
        if buf.id not in frag.nodes:
            frag.add_node(InputOp(), [], Tensor(buf.id, buf.output.shape, buf.output.dtype), node_id=buf.id)

    out_id = frag.add_node(kernel, list(kernel.body_inputs), Tensor(out.name, out_shape, out.dtype), node_id=f"kernel_{root.id}")
    frag.outputs = [out_id]
    return frag
