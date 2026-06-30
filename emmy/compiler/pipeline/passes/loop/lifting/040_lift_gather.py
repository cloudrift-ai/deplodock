"""Wrap a standalone GatherOp as a LoopOp copy kernel.

GatherOp is overloaded across three torch semantics — all routed here by
the tracer (``trace.torch`` maps ``embedding`` / ``index_select`` /
``gather`` to the same op). They differ in how the output rank relates to
the index and data ranks:

- **gather** (``torch.gather``): ``idx`` and ``data`` have the same rank;
  output shape equals ``idx`` shape. Each output coord reads ``data`` at
  ``(o_0, ..., idx[o_0,...,o_{n-1}], ..., o_{n-1})`` (``idx`` substituted
  at position ``axis``).
- **embedding** (``F.embedding`` / ``weight[idx]``): ``data`` is 2-D
  ``(V, H)``, ``idx`` is any rank, output shape is
  ``idx.shape + (data.shape[1:],)``. Generalizes to any ``axis``.
- **index_select** (``torch.index_select``): ``idx`` is 1-D, output shape
  is ``data.shape[:axis] + idx.shape + data.shape[axis+1:]``.

The last two share a unified formula: ``out_shape ==
data.shape[:axis] + idx.shape + data.shape[axis+1:]`` (embedding is the
special case ``axis=0``). They yield ``out_rank == idx_rank + data_rank
- 1`` — the marker used to distinguish them from same-rank gather.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import CastExpr, Var
from emmy.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import GatherOp
from emmy.compiler.pipeline import Pattern

PATTERN = [Pattern("root", GatherOp)]


def rewrite(root: Node, inp_data: Node, inp_idx: Node, out: Tensor) -> Graph | None:
    out_shape = tuple(out.shape)
    out_rank = len(out_shape)
    data_rank = len(inp_data.output.shape)
    idx_rank = len(inp_idx.output.shape)
    axis = int(root.op.axis) if int(root.op.axis) >= 0 else out_rank + int(root.op.axis)

    axes = tuple(Axis(name=f"a{i}", extent=d) for i, d in enumerate(out_shape))

    if out_rank == data_rank and out_rank == idx_rank:
        # torch.gather: idx, data, output all same rank.
        idx_index = tuple(Var(a.name) for a in axes)
        data_index = tuple(CastExpr("int", Var("idx")) if i == axis else Var(axes[i].name) for i in range(data_rank))
    elif out_rank == idx_rank + data_rank - 1:
        # Embedding / index_select: idx contributes ``idx_rank`` output
        # axes at positions [axis : axis + idx_rank]; the remaining
        # output axes map onto data's non-axis dims.
        idx_index = tuple(Var(axes[i].name) for i in range(axis, axis + idx_rank))
        data_index_l: list = []
        for j in range(data_rank):
            if j == axis:
                data_index_l.append(CastExpr("int", Var("idx")))
            elif j < axis:
                data_index_l.append(Var(axes[j].name))
            else:
                data_index_l.append(Var(axes[j + idx_rank - 1].name))
        data_index = tuple(data_index_l)
    else:
        raise ValueError(f"lift_gather: incompatible ranks — out_rank={out_rank}, data_rank={data_rank}, idx_rank={idx_rank}, axis={axis}")

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

    out_id = frag.add_node(kernel, list(kernel.inputs), Tensor(out.name, out_shape, out.dtype), node_id=f"kernel_{root.id}")
    frag.outputs = [out_id]
    return frag
