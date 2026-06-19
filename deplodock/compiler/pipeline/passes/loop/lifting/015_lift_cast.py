"""Lift ``CastOp`` to a single-op ``LoopOp`` copy-with-dtype kernel.

The cast is rank-preserving with no broadcast (input shape == output
shape), so the read is a plain identity index. The body is one
``Load`` → ``Assign(op="copy", dtype=<target>)`` → ``Write``; the
``copy`` Assign carries the target dtype so ``Assign.render`` takes its
copy/cast fast path (i32→f16 = a single ``__int2half_rn``, no f32
detour). The W4A16 int→fp16 boundary lowers through here.

Mergeable into its consumer (the group-scale multiply) by the later
fusion pass — after fusion the dequant cone holds no materialized cast
buffer; the int nibble casts to fp16 in registers right before the
multiply.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tensor.ir import CastOp
from deplodock.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", CastOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    out_shape = tuple(root.output.shape)
    axes = tuple(Axis(name=f"a{i}", extent=d) for i, d in enumerate(out_shape))
    write_index = tuple(Var(a.name) for a in axes)

    src_id = root.inputs[0]
    # Shape-preserving cast: identity read index (one axis Var per dim).
    idx = write_index

    inner: Body = (
        Load(name="in0", input=src_id, index=idx),
        Assign(name="v", op="copy", args=("in0",), dtype=root.output.dtype),
        Write(output=f"lift_{root.id}", index=write_index, value="v"),
    )
    body: Body = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    ext = graph.nodes.get(src_id)
    shape = ext.output.shape if ext is not None else ()
    dtype = ext.output.dtype if ext is not None else "f32"
    frag.add_node(InputOp(), [], Tensor(src_id, shape, dtype), node_id=src_id)
    out_id = frag.add_node(
        kernel,
        list(kernel.inputs),
        Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=f"lift_{root.id}",
    )
    frag.outputs = [out_id]
    return frag
