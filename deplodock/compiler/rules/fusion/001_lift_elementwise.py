"""Lift ``ElementwiseOp`` to a trivial single-op ``LoopOp``.

Every ElementwiseOp in the post-decomposition graph is wrapped as a one-op
kernel that reads its inputs via identity Ports, applies the op, and writes
the result. Broadcasts on input buffers are handled by right-aligning non-
size-1 dims onto kernel axes (matching the iteration space of the output).

Mergeable pairs (producer/consumer LoopOp) are collapsed later by the merge
rule; this pass only introduces the LoopOp wrapper.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Stmt, Write
from deplodock.compiler.ir.tensor_ir import ElementwiseOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", ElementwiseOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, ElementwiseOp):
        return None

    out_shape = tuple(node.output.shape)
    if out_shape and not all(isinstance(d, int) for d in out_shape):
        return None

    axes = tuple(Axis(name=f"a{i}", extent=int(d)) for i, d in enumerate(out_shape))

    load_stmts: list[Stmt] = []
    load_names: list[str] = []
    for i, inp_id in enumerate(node.inputs):
        inp_node = graph.nodes.get(inp_id)
        inp_shape = tuple(inp_node.output.shape) if inp_node is not None else ()
        idx = _identity_index(inp_shape, axes)
        name = f"in{i}"
        load_names.append(name)
        load_stmts.append(Load(name=name, source=i, index=idx))

    write_index = tuple(Var(a.name) for a in axes)
    inner: tuple[Stmt, ...] = (
        *load_stmts,
        Assign(name="v", op=node.op, args=tuple(load_names)),
        Write(output=0, index=write_index, value="v"),
    )
    # Nest the body in free-axis Loops (outer axis wraps the innermost).
    body: tuple[Stmt, ...] = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    for inp_id in node.inputs:
        if inp_id in frag.nodes:
            continue
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(
        kernel,
        list(node.inputs),
        Tensor(f"lift_{nid}", node.output.shape, node.output.dtype),
    )
    frag.outputs = [out_id]
    return frag


def _identity_index(src_shape: tuple, axes: tuple[Axis, ...]) -> tuple[Expr, ...]:
    """Build an identity read index for ``src_shape`` under ``axes``.

    Walks right-to-left so non-size-1 source dims latch onto the rightmost
    matching-extent axis. Size-1 source dims become ``Literal(0)`` (broadcast).
    Missing leading axes (scalar / fewer-dim inputs) contribute nothing.
    """
    if not src_shape:
        return ()

    result: list[Expr] = [Literal(0, "int")] * len(src_shape)
    cursor = len(axes) - 1

    for i in range(len(src_shape) - 1, -1, -1):
        dim = src_shape[i]
        if isinstance(dim, int) and dim == 1:
            continue
        while cursor >= 0 and int(axes[cursor].extent) != int(dim):
            cursor -= 1
        if cursor < 0:
            break
        result[i] = Var(axes[cursor].name)
        cursor -= 1
    return tuple(result)
