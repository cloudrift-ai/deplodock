"""Lift ``ReduceOp`` to a single-reduce-axis ``LoopOp``.

The lifted kernel iterates over the full input shape; the reduce axis is
kind ``"reduce"`` and contributes an accumulator ``LocalBuffer``. The Write
index covers only free axes (with size-1 placeholders for keep-dim outputs).
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Axis, LocalBuffer, LoopOp, Port, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", ReduceOp, "1")]

# Identity + combine tables (the same values the old assembler used).
_IDENTITY: dict[str, float] = {"sum": 0.0, "max": -1e30, "prod": 1.0, "min": 1e30}
_COMBINE: dict[str, str] = {"sum": "add", "max": "max", "prod": "mul", "min": "min"}


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    op = node.op
    if not isinstance(op, ReduceOp) or not node.inputs:
        return None

    src_id = node.inputs[0]
    src_node = graph.nodes.get(src_id)
    if src_node is None:
        return None
    src_shape = tuple(src_node.output.shape)
    if not src_shape or not all(isinstance(d, int) for d in src_shape):
        return None

    ndim = len(src_shape)
    axis_raw = op.axis
    axis = int(axis_raw) if isinstance(axis_raw, int) else 0
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        return None

    axes = tuple(Axis(name=f"a{i}", extent=int(d), kind="reduce" if i == axis else "free") for i, d in enumerate(src_shape))
    input_port = Port(index=tuple(Var(a.name) for a in axes))

    combine_fn = _COMBINE.get(op.fn, op.fn)
    init = _IDENTITY.get(op.fn, 0.0)
    acc = LocalBuffer(name="acc", combine=ElementwiseOp(combine_fn), init=Literal(init))

    write_index = _build_write_index(axes, tuple(node.output.shape))
    body = (
        Update(target="acc", value="$0"),
        Write(output=0, index=write_index, value="acc"),
    )
    kernel = LoopOp(axes=axes, inputs=(input_port,), locals=(acc,), body=body)

    frag = Graph()
    frag.add_node(InputOp(), [], Tensor(src_id, src_shape, src_node.output.dtype), node_id=src_id)
    out_id = frag.add_node(kernel, [src_id], Tensor(f"lift_{nid}", node.output.shape, node.output.dtype))
    frag.outputs = [out_id]
    return frag


def _build_write_index(axes: tuple[Axis, ...], out_shape: tuple) -> tuple[Expr, ...]:
    """Build the Write's index for a reduce kernel.

    - keepdim: ``len(out_shape) == len(axes)`` — reduce axes become ``Literal(0)``.
    - non-keepdim: ``len(out_shape) == free-axis-count`` — free axes only.
    - fallback: align non-size-1 output dims onto free axes right-to-left.
    """
    free_axes = tuple(a for a in axes if a.kind == "free")

    if len(out_shape) == len(axes):
        result: list[Expr] = []
        for a, d in zip(axes, out_shape, strict=True):
            if a.kind == "reduce" or (isinstance(d, int) and d == 1):
                result.append(Literal(0, "int"))
            else:
                result.append(Var(a.name))
        return tuple(result)

    if len(out_shape) == len(free_axes):
        return tuple(Var(a.name) for a in free_axes)

    # Fallback: right-align non-size-1 output dims onto free axes.
    out_list: list[Expr] = [Literal(0, "int")] * len(out_shape)
    cursor = len(free_axes) - 1
    for i in range(len(out_shape) - 1, -1, -1):
        d = out_shape[i]
        if isinstance(d, int) and d == 1:
            continue
        if cursor < 0:
            break
        out_list[i] = Var(free_axes[cursor].name)
        cursor -= 1
    return tuple(out_list)
