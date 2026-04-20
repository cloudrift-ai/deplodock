"""Lift ``ReduceOp`` to a single-reduce-axis ``LoopOp``.

The lifted kernel iterates over the full input shape; the reduce axis is
kind ``"reduce"`` and contributes an ``Accum`` + ``Accum`` pair. The
Write index covers only free axes (with size-1 placeholders for keep-dim
outputs).
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop_ir import Accum, Axis, Load, Loop, LoopOp, Stmt, Write
from deplodock.compiler.ir.tensor_ir import ElementwiseOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", ReduceOp, "1")]

# Map ReduceOp.fn ("sum"/"max"/"min"/"prod") to the combine op name that
# ``Accum`` carries. Accum init is derived from this op by the Loop IR
# identity table (``ACCUM_IDENTITY``), so no separate init lookup is needed.
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

    axes = tuple(Axis(name=f"a{i}", extent=int(d)) for i, d in enumerate(src_shape))
    reduce_axis_name = f"a{axis}"
    load_index = tuple(Var(a.name) for a in axes)

    combine = ElementwiseOp(_COMBINE.get(op.fn, op.fn))
    write_index = _build_write_index(axes, reduce_axis_name, tuple(node.output.shape))

    # Nested body: Loop(reduce_axis, [Load + Accum]) + Write wrapped in
    # Loop(free_axis, ...) blocks (outermost first). The Accum implicitly
    # declares the ``acc`` accumulator — its op defines the combine and,
    # via ACCUM_IDENTITY, the init value.
    reduce_axis = next(a for a in axes if a.name == reduce_axis_name)
    free_axes = [a for a in axes if a.name != reduce_axis_name]
    inner: tuple[Stmt, ...] = (
        Loop(
            axis=reduce_axis,
            body=(
                Load(name="in0", source=0, index=load_index),
                Accum(name="acc", value="in0", op=combine),
            ),
        ),
        Write(output=0, index=write_index, value="acc"),
    )
    body: tuple[Stmt, ...] = inner
    for a in reversed(free_axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    frag.add_node(InputOp(), [], Tensor(src_id, src_shape, src_node.output.dtype), node_id=src_id)
    out_id = frag.add_node(kernel, [src_id], Tensor(f"lift_{nid}", node.output.shape, node.output.dtype))
    frag.outputs = [out_id]
    return frag


def _build_write_index(axes: tuple[Axis, ...], reduce_axis_name: str, out_shape: tuple) -> tuple[Expr, ...]:
    """Build the Write's index for a reduce kernel under the keepdim invariant.

    Tensor IR's rank-preservation rule (enforced by the tracer and every
    decomposition rule) says reductions keep their reduced axis at size 1, so
    ``len(out_shape) == len(axes)``. The reduce axis and singleton dims become
    ``Literal(0)`` at their dim position; free axes become ``Var(axis.name)``.
    """
    if len(out_shape) != len(axes):
        raise ValueError(
            f"lift_reduce: out_shape rank {len(out_shape)} must equal axes count {len(axes)} "
            f"(keepdim invariant — wrap in a squeeze IndexMapOp if a dropped shape is needed)"
        )
    result: list[Expr] = []
    for a, d in zip(axes, out_shape, strict=True):
        if a.name == reduce_axis_name or (isinstance(d, int) and d == 1):
            result.append(Literal(0, "int"))
        else:
            result.append(Var(a.name))
    return tuple(result)
