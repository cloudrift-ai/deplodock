"""Decompose ReshapeOp into an IndexMapOp with linearize→delinearize coord_map.

ReshapeOp changes logical shape without moving data. The coord_map
linearizes the output coordinates using output strides, then
delinearizes into the input coordinate space using input strides.
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var, placeholder
from deplodock.compiler.ir.frontend.ir import ReshapeOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", ReshapeOp)]


def _reshape_coord_map(in_shape: tuple, out_shape: tuple):
    """Build coord_map entries that linearize output coords then delinearize to input coords.

    For reshape from in_shape to out_shape (same numel), each input
    coordinate is:  c_j = (flat // in_stride_j) % in_shape_j
    where flat = sum(out_coord_i * out_stride_i).

    Strides are built as ``Expr`` trees rather than Python ints so symbolic
    shape elements (``Dim('seq_len')``) thread through as ``Var('seq_len')``
    factors. The simplifier folds constant runs before the tree lands in
    the ``IndexMapOp.sources`` coord map.
    """
    out_ndim = len(out_shape)
    in_ndim = len(in_shape)

    if in_shape == out_shape:
        return tuple(placeholder(d) for d in range(out_ndim))

    flat = None
    out_stride: object = Literal(1, "int")
    for d in range(out_ndim - 1, -1, -1):
        stride_simplified = out_stride.simplify(SimplifyCtx.empty())
        term = placeholder(d) if _is_one(stride_simplified) else BinaryExpr("*", placeholder(d), stride_simplified)
        flat = term if flat is None else BinaryExpr("+", term, flat)
        out_stride = BinaryExpr("*", out_stride, _to_expr(out_shape[d]))

    if flat is None:
        flat = Literal(0, "int")

    coords = []
    in_stride: object = Literal(1, "int")
    for j in range(in_ndim - 1, -1, -1):
        stride_j = in_stride.simplify(SimplifyCtx.empty())
        coord = flat if _is_one(stride_j) else BinaryExpr("/", flat, stride_j)
        dim_j = _to_expr(in_shape[j])
        if j > 0:
            coord = BinaryExpr("%", coord, dim_j)
        coords.insert(0, coord.simplify(SimplifyCtx.empty()))
        in_stride = BinaryExpr("*", in_stride, dim_j)

    return tuple(coords)


def _to_expr(d) -> object:
    """Pull the ``Expr`` out of a shape element. ``Tensor.shape`` is always
    ``tuple[Dim, ...]`` so ``d.expr`` is the common path; bare ``int`` /
    ``str`` only show up for the un-coerced literal-shape passed via
    ``ReshapeOp.shape``."""
    if isinstance(d, Dim):
        return d.expr
    if isinstance(d, int):
        return Literal(d, "int")
    if isinstance(d, str):
        return Var(d)
    raise TypeError(f"_to_expr: unexpected shape element {d!r}")


def _is_one(e) -> bool:
    return isinstance(e, Literal) and e.value == 1


def rewrite(match: Match, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    in_shape = tuple(inp_x.output.shape)
    out_shape = root.op.infer_output_shape([in_shape])

    coord_map = _reshape_coord_map(in_shape, out_shape)

    frag = open_fragment(graph, [inp_x])
    new_node = single_indexmap(frag, inp_x, out_shape=out_shape, coord_map=coord_map, name=out.name)
    frag.outputs = [new_node.id]
    return frag
