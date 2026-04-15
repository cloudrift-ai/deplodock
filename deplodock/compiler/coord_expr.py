"""Coordinate expressions for IndexMapOp.

`IndexMapOp` describes layout-only ops (slice, cat, transpose, reshape,
unsqueeze) by mapping output coordinates to input coordinates via affine
arithmetic. We reuse the existing `LoopExpr` AST (`backend/ir/expr.py`) to
represent these expressions — no parallel AST.

Convention: an IndexMap's `coord_map[i]` is a `LoopExpr` over placeholder
variables ``Var("out_coord_0")``, ``Var("out_coord_1")``, ... — one per
output dimension. At lowering time the placeholders are substituted with
the LoopExprs that the kernel uses for its actual output coordinates.
The same substitution machinery composes adjacent IndexMaps in the
optimization pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.backend.ir.expr import BinOp, Builtin, FuncCall, Literal, Ternary, Var

if TYPE_CHECKING:
    from deplodock.compiler.backend.ir.loop_ir import LoopExpr


PLACEHOLDER_PREFIX = "out_coord_"


def placeholder(d: int) -> Var:
    """Return the placeholder variable for output coordinate axis ``d``."""
    return Var(f"{PLACEHOLDER_PREFIX}{d}")


def is_placeholder(expr: object, d: int | None = None) -> bool:
    """Check if ``expr`` is a placeholder ``Var``. If ``d`` is given, check that axis."""
    if not isinstance(expr, Var):
        return False
    if not expr.name.startswith(PLACEHOLDER_PREFIX):
        return False
    if d is None:
        return True
    return expr.name == f"{PLACEHOLDER_PREFIX}{d}"


def substitute(expr: LoopExpr, mapping: dict[str, LoopExpr]) -> LoopExpr:
    """Replace ``Var(name)`` nodes in ``expr`` with ``mapping[name]``.

    Walks the LoopExpr tree non-destructively. Used at:
    - **Lowering time**: substitute placeholder coords with the kernel's
      actual output-coord expressions.
    - **Composition time**: substitute outer IndexMap's placeholders with
      the inner IndexMap's coord_map entries.

    Variables not present in ``mapping`` are left unchanged.
    """
    if isinstance(expr, Var):
        return mapping.get(expr.name, expr)
    if isinstance(expr, (Literal, Builtin)):
        return expr
    if isinstance(expr, BinOp):
        return BinOp(expr.op, substitute(expr.left, mapping), substitute(expr.right, mapping))
    if isinstance(expr, Ternary):
        return Ternary(
            substitute(expr.cond, mapping),
            substitute(expr.if_true, mapping),
            substitute(expr.if_false, mapping),
        )
    if isinstance(expr, FuncCall):
        return FuncCall(expr.name, [substitute(a, mapping) for a in expr.args])
    return expr


def compose_index_maps(outer, inner):
    """Compose two adjacent IndexMapOps into one.

    Substitutes the outer's placeholder coords with the inner's coord_map.
    Both must be single-source (multi-source × multi-source composition is
    not supported — the optimization rule rejects that case).

    Returns a new ``IndexMapOp`` with:
    - ``out_shape`` from ``outer``
    - one source whose ``input_idx = inner.sources[0].input_idx``
    - ``coord_map`` = outer's coord_map composed with inner's coord_map
    - ``select`` = composed conjunction of outer's and inner's selects (if any)
    """
    from deplodock.compiler.ops import IndexMapOp, IndexSource

    if len(outer.sources) != 1 or len(inner.sources) != 1:
        raise ValueError("compose_index_maps only supports single-source IndexMaps")

    outer_src = outer.sources[0]
    inner_src = inner.sources[0]

    # Mapping: outer's placeholder for axis i → outer's coord_map[i] applied to inner's coords.
    # But we want the result to be expressed over the inner's input coords. So:
    #   merged_coord_map[i] = substitute(inner_src.coord_map[d], { f"out_coord_d": outer_src.coord_map[i] for d ... })
    # Actually simpler: walk the inner's coord_map, substituting its placeholders
    # with the outer's coord_map (which are themselves over the merged op's placeholders).
    outer_to_inner_mapping = {placeholder(d).name: outer_src.coord_map[d] for d in range(len(outer_src.coord_map))}
    merged_coord_map = tuple(substitute(c, outer_to_inner_mapping) for c in inner_src.coord_map)

    merged_select = None
    if outer_src.select is not None and inner_src.select is not None:
        # Both have selects — combine via logical AND, with outer's select transformed
        # through inner's coord_map.
        inner_select_under_outer = substitute(inner_src.select, outer_to_inner_mapping)
        merged_select = BinOp("&&", outer_src.select, inner_select_under_outer)
    elif outer_src.select is not None:
        merged_select = outer_src.select
    elif inner_src.select is not None:
        merged_select = substitute(inner_src.select, outer_to_inner_mapping)

    return IndexMapOp(
        out_shape=tuple(outer.out_shape),
        sources=(IndexSource(input_idx=inner_src.input_idx, coord_map=merged_coord_map, select=merged_select),),
    )
