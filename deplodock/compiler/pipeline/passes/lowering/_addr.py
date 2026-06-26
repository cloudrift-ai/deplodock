"""Flat-address `Expr` builders — fold-aware `sum` / `product` over int / `Expr` terms, used to
construct σ-tiled load/store indices in the lowering passes (`enumeration/_build` warp-tier σ-tiling,
`assembly/_assemble` carrier realization). Generic Expr algebra — no flash / attention / dialect
dependency. Lives in `lowering/` (a sibling of `_masking` / `_predicates`) so both the enumeration and
assembly layers import it without crossing the enumeration↔assembly boundary.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal


def add(*terms) -> Expr:
    """Sum int / Expr terms into one Expr (dropping literal zeros)."""
    out = None
    for t in terms:
        e = Literal(t, "int") if isinstance(t, int) else t
        if isinstance(e, Literal) and e.value == 0:
            continue
        out = e if out is None else BinaryExpr("+", out, e)
    return out if out is not None else Literal(0, "int")


def mul(a, b: int) -> Expr:
    """``a · b`` as an Expr, folding the ``b in {0, 1}`` degenerate cases."""
    return add() if b == 0 else (a if b == 1 else BinaryExpr("*", a if not isinstance(a, int) else Literal(a, "int"), Literal(b, "int")))
