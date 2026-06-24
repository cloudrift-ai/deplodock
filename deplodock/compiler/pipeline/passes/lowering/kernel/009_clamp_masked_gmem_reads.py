"""Clamp gmem-direct masked-tile operand reads (scalar tier).

A masked output axis (a non-divisor / symbolic ``M`` | ``N``) tiles past its real
extent; ``free_tile`` wraps the per-cell store in a boundary ``Cond(coord < real_extent)``
(``enumeration/_build._apply_masked_guards``), but the operand ``Load``s in the shared
K loop stay UNGUARDED. For a *staged* operand the cooperative slab fill is clamped
(``_stage_expand`` / ``Source.gmem_extents``); a gmem-DIRECT operand — one with no
intra-CTA fan-in reuse to stage (``enumeration/_stage``), or whose slab would blow the
smem budget — has no such clamp, so it reads ``coord`` past the buffer extent for the
masked overhang rows → ``CUDA_ERROR_ILLEGAL_ADDRESS``.

This pass restores the read-side clamp the staged path already has: each gmem-direct
``Load`` index dim that EQUALS a masked output coord ``coord`` becomes
``(coord < bound) ? coord : bound-1`` — a harmless in-bounds duplicate (the masked
output cell it feeds is never written, so the wrong value is dropped; the "M/N
edge-clamp" the ``plans/tile-ir-block-dag.md`` IR describes as ``AccessMap.clamp``).
A masked *reduce* (symbolic-K) axis is NOT an output mask — its overhang must be
zero-filled, not edge-clamped (a duplicate would corrupt the sum) — so only Conds whose
body stores a ``Write`` indexing ``coord`` are treated as output masks. The atom / MMA
tier has its own gmem-direct clamp (``005_lower_atom_tile``), so a kernel carrying an
``Mma`` is left untouched here.

Runs BEFORE ``010_split_register_axes`` (one logical cell — one coord, one guard, one
load — then replicated per cell with the clamp already in place); ``Load.index`` is
still per-dim here (the flatten to ``buf[d0*stride + d1]`` happens at the CUDA render),
so the match is exact dim equality, no subtree search.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr
from deplodock.compiler.ir.stmt import Body, Cond, Load, Mma, Stmt, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def _subexprs(e: Expr) -> tuple[Expr, ...]:
    """The direct child sub-expressions of ``e`` (frozen ``Expr`` nodes)."""
    if isinstance(e, BinaryExpr):
        return (e.left, e.right)
    if isinstance(e, TernaryExpr):
        return (e.cond, e.if_true, e.if_false)
    children = getattr(e, "expr", None)  # CastExpr
    if isinstance(children, Expr):
        return (children,)
    args = getattr(e, "args", None)  # FuncCallExpr
    if args:
        return tuple(a for a in args if isinstance(a, Expr))
    return ()


def _contains(e: Expr, sub: Expr) -> bool:
    """True iff ``sub`` occurs as ``e`` or one of its sub-expressions."""
    return e == sub or any(_contains(c, sub) for c in _subexprs(e))


def _masked_output_coords(body: Body) -> list[tuple[Expr, Expr]]:
    """``(coord, bound)`` for each output-mask boundary ``Cond`` — a ``coord < bound``
    whose guarded body stores a ``Write`` indexing ``coord`` (so it masks an OUTPUT
    axis, not a symbolic-K reduce). De-duplicated by structural identity."""
    pairs: list[tuple[Expr, Expr]] = []
    for s in body.iter():
        if not (isinstance(s, Cond) and isinstance(s.cond, BinaryExpr) and s.cond.op == "<"):
            continue
        coord, bound = s.cond.left, s.cond.right
        if not coord.free_vars():
            continue
        guards_a_store = any(_contains(idx, coord) for b in s.nested() for w in b.iter() if isinstance(w, Write) for idx in w.index)
        if guards_a_store and (coord, bound) not in pairs:
            pairs.append((coord, bound))
    return pairs


def _clamp_load(load: Load, pairs: list[tuple[Expr, Expr]]) -> Load:
    """Clamp each index dim that equals a masked output coord to ``[0, bound)``."""
    new_index = []
    changed = False
    for dim in load.index:
        repl = dim
        for coord, bound in pairs:
            if dim == coord:
                repl = TernaryExpr(
                    cond=BinaryExpr("<", coord, bound),
                    if_true=coord,
                    if_false=BinaryExpr("-", bound, Literal(1, "int")),
                )
                changed = True
                break
        new_index.append(repl)
    return dc_replace(load, index=tuple(new_index)) if changed else load


def _walk(body: Body, pairs: list[tuple[Expr, Expr]]) -> Body:
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Load):
            out.append(_clamp_load(s, pairs))
        elif s.nested():
            out.append(s.with_bodies(tuple(_walk(b, pairs) for b in s.nested())))
        else:
            out.append(s)
    return Body(out)


def rewrite(root: Node) -> TileOp:
    op: TileOp = root.op
    if any(isinstance(s, Mma) for s in op.body.iter()):
        raise RuleSkipped("atom/MMA tier — gmem-direct clamp owned by 005_lower_atom_tile")
    pairs = _masked_output_coords(op.body)
    if not pairs:
        raise RuleSkipped("no masked output-tile boundary guard — nothing to clamp")
    new_body = _walk(op.body, pairs)
    if new_body == op.body:
        raise RuleSkipped("no gmem-direct read indexes a masked output coord")
    return TileOp(body=new_body, name=op.name, knobs=dict(op.knobs))
