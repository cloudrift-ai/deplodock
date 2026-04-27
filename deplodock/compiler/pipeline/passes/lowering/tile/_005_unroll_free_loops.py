"""Unroll small-extent free Loops nested directly inside a TileOp body.

Runs after ``004_stage_inputs`` so staging decisions are made on the
un-unrolled form (one parameterized Load per buffer with its index
free-vars intact for cache-axis derivation). After staging, this pass
expands free Loops with extent ≤ ``_MAX_UNROLL`` into literal copies,
factoring axis-invariant stmts out of the copies (LICM).

Pre-rewrite::

    Tile(...):
        ... staged setup ...
        Loop(j, free, extent=4):
            Loop(k, reduce):
                ... j-invariant stmt A ...
                ... j-dependent stmt B(j) ...
                Accum(acc)
            Write[..., j] = acc

Post-rewrite::

    Tile(...):
        ... staged setup ...
        Loop(k, reduce):
            ... j-invariant stmt A (ONCE) ...
            ... j-dependent stmt B(0) ...
            Accum(acc_u0)
            ... j-dependent stmt B(1) ...
            Accum(acc_u1)
            ... j-dependent stmt B(2) ...
            Accum(acc_u2)
            ... j-dependent stmt B(3) ...
            Accum(acc_u3)
        Write[..., 0] = acc_u0
        Write[..., 1] = acc_u1
        Write[..., 2] = acc_u2
        Write[..., 3] = acc_u3

Init scoping is handled by the downstream ``kernel/000_place_inits``
pass; unrolling produces distinct SSA names per cell, so each cell's
Init lands at Tile body head and the per-cell accumulators are kept
separate.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Select, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_MAX_UNROLL = 64


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    new_tile_body = _unroll_in_body(tile.body)
    if new_tile_body is None:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _unroll_in_body(body: tuple) -> tuple | None:
    """Find the innermost unrollable Loop (free or reduce) and unroll it.
    Returns None when no unroll fires (rule terminates)."""
    for i, s in enumerate(body):
        if isinstance(s, Loop):
            descended = _unroll_in_body(s.body)
            if descended is not None:
                return body[:i] + (Loop(axis=s.axis, body=descended),) + body[i + 1 :]
            if int(s.axis.extent) <= _MAX_UNROLL:
                unrolled = _unroll(s)
                if unrolled is not None:
                    return body[:i] + tuple(unrolled) + body[i + 1 :]
    return None


def _unroll(loop: Loop) -> list[Stmt] | None:
    """Replicate ``loop.body`` ``loop.axis.extent`` times, with literal
    substitution and SSA tagging. F-invariant stmts emit once before
    the unrolled cells. Returns None when no F-invariant stmts exist
    (no LICM win — pure code expansion would just bloat the kernel),
    or when the body contains a Stage (replicating a Stage would
    re-declare its smem buffer with the same name)."""
    f_axis = loop.axis.name

    if _contains_stage(loop.body):
        return None

    # When unrolling a REDUCE loop, the Accum name represents a single
    # running accumulator shared across all unrolled cells (one reduction
    # in total). Don't rename it. When unrolling a FREE loop, each cell
    # is an independent reduction, so the Accum name DOES get renamed
    # per cell (acc -> acc_u0, acc_u1, ...).
    f_dep_names: set[str] = set()
    classified: list[tuple[Stmt, bool]] = []
    for s in loop.body:
        is_dep = _refs_axis(s, f_axis) or _reads_dep(s, f_dep_names)
        classified.append((s, is_dep))
        if is_dep:
            for name in _writes(s, exclude_accum=loop.is_reduce):
                f_dep_names.add(name)

    # Unroll fires unconditionally — F-invariant stmts (when present)
    # emit once at the front; F-dependent stmts emit ``extent`` copies.
    # Pure code expansion (no F-invariants) is still useful: NVCC may
    # schedule the unrolled body better, and downstream passes can do
    # CSE / register allocation across the unrolled cells.

    out: list[Stmt] = [s for s, dep in classified if not dep]
    extent = int(loop.axis.extent)
    for i in range(extent):
        sigma = Sigma({f_axis: Literal(i, "int")})
        tag = f"_u{i}"

        def rename(n: str, t: str = tag, deps: set[str] = f_dep_names) -> str:
            return n + t if n in deps else n

        for s, dep in classified:
            if dep:
                out.append(s.rewrite(rename, sigma))
    return out


def _contains_stage(stmts: tuple) -> bool:
    for s in stmts:
        if isinstance(s, Stage):
            return True
        for attr in ("body", "else_body"):
            val = getattr(s, attr, None)
            if val is None:
                continue
            if _contains_stage(val):
                return True
    return False


def _refs_axis(stmt: Stmt, axis_name: str) -> bool:
    """True if any Expr in ``stmt`` (or any nested body) references ``axis_name``.

    Inspects each Stmt subtype's Expr-bearing fields:

    - ``Load`` / ``Write``: ``index`` tuple of Exprs.
    - ``Select``: each branch's ``select`` predicate.
    - ``Cond``: ``cond`` predicate (its body / else_body recursed below).
    - ``Stage``: ``origin`` and ``source_index_template`` tuples of Exprs.
    """
    if isinstance(stmt, Select):
        for b in stmt.branches:
            if axis_name in b.select.free_vars():
                return True
    if isinstance(stmt, Cond):
        if axis_name in stmt.cond.free_vars():
            return True
    # Stage's Expr fields (origin, source_index_template) — we don't
    # import Stage at module top because the rule lives outside ir/tile,
    # so check by attr name instead.
    for attr in ("index", "origin", "source_index_template"):
        val = getattr(stmt, attr, None)
        if val is None:
            continue
        for e in val:
            if axis_name in e.free_vars():
                return True
    for attr in ("body", "else_body"):
        val = getattr(stmt, attr, None)
        if val is None:
            continue
        for child in val:
            if _refs_axis(child, axis_name):
                return True
    return False


def _reads_dep(stmt: Stmt, dep_names: set[str]) -> bool:
    if not dep_names:
        return False
    return any(d in dep_names for d in stmt.deps())


def _writes(stmt: Stmt, *, exclude_accum: bool = False) -> tuple[str, ...]:
    if exclude_accum and isinstance(stmt, Accum):
        return ()
    name = getattr(stmt, "name", None)
    if isinstance(name, str):
        return (name,)
    out: list[str] = []
    for attr in ("body", "else_body"):
        val = getattr(stmt, attr, None)
        if val is None:
            continue
        for child in val:
            out.extend(_writes(child, exclude_accum=exclude_accum))
    return tuple(out)
