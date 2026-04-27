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
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Select, Stmt, StridedLoop, Tile
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
    """Unroll loop.body F.extent times with literal substitution and SSA
    tagging. Pre-computes the F-dependent name set via fixpoint so
    F-invariant intermediate stmts (their values are the same across all
    F cells) keep their original SSA names — only F-dep names get tagged.
    Returns None when the body contains a Stage (replicating a Stage
    would re-declare its smem buffer with the same name)."""
    if _contains_stage(loop.body):
        return None
    f_dep_names = _compute_f_dep_names(loop.body, loop.axis.name, exclude_accum=loop.is_reduce)
    return _unroll_at_axis(loop.body, loop.axis.name, int(loop.axis.extent), f_dep_names)


def _compute_f_dep_names(body: tuple, f_axis: str, *, exclude_accum: bool) -> set[str]:
    """Fixpoint: a name is F-dep iff its producing stmt references
    ``f_axis`` or reads any already-F-dep name."""
    f_deps: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in _walk_all(body):
            if exclude_accum and isinstance(s, Accum):
                continue
            name = getattr(s, "name", None)
            if not isinstance(name, str) or name in f_deps:
                continue
            if _refs_axis(s, f_axis) or _reads_dep(s, f_deps):
                f_deps.add(name)
                changed = True
    return f_deps


def _walk_all(body: tuple):
    for s in body:
        yield s
        for attr in ("body", "else_body"):
            val = getattr(s, attr, None)
            if val:
                yield from _walk_all(val)


def _unroll_at_axis(body: tuple, f_axis: str, extent: int, f_dep_names: set[str]) -> list[Stmt]:
    """Per stmt:

    - F-invariant non-Loop: emit once unchanged.
    - F-invariant Loop: emit once unchanged.
    - F-dependent Loop: recurse (loop-swap pattern — body is unrolled
      inside the same Loop wrapper at this level).
    - F-dependent non-Loop: defer to per-cell replication.
    """
    out: list[Stmt] = []
    pending: list[Stmt] = []
    for s in body:
        is_dep = _stmt_is_f_dep(s, f_axis, f_dep_names)
        if not is_dep:
            out.append(s)
        elif isinstance(s, Loop) and not _contains_stage(s.body):
            inner = _unroll_at_axis(s.body, f_axis, extent, f_dep_names)
            out.append(Loop(axis=s.axis, body=tuple(inner)))
        else:
            pending.append(s)

    for i in range(extent):
        sigma = Sigma({f_axis: Literal(i, "int")})
        tag = f"_u{i}"

        def rename(n: str, t: str = tag, deps: set[str] = f_dep_names) -> str:
            return n + t if n in deps else n

        for s in pending:
            out.append(s.rewrite(rename, sigma))
    return out


def _stmt_is_f_dep(stmt: Stmt, f_axis: str, f_dep_names: set[str]) -> bool:
    if _refs_axis(stmt, f_axis):
        return True
    if _reads_dep(stmt, f_dep_names):
        return True
    if isinstance(stmt, Loop):
        for child in stmt.body:
            if _stmt_is_f_dep(child, f_axis, f_dep_names):
                return True
    return False


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
