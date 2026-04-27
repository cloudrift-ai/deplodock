"""Unroll free Loops with extent ≤ ``MAX_UNROLL`` into per-cell copies.

Free-Loop unrolling expands the loop body into ``extent`` copies, each
with the loop variable substituted by its literal value. Body-defined
SSA names get a ``_uN`` tag per cell so per-cell intermediates don't
collide. Stmts that are loop-invariant are emitted once (LICM).

Reduce loops are handled by ``005_unroll_small_reduce_loops``.
``Stage`` stmts in the body block unrolling at this level (replicating
a Stage would duplicate its smem buffer declaration).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Select, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

MAX_UNROLL = 64


def rewrite(graph: Graph, root: Node) -> Graph | None:
    body = root.op.body
    changed = False
    while True:
        new_body = _maybe_rewrite(body)
        if new_body is None:
            break
        body = new_body
        changed = True
    if not changed:
        return None
    root.op = TileOp(body=body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    new_tile_body = _unroll_in_body(tile.body)
    if new_tile_body is None:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _unroll_in_body(body: tuple) -> tuple | None:
    for i, s in enumerate(body):
        if isinstance(s, Loop):
            descended = _unroll_in_body(s.body)
            if descended is not None:
                return body[:i] + (Loop(axis=s.axis, body=descended),) + body[i + 1 :]
            if not s.is_reduce and int(s.axis.extent) <= MAX_UNROLL:
                unrolled = _unroll(s, exclude_accum=False)
                if unrolled is not None:
                    return body[:i] + tuple(unrolled) + body[i + 1 :]
    return None


def _unroll(loop: Loop, *, exclude_accum: bool) -> list[Stmt] | None:
    if _contains_stage(loop.body):
        return None
    f_dep_names = _compute_f_dep_names(loop.body, loop.axis.name, exclude_accum=exclude_accum)
    return _unroll_at_axis(loop.body, loop.axis.name, int(loop.axis.extent), f_dep_names)


def _compute_f_dep_names(body: tuple, f_axis: str, *, exclude_accum: bool) -> set[str]:
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


def _walk_all(body):
    for s in body:
        yield s
        for attr in ("body", "else_body"):
            val = getattr(s, attr, None)
            if val:
                yield from _walk_all(val)


def _unroll_at_axis(body: tuple, f_axis: str, extent: int, f_dep_names: set[str]) -> list[Stmt]:
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
    if isinstance(stmt, Select):
        for b in stmt.branches:
            if axis_name in b.select.free_vars():
                return True
    if isinstance(stmt, Cond):
        if axis_name in stmt.cond.free_vars():
            return True
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


_ = StridedLoop  # silence ruff
