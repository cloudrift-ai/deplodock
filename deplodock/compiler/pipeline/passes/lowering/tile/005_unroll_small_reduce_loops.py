"""Unroll innermost reduce ``Loop``s with extent ≤ ``MAX_UNROLL``.

Same body-replication idea as ``004_unroll_small_free_loops``, but
``Accum`` names are *not* tagged per cell — the single accumulator is
incremented ``extent`` times in sequence (manual unroll). After this
rule, an outer reduce ``Loop`` whose body was just an inner reduce
``Loop`` ends up with ``Accum`` directly in its immediate body, so
``Loop.is_reduce`` becomes True and downstream ``place_inits`` hoists
the ``Init`` to the right enclosing scope.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Select, Stmt, Tile
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
            if s.is_reduce and int(s.axis.extent) <= MAX_UNROLL and not _contains_stage(s.body):
                unrolled = _unroll(s)
                if unrolled is not None:
                    return body[:i] + tuple(unrolled) + body[i + 1 :]
    return None


def _unroll(loop: Loop) -> list[Stmt]:
    f_dep_names = _compute_f_dep_names(loop.body, loop.axis.name)
    return _unroll_at_axis(loop.body, loop.axis.name, int(loop.axis.extent), f_dep_names)


def _compute_f_dep_names(body: tuple, f_axis: str) -> set[str]:
    """Bottom-up fixpoint — Accum is excluded so a reduce loop's
    accumulator name is shared across all unrolled cells."""
    f_deps: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in _walk_all(body):
            if isinstance(s, Accum):
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
        is_dep = _stmt_is_f_dep(s, f_axis, f_dep_names) or isinstance(s, Accum)
        if not is_dep:
            out.append(s)
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
