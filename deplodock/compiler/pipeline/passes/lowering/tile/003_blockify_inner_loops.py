"""K-block — split each innermost serial reduce ``Loop`` into outer +
inner reduce pair (inner extent ``K_TILE``).

Pre-rewrite::

    for k in 0..K (K = 5632):  # reduce
        ... body ...

Post-rewrite (``K_TILE=32``)::

    for k_o in 0..K/K_TILE:  # reduce
        for k_i in 0..K_TILE:  # reduce
            ... body[k -> k_o*K_TILE + k_i] ...

Only fires on innermost serial reduce ``Loop`` stmts (``StridedLoop`` is
left alone — those are cooperative and have their own iteration shape).
Idempotent: a Loop whose body already contains a nested Loop is skipped.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

K_TILE = 32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body, changed = _walk(root.op.body)
    if not changed:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _walk(stmts: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], bool]:
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Tile):
            inner, c = _walk(s.body)
            if c:
                out.append(Tile(axes=s.axes, body=inner))
                changed = True
            else:
                out.append(s)
        elif isinstance(s, Loop):
            split = _maybe_split(s)
            if split is not None:
                out.append(split)
                changed = True
            else:
                inner, c = _walk(s.body)
                if c:
                    out.append(Loop(axis=s.axis, body=inner))
                    changed = True
                else:
                    out.append(s)
        elif isinstance(s, StridedLoop):
            inner, c = _walk(s.body)
            if c:
                out.append(StridedLoop(axis=s.axis, start=s.start, step=s.step, body=inner))
                changed = True
            else:
                out.append(s)
        elif isinstance(s, Cond):
            b1, c1 = _walk(s.body)
            b2, c2 = _walk(s.else_body)
            if c1 or c2:
                out.append(Cond(cond=s.cond, body=b1, else_body=b2))
                changed = True
            else:
                out.append(s)
        else:
            out.append(s)
    return tuple(out), changed


def _maybe_split(loop: Loop) -> Loop | None:
    if not loop.is_reduce:
        return None
    ext = int(loop.axis.extent)
    if ext <= K_TILE or ext % K_TILE != 0:
        return None
    if any(isinstance(c, (Loop, StridedLoop)) for c in loop.body):
        return None  # already nested
    outer = Axis(f"{loop.axis.name}_o", ext // K_TILE)
    inner = Axis(f"{loop.axis.name}_i", K_TILE)
    sigma = Sigma({loop.axis.name: Var(outer.name) * Literal(K_TILE, "int") + Var(inner.name)})
    new_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
    return Loop(axis=outer, body=(Loop(axis=inner, body=new_body),))


def _id(name: str) -> str:
    return name
