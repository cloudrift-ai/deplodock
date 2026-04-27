"""Place explicit ``Init`` Stmts at the correct scope for every Accum.

Runs before ``001_materialize_tile`` so materialize sees IR where each
Accum's lifetime is explicit. Removes the need for materialize to make
implicit scoping decisions.

Scope rule: walk up from each Accum, crossing reduce Loops freely;
stop at the nearest enclosing free Loop or at the Tile body. Insert
``Init(name, op)`` at that scope, just before the loop chain that
contains the Accum.

Why this is correct:

- **Matmul** (``Loop(k_o, reduce) > Loop(k_i, reduce) > Accum``):
  walk crosses both reduces, hits Tile. Init goes at Tile body head;
  ``acc`` persists across ``k_o`` iterations as needed.

- **SDPA k_reduce** (``Loop(a3, free) > Loop(a4, reduce) > Accum``):
  walk crosses ``a4`` (reduce ✓), stops at ``a3`` (free ✗). Init goes
  at the start of ``a3``'s body; ``acc`` resets per a3 iteration.

The renderer's ``RenderCtx.explicit_inits`` mechanism suppresses the
per-Loop default init when an explicit ``Init`` exists for that name,
so the standard render path produces the right code unmodified.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _place_inits_in_body(root.op.body)
    if new_body == root.op.body:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _place_inits_in_body(body: tuple) -> tuple:
    """Walk Tile bodies recursively. For each Tile we encounter, scan its
    body for the Accums whose Init should land at this Tile's body head
    (those whose nearest free-loop ancestor is the Tile itself), and for
    Accums under nested free Loops (Init goes inside the free loop body)."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Tile):
            new_tile_body = _place_inits_in_scope(s.body)
            out.append(Tile(axes=s.axes, body=new_tile_body))
        else:
            out.append(s)
    return tuple(out)


def _place_inits_in_scope(body: tuple) -> tuple:
    """At each scope (Tile body, or a free-Loop body), collect all Accum
    names whose Init belongs HERE (= no free-Loop interposed between this
    scope and the Accum). Insert an Init for each at the start of the
    scope body. Then recurse into any nested free Loops to repeat at
    their scope."""
    inits_here: dict[str, Accum] = {}
    for s in body:
        for a in _accums_under_reduces_only(s):
            inits_here.setdefault(a.name, a)

    new_body: list[Stmt] = []
    for s in body:
        new_body.append(_recurse(s))

    init_stmts: list[Stmt] = [Init(name=n, op=a.op) for n, a in inits_here.items()]
    return tuple(init_stmts + new_body)


def _accums_under_reduces_only(stmt: Stmt) -> list[Accum]:
    """Collect Accums reachable from ``stmt`` through reduce loops only.
    Stops at free loops — those Accums will be placed at the free loop's
    own scope when we recurse into it."""
    out: list[Accum] = []
    if isinstance(stmt, Accum):
        out.append(stmt)
    elif isinstance(stmt, (Loop, StridedLoop)):
        if stmt.is_reduce:
            for child in stmt.body:
                out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, Cond):
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
        for child in stmt.else_body:
            out.extend(_accums_under_reduces_only(child))
    return out


def _recurse(stmt: Stmt) -> Stmt:
    """Descend into block-structured stmts so nested free Loops get their
    own Init placement at their own scope."""
    if isinstance(stmt, Loop):
        if stmt.is_reduce:
            return Loop(axis=stmt.axis, body=tuple(_recurse(c) for c in stmt.body))
        # Free Loop — its body is its own scope; place Inits there.
        return Loop(axis=stmt.axis, body=_place_inits_in_scope(stmt.body))
    if isinstance(stmt, StridedLoop):
        return StridedLoop(axis=stmt.axis, start=stmt.start, step=stmt.step, body=tuple(_recurse(c) for c in stmt.body))
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_recurse(c) for c in stmt.body),
            else_body=tuple(_recurse(c) for c in stmt.else_body),
        )
    return stmt
