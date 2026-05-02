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

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _place_inits_in_body(root.op.body)
    if new_body == root.op.body:
        raise RuleSkipped("no Accum needs an Init placed (already placed or no reduce in body)")
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
    own scope when we recurse into it.

    "Reduce loop" here is the recursive notion: a Loop whose body has an
    Accum directly, *or* a nested reduce-Loop. This catches the
    K-chunked matmul shape where ``K_o``'s immediate body is ``[Stage,
    ..., Loop(K_i, reduce, body=[..., Accum])]`` — the inner Accum
    accumulates across ``K_o`` too, so ``K_o`` is crossable for Init
    placement and the Init lands at the surrounding Tile body head."""
    out: list[Accum] = []
    if isinstance(stmt, Accum):
        out.append(stmt)
    elif isinstance(stmt, (Loop, StridedLoop)):
        if _is_reduce_recursive(stmt):
            for child in stmt.body:
                out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, Cond):
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
        for child in stmt.else_body:
            out.extend(_accums_under_reduces_only(child))
    return out


def _is_reduce_recursive(loop: Loop | StridedLoop) -> bool:
    """Recursive reduce check (Init-scoping only). A Loop is "crossable"
    iff its body either has an Accum directly OR forms a pure
    reduce-passthrough — no Write / output-escape stmts in body, and at
    least one nested reduce-Loop carries the Accum.

    The Write check is what distinguishes:

    - **Matmul K-chunked** (``K_o`` body = [Stage, Stage, Loop(K_i,
      reduce)]): no Write directly in body → crossable, Init lives at
      the surrounding Tile.
    - **SDPA per-output free loop** (``a4`` body = [Loop(reduce, ...),
      Write[..., a4]]): the Write escapes per-iteration, so the Accum
      must reset per ``a4`` → not crossable, Init lives inside ``a4``."""
    has_inner_reduce = False
    for s in loop.body:
        if isinstance(s, Accum):
            return True
        if isinstance(s, Write):
            return False
        if isinstance(s, (Loop, StridedLoop)) and _is_reduce_recursive(s):
            has_inner_reduce = True
    return has_inner_reduce


def _recurse(stmt: Stmt) -> Stmt:
    """Descend into block-structured stmts so nested free Loops get their
    own Init placement at their own scope. Same recursive-reduce
    distinction as ``_accums_under_reduces_only``: a Loop that
    transitively wraps an Accum (matmul ``K_o`` chunking) descends
    without opening a new scope. ``StridedLoop`` is treated identically
    — SDPA's per-output free StridedLoop wraps a free Loop chain plus a
    Write, so it's not reduce-crossable and must open a scope so the
    inner Accum's Init lands inside (resetting per-iteration)."""
    if isinstance(stmt, (Loop, StridedLoop)):
        if _is_reduce_recursive(stmt):
            return replace(stmt, body=tuple(_recurse(c) for c in stmt.body))
        # Free loop — its body is its own scope; place Inits there.
        return replace(stmt, body=_place_inits_in_scope(stmt.body))
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_recurse(c) for c in stmt.body),
            else_body=tuple(_recurse(c) for c in stmt.else_body),
        )
    return stmt
