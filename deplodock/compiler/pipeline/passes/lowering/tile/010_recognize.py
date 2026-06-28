"""Recognize a ``LoopOp``'s algebraic structure → lift it to a ``TileOp`` carrying a
single op-tree ``AlgebraNode`` (``Map`` / ``Monoid`` / ``Semiring``) with an **empty
schedule**.

This is the Loop-IR → Tile-IR boundary: after this pass nothing downstream traffics
in ``LoopOp``. Recognition (here) reads the algebra and lifts; scheduling
(``020_schedule``) only moves the lifted node's ``free`` axes onto the thread grid;
materialization back to loop IR happens in ``lowering/kernel`` — so the tile passes
work purely with algebra primitives.

All recognition lives in THIS one rule (no separate flash / softmax pass), in order (each
step unconditional — no knobs):

1. **Flash attention** — a softmax-then-P@V kernel (+ its clean scaled-QK producer) is
   the online-softmax *monoid* over a streaming KV reduce; rewrite the pair to one fused
   flash ``TileOp`` (the ``(m, l, O)`` twisted monoid), with its free ``(batch…, m, d)``
   axes carried on the lifted ``Map`` node. Graph rewrite — consumes the score producer.
   Recognition + construction live in the ``_flash`` helper (``try_flash``). Because the
   fusion reads the score producer's Q/K as plain ``Load``\\ s, a node that IS such a
   producer is *deferred* (left a ``LoopOp``, :func:`is_flash_score_producer`) so step 4
   doesn't lift it out from under its consumer.
2. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same input
   fuses into one streaming online-softmax ``Monoid`` loop (the ``(m, d)`` twist). The
   ``_softmax`` helper (``_fuse``).
3. **Normalize** — every remaining scalar ``Accum`` (a plain sum / max / mean) becomes
   its **degenerate** monoid (``Accum.as_monoid`` — the identity twist, no rescale). A
   semiring contraction (``Semiring.match``) keeps its ``Accum`` so the contraction
   structure survives.
4. **Lift** — peel the free (parallel) axes off the kernel and lift the per-cell compute
   into ONE :data:`AlgebraNode`: a pure pointwise body is a ``Map``; a single flat reduce
   becomes a self-contained ``Monoid`` (plain reduce / online softmax) or ``Semiring``
   (contraction) node, with any pre/post pointwise stmts wrapped as a projection ``Map``
   over it. The free axes ride on the ``TileOp``'s ``Schedule`` (the root's concern, not the
   node's); ``020_schedule`` maps them onto the grid. A cell the lift can't cleanly factor (no
   reduce, several reduces, or a nested non-flash reduce) stays a flat ``Map`` of the
   per-cell stmts — still a valid op-tree node, just not factored.

Flash must precede online-softmax which must precede normalize: each later step consumes
the ``Accum``\\ s an earlier one matches. A **symbolic** axis (dynamic ``seq_len``) is left
un-lifted (the scalar ``Tile`` decode needs static extents) — the ``LoopOp`` stays put for
the dynamic-shape tier (``plans/tile-ir-rebuild.md``).

This is case-by-case recognition today (flash / online-softmax / contraction patterns);
the intent is to grow it toward ONE algorithmic algebra recognizer.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Monoid, Write
from deplodock.compiler.ir.stmt.algebra import AlgebraNode, Map, Semiring
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import Schedule, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._flash import is_flash_score_producer, try_flash
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import _fuse

PATTERN = [Pattern("root", LoopOp)]


def _normalize(stmts: list[Stmt]) -> tuple[list[Stmt], bool]:
    """Rewrite each plain reduce ``Loop``'s ``Accum``\\ s to their degenerate
    ``Monoid`` (deep), seeded by explicit ``Init`` stmts before the loop. A semiring
    contraction (``Semiring.match``) keeps its ``Accum`` — degenerate-monoidizing it would
    lose the contraction structure the matmul tier reads. The ``Init`` seeds are
    load-bearing only on the **flat-``Map`` fallback** path (a cell ``_lift`` keeps as
    loop-IR verbatim, where the ``Monoid`` renders standalone and its reassignments need
    the carried state declared); the lifted path strips them (``_lift_cell``) and reseeds
    from the carrier's fold ``Accum``\\ s. Returns ``(stmts, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and Semiring.match(s) is None:
            if any(isinstance(x, Accum) for x in s.body):
                new_body = [x.as_monoid() if isinstance(x, Accum) else x for x in s.body]
                out.extend(init for x in new_body if isinstance(x, Monoid) for init in x.state.inits())
                out.append(Loop(axis=s.axis, body=Body(tuple(new_body)), unroll=s.unroll))
                changed = True
                continue
        if s.nested():
            subs = []
            for b in s.nested():
                nb, ch = _normalize(list(b))
                subs.append(Body(tuple(nb)))
                changed = changed or ch
            s = s.with_bodies(tuple(subs))
        out.append(s)
    return out, changed


# --------------------------------------------------------------------------- #
# Lift — peel the free (parallel) axes off and lift the per-cell compute into ONE
# op-tree node carrying those free axes.
# --------------------------------------------------------------------------- #


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(free_axes, per_cell_stmts)``.

    A leading run of non-``Loop`` stmts (loop-invariant loads hoisted above the nest) is
    pulled into the per-cell body; then the outer chain of single-child **free** loops
    becomes the parallel axes. The chain stops at the first reduce loop or branch —
    everything from there down is the per-cell body (the fold and its epilogue / output
    sweep), run serially by one thread per cell."""
    stmts = list(body)
    i = 0
    while i < len(stmts) and not isinstance(stmts[i], Loop):
        i += 1
    prefix, rest = stmts[:i], stmts[i:]
    axes = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        axes.append(cur[0].axis)
        cur = list(cur[0].body)
    return axes, prefix + list(cur)


def _reduce_in(stmts) -> bool:
    """Any reduce ``Loop`` reachable in ``stmts`` (deep)."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            return True
        if any(_reduce_in(b) for b in s.nested()):
            return True
    return False


def _reads(stmts) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep — through ``deps`` + nested bodies)."""
    out: set[str] = set()
    for s in stmts:
        out.update(s.deps())
        for b in s.nested():
            out |= _reads(list(b))
    return out


def _clean_contraction(rloop: Loop, sem: Semiring) -> bool:
    """True iff lowering ``sem`` reproduces ``rloop`` — a clean contraction whose lift
    multiplies the operand loads **directly**. ``Semiring.match`` reconstructs each operand
    as a bare one-``Load`` ``Map``, so a contraction with per-operand preprocessing (e.g.
    pre-scaled inputs: ``sum_k (x·s)·(y·s)``) would silently drop that compute. Such a loop
    stays a flat ``Map`` instead (its loop-IR body lowers verbatim, preserving the scale)."""
    body = list(rloop.body)
    lift = next((s for s in body if isinstance(s, Assign) and s.name == sem.fold.value), None)
    if lift is None:
        return False
    load_names = [op.out for op in sem.operands]
    loads = [s for s in body if isinstance(s, Load)]
    # Body must be exactly the operand loads + the lift Assign + the fold Accum, and the
    # lift must read precisely those loads (no intervening per-operand compute).
    return len(body) == len(loads) + 2 and len(loads) == len(load_names) and set(lift.args) == set(load_names)


def _lift_reduce(rloop: Loop, pre_reduce: tuple[Stmt, ...] = ()) -> AlgebraNode | None:
    """Lift a single FLAT reduce ``Loop`` to a self-contained op-tree node: a
    ``Semiring`` for a clean contraction (``Semiring.match`` + :func:`_clean_contraction`),
    else a ``Monoid`` whose ``partial`` is the loop body's compute (everything but the
    carrier) wrapped as a ``Map``. ``pre_reduce`` are loop-invariant prologue stmts that feed
    the partial (e.g. a scale-constant load) — prepended to the partial ``Map`` so they lower
    INSIDE the loop (a clean contraction has none, its operands being direct loads). Returns
    ``None`` if the carrier can't be read (the caller falls back to a flat ``Map``)."""
    sem = Semiring.match(rloop)
    if sem is not None and _clean_contraction(rloop, sem):
        return sem
    carriers = [s for s in rloop.body if isinstance(s, (Monoid, Accum))]
    if len(carriers) != 1:
        return None
    carrier = carriers[0]
    mono = carrier if isinstance(carrier, Monoid) else carrier.as_monoid()
    partial_stmts = pre_reduce + tuple(s for s in rloop.body if s is not carrier)
    return replace(mono, partial=(Map(body=partial_stmts),), axis=rloop.axis)


def _lift_cell(cell: list[Stmt], free: list, output: str) -> AlgebraNode:
    """Lift the per-cell stmts into one op-tree node. A pure pointwise cell (no reduce) is
    a flat ``Map``; a single flat reduce becomes a ``Monoid`` / ``Semiring`` node, bare when
    its only epilogue is the grid-cell output ``Write`` (materialize stores ``out`` as glue)
    or wrapped in a projection ``Map`` (the pre/post pointwise stmts) otherwise. A cell with
    no, or several, or a nested reduce stays a flat ``Map`` (still a valid op-tree node)."""
    reduces = [i for i, s in enumerate(cell) if isinstance(s, Loop) and s.is_reduce]
    if len(reduces) != 1:
        return Map(body=tuple(cell))
    idx = reduces[0]
    rloop = cell[idx]
    if _reduce_in(list(rloop.body)):
        return Map(body=tuple(cell))  # nested (non-flash) reduce — keep loop-IR form
    # Route the loop-invariant prologue (stmts above the reduce, sans the regenerated ``Init``
    # seeds): a stmt feeding the reduce must lower inside the loop (``pre_reduce``); one feeding
    # only the epilogue stays after it. A stmt feeding BOTH can't be placed by reordering — keep
    # the whole cell as a flat ``Map`` (its loop-IR order is preserved verbatim).
    before = [s for s in cell[:idx] if not isinstance(s, Init)]
    after = list(cell[idx + 1 :])
    before_defs = {n for s in before for n in s.defines()}
    feeds_reduce = bool(before_defs & _reads(list(rloop.body)))
    feeds_epilogue = bool(before_defs & _reads(after))
    if feeds_reduce and feeds_epilogue:
        return Map(body=tuple(cell))
    pre_reduce = tuple(before) if feeds_reduce else ()
    pre_epilogue = () if feeds_reduce else tuple(before)
    node = _lift_reduce(rloop, pre_reduce)
    if node is None:
        return Map(body=tuple(cell))
    grid_index = tuple(Var(ax.name) for ax in free)
    bare = (
        not before
        and len(after) == 1
        and isinstance(after[0], Write)
        and after[0].is_scalar
        and after[0].value == node.out
        and after[0].output == output
        and after[0].index == grid_index
    )
    if bare:
        return node  # materialize writes ``node.out`` at the grid cell
    return Map(source=node, body=pre_epilogue + tuple(after))


def _lift(stmts: list[Stmt], output: str) -> tuple[AlgebraNode, tuple]:
    """Peel the free axes and lift the per-cell compute, returning ``(root node, free
    axes)``. The free axes are the schedule's (carried on the ``TileOp``, not the node);
    ``020_schedule`` maps them onto the grid."""
    free, cell = _peel(Body(tuple(stmts)))
    return _lift_cell(cell, free, output), tuple(free)


def rewrite(match: Match, root: Node) -> TileOp | Graph | None:
    # (1) Flash attention — a graph rewrite that fuses a softmax-then-P@V kernel with its
    # scaled-QK producer. Tried first on every node; flash precedes online-softmax precedes
    # normalize, each consuming the Accums the next would match.
    graph = match.graph
    flash = try_flash(graph, root)
    if flash is not None:
        return flash
    # (2) Defer a flash score producer: the general lift below would turn this scaled-QK
    # matmul into a ``TileOp`` before its softmax-then-P@V consumer fuses, and that fusion
    # reads the producer's Q/K as plain ``Load``s. Leave it a ``LoopOp`` until the consumer
    # has had its chance to consume it (a later scan re-visits this node, by then removed).
    if is_flash_score_producer(graph, root):
        raise RuleSkipped("flash score producer — defer to its consumer's fusion")
    loop: LoopOp = root.op
    # The scalar tier materializes a static thread grid (``Tile`` decodes literal extents).
    # A symbolic axis (dynamic ``seq_len``) needs the runtime-arg decode — leave the
    # ``LoopOp`` un-lifted for the dynamic-shape tier.
    if any(not a.extent.is_static for a in loop.axes):
        raise RuleSkipped("symbolic axis — scalar tier needs static extents (dynamic tier not built)")
    fused, _ = _fuse(loop.body)
    normed, _ = _normalize(list(fused))
    node, free = _lift(normed, root.output.name)
    return TileOp(op=node, schedule=Schedule(free=free), name=loop.name)
