"""Recognize a ``LoopOp``'s algebraic structure, lift it to a ``TileOp``, AND schedule it —
the merged Loop-IR → Tile-IR pass (recognition + scheduling in one rewrite, no separate
``020`` step).

This is the Loop-IR → Tile-IR boundary: after this pass nothing downstream traffics in
``LoopOp``. **Recognition** (here) reads the algebra off the body and lifts the per-cell
compute into a :class:`~deplodock.compiler.ir.stmt.algebra.Map` whose body is the **annotated
loop nest** (the reduce ``Loop`` stamped with its
:class:`~deplodock.compiler.ir.axis.AxisRole` + :class:`~deplodock.compiler.ir.stmt.algebra.Carrier`)
on a ``Kernel`` with an UNMAPPED placement; the final step hands that kernel to **scheduling**
(:func:`~deplodock.compiler.pipeline.passes.lowering.tile._schedule.schedule`, the
``_schedule`` helper) which maps the free axes onto the grid and offers the per-axis
scheduling forks (``REDUCE`` partition / ``TILE`` output tile), dispatched on the axes'
``AxisRole``. Materialization back to loop IR happens in ``lowering/kernel``.

All recognition lives in THIS one rule (no separate flash / softmax pass), in order (each
step unconditional — no knobs):

1. **Flash attention** — a softmax-then-P@V kernel (+ its clean scaled-QK producer) is
   the online-softmax twisted reduce over a streaming KV axis; rewrite the pair to one fused
   flash ``TileOp`` (the ``(m, l, O)`` ``TWISTED`` kv loop over a nested ``CONTRACTION`` score
   loop), with its free ``(batch…, m, d)`` axes carried on the schedule. Graph rewrite —
   consumes the score producer. Recognition + construction live in the ``_flash`` helper
   (``try_flash``). Because the fusion reads the score producer's Q/K as plain ``Load``\\ s, a
   node that IS such a producer is *deferred* (left a ``LoopOp``, :func:`is_flash_score_producer`)
   so step 3 doesn't lift it out from under its consumer.
2. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same input fuses
   into one streaming online-softmax loop: a ``TWISTED`` reduce ``Loop`` carrying the ``(m, d)``
   exp-family ``Carrier`` (its dissolved ``merge`` in the body). The ``_softmax`` helper
   (``_fuse``).
3. **Lift** — peel the free (parallel) axes off the kernel and lift the per-cell compute into a
   ``Map`` whose body holds the annotated reduce ``Loop`` + projection: a pure pointwise body is a
   flat ``Map``; a single flat reduce is annotated in place — ``CONTRACTION`` (clean contraction)
   / ``PLANAR`` (plain ``sum`` / ``max`` / ``mean``) / pre-annotated ``TWISTED`` (online softmax) —
   with its degenerate / exp-family ``Carrier`` and the projection after it. The free axes ride on
   the ``TileOp``'s schedule (the root's concern); ``_schedule`` maps them onto the grid. A cell
   the lift can't cleanly factor (no reduce, several reduces, or a nested non-flash reduce) stays a
   flat un-annotated ``Map`` (→ the scalar tier).

Flash must precede online-softmax which must precede the lift: each later step consumes the
``Accum``\\ s an earlier one matches. A **symbolic** axis (dynamic ``seq_len``) is left
un-lifted (the scalar ``Tile`` decode needs static extents) — the ``LoopOp`` stays put for
the dynamic-shape tier.

This is case-by-case recognition today (flash / online-softmax / contraction patterns);
the intent is to grow it toward ONE algorithmic algebra recognizer.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import AxisRole
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import Placement, TileOp, kernel_for
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._flash import is_flash_score_producer, try_flash
from deplodock.compiler.pipeline.passes.lowering.tile._schedule import schedule
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import _fuse

PATTERN = [Pattern("root", (LoopOp, TileOp))]


# --------------------------------------------------------------------------- #
# Lift — peel the free (parallel) axes off and lift the per-cell compute into ONE
# ``Map`` whose body holds the annotated reduce ``Loop`` + projection.
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


def _is_clean_contraction(body: list[Stmt], k_name: str) -> bool:
    """True iff ``body`` (a reduce loop's body, possibly with a moved-in prologue) is a clean
    contraction whose lift multiplies the operand loads **directly** — body is exactly the
    K-indexed operand loads + the ``⊗`` lift ``Assign`` (distributing over the fold) + the
    additive fold ``Accum``, contracting ≥ 2 distinct operand buffers, with NO loop-invariant
    load or per-operand preprocessing (a pre-scaled ``sum_k (x·s)·(y·s)`` is NOT clean — it
    becomes a degenerate ``PLANAR`` reduce so the scale survives in the loop body)."""
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return False
    fold = accs[0]
    lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
    if lift is None or not lift.op.distributes_over(fold.op):
        return False
    k_loads = [ld for ld in body if isinstance(ld, Load) and k_name in {v for e in ld.index for v in e.free_vars()}]
    if len({ld.input for ld in k_loads}) < 2:
        return False
    all_loads = [s for s in body if isinstance(s, Load)]
    return len(body) == len(all_loads) + 2 and len(all_loads) == len(k_loads) and set(lift.args) == {ld.names[0] for ld in k_loads}


def _annotate_reduce(rloop: Loop, pre_reduce: tuple[Stmt, ...]) -> Loop | None:
    """Annotate a single FLAT reduce ``Loop`` with its :class:`AxisRole` + :class:`Carrier`,
    moving any reduce-feeding ``pre_reduce`` prologue INTO the loop body (so the cooperative
    register fold replicates it per accumulator chain). An already-annotated loop (online-softmax
    / flash from ``_fuse``) keeps its carrier; a clean contraction becomes ``CONTRACTION`` + the
    additive fold's degenerate carrier; a single-``Accum`` reduce becomes ``PLANAR`` + that
    ``Accum``'s degenerate carrier. Returns ``None`` (→ flat ``Map`` fallback) when the carrier
    can't be read (several ``Accum``\\ s, no fold)."""
    body = (*pre_reduce, *rloop.body)
    if rloop.carrier is not None:
        return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=rloop.role, carrier=rloop.carrier)
    if _is_clean_contraction(list(body), rloop.axis.name):
        fold = next(s for s in body if isinstance(s, Accum))
        return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.CONTRACTION, carrier=fold.as_carrier())
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return None
    return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.PLANAR, carrier=accs[0].as_carrier())


def _lift_cell(cell: list[Stmt], free: list, output: str) -> Map:
    """Lift the per-cell stmts into a ``Map`` whose body is the annotated loop nest. A pure
    pointwise cell (no reduce) is a flat ``Map`` of its stmts; a single flat reduce annotates that
    reduce ``Loop`` in place (``CONTRACTION`` / ``PLANAR`` / pre-annotated ``TWISTED``), its body
    holding the reduce loop followed by the projection — stripped to just the loop when the only
    epilogue is the grid-cell ``Write`` (materialize stores ``out`` as glue). A cell with no, or
    several, or a nested reduce stays a flat ``Map`` (un-annotated → the scalar tier)."""
    reduces = [i for i, s in enumerate(cell) if isinstance(s, Loop) and s.is_reduce]
    if len(reduces) != 1:
        return Map(body=tuple(cell))
    idx = reduces[0]
    rloop = cell[idx]
    if _reduce_in(list(rloop.body)):
        return Map(body=tuple(cell))  # nested (non-flash) reduce — keep loop-IR form
    # Route the loop-invariant prologue (stmts above the reduce, sans the regenerated ``Init``
    # seeds): a stmt feeding the reduce moves INTO the loop (``pre_reduce``); one feeding only the
    # epilogue stays as a sibling after it. A stmt feeding BOTH can't be placed by reordering —
    # keep the whole cell as a flat ``Map`` (its loop-IR order is preserved verbatim).
    before = [s for s in cell[:idx] if not isinstance(s, Init)]
    after = list(cell[idx + 1 :])
    before_defs = {n for s in before for n in s.defines()}
    feeds_reduce = bool(before_defs & _reads(list(rloop.body)))
    feeds_epilogue = bool(before_defs & _reads(after))
    if feeds_reduce and feeds_epilogue:
        return Map(body=tuple(cell))
    pre_reduce = tuple(before) if feeds_reduce else ()
    pre_epilogue = () if feeds_reduce else tuple(before)
    annotated = _annotate_reduce(rloop, pre_reduce)
    if annotated is None:
        return Map(body=tuple(cell))
    grid_index = tuple(Var(ax.name) for ax in free)
    bare = (
        not before
        and len(after) == 1
        and isinstance(after[0], Write)
        and after[0].is_scalar
        and after[0].value == annotated.carrier.out
        and after[0].output == output
        and after[0].index == grid_index
    )
    if bare:
        return Map(body=(annotated,))  # materialize writes ``carrier.out`` at the grid cell
    return Map(body=(annotated, *pre_epilogue, *after))


def _lift(stmts: list[Stmt], output: str) -> tuple[Map, tuple]:
    """Peel the free axes and lift the per-cell compute, returning ``(root node, free
    axes)``. The free axes are the schedule's (carried on the ``TileOp``, not the node);
    ``020_schedule`` maps them onto the grid."""
    free, cell = _peel(Body(tuple(stmts)))
    node = _lift_cell(cell, free, output)
    return node, _order_free_by_output(node, free)


def _order_free_by_output(node: Map, free: list) -> tuple:
    """Order the free (grid) axes to match the **output Write's index order**, so the innermost
    grid axis is the output's *contiguous* dim. The contraction tier needs ``n_axis == grid[-1] ==``
    the contiguous output axis — the mma fragment store coalesces a ``float2`` along it, and the
    cuda lowering's ``ldm`` is the output's inner extent — but the peel / loop-naming order can
    diverge from the output layout (e.g. a batched ``Q@Kᵀ`` whose ``kv`` got named before ``m``).
    A node with no explicit output ``Write`` (a bare contraction whose grid-cell store is synthesized
    at materialize, already in free order) is left as-is."""
    body = getattr(node, "body", ())
    write = next((s for s in body if isinstance(s, Write)), None)
    if write is None:
        return tuple(free)
    pos = {e.name: i for i, e in enumerate(write.index) if isinstance(e, Var)}
    if not all(ax.name in pos for ax in free):
        return tuple(free)  # a free axis absent from the output index — leave the peel order
    return tuple(sorted(free, key=lambda ax: pos[ax.name]))


def rewrite(match: Match, root: Node) -> list[TileOp] | TileOp | Graph | None:
    # (0) Schedule an UNMAPPED ``TileOp`` — a kernel that recognition emitted as a *graph
    # rewrite* (flash's fused fragment, ``try_flash``) rather than scheduling inline, because a
    # graph fragment can't embed a scheduling fork. The fused ``TileOp`` re-enters this same pass
    # and is scheduled here, the same ``_schedule.schedule`` the inline path uses. A mapped /
    # kernel-less ``TileOp`` (already scheduled, or ``030_split``'s output) is left for materialize.
    if isinstance(root.op, TileOp):
        tile: TileOp = root.op
        if tile.kernel is None or tile.kernel.schedule.place.is_mapped:
            raise RuleSkipped("TileOp already scheduled / nothing to map")
        return schedule(tile.kernel, tile.name, tile.knobs)
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
    fused, _ = _fuse(loop.body)
    node, free = _lift(list(fused), root.output.name)
    # A symbolic FREE (parallel) axis rides a **symbolic grid**: the ``Tile`` decode sizes the
    # launch from the runtime extent (``_gid < ∏extents``, the ``Dim`` name threaded as an
    # ``int`` arg by the cuda lowering) — the dynamic-grid tier. A symbolic REDUCE /
    # output-sweep axis is likewise supported (the reduce loop strides to the runtime extent,
    # the ``< seq_len`` cap masking the tail). Register-tiled symbolic axes mask their tail
    # cell (clamp-read + guarded write) in ``lowering/kernel``.
    # Wrap the lifted node + its unmapped placement in a ``Kernel``, then schedule it inline
    # (the merged second half, ``_schedule.schedule``): map the free axes onto the grid and offer
    # the per-axis scheduling forks (``REDUCE`` partition / ``TILE`` output tile), dispatched on
    # the axes' ``AxisRole``. Returns the scheduled ``TileOp`` (or a fork list of candidates).
    return schedule(kernel_for(node, Placement(free=free)), loop.name, {})
