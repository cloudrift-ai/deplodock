"""Recognize a ``LoopOp``'s algebraic structure, lift it to a ``TileOp``, AND schedule it —
the merged Loop-IR → Tile-IR pass (recognition + scheduling in one rewrite, no separate
``020`` step).

This is the Loop-IR → Tile-IR boundary: after this pass nothing downstream traffics in
``LoopOp``. **Recognition** (here) reads the algebra off the body and lifts the per-cell
compute into a :class:`~emmy.compiler.ir.tile.ir.Map` whose body is the **annotated
loop nest** (the reduce ``Loop`` stamped with its
:class:`~emmy.compiler.ir.axis.AxisRole` + :class:`~emmy.compiler.ir.stmt.algebra.Carrier`)
on an UNMAPPED :class:`~emmy.compiler.ir.tile.ir.TileOp`; the final step hands that tile op to
**scheduling** (:func:`~emmy.compiler.pipeline.passes.lowering.tile._schedule.schedule`, the
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

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import Contraction, Map, Placement, Reduction, TileOp, TilePlan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction
from emmy.compiler.pipeline.passes.lowering.tile._flash import is_flash_score_producer, is_fold_offer_site, try_flash
from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule
from emmy.compiler.pipeline.passes.lowering.tile._softmax import _fuse
from emmy.compiler.pipeline.pipeline import LoweringError
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("root", (LoopOp, TileOp))]


# --------------------------------------------------------------------------- #
# Lift — peel the free (parallel) axes off and lift the per-cell compute into ONE
# ``Map`` whose body holds the annotated reduce ``Loop`` + projection.
# --------------------------------------------------------------------------- #


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(free_axes, per_cell_stmts)``.

    The outer chain of **free** loops becomes the parallel axes. At every level of the
    chain a leading run of pure stmts (``Load`` / ``Assign`` / ``Init`` — loop-invariant
    loads hoisted above or between the free loops, e.g. a broadcast row scale ``rs[m]``
    read once per ``m``) is sunk into the per-cell body, re-evaluated per cell. The chain
    stops at the first reduce loop, branch, or non-pure stmt — everything from there down
    is the per-cell body (the fold and its epilogue / output sweep), run serially by one
    thread per cell."""
    axes: list = []
    prefix: list[Stmt] = []
    cur = list(body)
    while True:
        i = 0
        while i < len(cur) and isinstance(cur[i], (Load, Assign, Init)):
            i += 1
        head, rest = cur[:i], cur[i:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + cur
        prefix += head
        axes.append(rest[0].axis)
        cur = list(rest[0].body)


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


def _lift_cell(cell: list[Stmt], free: list, output: str) -> Map | Reduction:
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
    # ``bare`` ⇒ materialize writes ``carrier.out`` at the grid cell (empty projection).
    projection = () if bare else (*pre_epilogue, *after)
    # A PLANAR / TWISTED reduce lifts to a typed ``Reduction`` node (its ⊕ carrier + structure split
    # out, the fold loop synthesized on demand); a ``CONTRACTION`` is nodified to a
    # :class:`Contraction` right after the free axes are ordered (:func:`_nodify_contraction`).
    # ``lower`` flattens either back identically.
    if annotated.role in (AxisRole.PLANAR, AxisRole.TWISTED):
        reduction = Reduction.from_loop(annotated)
        # A bare reduce is the kernel root (its grid ``Write`` is glue); a projected reduce
        # (softmax / RMSNorm) is the ``source`` of a ``Map`` whose body IS that projection.
        return reduction if bare else Map(body=Body(projection), source=reduction)
    return Map(body=(annotated, *projection))


def _nodify_contraction(node, free: tuple):
    """Nodify a freshly-lifted flat ``CONTRACTION`` ``Map`` into the :class:`Contraction`
    structural node with a **deferred** per-cell ``TilePlan()`` (the schedule fork re-tiles it),
    resolving the operand→role binding ONCE, recognize-side (:func:`bind_contraction` over the
    ordered ``free`` axes' trailing ``(m, n)``). An unbindable contraction — a 1-D output (a
    matvec-shaped cell) or no (m, n)-bearing K-loads — **demotes to PLANAR**: its carrier is
    already the additive fold, so it becomes an ordinary :class:`Reduction` (gaining the
    cooperative / ILP partitions a per-cell serial fold never offered). After this step no flat
    ``Map`` carries an annotated ``CONTRACTION`` loop — the scheduler and materializer read
    contraction structure only off the node."""
    if not isinstance(node, Map) or node.source is not None or len(node.body) == 0:
        return node
    rloop = node.body[0]
    if not isinstance(rloop, Loop) or rloop.role is not AxisRole.CONTRACTION:
        return node
    projection = Body(tuple(node.body[1:]))
    if len(free) >= 2:
        try:
            a_load, b_load, acc, epi = bind_contraction(rloop, free[-2].name, free[-1].name, projection)
        except LoweringError:
            pass
        else:
            return Contraction(
                axes=(free[-2], free[-1]),
                k_axis=rloop.axis,
                a_operand=a_load,
                b_load=b_load,
                acc=acc,
                tile=TilePlan(),
                lead_axes=tuple(free[:-2]),
                epilogue=epi,
            )
    demoted = Loop(axis=rloop.axis, body=rloop.body, unroll=rloop.unroll, role=AxisRole.PLANAR, carrier=rloop.carrier)
    red = Reduction.from_loop(demoted)
    return Map(body=projection, source=red) if len(projection) else red


def _lift(stmts: list[Stmt], output: str) -> tuple[Map | Reduction | Contraction, tuple]:
    """Peel the free axes and lift the per-cell compute, returning ``(root node, free
    axes)``. The free axes are the schedule's (carried on the ``TileOp``, not the node);
    ``_schedule`` (inside ``010_recognize``) maps them onto the grid. A ``CONTRACTION`` cell
    nodifies to a :class:`Contraction` once the free axes are output-ordered (the binding needs
    the final ``(m, n)``)."""
    free, cell = _peel(Body(tuple(stmts)))
    node = _lift_cell(cell, free, output)
    free = _order_free_by_output(node, free)
    return _nodify_contraction(node, free), free


def _order_free_by_output(node: Map | Reduction, free: list) -> tuple:
    """Order the free (grid) axes to match the **output Write's index order**, so the innermost
    grid axis is the output's *contiguous* dim. The contraction tier needs ``n_axis == grid[-1] ==``
    the contiguous output axis — the mma fragment store coalesces a ``float2`` along it, and the
    cuda lowering's ``ldm`` is the output's inner extent — but the peel / loop-naming order can
    diverge from the output layout (e.g. a batched ``Q@Kᵀ`` whose ``kv`` got named before ``m``).
    A node with no explicit output ``Write`` (a bare contraction whose grid-cell store is synthesized
    at materialize, already in free order) is left as-is."""
    body = node.lower() if isinstance(node, Reduction) else getattr(node, "body", ())
    write = next((s for s in body if isinstance(s, Write)), None)
    if write is None:
        return tuple(free)
    pos = {e.name: i for i, e in enumerate(write.index) if isinstance(e, Var)}
    if not all(ax.name in pos for ax in free):
        return tuple(free)  # a free axis absent from the output index — leave the peel order
    return tuple(sorted(free, key=lambda ax: pos[ax.name]))


def rewrite(match: Match, root: Node) -> Fork | list[TileOp] | TileOp | Graph | None:
    # (0) Schedule an UNMAPPED ``TileOp`` — a kernel that recognition emitted as a *graph
    # rewrite* (flash's fused fragment, ``try_flash``) rather than scheduling inline, because a
    # graph fragment can't embed a scheduling fork. The fused ``TileOp`` re-enters this same pass
    # and is scheduled here, the same ``_schedule.schedule`` the inline path uses. A mapped /
    # kernel-less ``TileOp`` (already scheduled, or ``030_split``'s output) is left for materialize.
    if isinstance(root.op, TileOp):
        tile: TileOp = root.op
        if tile.op is None or tile.place.is_mapped:
            raise RuleSkipped("TileOp already scheduled / nothing to map")
        return schedule(tile, tile.name, tile.knobs)
    # (1) Flash attention — a graph rewrite that fuses a softmax-then-P@V kernel with its
    # scaled-QK producer. Tried first on every node; flash precedes online-softmax precedes
    # normalize, each consuming the Accums the next would match. The downstream-fold
    # absorption is the ``PLACE@fold`` placement: ``cut`` keeps the score producer and the
    # softmax-then-P@V as separate kernels (the multi-kernel attention escape).
    graph = match.graph
    if place_decision("fold") == "fuse":
        flash = try_flash(graph, root)
        if flash is not None:
            return flash
        # (2) Defer a flash score producer: the general lift below would turn this scaled-QK
        # matmul into a ``TileOp`` before its softmax-then-P@V consumer fuses, and that fusion
        # reads the producer's Q/K as plain ``Load``s. Leave it a ``LoopOp`` until the consumer
        # has had its chance to consume it (a later scan re-visits this node, by then removed).
        if is_flash_score_producer(graph, root):
            raise RuleSkipped("flash score producer — defer to its consumer's fusion")
    # A fold offer site (a softmax-then-P@V consumer) that did NOT fuse — the pin forced the cut,
    # or the fuse degraded (uncertifiable) — stamps the RESOLVED ``cut`` so its DB row is
    # distinguishable from a never-offered kernel (the fuse side stamps ``fuse`` on the fused
    # fragment in ``build_flash_frag``).
    knob_base: dict = {"PLACE@fold": "cut"} if is_fold_offer_site(graph, root) else {}
    loop: LoopOp = root.op
    # (3) Online softmax — the sibling-fold tupling (``PLACE@tuple``): fuse the adjacent
    # (rowmax, Σexp) reduce pair into one streaming pass; ``cut`` keeps the two-pass stats.
    fused, _ = _fuse(loop.body) if place_decision("tuple") == "fuse" else (loop.body, False)
    node, free = _lift(list(fused), root.output.name)
    # A symbolic FREE (parallel) axis rides a **symbolic grid**: the ``Tile`` decode sizes the
    # launch from the runtime extent (``_gid < ∏extents``, the ``Dim`` name threaded as an
    # ``int`` arg by the cuda lowering) — the dynamic-grid tier. A symbolic REDUCE /
    # output-sweep axis is likewise supported (the reduce loop strides to the runtime extent,
    # the ``< seq_len`` cap masking the tail). Register-tiled symbolic axes mask their tail
    # cell (clamp-read + guarded write) in ``lowering/kernel``.
    # Wrap the lifted node + its unmapped placement in an UNMAPPED ``TileOp``, then schedule it inline
    # (the merged second half, ``_schedule.schedule``): map the free axes onto the grid and offer
    # the per-axis scheduling forks (``REDUCE`` partition / ``TILE`` output tile), dispatched on
    # the axes' ``AxisRole``. Returns the scheduled ``TileOp`` (or a fork list of candidates).
    # ``inputs`` is seeded from the matched ``LoopOp`` (the matcher populated its real Tensors) so
    # the scheduler can read operand shapes (the shared-row stage detection); the matcher refreshes
    # it from the graph again when a later pass matches the scheduled op.
    return schedule(TileOp(op=node, place=Placement(free=free), inputs=dict(loop.inputs)), loop.name, knob_base)
