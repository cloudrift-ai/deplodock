"""Warp specialization — emit a WS=0/1 Fork and (for WS=1) rewrite
the ThreadTile body into a ``WarpTile(role) > WarpSpecialize`` shape.

Pattern-matches a ``TileOp`` whose body contains a TMA ``StageBundle``
with ``pipeline_depth == 2`` inside a ``SerialTile(serial_outer)`` —
i.e. the post-``080_pipeline_stages`` shape of a pipelined matmul
kernel. Emits a 2-child ``Fork``: ``WS=0`` returns the input unchanged,
``WS=1`` returns a rewritten ``TileOp``.

**MMA interaction** (warp-tier MMA tower — planner-emitted ``WarpTile`` >
``RegisterTile`` > ``AtomTile`` + fragment chain). This pass handles both
consumer tiers: ``_find_consumer_tile`` returns the scalar ``ThreadTile``
(``is_warp=False``, pointwise / cooperative-reduce) or, for MMA, the existing
``WarpTile`` (``is_warp=True``). The warp arm consumes the planner-emitted warp
tier directly — no fresh tier synthesis — so the WM×WN warp axes stay in scope
for the RegisterTile / AtomTile lowering; only the producer/consumer body split
is applied (``_split_by_role`` hoists the cooperative TMA ``StageBundle`` up
across the RegisterTile / AtomTile into the producer). ``WarpSpecialize``
carries ``consumer_is_warp`` so the materializer wraps the consumer in a
``WarpTile(tid_offset)`` decode and scales the ``bar.sync`` participant count
by 32 (warp axes count warps, not threads). Validated ``max_diff=0`` across
256²-2048²; ~no latency change vs WS=0 on GeForce s16816 (its cuBLAS gap is the
SASS mma schedule below the IR, not warp-level producer/consumer overlap), so
the autotuner will normally pick WS=0 — the warp arm exists for parity /
investigation and for Hopper-class parts where the split should pay off.

The structural rewrite (WS=1 only):

1. **Replace the inner ``ThreadTile`` with a ``WarpTile``** whose single
   ``role`` axis has extent equal to total CTA warps
   (``n_producer_warps + n_consumer_warps``). Producer warps own
   ``role ∈ [0, n_producer_warps)``; consumer warps own
   ``role ∈ [n_producer_warps, total_warps)``. Total CTA threads =
   ``role.extent × 32``, which the kernel renderer picks up
   automatically for ``__launch_bounds__`` (see
   ``ir/kernel/render._launch_bounds_for``'s ``WarpTile`` branch).
2. **Split the body by role** into producer / consumer subtrees:
   - Producer subtree keeps ``StageBundle``s.
   - Consumer subtree keeps ``AsyncWait`` + reduce ``SerialTile`` +
     output ``Write``s.
3. **Carry the original ``ThreadTile.axes``** on the ``WarpSpecialize``
   Stmt as ``consumer_thread_axes``. The consumer body keeps its
   references to those axis Vars *unshifted* — the materializer drops a
   ``ThreadTile(consumer_thread_axes, tid_offset=n_producer_threads, …)``
   inside the consumer ``Cond.else_body`` so consumer threads decode
   ``threadIdx.x - n_producer_threads`` back into the original axis
   range ``[0, n_consumer_threads)``. No σ-shift on the consumer subtree.
4. **Package as a single Tile-IR ``WarpSpecialize`` Stmt** carrying the
   role bodies, ring depth (= TMA buffer_count), producer thread count,
   and ``consumer_thread_axes``. The materializer
   (``100_materialize_tile.emit_warp_specialize``) expands this into the
   empty-mbarrier ring, per-K_o handshake, producer/consumer ``Cond``
   wrapper, ``SetMaxNReg`` register-budget framing, and the
   consumer-side ``ThreadTile(tid_offset=…)`` decode — none of which
   appear at Tile level.

This pass stays inside the Tile-IR dialect — no ``from
deplodock.compiler.ir.kernel.ir import …``. The boundary mirrors what
``080_pipeline_stages`` already does for ``AsyncWait``: declare
scheduling intent at Tile level, materializer fills in the hardware
primitives.

Eligibility:

- TMA policy + ``pipeline_depth == 2`` (the producer/consumer split
  only buys schedule overlap on a depth-2 ring).
- The TMA bundle must be **reachable by the producer split**: ``_split_by_role``
  recurses only through ``SerialTile(serial_outer)`` / ``RegisterTile`` /
  ``AtomTile`` from the consumer tile's top level, so a bundle nested under any
  other wrapper (e.g. the ``SerialTile(kind='plain')`` per-thread M-fragment
  loop of a fused linear+mean kernel) would be stranded in the **consumer**
  branch — where its ``threadIdx.x == issuer`` TMA guard can never fire (thread
  0 is a producer-warp thread) — and every consumer ``mbarrier.wait`` would
  deadlock (the Qwen3 ``k_linear_mean_reduce`` hang). ``_eligible`` runs the
  same split the transform uses and rejects when no TMA depth-2 bundle lands
  producer-side.
- No cooperative ``Accum`` in the body — SDPA cooperative reductions
  need per-thread index remap that doesn't compose cleanly with the
  role-split decode; deferred.
- ``n_producer_threads`` divides the inner thread-axis extent. Today
  only ``producer_warps=1`` (32 threads) is emitted, so the inner
  axis extent must divide 32 (any power-of-two 1..32 inclusive).
- Total CTA threads (``prod(thread_axes.extent) + n_producer_threads``)
  must be a multiple of 32 so the ``WarpTile`` role axis has integer
  extent. Pre-extension matmul tiles satisfy this by construction
  (``BM × BN`` is always a multiple of 32 for matmul shapes).

``DEPLODOCK_WARP_SPECIALIZE=0`` / ``DEPLODOCK_WARP_SPECIALIZE=1`` env pins narrow
the fork via ``WARP_SPECIALIZE.narrow``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    AtomTile,
    GridTile,
    RegisterTile,
    SerialTile,
    StageBundle,
    StagePolicy,
    ThreadTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import ThunkFork
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]


WARP_SPECIALIZE = Knob(
    "WARPSPEC",
    KnobType.BOOL,
    hints=(False, True),
    help="Warp-specialize TMA staging: producer warps issue TMA, consumer warps wait + reduce",
    aliases=("WARP_SPECIALIZE",),
    off=False,
)

# v1 — single producer warp on sm_90+ (warp size 32). If WS_PRODUCER_WARPS
# becomes a knob, derive these from the knob value.
_PRODUCER_WARPS = 1
_WARP_SIZE = 32
_N_PRODUCER_THREADS = _PRODUCER_WARPS * _WARP_SIZE


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def _find_thread_tile(body: Body) -> ThreadTile | None:
    """Locate the (single) ThreadTile in a TileOp body — either directly
    at the top level (pointwise shape) or nested inside a GridTile
    (cooperative shape).

    Matches ``ThreadTile`` only (not ``WarpTile``) — the scalar (pointwise /
    cooperative-reduce) consumer tier. The warp-tier MMA matmul carries a
    planner-emitted ``WarpTile`` instead; ``_find_warp_tile`` locates that and
    ``_find_consumer_tile`` reports which tier was found. ``_ws_transform``
    consumes the existing warp tier directly (``consumer_is_warp=True``) rather
    than synthesising one — the producer warp issues TMA, the WM×WN consumer
    warps run the RegisterTile > AtomTile MMA tower. (Validated: ``max_diff=0``
    across 256²-2048²; ~no latency change vs WS=0 on GeForce s16816, whose gap
    to cuBLAS is the SASS mma schedule below the IR, not producer/consumer
    overlap.)"""
    for s in body:
        if isinstance(s, ThreadTile):
            return s
        if isinstance(s, GridTile):
            for c in s.body:
                if isinstance(c, ThreadTile):
                    return c
    return None


def _find_warp_tile(body: Body) -> WarpTile | None:
    """Locate the (single) planner-emitted ``WarpTile`` — the consumer tower of
    a warp-tier MMA matmul (``GridTile > WarpTile > RegisterTile > AtomTile``).
    Mutually exclusive with ``ThreadTile`` (``TileOp.__post_init__`` enforces)."""
    for s in body:
        if isinstance(s, WarpTile):
            return s
        if isinstance(s, GridTile):
            for c in s.body:
                if isinstance(c, WarpTile):
                    return c
    return None


def _find_consumer_tile(body: Body) -> tuple[ThreadTile | WarpTile | None, bool]:
    """The tile the WS consumer decodes against: a scalar ``ThreadTile``
    (``is_warp=False``) or a warp-tier ``WarpTile`` (``is_warp=True``)."""
    tt = _find_thread_tile(body)
    if tt is not None:
        return tt, False
    wt = _find_warp_tile(body)
    if wt is not None:
        return wt, True
    return None, False


def _has_tma_depth2(stmts: tuple[Stmt, ...] | list[Stmt]) -> bool:
    """True if a TMA ``StageBundle`` with ``pipeline_depth == 2`` appears
    anywhere under ``stmts``."""
    for s in Body(tuple(stmts)).iter():
        if isinstance(s, StageBundle) and s.policy == StagePolicy.TMA and s.pipeline_depth == 2:
            return True
    return False


def _eligible(op: TileOp) -> tuple[bool, str]:
    """Return ``(True, "")`` if WS=1 can fire, otherwise
    ``(False, reason)``."""
    has_tma_depth2 = False
    for stmt in op.body.iter():
        if isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
            for inner in stmt.body:
                if isinstance(inner, StageBundle) and inner.policy == StagePolicy.TMA and inner.pipeline_depth == 2:
                    has_tma_depth2 = True
                    break
        if has_tma_depth2:
            break
    if not has_tma_depth2:
        return False, "no TMA StageBundle with pipeline_depth=2 inside serial_outer"

    escape = op.body.coordination
    if any(escape.accum_cooperative_axes.values()):
        return False, "cooperative Accum present — consumer-tid remap pending"

    tile, is_warp = _find_consumer_tile(op.body)
    if tile is None:
        return False, "no ThreadTile or WarpTile in body"

    # The bundle must actually land producer-side under the same split the
    # transform performs — `_split_by_role` only recurses through
    # serial_outer / RegisterTile / AtomTile, so a bundle nested under any
    # other wrapper (e.g. a `SerialTile(kind='plain')` fragment loop) stays
    # in the consumer branch, whose `threadIdx.x == issuer` TMA guard is
    # unreachable from consumer warps: every mbarrier.wait would deadlock.
    prod_stmts, _ = _split_by_role(tuple(tile.body))
    if not _has_tma_depth2(prod_stmts):
        return False, "TMA StageBundle not reachable by the producer split — TMA issues would strand in the consumer branch"
    for ax in tile.axes:
        if not ax.extent.is_static:
            return False, "consumer tile axis has symbolic extent"

    if is_warp:
        # Warp-tier MMA: producer = 1 warp, consumer = ∏(warp axes) warps. Total
        # = (n_consumer + 1) warps → always a multiple of warp_size, and warp-
        # granularity decode needs no inner-thread divisibility constraint.
        return True, ""

    # Scalar (pointwise / cooperative-reduce) ThreadTile path:
    consumer_threads = 1
    for ax in tile.axes:
        consumer_threads *= ax.extent.as_static()
    # Inner-axis divisibility: ``n_producer_threads`` must split cleanly
    # across the inner-axes product so the per-warp coord can replace
    # the σ-arithmetic the old shape relied on.
    inner_product = 1
    for ax in tile.axes[1:]:
        inner_product *= ax.extent.as_static()
    if _N_PRODUCER_THREADS % inner_product != 0:
        return False, (f"producer_threads ({_N_PRODUCER_THREADS}) must be divisible by inner thread-axes product ({inner_product})")
    # Total CTA threads must be a multiple of warp_size so the role axis
    # has integer extent. Matmul shapes satisfy this by construction;
    # check explicitly for the corner cases.
    total_threads = consumer_threads + _N_PRODUCER_THREADS
    if total_threads % _WARP_SIZE != 0:
        return False, f"total CTA threads ({total_threads}) not divisible by warp_size={_WARP_SIZE}"
    return True, ""


# ---------------------------------------------------------------------------
# Structural transform
# ---------------------------------------------------------------------------


def _first_buffer_count(body: Body) -> int:
    for s in body.iter():
        if isinstance(s, StageBundle) and s.policy == StagePolicy.TMA:
            return s.buffer_count
    raise AssertionError("no TMA bundle found — _eligible should have caught this")


def _split_by_role(stmts: tuple[Stmt, ...]) -> tuple[list[Stmt], list[Stmt]]:
    """Classify each top-level stmt: ``StageBundle``s and the
    producer-side scaffolding inside ``SerialTile(serial_outer)`` →
    producer; ``AsyncWait`` + reduce ``SerialTile``s + output stmts →
    consumer.

    Recurse into ``serial_outer`` to build two K_o loops (one per role,
    both iterating the same axis with role-specific bodies). Recurse
    into ``RegisterTile`` to **hoist** producer stmts out of the
    register-cell wrapper — TMA loads don't depend on register-cell
    axes, so duplicating them per cell (which ``010_split_register_axes``
    would do if they stayed inside) would issue N redundant copies of
    the same TMA. Consumer-side compute stays in its RegisterTile."""
    producer: list[Stmt] = []
    consumer: list[Stmt] = []
    for stmt in stmts:
        if isinstance(stmt, StageBundle):
            producer.append(stmt)
        elif isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
            inner_prod, inner_cons = _split_by_role(tuple(stmt.body))
            if inner_prod:
                producer.append(stmt.with_bodies((Body(tuple(inner_prod)),)))
            if inner_cons:
                consumer.append(stmt.with_bodies((Body(tuple(inner_cons)),)))
        elif isinstance(stmt, (RegisterTile, AtomTile)):
            inner_prod, inner_cons = _split_by_role(tuple(stmt.body))
            # Hoist producer stmts out of RegisterTile / AtomTile — the
            # cooperative TMA loads are CTA-wide and don't depend on the
            # register-cell / atom axes, so running them once at the WS
            # producer level (rather than per cell / per atom) is both
            # correct and what avoids 010_split_register_axes replicating
            # them. The consumer keeps its RegisterTile / AtomTile wrapper so
            # 005_lower_atom_tile / 010 still see the warp-tier MMA tower.
            producer.extend(inner_prod)
            if inner_cons:
                consumer.append(stmt.with_bodies((Body(tuple(inner_cons)),)))
        else:
            # AsyncWait, SerialTile(stage_inner / plain), Write, Accum,
            # Cond — all consumer-side.
            consumer.append(stmt)
    return producer, consumer


def _ws_transform(op: TileOp) -> TileOp:
    """Build the WS=1 TileOp by structurally rewriting ``op``'s body.

    Output shape:

        GridTile(M_b, N_b, …)
          WarpTile(role: extent = total_warps)
            WarpSpecialize(producer_body, consumer_body, ring_depth,
                           n_producer_threads, consumer_thread_axes)

    The materializer fabricates the mbarrier ring, role-split ``Cond``,
    register-budget framing, and the consumer-side ``ThreadTile(tid_offset
    = n_producer_threads, …)`` decode from this marker. The consumer
    body's references to the original thread-axis Vars survive
    unchanged — no σ-shift at the Tile-IR level (the decode in the
    materializer's nested ``ThreadTile`` rebinds them to the consumer-
    relative range)."""
    tile, is_warp = _find_consumer_tile(op.body)
    assert tile is not None  # _eligible enforces

    consumer_count = 1
    for ax in tile.axes:
        consumer_count *= ax.extent.as_static()
    # Scalar tile axes count *threads*; a warp tile counts *warps*.
    n_consumer_warps = consumer_count if is_warp else consumer_count // _WARP_SIZE
    total_warps = n_consumer_warps + _PRODUCER_WARPS
    # Synthesise the role axis as a fresh name — ``normalize_body``
    # canonicalises it to ``a<N>`` during ``TileOp.__post_init__`` so the
    # placeholder name only matters for pretty-print of the un-normalized form.
    role_axis = Axis(name="ws_role", extent=Dim(total_warps))

    bc = _first_buffer_count(op.body)

    # Split the original consumer-tile body into producer / consumer halves.
    # For the warp tier this hoists the cooperative TMA loads up across the
    # RegisterTile / AtomTile (see ``_split_by_role``); the consumer keeps the
    # RegisterTile > AtomTile > reduce tower the MMA lowering expects.
    prod_stmts, cons_stmts = _split_by_role(tuple(tile.body))
    assert _has_tma_depth2(prod_stmts), "WS split stranded every TMA StageBundle in the consumer branch — _eligible should have caught this"

    ws = WarpSpecialize(
        producer_body=Body(tuple(prod_stmts)),
        consumer_body=Body(tuple(cons_stmts)),
        ring_depth=bc,
        n_producer_threads=_N_PRODUCER_THREADS,
        consumer_thread_axes=tile.axes,
        consumer_is_warp=is_warp,
    )

    new_warp_tile = WarpTile(axes=(role_axis,), body=Body((ws,)))

    # Replace the original consumer tile (ThreadTile or warp-tier WarpTile) with
    # the role-bearing WarpTile, preserving the GridTile wrapper (if any).
    def _is_consumer_tile(s: Stmt) -> bool:
        return isinstance(s, (ThreadTile, WarpTile))

    new_outer: list[Stmt] = []
    for s in op.body:
        if _is_consumer_tile(s):
            new_outer.append(new_warp_tile)
        elif isinstance(s, GridTile):
            new_children = [new_warp_tile if _is_consumer_tile(c) else c for c in s.body]
            new_outer.append(GridTile(axes=s.axes, body=Body(new_children), swizzle_group_m=s.swizzle_group_m))
        else:
            new_outer.append(s)

    new_knobs = dict(op.knobs)
    new_knobs[WARP_SPECIALIZE.name] = True
    return TileOp(body=Body(new_outer), name=op.name, knobs=new_knobs)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


def rewrite(ctx: Context, root: Node) -> TileOp | ThunkFork | list[ThunkFork] | None:
    op: TileOp = root.op

    if WARP_SPECIALIZE.name in op.knobs:
        raise RuleSkipped("WARP_SPECIALIZE knob already set")

    ok, reason = _eligible(op)
    if not ok:
        # Fail loudly if the user explicitly pinned ``DEPLODOCK_WARP_SPECIALIZE=1``
        # but the kernel can't be warp-specialized (e.g. a warp-tier MMA matmul
        # — ``no ThreadTile in body``, WS+MMA is out of v1 scope). A
        # pinned-but-unhonorable knob raises — same policy as the BUFFER_COUNT /
        # TMA pins.
        pin = WARP_SPECIALIZE.raw()
        if pin is not None and WARP_SPECIALIZE.parse(pin):
            raise ValueError(f"{WARP_SPECIALIZE.env}=1 pinned but warp specialization cannot fire: {reason}")
        # Ineligible (not pinned on) — record the off decision (body unchanged) so
        # the realized config keeps a uniform knob set instead of leaving it absent.
        return TileOp(body=op.body, name=op.name, knobs={**op.knobs, WARP_SPECIALIZE.name: False})

    ws_choices = WARP_SPECIALIZE.narrow(WARP_SPECIALIZE.hints)
    if not ws_choices:
        raise RuleSkipped("DEPLODOCK_WARP_SPECIALIZE env pin removed all choices")

    def _stamp(ws: bool) -> TileOp:
        if not ws:
            new_knobs = dict(op.knobs)
            new_knobs[WARP_SPECIALIZE.name] = False
            return TileOp(body=op.body, name=op.name, knobs=new_knobs)
        return _ws_transform(op)

    if len(ws_choices) == 1:
        return _stamp(ws_choices[0])

    # For the warp-tier MMA tower, WS=1 is a measured win (≈17% at 64×64, neutral
    # at 128×128 — the RTX 5090 warp sweep, see ``plans/golden-sweep-report.md``),
    # so it should be the first guess. Ordering is the ONLY lever that does this:
    # the policies select on the learned prior when fitted and on **emission
    # order** otherwise (option-0 / PUCT tie) — Forks carry no heuristic score.
    # So a cold MCTS (a fresh ``tune --clean``) and a no-prior
    # greedy ``compile`` would take WS=0 and, under patience, never bench WS=1
    # (the measured fp16 gap in ``plans/golden-sweep-report.md``). Emit WS=1 FIRST
    # for the warp tier — emission order deploys the win cold, and the -O3 re-bench
    # tolerance feeds the prior its deployable latency. The scalar (pointwise /
    # coop-reduce) tier has no such consensus → keep the hint order (WS=0 first).
    _, is_warp = _find_consumer_tile(op.body)
    if is_warp and set(ws_choices) == {False, True}:
        ws_choices = (True, False)

    def _ws_expand(knobs: dict) -> list[TileOp]:
        return [_stamp(knobs[WARP_SPECIALIZE.name])]

    return [ThunkFork(knobs={WARP_SPECIALIZE.name: ws}, expand_fn=_ws_expand, is_leaf=True) for ws in ws_choices]
