"""Warp specialization — emit a WS=0/1 Fork and (for WS=1) rewrite
the ThreadTile body into a ``WarpTile(role) > WarpSpecialize`` shape.

Pattern-matches a ``TileOp`` whose body contains a TMA ``StageBundle``
with ``pipeline_depth == 2`` inside a ``SerialTile(serial_outer)`` —
i.e. the post-``080_pipeline_stages`` shape of a pipelined matmul
kernel. Emits a 2-child ``Fork``: ``WS=0`` returns the input unchanged,
``WS=1`` returns a rewritten ``TileOp``.

**MMA interaction** (see ``plans/mma-fragment-factorization.md`` M11).
The warp-tier MMA path (planner-emitted ``WarpTile`` + ``AtomTile`` +
fragment chain) bypasses this pass: ``_find_thread_tile`` returns ``None``
on a body whose binding tier is already a ``WarpTile`` (the MMA cell
materializer in ``kernel/005_lower_atom_tile`` consumes the AtomTile but
preserves the WarpTile wrapper). For v1, MMA + WS is out of scope; 085
silently skipping MMA kernels is the desired behaviour. The M11 followup
splits ``_ws_transform`` into a ThreadTile arm (today's body) and a
WarpTile arm (which would consume the planner-emitted warp tier
directly, no fresh tier synthesis) once MMA + WS is a validated combined
target — the WarpTile arm would only need the producer/consumer body
split, since the warp axes are already in scope.

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

``DEPLODOCK_WS=0`` / ``DEPLODOCK_WS=1`` env pins narrow the fork via
``WS.narrow``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
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
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.pipeline import Fork

PATTERN = [Pattern("root", TileOp)]


WS = Knob(
    "WS",
    KnobType.BOOL,
    hints=(0, 1),
    help="Warp-specialize TMA staging: producer warps issue TMA, consumer warps wait + reduce",
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

    Defensive: matches ``ThreadTile`` only (not ``WarpTile``). The pass
    rewrites a scalar-tower input into the warp-tower output; a MMA path
    TileOp (warp-tier matmul with planner-emitted ``WarpTile``) sees no
    ThreadTile and skips cleanly — by design, since MMA + WS is out of
    scope for v1 (see ``plans/mma-fragment-factorization.md`` M11 and
    Failure modes). When MMA + WS is a validated combined target, this
    helper splits into ``_find_thread_or_warp_tile`` and ``_ws_transform``
    grows a WarpTile arm that consumes the existing warp tier instead of
    synthesising one."""
    for s in body:
        if isinstance(s, ThreadTile):
            return s
        if isinstance(s, GridTile):
            for c in s.body:
                if isinstance(c, ThreadTile):
                    return c
    return None


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

    tt = _find_thread_tile(op.body)
    if tt is None:
        return False, "no ThreadTile in body"
    for ax in tt.axes:
        if not ax.extent.is_static:
            return False, "ThreadTile axis has symbolic extent"
    consumer_threads = 1
    for ax in tt.axes:
        consumer_threads *= ax.extent.as_static()
    # Inner-axis divisibility: ``n_producer_threads`` must split cleanly
    # across the inner-axes product so the per-warp coord can replace
    # the σ-arithmetic the old shape relied on. The historical extension
    # arithmetic (extending the outer axis by ``n_producer_threads /
    # inner_product``) needs the same divisibility — keeping it makes
    # eligibility match the prior pass exactly even though the rewrite
    # shape changed.
    inner_product = 1
    for ax in tt.axes[1:]:
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
        elif isinstance(stmt, RegisterTile):
            inner_prod, inner_cons = _split_by_role(tuple(stmt.body))
            # Hoist producer stmts out of RegisterTile — they don't
            # depend on the register-cell axes; running them once at
            # top level avoids 010_split_register_axes replicating
            # them per cell.
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
    tt = _find_thread_tile(op.body)
    assert tt is not None  # _eligible enforces

    consumer_threads = 1
    for ax in tt.axes:
        consumer_threads *= ax.extent.as_static()
    total_warps = (consumer_threads + _N_PRODUCER_THREADS) // _WARP_SIZE
    # Synthesise the role axis as a fresh name — ``normalize_body``
    # canonicalises it to ``a<N>`` during ``TileOp.__post_init__`` so the
    # placeholder name only matters for pretty-print of the un-normalized
    # form.
    role_axis = Axis(name="ws_role", extent=Dim(total_warps))

    bc = _first_buffer_count(op.body)

    # Split the original ThreadTile body into producer / consumer halves.
    prod_stmts, cons_stmts = _split_by_role(tuple(tt.body))

    ws = WarpSpecialize(
        producer_body=Body(tuple(prod_stmts)),
        consumer_body=Body(tuple(cons_stmts)),
        ring_depth=bc,
        n_producer_threads=_N_PRODUCER_THREADS,
        consumer_thread_axes=tt.axes,
    )

    new_warp_tile = WarpTile(axes=(role_axis,), body=Body((ws,)))

    # Replace the inner ThreadTile with the new WarpTile, preserving the
    # GridTile wrapper (if any).
    new_outer: list[Stmt] = []
    for s in op.body:
        if isinstance(s, ThreadTile):
            new_outer.append(new_warp_tile)
        elif isinstance(s, GridTile):
            new_children = [new_warp_tile if isinstance(c, ThreadTile) else c for c in s.body]
            new_outer.append(GridTile(axes=s.axes, body=Body(new_children), swizzle_group_m=s.swizzle_group_m))
        else:
            new_outer.append(s)

    new_knobs = dict(op.knobs)
    new_knobs["WS"] = 1
    return TileOp(body=Body(new_outer), name=op.name, knobs=new_knobs)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


def rewrite(ctx: Context, root: Node) -> TileOp | Fork | list[Fork] | None:
    op: TileOp = root.op

    if "WS" in op.knobs:
        raise RuleSkipped("WS knob already set")

    ok, reason = _eligible(op)
    if not ok:
        raise RuleSkipped(reason)

    ws_choices = WS.narrow((0, 1))
    if not ws_choices:
        raise RuleSkipped("DEPLODOCK_WS env pin removed all WS choices")

    def _stamp(ws: int) -> TileOp:
        if ws == 0:
            new_knobs = dict(op.knobs)
            new_knobs["WS"] = 0
            return TileOp(body=op.body, name=op.name, knobs=new_knobs)
        return _ws_transform(op)

    if len(ws_choices) == 1:
        return _stamp(ws_choices[0])

    return [
        Fork(
            knobs={WS.name: ws},
            expand=(lambda ws=ws: [_stamp(ws)]),
            score=0.0,
            is_leaf=True,
        )
        for ws in ws_choices
    ]
