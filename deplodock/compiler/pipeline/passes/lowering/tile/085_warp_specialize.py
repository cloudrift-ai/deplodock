"""Warp specialization — emit a WS=0/1 Fork and (for WS=1) rewrite
the ThreadTile body into the warp-specialized shape.

Pattern-matches a ``TileOp`` whose body contains a TMA ``StageBundle``
with ``pipeline_depth == 2`` inside a ``SerialTile(serial_outer)`` —
i.e. the post-``080_pipeline_stages`` shape of a pipelined matmul
kernel. Emits a 2-child ``Fork``: ``WS=0`` returns the input unchanged,
``WS=1`` returns a rewritten ``TileOp``.

The structural rewrite (WS=1 only):

1. **Extend the inner thread axis** by ``n_producer_threads /
   inner_extent`` slots so the new axis range covers both producer
   warps (first ``extension`` rows) and consumer warps (remaining rows).
   Total CTA threads = product of (extended) ThreadTile axis extents,
   which the kernel renderer picks up automatically for ``__launch_bounds__``.
2. **σ-shift the consumer subtree** so every reference to the extended
   axis sees its original (pre-extension) range. Producer threads have
   axis values in ``[0, extension)``; consumer threads in ``[extension,
   new_extent)`` which the σ shifts back to ``[0, original_extent)``
   inside the consumer body — every Load/Write index, every Loop bound,
   every cooperative-thread reference stays correct.
3. **Allocate the empty mbarrier ring** (``Smem`` +
   per-slot ``MbarrierInit`` inside a single-thread ``Cond``,
   followed by ``Sync``) at the top of the new ThreadTile body. Init
   count = 1 — exactly one consumer thread arrives per slot release.
4. **Split the body by role** into producer / consumer subtrees:
   - Producer subtree keeps ``StageBundle``s; inside each
     ``SerialTile(serial_outer)`` it inserts ``MbarrierWait(empty[..],
     phase=..)`` before the issue, gated by ``Cond(K_o >= bc-1, ...)``
     so the first ``bc-1`` iters skip (those slots were unfilled by
     prologue).
   - Consumer subtree keeps ``AsyncWait`` + reduce ``SerialTile`` +
     output ``Write``s; inside each ``SerialTile(serial_outer)`` it
     appends ``MbarrierArrive(empty[slot])`` after the reduce, gated
     by ``Cond(thread_idx == n_producer_threads, ...)`` so exactly one
     consumer thread arrives. Consumer-side ``AsyncWait``s carry
     ``barrier_id=1, barrier_count=n_consumer_threads`` so the
     materializer's lowered trailing ``Sync`` is a named ``bar.sync``
     (``__syncthreads`` is CUDA UB on the warp-divergent ``Cond``).
5. **Wrap producer + consumer in ``Cond(inner_axis < extension, ...,
   ...)``** with ``SetMaxNReg(24, "dec")`` and ``SetMaxNReg(240,
   "inc")`` at the top of each branch.

The materializer (``100_materialize_tile``) sees standard tile IR plus
the new Cond / Smem / MbarrierInit / MbarrierArrive / SetMaxNReg
primitives in the body and lowers each 1:1 without any WS-awareness.

Eligibility:

- TMA policy + ``pipeline_depth == 2`` (the producer/consumer split
  only buys schedule overlap on a depth-2 ring).
- No cooperative ``Accum`` in the body — SDPA cooperative reductions
  need per-thread index remap that doesn't compose cleanly with the
  σ-shift; deferred.
- ``n_producer_threads`` divides the inner thread-axis extent. Today
  only ``producer_warps=1`` (32 threads) is emitted, so the inner
  axis extent must divide 32 (any power-of-two 1..32 inclusive).

``DEPLODOCK_WS=0`` / ``DEPLODOCK_WS=1`` env pins narrow the fork via
``WS.narrow``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import MbarrierArrive, MbarrierInit, MbarrierWait, SetMaxNReg, Smem, Sync
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Stmt
from deplodock.compiler.ir.tile.ir import (
    AsyncWait,
    GridTile,
    RegisterTile,
    SerialTile,
    StageBundle,
    StagePolicy,
    ThreadTile,
    TileOp,
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
    (cooperative shape)."""
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
                if (
                    isinstance(inner, StageBundle)
                    and inner.policy == StagePolicy.TMA
                    and inner.pipeline_depth == 2
                ):
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
    outer = tt.axes[0]
    if not outer.extent.is_static:
        return False, "ThreadTile outer axis has symbolic extent"
    # Extend the OUTERMOST axis by ``extension = n_producer_threads /
    # inner_product``. The linear thread decode is row-major
    # (outer = tid / inner_product; inner = tid % inner_product), so
    # extending the outermost axis keeps producer threads contiguous
    # as ``tid ∈ [0, n_producer_threads)`` — i.e. a single warp 0 for
    # n_producer_threads=32. Extending an inner axis instead would
    # scatter producer threads (tid 0, 1, BN, BN+1, ...) which breaks
    # the warp-specialization invariant. The inner-axes product must
    # divide n_producer_threads for the extension to be integer.
    inner_product = 1
    for ax in tt.axes[1:]:
        if not ax.extent.is_static:
            return False, "ThreadTile inner axis has symbolic extent"
        inner_product *= ax.extent.as_static()
    if _N_PRODUCER_THREADS % inner_product != 0:
        return False, (
            f"producer_threads ({_N_PRODUCER_THREADS}) must be divisible by inner thread-axes product ({inner_product})"
        )
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


def _wire_producer_wait(stmts: list[Stmt], empty_mbar: str, bc: int) -> list[Stmt]:
    """Inside each producer K_o body, prepend a ``MbarrierWait`` on the
    empty mbarrier slot, gated by ``Cond(K_o >= bc-1)`` (first ``bc-1``
    iters fill slots that weren't touched by the prologue)."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            k_var = s.axis.name
            k_plus_1 = Var(k_var) + Literal(1, "int")
            slot_expr = k_plus_1 % Literal(bc, "int")
            phase_expr = (k_plus_1 / Literal(bc, "int") - Literal(1, "int")) % Literal(2, "int")
            wait_cond = Cond(
                cond=BinaryExpr(">=", Var(k_var), Literal(bc - 1, "int")),
                body=(MbarrierWait(mbar=empty_mbar, phase=phase_expr, slot=slot_expr),),
            )
            new_inner = Body((wait_cond, *s.body))
            out.append(s.with_bodies((new_inner,)))
        else:
            out.append(s)
    return out


def _wire_consumer_arrive(
    stmts: list[Stmt], empty_mbar: str, bc: int, first_consumer_tid: int, n_consumer_threads: int
) -> list[Stmt]:
    """Recursively walk the consumer subtree; inside each
    ``SerialTile(serial_outer)`` body, append a named-barrier ``Sync``
    + a single-thread ``MbarrierArrive`` on the empty-mbar slot.

    The Sync is critical: without it, the chosen arriving thread
    (``threadIdx.x == first_consumer_tid``) can race ahead of slower
    consumer threads still reading the slot's smem. The arrive flips
    the empty mbarrier → producer can refill the slot → racing reads
    see corrupted data. The named ``bar.sync barrier_id=1,
    n_consumer_threads`` blocks all consumer threads at the loop's
    tail before the arrive, so the slot's reads are all finished by
    the time the producer is free to refill.

    Recursion descends into any wrapper (e.g. ``RegisterTile`` around
    the K_o loop in blocked matmul); the K_o loop itself isn't
    recursed into."""

    def _augment(stmt: Stmt) -> Stmt:
        if isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
            k_var = stmt.axis.name
            slot_expr = Var(k_var) % Literal(bc, "int")
            barrier_sync = Sync(barrier_id=1, count=n_consumer_threads)
            arrive_cond = Cond(
                cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(first_consumer_tid, "int")),
                body=(MbarrierArrive(mbar=empty_mbar, slot=slot_expr),),
            )
            return stmt.with_bodies((Body((*stmt.body, barrier_sync, arrive_cond)),))
        nested = stmt.nested()
        if nested:
            return stmt.with_bodies(tuple(Body(tuple(_augment(c) for c in b)) for b in nested))
        return stmt

    return [_augment(s) for s in stmts]


def _stamp_async_wait_barrier(body: Body, count: int) -> Body:
    """Recursively stamp ``barrier_id=1, barrier_count=count`` onto every
    ``AsyncWait`` in the body. The materializer's lowered trailing
    ``Sync`` then becomes a named ``bar.sync`` instead of the default
    ``__syncthreads()`` (UB on the warp-divergent consumer Cond)."""

    def _xform(s: Stmt) -> Stmt:
        if isinstance(s, AsyncWait):
            return replace(s, barrier_id=1, barrier_count=count)
        return s

    return body.map(_xform)


def _apply_sigma(body: Body, sigma: Sigma) -> Body:
    """Apply ``sigma`` to every Stmt in ``body`` — substitutes the
    extended-axis Var with its shifted form throughout. ``Stmt.rewrite``
    on wrapper stmts (SerialTile, Cond, Loop, …) recursively rewrites
    their bodies, so we only iterate the top level — using ``Body.map``
    here would double-apply σ at every nesting level."""
    return Body(tuple(s.rewrite(lambda n: n, sigma) for s in body))


def _ws_transform(op: TileOp) -> TileOp:
    """Build the WS=1 TileOp by structurally rewriting ``op``'s body."""
    tt = _find_thread_tile(op.body)
    assert tt is not None  # _eligible enforces

    outer = tt.axes[0]
    outer_extent = outer.extent.as_static()
    inner_product = 1
    for ax in tt.axes[1:]:
        inner_product *= ax.extent.as_static()
    extension = _N_PRODUCER_THREADS // inner_product
    extended_outer = Axis(name=outer.name, extent=outer.extent + Literal(extension, "int"))
    new_axes = (extended_outer, *tt.axes[1:])

    # Consumer threads = original total (we extended by exactly the
    # producer count, so the consumer range still spans the original
    # axis product). Used for the named-barrier participant count.
    n_consumer_threads = outer_extent * inner_product

    # Sigma to shift the extended outer axis back into [0, outer_extent)
    # inside the consumer body. Producer threads have outer-axis values
    # in [0, extension); they never enter the consumer branch so the
    # negative shift never appears in their executed code.
    sigma = Sigma({outer.name: Var(outer.name) - Literal(extension, "int")})

    bc = _first_buffer_count(op.body)
    empty_mbar = "tma_mbar_empty"

    # Empty mbarrier prologue: Smem + per-slot MbarrierInit (single-thread
    # gated) + Sync. Placed at the top of the new ThreadTile body so the
    # materializer's regular flow picks up the Smem decl and the
    # MbarrierInit Conds + Sync render as-is.
    empty_decl = Smem(name=empty_mbar, extents=(bc,), dtype="unsigned long long")
    empty_inits = tuple(MbarrierInit(mbar=empty_mbar, count=1, slot=Literal(s, "int")) for s in range(bc))
    init_cond = Cond(cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(0, "int")), body=empty_inits)

    # Split the original ThreadTile body into producer / consumer halves.
    prod_stmts, cons_stmts = _split_by_role(tuple(tt.body))

    # Wire empty-mbarrier wait/arrive into the role-specific K_o bodies.
    prod_stmts = _wire_producer_wait(prod_stmts, empty_mbar, bc)
    # First consumer thread = extension * inner_product = _N_PRODUCER_THREADS.
    cons_stmts = _wire_consumer_arrive(cons_stmts, empty_mbar, bc, _N_PRODUCER_THREADS, n_consumer_threads)

    # Apply the σ-shift to the consumer subtree (every Load/Write index,
    # Loop bound, Accum ref), then stamp consumer-side AsyncWaits with
    # the named-barrier params.
    cons_body = _apply_sigma(Body(tuple(cons_stmts)), sigma)
    cons_body = _stamp_async_wait_barrier(cons_body, count=n_consumer_threads)

    # Wrap producer + consumer in Cond(outer < extension, prod, σ(cons))
    # with SetMaxNReg framing each branch. The outermost axis controls
    # producer/consumer membership; row-major decode places producer
    # threads (outer ∈ [0, extension)) at contiguous tid [0, n_producer).
    ws_cond = Cond(
        cond=BinaryExpr("<", Var(outer.name), Literal(extension, "int")),
        body=Body((SetMaxNReg(24, "dec"), *prod_stmts)),
        else_body=Body((SetMaxNReg(240, "inc"), *cons_body)),
    )

    new_tt_body = Body((empty_decl, init_cond, Sync(), ws_cond))
    new_tt = ThreadTile(axes=new_axes, body=new_tt_body)

    # Rewrap in GridTile if needed.
    new_outer: list[Stmt] = []
    for s in op.body:
        if isinstance(s, ThreadTile):
            new_outer.append(new_tt)
        elif isinstance(s, GridTile):
            new_children = [new_tt if isinstance(c, ThreadTile) else c for c in s.body]
            new_outer.append(GridTile(axes=s.axes, body=Body(new_children)))
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
