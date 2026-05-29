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
3. **Split the body by role** into producer / consumer subtrees:
   - Producer subtree keeps ``StageBundle``s.
   - Consumer subtree keeps ``AsyncWait`` + reduce ``SerialTile`` +
     output ``Write``s.
4. **Package as a single Tile-IR ``WarpSpecialize`` Stmt** carrying the
   role bodies, ring depth (= TMA buffer_count), and producer thread
   count. The materializer (``100_materialize_tile.emit_warp_specialize``)
   expands this into the empty-mbarrier ring, per-K_o handshake,
   producer/consumer ``Cond`` wrapper, and ``SetMaxNReg`` register-
   budget framing — none of which appear at Tile level.

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
  σ-shift; deferred.
- ``n_producer_threads`` divides the inner thread-axis extent. Today
  only ``producer_warps=1`` (32 threads) is emitted, so the inner
  axis extent must divide 32 (any power-of-two 1..32 inclusive).

``DEPLODOCK_WS=0`` / ``DEPLODOCK_WS=1`` env pins narrow the fork via
``WS.narrow``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
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
        return False, (f"producer_threads ({_N_PRODUCER_THREADS}) must be divisible by inner thread-axes product ({inner_product})")
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


def _apply_sigma(body: Body, sigma: Sigma) -> Body:
    """Apply ``sigma`` to every Stmt in ``body`` — substitutes the
    extended-axis Var with its shifted form throughout. ``Stmt.rewrite``
    on wrapper stmts (SerialTile, Cond, Loop, …) recursively rewrites
    their bodies, so we only iterate the top level — using ``Body.map``
    here would double-apply σ at every nesting level."""
    return Body(tuple(s.rewrite(lambda n: n, sigma) for s in body))


def _ws_transform(op: TileOp) -> TileOp:
    """Build the WS=1 TileOp by structurally rewriting ``op``'s body.

    Output shape: the extended ThreadTile holds a single
    ``WarpSpecialize`` carrying the two role bodies. The materializer
    fabricates the mbarrier ring, handshake, register-budget framing,
    and producer/consumer Cond wrapper from this marker."""
    tt = _find_thread_tile(op.body)
    assert tt is not None  # _eligible enforces

    outer = tt.axes[0]
    inner_product = 1
    for ax in tt.axes[1:]:
        inner_product *= ax.extent.as_static()
    extension = _N_PRODUCER_THREADS // inner_product
    extended_outer = Axis(name=outer.name, extent=outer.extent + Literal(extension, "int"))
    new_axes = (extended_outer, *tt.axes[1:])

    # Sigma to shift the extended outer axis back into [0, outer_extent)
    # inside the consumer body. Producer threads have outer-axis values
    # in [0, extension); they never enter the consumer branch so the
    # negative shift never appears in their executed code.
    sigma = Sigma({outer.name: Var(outer.name) - Literal(extension, "int")})

    bc = _first_buffer_count(op.body)

    # Split the original ThreadTile body into producer / consumer halves.
    prod_stmts, cons_stmts = _split_by_role(tuple(tt.body))

    # Apply the σ-shift to the consumer subtree (every Load/Write index,
    # Loop bound, Accum ref). Producer stays unshifted because producer
    # threads have outer ∈ [0, extension) — exactly the pre-extension
    # zero-based range.
    cons_body = _apply_sigma(Body(tuple(cons_stmts)), sigma)

    ws = WarpSpecialize(
        producer_body=Body(tuple(prod_stmts)),
        consumer_body=cons_body,
        ring_depth=bc,
        n_producer_threads=_N_PRODUCER_THREADS,
    )

    new_tt = ThreadTile(axes=new_axes, body=Body((ws,)))

    # Rewrap in GridTile if needed.
    new_outer: list[Stmt] = []
    for s in op.body:
        if isinstance(s, ThreadTile):
            new_outer.append(new_tt)
        elif isinstance(s, GridTile):
            new_children = [new_tt if isinstance(c, ThreadTile) else c for c in s.body]
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
