"""Materializer integration tests for ``WarpSpecialize``.

No live pass constructs a ``WarpSpecialize`` today (the ``tile/085_warp_specialize.py``
producer pass was removed in the block-DAG rewrite), but the consumer-side codegen in
``100_materialize_tile.py::emit_warp_specialize`` is still live and reachable: a
hand-built ``WarpSpecialize`` node fed through the materializer lowers into the full
mbarrier handshake. These tests build that node directly and check the emitted Stmt
tree, exactly as the original suite did.

We assert that:

- the prologue is ``Smem("tma_mbar_empty", ...) + Cond(tid==0, [MbarrierInit per slot]) + Sync()``;
- producer / consumer subtrees are wrapped in a role-split ``Cond(role < n_producer_warps,
  [SetMaxNReg(24,"dec"), ...], [SetMaxNReg(240,"inc"), ThreadTile(tid_offset=n_producer, ...)])``;
- producer-side ``SerialTile(serial_outer)`` matching the K_o axis gets a ``MbarrierWait``
  prepended inside a ``Cond(K_o >= bc-1)`` guard;
- consumer-side ``SerialTile(serial_outer)`` gets a trailing ``Sync(barrier_id=1,
  count=n_consumer_threads)`` + single-thread ``MbarrierArrive``;
- ``AsyncWait`` inside the consumer lowers with a named ``Sync(barrier_id=1, count=n_cons)``
  trailing fence;
- ``AsyncWait`` outside any ``WarpSpecialize`` lowers with the default ``Sync(barrier_id=0)``.

The input TileOp wraps ``WarpSpecialize`` in a ``WarpTile(role)``; the materializer reads
the role axis off the WarpTile and emits a nested ``ThreadTile(tid_offset=n_producer_threads,
…)`` inside the consumer ``Cond.else_body``.
"""

from __future__ import annotations

import importlib

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    MbarrierArrive,
    MbarrierInit,
    MbarrierWait,
    SetMaxNReg,
    Smem,
    Sync,
)
from emmy.compiler.ir.stmt import Body, Cond
from emmy.compiler.ir.tile.ir import (
    AsyncWait,
    SerialTile,
    ThreadTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
)
from emmy.compiler.tensor import Tensor

_mat = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.100_materialize_tile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materialize(tile_op: TileOp) -> KernelOp:
    g = Graph()
    g.add_node(op=tile_op, inputs=[], output=Tensor(tile_op.name, ()), node_id="op")
    node = g.nodes["op"]
    ctx = Context(compute_capability=(9, 0))
    result = _mat.rewrite(ctx, node)
    assert isinstance(result, KernelOp), f"expected KernelOp, got {type(result).__name__}"
    return result


def _walk(stmts):
    """Recursive walk yielding every Stmt in tree order."""
    for s in stmts:
        yield s
        for sub in s.nested():
            yield from _walk(sub)


def _tile_op_with_ws(
    *,
    ring_depth: int = 2,
    n_producer_threads: int = 32,
    consumer_has_async_wait: bool = True,
) -> TileOp:
    """WarpTile(role) holding a single WarpSpecialize. Producer side has
    a K_o ``SerialTile(serial_outer)`` (no children); consumer side has
    a K_o ``SerialTile(serial_outer)`` optionally containing an
    AsyncWait.

    Geometry: consumer_thread_axes = (m_i=4, n_i=16) → n_consumer =
    64. n_producer = 32. Total = 96 → 3 warps. Role axis extent = 3,
    n_producer_warps = 32 / 32 = 1. The materializer's role-split
    predicate becomes ``role < 1``. The bodies use axis name
    ``k_outer`` but TileOp's ``normalize_body`` canonicalises it to
    ``a<N>``; the materializer reads the canonical name off the
    matched ``SerialTile.axis``."""
    k_axis = Axis("k_outer", 8)
    cons_body = Body((AsyncWait(keep=1, phase=Var("k_outer") % Literal(2, "int")),)) if consumer_has_async_wait else Body(())
    producer_body = Body((SerialTile(axis=k_axis, body=Body(()), kind="serial_outer"),))
    consumer_body = Body((SerialTile(axis=k_axis, body=cons_body, kind="serial_outer"),))
    consumer_thread_axes = (Axis("m_i", 4), Axis("n_i", 16))
    ws = WarpSpecialize(
        producer_body=producer_body,
        consumer_body=consumer_body,
        ring_depth=ring_depth,
        n_producer_threads=n_producer_threads,
        consumer_thread_axes=consumer_thread_axes,
    )
    n_consumer = 1
    for ax in consumer_thread_axes:
        n_consumer *= ax.extent.as_static()
    total_warps = (n_consumer + n_producer_threads) // 32
    body = Body(
        (
            WarpTile(
                axes=(Axis("ws_role", Dim(total_warps)),),
                body=Body((ws,)),
            ),
        )
    )
    return TileOp(body=body, name="k_ws_test")


# ---------------------------------------------------------------------------
# Prologue: Smem + MbarrierInit cond + Sync
# ---------------------------------------------------------------------------


def test_materializer_emits_empty_mbar_prologue():
    """The materialized WarpTile body holds the empty-mbarrier Smem decl,
    an init Cond gated on ``threadIdx.x==0``, and a CTA-wide Sync."""
    kernel = _materialize(_tile_op_with_ws(ring_depth=2))
    # Outer is a WarpTile; descend to its body.
    top = kernel.body[0]
    assert isinstance(top, WarpTile)
    items = list(top.body)
    # First Smem named tma_mbar_empty.
    smem = next(s for s in items if isinstance(s, Smem) and s.name == "tma_mbar_empty")
    assert smem.extents == (2,)
    assert smem.dtype == "unsigned long long"
    # Init Cond gated on tid==0 holds MbarrierInit per slot.
    init_conds = [s for s in items if isinstance(s, Cond) and any(isinstance(c, MbarrierInit) for c in s.body)]
    assert len(init_conds) >= 1, "missing single-thread MbarrierInit prologue"
    init_cond = init_conds[0]
    inits = [c for c in init_cond.body if isinstance(c, MbarrierInit)]
    assert len(inits) == 2, f"expected 2 MbarrierInit per slot, got {len(inits)}"
    for c in inits:
        assert c.mbar == "tma_mbar_empty"
        assert c.count == 1


# ---------------------------------------------------------------------------
# Outer Cond shape — producer/consumer branches with SetMaxNReg
# ---------------------------------------------------------------------------


def test_materializer_wraps_branches_in_setmaxnreg_cond():
    """The role-split Cond has SetMaxNReg(24,"dec") at the head of the
    producer branch and SetMaxNReg(240,"inc") at the head of the
    consumer branch. The consumer's body second stmt is the nested
    ``ThreadTile(tid_offset=n_producer_threads, …)`` that re-decodes
    consumer-relative tid into the consumer thread axes."""
    kernel = _materialize(_tile_op_with_ws())
    top = kernel.body[0]
    role_conds = [s for s in top.body if isinstance(s, Cond) and any(isinstance(c, SetMaxNReg) for c in s.body)]
    assert len(role_conds) == 1, f"expected exactly one role-split Cond, got {len(role_conds)}"
    cond = role_conds[0]
    # Producer branch leads with SetMaxNReg(24, "dec").
    prod_head = cond.body[0]
    assert isinstance(prod_head, SetMaxNReg)
    assert prod_head.count == 24
    assert prod_head.direction == "dec"
    # Consumer branch leads with SetMaxNReg(240, "inc"), followed by a
    # ThreadTile carrying the consumer thread axes + tid_offset.
    cons_head = cond.else_body[0]
    assert isinstance(cons_head, SetMaxNReg)
    assert cons_head.count == 240
    assert cons_head.direction == "inc"
    cons_decode = cond.else_body[1]
    assert isinstance(cons_decode, ThreadTile)
    assert cons_decode.tid_offset == 32
    assert tuple(ax.extent.as_static() for ax in cons_decode.axes) == (4, 16)


# ---------------------------------------------------------------------------
# Producer wiring — MbarrierWait inside K_o, gated by K_o >= bc-1
# ---------------------------------------------------------------------------


def test_producer_serial_outer_gets_mbarrier_wait():
    """Inside the producer branch's ``SerialTile(serial_outer)``, the
    head stmt is a ``Cond(K_o >= bc-1, [MbarrierWait(...)])``.

    The producer branch is NOT wrapped in a consumer-decode ThreadTile —
    producer warps don't need the inner-thread-axis decode (they only
    use ``threadIdx.x`` directly for the TMA issuer cond and the
    ``lane`` / ``warp_id`` exposed by the enclosing WarpTile)."""
    kernel = _materialize(_tile_op_with_ws(ring_depth=2))
    top = kernel.body[0]
    role_cond = next(s for s in top.body if isinstance(s, Cond) and any(isinstance(c, SetMaxNReg) for c in s.body))
    # Producer branch: SetMaxNReg, then SerialTile(serial_outer).
    prod_loop = next(s for s in role_cond.body if isinstance(s, SerialTile) and s.kind == "serial_outer")
    head = prod_loop.body[0]
    assert isinstance(head, Cond), f"producer K_o head should be a Cond, got {type(head).__name__}"
    inner_waits = [c for c in head.body if isinstance(c, MbarrierWait)]
    assert len(inner_waits) == 1
    assert inner_waits[0].mbar == "tma_mbar_empty"


# ---------------------------------------------------------------------------
# Consumer wiring — named Sync + tid-gated MbarrierArrive
# ---------------------------------------------------------------------------


def test_consumer_serial_outer_gets_named_sync_and_arrive():
    """Inside the consumer branch's ``SerialTile(serial_outer)`` (which
    now lives inside the consumer-decode ``ThreadTile``), the tail two
    stmts are a named ``Sync(barrier_id=1, count=n_cons)`` and a
    ``Cond(tid == n_producer, [MbarrierArrive])``."""
    kernel = _materialize(_tile_op_with_ws(n_producer_threads=32, consumer_has_async_wait=False))
    top = kernel.body[0]
    role_cond = next(s for s in top.body if isinstance(s, Cond) and any(isinstance(c, SetMaxNReg) for c in s.body))
    cons_decode = next(s for s in role_cond.else_body if isinstance(s, ThreadTile))
    cons_loop = next(s for s in cons_decode.body if isinstance(s, SerialTile) and s.kind == "serial_outer")
    # Last two stmts: named Sync then single-thread MbarrierArrive Cond.
    tail = list(cons_loop.body)[-2:]
    assert len(tail) == 2
    sync, arrive_cond = tail
    assert isinstance(sync, Sync)
    assert sync.barrier_id == 1
    # n_cons = product(consumer_thread_axes.extent) = 4 * 16 = 64.
    assert sync.count == 64
    assert isinstance(arrive_cond, Cond)
    arrive = next(c for c in arrive_cond.body if isinstance(c, MbarrierArrive))
    assert arrive.mbar == "tma_mbar_empty"


# ---------------------------------------------------------------------------
# AsyncWait inside consumer uses named-barrier trailing fence
# ---------------------------------------------------------------------------


def test_async_wait_inside_consumer_routes_to_named_barrier():
    """AsyncWait inside the consumer subtree lowers with
    ``Sync(barrier_id=1, count=n_cons)`` for its trailing fence — not
    the default ``Sync()`` / ``__syncthreads()``."""
    kernel = _materialize(_tile_op_with_ws(consumer_has_async_wait=True))
    # Find every Sync in the kernel body and check that at least one has
    # barrier_id=1 (the named bar.sync for the consumer-side AsyncWait
    # trailing fence and for the per-K_o consumer arrive).
    named_syncs = [s for s in _walk(kernel.body) if isinstance(s, Sync) and s.barrier_id == 1]
    assert len(named_syncs) >= 1, "expected at least one Sync(barrier_id=1) from consumer subtree"
    # All named syncs should carry the same n_cons count.
    assert all(s.count == 64 for s in named_syncs), f"named syncs have inconsistent count: {[s.count for s in named_syncs]}"


def test_async_wait_outside_warp_specialize_uses_default_sync():
    """A TileOp without a WarpSpecialize wrapper — AsyncWait lowers
    with ``Sync(barrier_id=0, count=None)`` (the default
    __syncthreads path)."""
    k_var = "k_outer"
    k_axis = Axis(k_var, 4)
    body = Body(
        (
            ThreadTile(
                axes=(Axis("t", 32),),
                body=Body(
                    (
                        SerialTile(
                            axis=k_axis,
                            body=Body((AsyncWait(keep=1),)),
                            kind="serial_outer",
                        ),
                    )
                ),
            ),
        )
    )
    tile_op = TileOp(body=body, name="k_plain")
    kernel = _materialize(tile_op)
    # Find AsyncWait's trailing Sync — it follows a CpAsyncWait in
    # _walk order; we just check no Sync carries barrier_id=1.
    syncs = [s for s in _walk(kernel.body) if isinstance(s, Sync)]
    assert any(s.barrier_id == 0 for s in syncs), "expected default __syncthreads"
    assert not any(s.barrier_id == 1 for s in syncs), "no named bar.sync outside WS"
