"""Tests for the smem bank-conflict simulator.

Each fixture builds a minimal ``TileOp`` with a known ``Stage`` + body
``Load`` and asserts the predicted lane→bank distribution.

Expected ``max_way`` values are derived from first principles. To
verify on real hardware, dump the equivalent kernel and run::

    ncu --metrics smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum

A 32-way conflict in this simulator predicts 31 bank-conflict events
per warp-LDS instruction.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.diagnostics.bank_conflicts import (
    BANKS,
    WARP_SIZE,
    StageBinding,
    simulate,
    simulate_graph,
    thread_axis_env,
)
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Load, Loop, Tile
from deplodock.compiler.ir.tile.ir import AffineAddressing, Stage, TileOp

# ---------------------------------------------------------------------------
# Fixture builder — a 1-Stage, 1-Load TileOp wrapped in a Tile with a
# single thread axis (so threadIdx.x maps directly to that axis).
# ---------------------------------------------------------------------------


def _build_tileop(
    *,
    thread_axes: tuple[BoundAxis, ...],
    stage_axes: tuple[Axis, ...],
    pad: tuple[int, ...] = (),
    load_index,
    inner_loop_axis: Axis | None = None,
    stage_cls=Stage,
) -> TileOp:
    """Build ``TileOp(Tile(Stage; [Loop?](Load)))``.

    ``load_index`` is a tuple of ``Expr`` matching the staged buffer's
    expected access (rank == ``len(stage_axes)``).
    ``inner_loop_axis`` wraps the Load in a serial Loop — used when the
    test wants to bind a k-iter; pass ``None`` for a Load directly in
    the Tile body.
    """
    addressing = AffineAddressing(dims=tuple(range(len(stage_axes))))
    stage = stage_cls(
        name="smem",
        buf="src",
        origin=tuple(Literal(0) for _ in range(len(stage_axes))),
        axes=stage_axes,
        addressing=addressing,
        pad=pad,
    )
    load = Load(name="x", input="smem", index=tuple(load_index))
    if inner_loop_axis is not None:
        body = (stage, Loop(axis=inner_loop_axis, body=(load,)))
    else:
        body = (stage, load)
    tile = Tile(axes=thread_axes, body=body)
    return TileOp(body=(tile,), name="probe")


def _binding_of(tile_op: TileOp) -> StageBinding:
    from deplodock.compiler.diagnostics.bank_conflicts import find_stage_bindings

    bindings = find_stage_bindings(tile_op)
    assert len(bindings) == 1, f"expected exactly one binding, got {len(bindings)}"
    return bindings[0]


# ---------------------------------------------------------------------------
# Direct unit tests for thread_axis_env (mirrors the runtime decode).
# ---------------------------------------------------------------------------


def test_thread_axis_env_single_axis():
    ax = (Axis("a", 32),)
    assert thread_axis_env(ax, 0) == {"a": 0}
    assert thread_axis_env(ax, 31) == {"a": 31}


def test_thread_axis_env_two_axes_outer_first():
    """``Tile.thread_axes = (outer:8, inner:4)`` flattens via tid =
    outer*4 + inner. Lane 0 → (0,0); lane 4 → (1,0); lane 7 → (1,3)."""
    ax = (Axis("o", 8), Axis("i", 4))
    assert thread_axis_env(ax, 0) == {"o": 0, "i": 0}
    assert thread_axis_env(ax, 4) == {"o": 1, "i": 0}
    assert thread_axis_env(ax, 7) == {"o": 1, "i": 3}
    assert thread_axis_env(ax, 31) == {"o": 7, "i": 3}


# ---------------------------------------------------------------------------
# Bank-conflict fixtures
# ---------------------------------------------------------------------------


def test_row_strided_load_no_pad_is_32_way_conflict():
    """All 32 lanes read different rows at fixed col=0 from a (32,32)
    smem; row stride = 32 = #banks → every lane lands on bank 0."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        pad=(),
        load_index=(Var("a"), Literal(0)),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 32
    assert r.lane_banks == [0] * WARP_SIZE
    assert r.counts[0] == 32


def test_row_strided_load_pad_one_clears_conflict():
    """Same access, pad=(0,1) makes row stride 33; bank = (lane*33)%32 =
    lane%32, all distinct → no conflict."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        pad=(0, 1),
        load_index=(Var("a"), Literal(0)),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 1
    assert sorted(r.lane_banks) == list(range(32))


def test_col_coalesced_load_no_conflict():
    """All lanes read row 0 at column = lane → bank = lane → 1-way."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        pad=(),
        load_index=(Literal(0), Var("a")),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 1


def test_register_tile_F4_row_stride_16_8way_conflict():
    """Pattern matmul-A loads with F=4: lane reads row=lane*4 in a
    (128,16) smem at fixed col=0. row_stride=16 → bank=(lane*4*16)%32 =
    (lane*64)%32 = 0 → 32-way."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 128), Axis("c", 16)),
        pad=(),
        load_index=(Var("a") * Literal(4), Literal(0)),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 32
    assert r.counts[0] == 32


def test_register_tile_F4_pad_one_partial_fix():
    """Same F=4 pattern with pad=(0,1) → row stride 17.
    addr = lane*4*17 = lane*68; bank = (lane*68)%32 = (lane*4)%32.
    lane*4 mod 32 cycles through {0,4,8,...,28} (8 banks) with 4 lanes
    each → max_way 4 (mild conflict — pad=1 is not enough for F=4)."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 128), Axis("c", 16)),
        pad=(0, 1),
        load_index=(Var("a") * Literal(4), Literal(0)),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 4
    nonzero = [c for c in r.counts if c > 0]
    assert len(nonzero) == 8
    assert all(c == 4 for c in nonzero)


def test_two_thread_axes_decode_into_index():
    """``Tile.thread_axes=(o:8, i:4)`` flattens via tid=o*4+i.
    Stage(8,4); Load `[o,i]` → lane reads `[lane//4, lane%4]`.
    addr = (lane//4)*4 + (lane%4) = lane → bank = lane → 1-way."""
    o = Axis("o", 8)
    i = Axis("i", 4)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(o, BIND_THREAD), BoundAxis(i, BIND_THREAD)),
        stage_axes=(Axis("r", 8), Axis("c", 4)),
        pad=(),
        load_index=(Var("o"), Var("i")),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    assert r.max_way == 1
    assert sorted(r.lane_banks) == list(range(32))


def test_k_iter_changes_bank_for_col_coalesced():
    """Load `[0, a + k_iter]`: shifting k_iter shifts the bank by k_iter
    but the per-lane spread is unchanged (still 1-way). Confirms k_iter
    propagates into the eval env."""
    a = Axis("a", 32)
    k = Axis("k", 4)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        pad=(),
        load_index=(Literal(0), Var("a") + Var("k")),
        inner_loop_axis=k,
    )
    binding = _binding_of(tile_op)
    r0 = simulate(binding, k_iter=0)
    r2 = simulate(binding, k_iter=2)
    assert r0 is not None and r2 is not None
    assert r0.max_way == 1 and r2.max_way == 1
    # Each lane shifts by 2 banks when k_iter goes 0→2.
    assert r2.lane_banks == [(b + 2) % BANKS for b in r0.lane_banks]


def test_buffered_stage_slot_dim_dropped():
    """When the Load index has a leading slot dim that exceeds Stage.axes
    rank, simulator drops it (slot is CTA-uniform, doesn't change banks).
    Verifies the slot=0 vs slot=1 case yields identical bank spread."""

    a = Axis("a", 32)
    base_kwargs = dict(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        pad=(),
    )
    tile_op_slot0 = _build_tileop(**base_kwargs, load_index=(Literal(0), Var("a"), Literal(0)))
    tile_op_slot1 = _build_tileop(**base_kwargs, load_index=(Literal(1), Var("a"), Literal(0)))
    r0 = simulate(_binding_of(tile_op_slot0))
    r1 = simulate(_binding_of(tile_op_slot1))
    assert r0 is not None and r1 is not None
    assert r0.lane_banks == r1.lane_banks
    assert r0.max_way == 32  # row-strided → 32-way as in fixture 1


def test_simulate_graph_walks_multiple_tileops():
    """``simulate_graph`` should walk every TileOp in a Graph."""
    g = Graph()
    a = Axis("a", 32)
    for nid in ("k0", "k1"):
        op = _build_tileop(
            thread_axes=(BoundAxis(a, BIND_THREAD),),
            stage_axes=(Axis("r", 32), Axis("c", 32)),
            load_index=(Var("a"), Literal(0)),
        )
        op.name = nid
        g.add_node(op, [], None, node_id=nid)
    results = simulate_graph(g)
    # Two TileOps × one (Stage, Load) each = 2 results.
    assert len(results) == 2
    assert all(r.max_way == 32 for r in results)


def test_stage_filter_restricts_results():
    """``stage_filter`` excludes non-matching stage names."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", 32), Axis("c", 32)),
        load_index=(Var("a"), Literal(0)),
    )
    g = Graph()
    g.add_node(tile_op, [], None, node_id="k0")
    assert simulate_graph(g, stage_filter={"smem"}) != []
    assert simulate_graph(g, stage_filter={"other"}) == []


@pytest.mark.parametrize("rows", [16, 32, 64, 128])
def test_avg_lane_per_bank_consistent(rows: int):
    """``avg_way`` equals 32/distinct_banks. Sanity: row-strided no-pad
    always lands on 1 bank → avg=32."""
    a = Axis("a", 32)
    tile_op = _build_tileop(
        thread_axes=(BoundAxis(a, BIND_THREAD),),
        stage_axes=(Axis("r", rows), Axis("c", 32)),
        load_index=(Var("a"), Literal(0)),
    )
    r = simulate(_binding_of(tile_op))
    assert r is not None
    nonzero = [c for c in r.counts if c > 0]
    assert r.avg_way == pytest.approx(WARP_SIZE / len(nonzero))
