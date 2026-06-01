"""Tests for ``085_warp_specialize`` — the WARP_SPECIALIZE Fork-emitting tile-IR pass.

The pass matches ``TileOp`` and emits a 2-child ``Fork`` (``WARP_SPECIALIZE=0`` /
``WARP_SPECIALIZE=1``) when the body contains a TMA ``StageBundle`` with
``pipeline_depth == 2`` inside a ``SerialTile(serial_outer)``. Otherwise
``RuleSkipped``.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import (
    CacheDim,
    SerialTile,
    Source,
    Stage,
    StageBundle,
    StagePolicy,
    SwizzleMode,
    ThreadTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
)
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.pipeline.pipeline import Fork
from deplodock.compiler.tensor import Tensor

_ws = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.085_warp_specialize")


def _ctx() -> Context:
    return Context(compute_capability="sm_90")


def _tile_op_pointwise() -> TileOp:
    """Trivial pointwise TileOp with no StageBundle — should be RuleSkipped."""
    ax = Axis("t", 4)
    body = Body(
        (
            ThreadTile(
                axes=(ax,),
                body=Body(
                    (
                        Load(name="v", input="x", index=(Var("t"),)),
                        Write(output="o", index=(Var("t"),), value="v"),
                    )
                ),
            ),
        )
    )
    return TileOp(body=body, name="k_pointwise")


def _tile_op_tma_pipelined() -> TileOp:
    """Synthetic TileOp shaped like post-080: SerialTile(serial_outer)
    containing a TMA StageBundle with pipeline_depth=2."""
    k_outer = Axis("k_outer", 8)
    src = Source(
        name="a_smem",
        buf="a",
        cache_dims=(
            CacheDim(axis=Axis("a_c0", 16), source_dim=0),
            CacheDim(axis=Axis("a_c1", 16), source_dim=1),
        ),
        origin=(Var("k_outer") * Literal(16, "int"), Literal(0, "int")),
    )
    tma_bundle = StageBundle(
        stages=(Stage(sources=(src,)),),
        body=Body(()),
        policy=StagePolicy.TMA,
        buffer_count=2,
        phase=Var("k_outer") % Literal(2, "int"),
        pipeline_depth=2,
        swizzle=SwizzleMode.NONE,
    )
    outer_loop = SerialTile(
        axis=k_outer,
        body=Body((tma_bundle,)),
        kind="serial_outer",
    )
    # 2D ThreadTile so the inner axis (n_i, extent 16) divides
    # n_producer_threads=32 cleanly (extension=2 rows). Single-axis
    # extent 128 would not satisfy 32 % inner_extent == 0.
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m_i", 16), Axis("n_i", 16)),
                body=Body((outer_loop,)),
            ),
        )
    )
    return TileOp(body=body, name="k_tma_pipelined")


def _run_rule(op: TileOp):
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(op.name, ()), node_id="op")
    node = g.nodes["op"]
    return _ws.rewrite(_ctx(), node)


def test_rule_skipped_on_pointwise():
    op = _tile_op_pointwise()
    with pytest.raises(RuleSkipped, match="no TMA"):
        _run_rule(op)


def test_pinned_ws1_on_ineligible_fails_loudly(monkeypatch):
    """A pinned ``DEPLODOCK_WARP_SPECIALIZE=1`` that can't be honored must
    **raise**, not silently RuleSkip into the non-WS kernel. (The motivating
    case is a warp-tier MMA matmul — ``no ThreadTile in body`` — but any
    ineligible op exercises it; the pointwise op has no TMA bundle.) A
    pinned-but-unhonorable knob erroring matches the BUFFER_COUNT / TMA pin
    policy, instead of handing back a kernel the pin never shaped. Without the
    pin the same op RuleSkips cleanly (see ``test_rule_skipped_on_pointwise``)."""
    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "1")
    op = _tile_op_pointwise()
    with pytest.raises(ValueError, match="WARP_SPECIALIZE=1 pinned but warp specialization cannot fire"):
        _run_rule(op)


def test_rule_skipped_when_ws_already_set():
    op = _tile_op_tma_pipelined()
    op_with_ws = TileOp(body=op.body, name=op.name, knobs={"WARP_SPECIALIZE": True})
    with pytest.raises(RuleSkipped, match="WARP_SPECIALIZE knob already set"):
        _run_rule(op_with_ws)


def test_emits_two_child_fork_on_tma_pipelined():
    op = _tile_op_tma_pipelined()
    result = _run_rule(op)
    assert isinstance(result, list), f"expected list[Fork], got {type(result).__name__}"
    assert len(result) == 2, f"expected 2 forks, got {len(result)}"
    knob_values = sorted(f.knobs["WARP_SPECIALIZE"] for f in result)
    assert knob_values == [False, True]
    for fork in result:
        assert isinstance(fork, Fork)
        assert fork.is_leaf
        # Each leaf expand returns one TileOp with WS knob stamped
        children = fork.expand()
        assert len(children) == 1
        assert isinstance(children[0], TileOp)
        assert children[0].knobs["WARP_SPECIALIZE"] == fork.knobs["WARP_SPECIALIZE"]


def test_env_pin_narrows_to_single_choice(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "1")
    op = _tile_op_tma_pipelined()
    result = _run_rule(op)
    # Single-choice short-circuit: return a bare TileOp, not a Fork list.
    assert isinstance(result, TileOp)
    assert result.knobs["WARP_SPECIALIZE"] is True


def test_env_pin_zero_returns_bare_tileop(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "0")
    op = _tile_op_tma_pipelined()
    result = _run_rule(op)
    assert isinstance(result, TileOp)
    assert result.knobs["WARP_SPECIALIZE"] is False


# ---------------------------------------------------------------------------
# Boundary checks — Tile pass emits only Tile IR
# ---------------------------------------------------------------------------


def test_ws1_emits_warp_specialize_stmt(monkeypatch):
    """WS=1 path produces a TileOp whose **WarpTile** body contains a
    single ``WarpSpecialize`` Stmt — not a hand-rolled
    Cond/Smem/MbarrierInit tree. The materializer is responsible for the
    lowering. Post-refactor (M6 of ``plans/warptile-primitive.md``) the
    inner tile is a ``WarpTile`` carrying a single role axis, not the
    extended ``ThreadTile`` the previous shape used."""
    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "1")
    op = _tile_op_tma_pipelined()
    result = _run_rule(op)
    assert isinstance(result, TileOp)
    # The inner tile is now a WarpTile (role-binding); no ThreadTile in
    # the rewritten body.
    assert not any(isinstance(s, ThreadTile) for s in result.body.iter()), "ThreadTile should be gone from WS=1 output"
    wt = next(s for s in result.body if isinstance(s, WarpTile))
    assert len(wt.axes) == 1, f"WarpTile should carry one role axis, got {len(wt.axes)}"
    body = list(wt.body)
    assert len(body) == 1, f"expected single WarpSpecialize in WarpTile body, got {len(body)}"
    ws = body[0]
    assert isinstance(ws, WarpSpecialize)
    # consumer_thread_axes carries the original ThreadTile axes (unshifted).
    assert tuple(ax.extent.as_static() for ax in ws.consumer_thread_axes) == (16, 16)
    # Role axis extent = (n_consumer + n_producer) / 32 = (256 + 32) / 32 = 9.
    assert wt.axes[0].extent.as_static() == 9


def test_ws1_tile_body_contains_no_kernel_ir(monkeypatch):
    """No Kernel-IR types (Smem / Sync / MbarrierInit / MbarrierWait /
    MbarrierArrive / SetMaxNReg) appear anywhere inside the WS=1
    TileOp body. The layering boundary is intact."""
    # Local imports here — the check enforces that Kernel-IR types do
    # not appear in the body, but we still need the types to assert.
    from deplodock.compiler.ir.kernel.ir import (
        MbarrierArrive,
        MbarrierInit,
        MbarrierWait,
        SetMaxNReg,
        Smem,
        Sync,
    )

    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "1")
    op = _tile_op_tma_pipelined()
    result = _run_rule(op)
    assert isinstance(result, TileOp)
    forbidden = (Smem, Sync, MbarrierInit, MbarrierWait, MbarrierArrive, SetMaxNReg)
    for stmt in result.body.iter():
        assert not isinstance(stmt, forbidden), f"Tile-IR pass leaked Kernel-IR type {type(stmt).__name__} into TileOp body"
