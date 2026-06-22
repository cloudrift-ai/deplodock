"""Tests for the Phase-3b cooperative-reduce move composer.

Covers ``skeleton.lift_coop_reduce`` (MONOID regime, K ≥ warp_size) and
``materialize.build_coop_reduce_tile`` (whole-CTA tower with a ``K_c`` THREAD
axis). End-to-end combine correctness is covered by the accuracy oracle (a real
``torch.sum`` run through the composer).
"""

from __future__ import annotations

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import ThreadTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import COOP_BR, RED_BK, RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_coop_reduce_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import CoopReduceSkeleton
from deplodock.compiler.pipeline.passes.lowering.tile.partition.walk import walk_nest


def _reduce_epilogue_map(rows: int, k: int) -> LoopOp:
    """`inv=1/sum_k x; o[i,k]=x[i,k]*inv` — a reduce, a scalar epilogue, and a
    second-pass map loop over a *different-named* k axis of the same extent
    (the RMSNorm/softmax shape)."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("i", rows),
                body=(
                    Loop(
                        axis=Axis("kr", k),
                        body=(
                            Load(name="xv", input="x", index=(Var("i"), Var("kr"))),
                            Accum(name="acc", value="xv", op=ElementwiseImpl("add"), axes=("kr",)),
                        ),
                    ),
                    Assign(name="inv", op=ElementwiseImpl("reciprocal"), args=("acc",)),
                    Loop(
                        axis=Axis("km", k),
                        body=(
                            Load(name="xv2", input="x", index=(Var("i"), Var("km"))),
                            Assign(name="ov", op=ElementwiseImpl("multiply"), args=("xv2", "inv")),
                            Write(output="o", index=(Var("i"), Var("km")), value="ov"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_walk_recognizes_reduce_with_epilogue_and_map():
    nest = walk_nest(_reduce_epilogue_map(64, 128))
    assert nest is not None and nest.k_extent == 128
    # Both the reduce and the second-pass map loop (different names, same extent)
    # are cooperative-split targets.
    assert len(nest.target_names) == 2, f"reduce + map should both be K targets: {nest.target_names}"
    # The scalar epilogue (reciprocal) survives in the body to ride the row tile.
    assert any(isinstance(s, Assign) for s in nest.inner_body), "epilogue Assign must survive the walk"


def _sum(rows: int, k: int) -> LoopOp:
    """``o[i] = sum_k x[i,k]`` — a plain MONOID reduce."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("i", rows),
                body=(
                    Loop(
                        axis=Axis("k", k),
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                            Accum(name="acc", value="x_v", op=ElementwiseImpl("add"), axes=("k",)),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="acc"),
                ),
            ),
        ),
    )


def test_lift_coop_reduce_detects_monoid():
    skel = walk_nest(_sum(64, 128))
    assert skel is not None
    assert skel.k_extent == 128
    assert skel.inner_n.extent == 64


def test_lift_coop_reduce_accepts_small_k():
    # K=16 < warp_size still composes — the builder deploys COOP_BR=1, a plain
    # per-row serial reduce (offers bound br·bk·fk ≤ k_extent), no warp shuffle.
    assert isinstance(walk_nest(_sum(64, 16)), CoopReduceSkeleton)


def test_lift_coop_reduce_rejects_matmul():
    from tests.compiler.passes.test_move_composer_matmul import _matmul  # noqa: PLC0415

    assert not isinstance(walk_nest(_matmul(64, 64, 128)), CoopReduceSkeleton)


def test_build_coop_tile_has_kc_thread_axis():
    lo = _sum(64, 128)
    skel = walk_nest(lo)
    knobs = {RED_BK.name: 1, RED_FK.name: 1, COOP_BR.name: 128}
    tile = build_coop_reduce_tile(skel, knobs, kernel_name="k", base_knobs={}, dag=iter_dag(lo))
    assert isinstance(tile, TileOp)
    # Axis names canonicalize to a0/a1/… — match on structure, not name. The
    # cooperative BR threads bind to a THREAD axis of extent 128.
    thread_axes = {ax.name: ax.extent.as_static() for tt in tile.body.iter_of_type(ThreadTile) for ax in tt.axes}
    assert 128 in thread_axes.values(), "coop reduce must bind BR=128 threads to a K_c THREAD axis"
    # The combine fires iff the Accum reduces over a THREAD axis
    # (Accum.axes ∩ ThreadTile.axes ≠ ∅) — the σ propagated K_c into Accum.axes.
    accums = list(tile.body.iter_of_type(Accum))
    assert accums and any(set(a.axes) & set(thread_axes) for a in accums), "Accum.axes must intersect a THREAD axis for the combine"
