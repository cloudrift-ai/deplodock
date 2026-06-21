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
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.ir.tile.ir import ThreadTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_coop_reduce_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import lift_coop_reduce


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
    skel = lift_coop_reduce(_sum(64, 128))
    assert skel is not None
    assert skel.k_extent == 128
    assert skel.inner_n.extent == 64


def test_lift_coop_reduce_rejects_small_k():
    # K=16 < warp_size → stays on the legacy / pointwise path
    assert lift_coop_reduce(_sum(64, 16)) is None


def test_lift_coop_reduce_rejects_matmul():
    from tests.compiler.passes.test_move_composer_matmul import _matmul  # noqa: PLC0415

    assert lift_coop_reduce(_matmul(64, 64, 128)) is None


def test_build_coop_tile_has_kc_thread_axis():
    skel = lift_coop_reduce(_sum(64, 128))
    knobs = {"RED_BK": 1, "RED_FK": 1, "COOP_BR": 128}
    tile = build_coop_reduce_tile(skel, knobs, kernel_name="k", base_knobs={})
    assert isinstance(tile, TileOp)
    # Axis names canonicalize to a0/a1/… — match on structure, not name. The
    # cooperative BR threads bind to a THREAD axis of extent 128.
    thread_axes = {ax.name: ax.extent.as_static() for tt in tile.body.iter_of_type(ThreadTile) for ax in tt.axes}
    assert 128 in thread_axes.values(), "coop reduce must bind BR=128 threads to a K_c THREAD axis"
    # The combine fires iff the Accum reduces over a THREAD axis
    # (Accum.axes ∩ ThreadTile.axes ≠ ∅) — the σ propagated K_c into Accum.axes.
    accums = list(tile.body.iter_of_type(Accum))
    assert accums and any(set(a.axes) & set(thread_axes) for a in accums), "Accum.axes must intersect a THREAD axis for the combine"
