"""Tests for the Phase-2a scalar-matmul move composer.

Covers ``skeleton.lift_matmul`` (SEMIRING regime detection),
``tree.build_matmul_tree`` (generative reduce → thread → register Fork tree),
and ``materialize.build_matmul_tile`` (the K_o/K_i serial tower + masked-axis
guards). A final test drives the whole tile pipeline with the composer enabled.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import GridTile, SerialTile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_matmul_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import lift_matmul, lift_pointwise
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_matmul_tree

_MM_KEYS = {"MAP_N_THREAD", "MAP_N_REG", "MAP_M_THREAD", "MAP_M_REG", "RED_BK", "RED_FK"}


def _matmul(m: int, n: int, k: int) -> LoopOp:
    """Canonical decomposed matmul ``o[m,n] = sum_k a[m,k]·b[k,n]``."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("m", m),
                body=(
                    Loop(
                        axis=Axis("n", n),
                        body=(
                            Loop(
                                axis=Axis("k", k),
                                body=(
                                    Load(name="a", input="a", index=(Var("m"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("n"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Write(output="o", index=(Var("m"), Var("n")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _sum_reduce() -> LoopOp:
    """A plain (non-matmul) reduce — single K-indexed load → MONOID, not SEMIRING."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("i", 8),
                body=(
                    Loop(
                        axis=Axis("k", 16),
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="acc"),
                ),
            ),
        ),
    )


def test_lift_matmul_names_axes():
    skel = lift_matmul(_matmul(64, 96, 128))
    assert skel is not None
    assert skel.inner_n.extent == 96
    assert skel.outer_m.extent == 64
    # axis names are canonicalized (a0/a1/a2…) by LoopOp normalization
    assert skel.k_extent == 128
    assert skel.k_name == skel.k_loop.axis.name


def test_lift_matmul_rejects_non_matmul_reduce():
    assert lift_matmul(_sum_reduce()) is None


def test_lift_pointwise_rejects_matmul():
    # the matmul must NOT be claimed by the pointwise regime (it has a reduce)
    assert lift_pointwise(_matmul(64, 96, 128)) is None


def test_tree_leaves_complete_and_within_budget():
    skel = lift_matmul(_matmul(128, 128, 128))
    tree = build_matmul_tree(skel, base_knobs={}, kernel_name="k")
    leaves = flatten_leaves([tree])
    assert len(leaves) > 1
    for leaf in leaves:
        kn = leaf.knobs
        assert _MM_KEYS <= set(kn), f"leaf missing matmul knobs: {kn}"
        assert kn["MAP_N_THREAD"] * kn["MAP_M_THREAD"] <= 1024, kn
        assert kn["RED_FK"] * kn["MAP_N_REG"] * kn["MAP_M_REG"] <= 128, kn
        assert 128 % (kn["RED_BK"] * kn["RED_FK"]) == 0, f"bk·fk must divide K: {kn}"


def test_materialize_emits_k_serial_tower():
    skel = lift_matmul(_matmul(128, 128, 128))
    knobs = {"MAP_N_THREAD": 8, "MAP_N_REG": 1, "MAP_M_THREAD": 16, "MAP_M_REG": 4, "RED_BK": 32, "RED_FK": 1}
    tile = build_matmul_tile(skel, knobs, kernel_name="k", base_knobs={})
    assert isinstance(tile, TileOp)
    serials = list(tile.body.iter_of_type(SerialTile))
    kinds = {s.kind for s in serials}
    assert "serial_outer" in kinds and "stage_inner" in kinds, f"K tower missing: {kinds}"
    # K_o = K / BK = 128 / 32 = 4 ; K_i = BK = 32
    outer = next(s for s in serials if s.kind == "serial_outer")
    inner = next(s for s in serials if s.kind == "stage_inner")
    assert outer.axis.extent.as_static() == 4
    assert inner.axis.extent.as_static() == 32


def test_pipeline_uses_composer_for_matmul(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_MOVE_COMPOSER", "1")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (64, 128)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (128, 96)), node_id="b")
    g.add_node(op=_matmul(64, 96, 128), inputs=["a", "b"], output=Tensor("o", (64, 96)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    out = Pipeline.build(TILE_PASSES).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.knobs.get("RED_BK", 0) >= 1, "composer should stamp the reduce-tile vocabulary"
    assert tile_op.knobs.get("BN", 0) == 0, "legacy planner must not have run for a composer-covered matmul"
    assert isinstance(tile_op.body[0], GridTile)
