"""Tests for the Phase-1 pointwise move composer
(``passes/lowering/tile/partition/``).

Covers the three new layers: ``skeleton.lift_pointwise`` (regime detection),
``tree.build_pointwise_tree`` (generative Fork tree — complete leaves, budget
pruning), and ``materialize.build_pointwise_tile`` (the ``TileOp`` tower +
masked-axis store guards). A final test drives the whole tile pipeline with the
composer enabled and asserts the greenfield knob vocabulary reaches the
``TileOp``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Cond
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_pointwise_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import PointwiseSkeleton
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_pointwise_tree
from deplodock.compiler.pipeline.passes.lowering.tile.partition.walk import walk_nest

_MAP_KEYS = {"MAP_N_THREAD", "MAP_N_REG", "MAP_M_THREAD", "MAP_M_REG"}


def _pointwise(m: int, n: int) -> LoopOp:
    """A 2-D elementwise copy ``o[i,j] = x[i,j]`` (inner axis ``j`` = N)."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("i", m),
                body=(
                    Loop(
                        axis=Axis("j", n),
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _reduce() -> LoopOp:
    """A reduce kernel ``o[i] = sum_k x[i,k]`` — not pointwise."""
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


def test_lift_pointwise_names_axes():
    skel = walk_nest(_pointwise(4, 8))
    assert skel is not None
    assert (skel.inner_n.extent, skel.inner_n.symbolic) == (8, False)
    assert skel.outer_m is not None and skel.outer_m.extent == 4
    assert skel.extra_outer == ()


def test_lift_rejects_reduce():
    assert not isinstance(walk_nest(_reduce()), PointwiseSkeleton)


def test_tree_leaves_complete_and_within_budget():
    skel = walk_nest(_pointwise(64, 64))
    tree = build_pointwise_tree(skel, base_knobs={}, kernel_name="k")
    leaves = flatten_leaves([tree])
    assert len(leaves) > 1, "64x64 has many legal variants → a real fork tree"
    for leaf in leaves:
        kn = leaf.knobs
        assert _MAP_KEYS <= set(kn), f"leaf missing map knobs: {kn}"
        assert kn["MAP_N_THREAD"] * kn["MAP_M_THREAD"] <= 1024, kn
        assert kn["MAP_N_REG"] * kn["MAP_M_REG"] <= 128, kn


def test_materialize_clean_tower_no_guard():
    skel = walk_nest(_pointwise(64, 64))
    knobs = {"MAP_N_THREAD": 32, "MAP_N_REG": 1, "MAP_M_THREAD": 8, "MAP_M_REG": 1}
    tile = build_pointwise_tile(skel, knobs, kernel_name="k", base_knobs={})
    assert isinstance(tile, TileOp)
    grid = next(s for s in tile.body if isinstance(s, GridTile))
    assert list(grid.body.iter_of_type(ThreadTile)), "tower should nest a ThreadTile"
    assert not list(tile.body.iter_of_type(Cond)), "a clean (divisible) tile needs no store guard"


def test_materialize_masked_has_store_guard():
    skel = walk_nest(_pointwise(70, 130))  # neither axis divides the tile
    knobs = {"MAP_N_THREAD": 8, "MAP_N_REG": 1, "MAP_M_THREAD": 32, "MAP_M_REG": 1}
    tile = build_pointwise_tile(skel, knobs, kernel_name="k", base_knobs={})
    conds = list(tile.body.iter_of_type(Cond))
    assert len(conds) == 2, f"masked N and M each get a boundary guard, got {len(conds)}"


def test_pipeline_uses_composer_when_enabled(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_MOVE_COMPOSER", "1")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=_pointwise(4, 8), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    out = Pipeline.build(TILE_PASSES).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    # The composer stamped a real map tile...
    assert tile_op.knobs.get("MAP_N_THREAD", 0) >= 1, "composer should stamp the greenfield vocabulary"
    # ...and the legacy planner did NOT run: legacy ``BN`` stays at its OFF
    # sentinel (0), stamped by the downstream ``apply_off_defaults`` for every
    # still-registered knob during the transition, never a real thread size.
    assert tile_op.knobs.get("BN", 0) == 0, "legacy planner must not have run for a composer-covered regime"
