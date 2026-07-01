"""Structural-coverage test for the permitted-move catalog (``lowering/tile/_catalog``).

The scheduling emit (``010_recognize`` → ``_schedule``) enumerates the catalog into the tile fork; this
file pins the catalog's **legal product** two ways:

- the catalog function ``scalar_tile_moves()`` equals the hand-computed ``(par × reg)`` grid + per-cell,
  option-0-first and legality-guarded (``par_n·par_m ≤ 1024``);
- the **leaf set** the scheduler actually emits over a matmul fixture equals that product (keyed
  ``TILE@<k_axis>``) — so a missing / extra move is caught structurally, without lowering a kernel.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.schedule import TilePlan
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.knob import axis_of, family_of, family_value
from deplodock.compiler.pipeline.passes.lowering.tile._catalog import _MAX_BLOCK_THREADS, scalar_tile_moves
from deplodock.compiler.pipeline.pipeline import Run

# The hand-computed legal product as explicit literals — per-cell option-0, then the (par × reg) grid
# spelled through the codec (the default ``f1x1`` register sub-tile suppresses, so ``reg=(1,1)`` is the
# bare ``n<par_n>x<par_m>``). Written out (not recomputed from ``_SCALAR_*``) so a change to the grid,
# the ordering, or the legality filter is caught here explicitly.
_EXPECTED_MOVES = [
    "",
    "n16x8", "n16x8/f2x2", "n16x8/f4x4", "n16x8/f2x4", "n16x8/f4x2",
    "n16x16", "n16x16/f2x2", "n16x16/f4x4", "n16x16/f2x4", "n16x16/f4x2",
    "n32x8", "n32x8/f2x2", "n32x8/f4x4", "n32x8/f2x4", "n32x8/f4x2",
    "n32x16", "n32x16/f2x2", "n32x16/f4x4", "n32x16/f2x4", "n32x16/f4x2",
]  # fmt: skip


def test_scalar_tile_moves_equals_hand_product():
    moves = scalar_tile_moves()
    assert moves == _EXPECTED_MOVES
    assert moves[0] == ""  # conservative per-cell option-0 leads (cold greedy stays stable)
    assert len(set(moves)) == len(moves)  # no duplicate candidates
    # Every non-empty move round-trips the codec grammar and stays inside the thread budget.
    for spec in moves[1:]:
        plan = TilePlan.parse(spec)
        assert plan.spell() == spec
        assert plan.units_n * plan.units_m <= _MAX_BLOCK_THREADS


def _matmul_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Dim(64), Dim(64))), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(64), Dim(64))), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Dim(64), Dim(64))), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def test_schedule_leaf_set_equals_catalog():
    """The tile scheduler's emitted leaf set over a static matmul equals the catalog product, keyed
    ``TILE@<k_axis>`` — the enumeration IS the legal move product."""
    captured: list[str] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        rows = [dict(leaf.knobs) if hasattr(leaf, "knobs") else {} for leaf in leaves]
        # The contraction tile fork is the one whose leaves carry a ``TILE@<k>`` key.
        if any("TILE" in family_of(k) for row in rows for k in row):
            captured.extend(str(family_value(row, "TILE")) for row in rows)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert captured, "no TILE fork was emitted for the matmul"
    assert sorted(captured) == sorted(scalar_tile_moves())


def test_schedule_leaves_key_tile_by_contraction_axis():
    """Each emitted contraction leaf keys its output tile ``TILE@<k_axis>`` (a single eligible axis),
    so the bare catalog spelling canonicalizes onto the node's contraction axis."""
    axes: set[str | None] = set()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            for k in getattr(leaf, "knobs", {}):
                if family_of(k) == "TILE":
                    axes.add(axis_of(k))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert axes and None not in axes  # every TILE key is axis-named (TILE@<k>), never bare
    assert len(axes) == 1  # one contraction → one eligible k-axis
