"""Structural-coverage test for the permitted-move catalog (``search/space.py``).

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
from deplodock.compiler.ir.schedule import TilePlan
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.knob import axis_of, family_of, family_value
from deplodock.compiler.pipeline.pipeline import Run
from deplodock.compiler.pipeline.search.space import MAX_BLOCK_THREADS as _MAX_BLOCK_THREADS
from deplodock.compiler.pipeline.search.space import scalar_tile_moves

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
    """The tile scheduler's emitted leaf set over a static f32 matmul equals the catalog's legal
    product (keyed ``FAMILY@<k_axis>``) — the enumeration IS the tile × stage × reduce move product:

    - distinct ``TILE`` values = exactly ``scalar_tile_moves()`` (f32 → no warp moves);
    - the per-cell tile rides serial + the coop/ILP moves (non-output-tiled tier only), no split-K
      (one thread per cell already saturates the 64×64 grid — the occupancy gate);
    - every tiled tile rides serial × {gmem-direct + the resolved d1 stages} + the divisor-guarded
      split-K widths on the gmem-direct row only (split partials are gmem-direct).
    """
    from deplodock.compiler.pipeline.search.space import coop_reduce_moves, splitk_moves

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if any("TILE" in family_of(k) for k in row):
                rows.append(row)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert rows, "no TILE fork was emitted for the matmul"
    tiles = {str(family_value(r, "TILE")) for r in rows}
    assert tiles == set(scalar_tile_moves())
    # The hand-computed legal product: per-cell = serial + the 5 coop/ILP moves (6 rows); each of
    # the 20 tiled tiles = gmem-direct × (serial + 6 split moves) + 2 resolved d1 stages × serial.
    n_percell = 1 + len(coop_reduce_moves())
    n_tiled = (1 + len(splitk_moves(warp=False))) + 2
    assert len(rows) == n_percell + 20 * n_tiled, f"got {len(rows)} rows"
    by_tile: dict[str, list[dict]] = {}
    for r in rows:
        by_tile.setdefault(str(family_value(r, "TILE")), []).append(r)
    percell = by_tile[""]
    assert {str(family_value(r, "REDUCE")) for r in percell} == {"None", *coop_reduce_moves()}
    assert all(family_value(r, "STAGE") is None for r in percell), "per-cell has no operand slab to stage"
    tiled = by_tile["n16x8/f2x4"]
    assert {str(family_value(r, "STAGE")) for r in tiled} == {"None", "d1/cp", "d1/tma"}, "resolved d1 stages only"
    splits = {str(family_value(r, "REDUCE")) for r in tiled if family_value(r, "REDUCE")}
    assert splits == set(splitk_moves(warp=False))
    assert all(family_value(r, "STAGE") is None for r in tiled if family_value(r, "REDUCE")), (
        "split rows are gmem-direct (030_split drops the stage)"
    )


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
