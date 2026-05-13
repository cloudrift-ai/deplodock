"""GreedySearch consults the DB at fork points.

Drives a real fork point (``005_blockify_launch`` on a matmul) through
``run_pipeline`` and asserts:

1. **Baseline**: fresh DB → greedy picks option 0 (heuristic order).
2. **Seeded DB**: write a ``lowering`` row pointing at a non-option-0
   variant → greedy picks that variant instead.
3. **No seed for an unrelated parent key**: the seed is ignored when
   ``op_cache_key`` doesn't match, so option 0 is still chosen.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp, Op
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import TILE_PASSES, TuningSearch, run_autotune, run_pipeline
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key

# Same shape as ``tests/compiler/passes/test_matmul_rules.py`` — large
# enough that ``005_blockify_launch`` actually forks over multiple
# ``(BN, BM)`` variants.
_M, _K, _N = 256, 64, 256


def _make_matmul() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (_M, _K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_K, _N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _final_tile_op(g: Graph) -> TileOp:
    """Return the (single) TileOp in ``g`` post-TILE_PASSES."""
    tiles = [n.op for n in g.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tiles) == 1, f"expected one TileOp, got {len(tiles)}"
    return tiles[0]


def _blockify_pair(op: Op) -> tuple[Op, Op] | None:
    """Walk ``op.source`` chain and return ``(parent, child)`` at the
    transition where ``BN`` first appears in knobs — that's the
    ``005_blockify_launch`` rewrite point. ``parent`` is the pre-blockify
    TileOp, ``child`` is the one introduced by the rule. Knobs propagate
    forward via ``_apply_one`` so anything later in the chain inherits
    ``BN``; only the deepest such transition is the actual rewrite."""
    cur = op
    while cur is not None and cur.source is not None:
        cur_has = cur.knobs.get("BN") is not None
        src_has = cur.source.knobs.get("BN") is not None
        if cur_has and not src_has:
            return cur.source, cur
        cur = cur.source
    return None


def _enumerate_blockify_variants() -> list[tuple[str, str, dict]]:
    """Sweep the autotune space (no backend → stub latencies) and
    collect every distinct ``(parent_key, child_key, knobs)`` produced
    at the blockify_launch fork point.

    All matmul-graph terminal candidates share the same parent op for
    blockify (the pre-blockify matmul TileOp), so deduping on
    ``child_key`` is enough."""
    search = TuningSearch(db=SearchDB(), budget_s=float("inf"), patience=10**6, min_coverage=0.0)
    candidates = list(run_autotune(_make_matmul(), TILE_PASSES, search=search, db=SearchDB()))
    out: dict[str, tuple[str, str, dict]] = {}
    for cand in candidates:
        op = _final_tile_op(cand.graph)
        pair = _blockify_pair(op)
        if pair is None:
            continue
        parent, child = pair
        pk = op_cache_key(parent)
        ck = op_cache_key(child)
        if pk is None or ck is None or ck in out:
            continue
        out[ck] = (pk, ck, {k: v for k, v in child.knobs.items() if k in ("BN", "BM")})
    return list(out.values())


def test_baseline_picks_option_zero() -> None:
    """With a fresh in-memory DB, GreedySearch falls back to the rule's
    heuristic-first ordering (option 0)."""
    variants = _enumerate_blockify_variants()
    assert len(variants) >= 2, f"need ≥2 variants to test; got {len(variants)}"
    option_zero_knobs = variants[0][2]

    out = run_pipeline(_make_matmul(), TILE_PASSES, db=SearchDB())
    bn_bm = {k: _final_tile_op(out).knobs.get(k) for k in ("BN", "BM")}
    assert bn_bm == option_zero_knobs


def test_greedy_picks_db_winner() -> None:
    """Seed the lowering table with a non-option-0 child. Greedy should
    pick that variant instead of option-0."""
    variants = _enumerate_blockify_variants()
    assert len(variants) >= 2
    option_zero = variants[0]
    winner = variants[1]
    parent_key, winner_key, winner_knobs = winner
    # Sanity: knobs really do differ.
    assert option_zero[2] != winner_knobs

    db = SearchDB()
    # ``record_lowering`` early-returns for tile→tile (first-write-wins),
    # so write directly. ``best_median_us`` only needs to be non-NULL for
    # the policy to consider the row.
    db._conn.execute(
        "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, best_median_us) VALUES (?, 'tile', ?, 'tile', ?)",
        (parent_key, winner_key, 1.0),
    )

    out = run_pipeline(_make_matmul(), TILE_PASSES, db=db)
    bn_bm = {k: _final_tile_op(out).knobs.get(k) for k in ("BN", "BM")}
    assert bn_bm == winner_knobs


def test_irrelevant_seed_is_ignored() -> None:
    """A ``lowering`` row keyed on an op we never see leaves option-0
    behavior unchanged."""
    variants = _enumerate_blockify_variants()
    option_zero_knobs = variants[0][2]

    db = SearchDB()
    db._conn.execute(
        "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, best_median_us) "
        "VALUES ('unrelated-parent', 'tile', 'unrelated-child', 'tile', 1.0)",
    )

    out = run_pipeline(_make_matmul(), TILE_PASSES, db=db)
    bn_bm = {k: _final_tile_op(out).knobs.get(k) for k in ("BN", "BM")}
    assert bn_bm == option_zero_knobs
