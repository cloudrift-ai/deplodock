"""GreedySearch always picks option-0 at fork points.

Drives a real fork point (``005_blockify_launch`` on a matmul) through
``run_pipeline`` and asserts:

1. **Baseline**: fresh DB → greedy picks option 0 (rule's heuristic
   ordering).
2. **DB seeded with an unrelated row**: greedy still picks option 0
   (DB lookup was removed; greedy is purely option-0).
"""

from __future__ import annotations

import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp, Op
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, TuningSearch
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key

# Same shape as ``tests/compiler/passes/test_matmul_rules.py`` — large
# enough that ``005_blockify_launch`` actually forks over multiple
# ``(BN, BM)`` variants.
_M, _K, _N = 256, 64, 256


@pytest.fixture(autouse=True)
def _shrink_autotune_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap the autotune search space so the exhaustive walk in
    ``_enumerate_blockify_variants`` stays tractable. Pins everything
    that this test doesn't care about (only blockify ``(BN, BM)`` needs
    to fork) via ``DEPLODOCK_*`` env knobs and disables the per-buffer
    staging power-set so ``007_stage_inputs`` emits a single variant
    (its pre-knob behavior)."""
    monkeypatch.setenv("DEPLODOCK_STAGE_INPUTS", "0xFFFF")
    monkeypatch.setenv("DEPLODOCK_TMA", "0")


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


def _record_pair(out: dict[str, tuple[str, str, dict]], op: Op) -> None:
    pair = _blockify_pair(op)
    if pair is None:
        return
    parent, child = pair
    pk = op_cache_key(parent)
    ck = op_cache_key(child)
    if pk is None or ck is None or ck in out:
        return
    out[ck] = (pk, ck, {k: v for k, v in child.knobs.items() if k in ("BN", "BM")})


def _enumerate_blockify_variants() -> list[tuple[str, str, dict]]:
    """Collect every distinct ``(parent_key, child_key, knobs)`` produced
    at the blockify_launch fork point.

    Index 0 is anchored on the greedy/heuristic variant (what option-0
    in the rule emits) by recording the greedy ``run_pipeline`` result
    first; the autotune sweep then appends alternatives. This is stable
    across changes to the MCTS walk order, which can shift if e.g.
    ``op_cache_key`` partitions the search frontier differently."""
    out: dict[str, tuple[str, str, dict]] = {}
    greedy = Pipeline.build(TILE_PASSES).run(_make_matmul(), db=SearchDB())
    _record_pair(out, _final_tile_op(greedy))
    search = TuningSearch(db=SearchDB(), patience=10**6)
    candidates = list(Pipeline.build(TILE_PASSES).tune(_make_matmul(), search=search, db=SearchDB()))
    for cand in candidates:
        _record_pair(out, _final_tile_op(cand.graph))
    return list(out.values())


def test_baseline_picks_option_zero() -> None:
    """With a fresh in-memory DB, GreedySearch falls back to the rule's
    heuristic-first ordering (option 0)."""
    variants = _enumerate_blockify_variants()
    assert len(variants) >= 2, f"need ≥2 variants to test; got {len(variants)}"
    option_zero_knobs = variants[0][2]

    out = Pipeline.build(TILE_PASSES).run(_make_matmul(), db=SearchDB())
    bn_bm = {k: _final_tile_op(out).knobs.get(k) for k in ("BN", "BM")}
    assert bn_bm == option_zero_knobs


def test_irrelevant_seed_is_ignored() -> None:
    """A ``lowering`` row keyed on an op we never see leaves option-0
    behavior unchanged."""
    variants = _enumerate_blockify_variants()
    option_zero_knobs = variants[0][2]

    db = SearchDB()
    db.record_lowering(
        "unrelated-parent",
        "tile",
        "unrelated-child",
        "tile",
        knobs={"BN": 999, "BM": 999, "blockify": True},
        measured_median_us=1.0,
    )

    out = Pipeline.build(TILE_PASSES).run(_make_matmul(), db=db)
    bn_bm = {k: _final_tile_op(out).knobs.get(k) for k in ("BN", "BM")}
    assert bn_bm == option_zero_knobs
