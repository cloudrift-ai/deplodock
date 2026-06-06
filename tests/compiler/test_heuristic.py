"""Hardcoded prior-free knob heuristic (``search/heuristic``).

CPU-only (no CUDA): the heuristic reconstructs the planner's enumeration and
scores it analytically. These pin the load-bearing properties — degenerate tiles
score below golden-shaped ones, picks land in the geometry band, and the warp
tier dispatches by dtype — without re-running the offline weight fit (that lives
in ``scripts/golden_knob_heuristics.py``).
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.search.heuristic import (
    THREAD_KNOBS,
    pick_matmul,
    score_matmul_thread,
)


def _ctx() -> Context:
    # sm_120 (RTX 5090) — the regime the golden set + heuristic weights target.
    return Context.from_target((12, 0))


def test_golden_row_outscores_degenerate_tile():
    # The current static priority's top pick for 2048² is the degenerate
    # BN=1, BM=256 tile; the recorded golden is BN=32, BM=8, FM=26, FN=4.
    # The heuristic must rank the golden strictly above the degenerate one.
    shape = (2048, 2048, 2048)
    golden = {"BN": 32, "BM": 8, "FM": 26, "FN": 4, "FK": 1, "BK": 32, "SPLITK": 1, "BR": 1}
    degenerate = {"BN": 1, "BM": 256, "FM": 1, "FN": 128, "FK": 1, "BK": 32, "SPLITK": 1, "BR": 1}
    assert score_matmul_thread(golden, *shape) > score_matmul_thread(degenerate, *shape)


def test_pick_matmul_lands_in_geometry_band():
    # Across a spread of shapes the argmax pick should respect the heuristic's
    # dominant bands: coalesced inner axis 16..64, short outer axis 8..16,
    # large K-chunk, light split-K — never a degenerate BN=1 / BM=256 tile.
    ctx = _ctx()
    for M, N, K in [(64, 64, 64), (128, 256, 128), (512, 1024, 1024)]:
        r = pick_matmul(M, N, K, "fp32", ctx)
        assert 16 <= r["BN"] <= 64, (M, N, K, r)
        assert 8 <= r["BM"] <= 16, (M, N, K, r)
        assert r["BK"] >= 32, (M, N, K, r)
        assert r["SPLITK"] <= 2, (M, N, K, r)
        assert set(THREAD_KNOBS) <= set(r)


def test_pick_matmul_warp_dispatch_by_dtype():
    ctx = _ctx()
    r16 = pick_matmul(256, 256, 256, "fp16", ctx)
    assert r16.get("MMA") == "mma_m16n8k16_f16"
    assert r16["WM"] * r16["WN"] != 1  # single-warp tiles are pruned
    r_bf = pick_matmul(256, 256, 256, "bf16", ctx)
    assert r_bf.get("MMA") == "mma_m16n8k16_bf16"
