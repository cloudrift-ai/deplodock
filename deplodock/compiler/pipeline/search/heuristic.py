"""Hardcoded analytic knob heuristic — a *prior-free* ranking of matmul tile
configs.

The two-level autotuner's inner search seeds from an ordering of the enumerated
knob rows. The learned ``CatBoostPrior`` provides that ordering once it has been
tuned; before any tuning (a cold prior, or a deliberate no-prior baseline) the
only signal is :mod:`_enumeration`'s ``_priority_matmul_thread``, whose top picks
are degenerate (``BN=1, BM=256``) and rank the recorded goldens at ~10⁴–10⁵.

This module is the *hardcoded* alternative: a closed-form score over a knob row +
shape that ranks the recorded :data:`GOLDEN_CONFIGS` near the top **without any
measurements**. The weights and feature set were found offline by
``scripts/golden_knob_heuristics.py`` (learning-to-rank over the reconstructed
enumeration); the dominant term is *occupancy* — keep the CTA count near ~2 waves
over the device's SMs — which is what lets one formula adapt the tile size to the
shape (tiny tiles for ``M=32`` projections, fat tiles for ``4096²``).

Scope: thread-tier fp32 matmul (where the static priority is worst). The warp
tier (fp16/bf16 MMA) already ranks goldens at ~7 via ``_priority_matmul_warp``,
so :func:`pick_matmul` defers to the enumeration's own order there.
"""

from __future__ import annotations

import math

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import enumerate_cartesian

# Occupancy reference. RTX 5090 / sm_120 has 170 SMs; the golden set was measured
# there. The dominant heuristic term targets ~2 waves of CTAs over this count, so
# the right value matters for the absolute occupancy band (a future planner-side
# wiring should pass the live device's SM count instead of this constant).
DEFAULT_SM_COUNT = 170

# Knobs the thread-tier / warp-tier enumeration actually decides — the only ones
# this heuristic can be scored against (STAGE / BUFFER_COUNT / WARP_SPECIALIZE /
# ATOMIC_FREE_SPLITK are stamped by later passes, not chosen here). FK is omitted
# from the thread set: the matmul enumeration always emits FK=1, so it never
# distinguishes a golden.
THREAD_KNOBS = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
WARP_KNOBS = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")

# Per-feature weights of the offline-learned linear ranker, folded so they apply
# to the RAW (un-normalized) features — i.e. the score is a plain dot product and
# its argmax over a shape's enumeration is the heuristic's pick. Regenerate with
# ``scripts/golden_knob_heuristics.py`` (it writes /tmp/golden_heuristic_weights.json
# and reports the top-k golden coverage these reproduce: 18/19 within top-200,
# 14/19 within top-50, median rank 8 on the RTX 5090 golden set). The dominant
# terms are occupancy (``near_waves``: keep #CTAs near ~2 waves over the SMs) and
# the geometry bands (``bn_band``/``bm_band``/``bk_ge_32``/``tilen_clean``) — that
# occupancy term is what makes one formula adapt tile size from ``M=32``
# projections up to ``4096²``.
_W: dict[str, float] = {
    "l2_threads": -1.0680490980612438,
    "near_threads256": 0.4082401394250646,
    "pow2_threads": -0.4401714077219094,
    "l2_bn": -0.3805032925404449,
    "l2_bm": 0.4154495654041388,
    "bn_ge_bm": 6.278827239141573,
    "bn_band": 12.473490186117838,
    "bm_band": 10.917382853192994,
    "l2_bk": 0.5994873368224478,
    "bk_ge_32": 11.772583918770707,
    "cells": -0.07273396778190817,
    "near_cells16": 0.008924878335898901,
    "splitk": -0.08033523591907027,
    "splitk_le2": 8.407426536981845,
    "l2_area": -0.2078815594544474,
    "near_area8192": -0.4466255401802602,
    "tilen_clean": 10.411743172272452,
    "near_tilen64": -0.8302436472339884,
    "neg_overhang": 0.1648645127840344,
    "near_kchunks32": -0.14573975724955723,
    "square_tile": 1.0234675229500252,
    "l2_ctas": 0.868682587402858,
    "ctas_ge_sm": -8.471062355378367,
    "near_waves": 3.1056133897011744,
    "l2_reuse": -0.00890648854922402,
    "near_intensity": 0.30289850357623,
}


def _l2(x: float) -> float:
    return math.log2(max(x, 1.0))


def score_matmul_thread(row: dict, M: int, N: int, K: int, sm_count: int = DEFAULT_SM_COUNT) -> float:
    """Closed-form quality score for one thread-tier matmul knob row on an
    ``(M, K) @ (K, N)`` shape — higher is better. Linear in the features below;
    weights are :data:`_W` (see its provenance note). The feature definitions
    MUST stay in lockstep with ``scripts/golden_knob_heuristics.py:_featurize``,
    which is where the weights were fit."""
    bn, bm, fm, fn = row["BN"], row["BM"], row["FM"], row["FN"]
    bk, sk, br = row["BK"], row["SPLITK"], row["BR"]
    overhang = len(row.get("OVERHANG", ()))

    threads = bn * bm
    cells = fm * fn
    tile_m, tile_n = bm * fm, bn * fn
    area = tile_m * tile_n
    kchunks = max((K / br) / bk, 1.0)
    ctas = math.ceil(M / tile_m) * math.ceil(N / tile_n) * sk
    reuse = area / (tile_m + tile_n)

    feats = {
        "l2_threads": _l2(threads),
        "near_threads256": -abs(_l2(threads) - 8.0),
        "pow2_threads": 1.0 if threads & (threads - 1) == 0 else 0.0,
        "l2_bn": _l2(bn),
        "l2_bm": _l2(bm),
        "bn_ge_bm": 1.0 if bn >= bm else 0.0,
        "bn_band": 1.0 if 16 <= bn <= 64 else 0.0,
        "bm_band": 1.0 if 8 <= bm <= 16 else 0.0,
        "l2_bk": _l2(bk),
        "bk_ge_32": 1.0 if bk >= 32 else 0.0,
        "cells": min(cells, 128.0),
        "near_cells16": -abs(cells - 16.0),
        "splitk": float(sk),
        "splitk_le2": 1.0 if sk <= 2 else 0.0,
        "l2_area": _l2(area),
        "near_area8192": -abs(_l2(area) - 13.0),
        "tilen_clean": 1.0 if tile_n in (32, 64, 128) else 0.0,
        "near_tilen64": -abs(_l2(tile_n) - 6.0),
        "neg_overhang": float(-overhang),
        "near_kchunks32": -abs(_l2(kchunks) - 5.0),
        "square_tile": -abs(_l2(tile_m) - _l2(tile_n)),
        "l2_ctas": _l2(ctas),
        "ctas_ge_sm": 1.0 if ctas >= sm_count else 0.0,
        "near_waves": -abs(_l2(ctas / sm_count) - 1.0),
        "l2_reuse": _l2(reuse),
        "near_intensity": -abs(_l2(reuse) - 5.0),
    }
    return sum(_W[k] * feats[k] for k in _W)


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> tuple[list[dict], tuple[str, ...], object]:
    """Reconstruct the planner's enumeration for a matmul shape, plus the knob
    tuple to match a golden on and a per-row score key (higher first). fp32 scores
    the thread tier by :func:`score_matmul_thread`; fp16/bf16 use the warp tier's
    own enumeration order (``_priority_matmul_warp``, already golden-near)."""
    if dtype == "fp32":
        rows = enumerate_cartesian(E_M=M, E_N=N, E_K=K, ctx=ctx, priority_mode="matmul", m_axis_name="m", n_axis_name="n")
        return rows, THREAD_KNOBS, None  # None score key → caller scores by heuristic

    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get({"fp16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(dtype, ""))
    if atom is None:
        return [], WARP_KNOBS, None
    rows = enumerate_cartesian(E_M=M, E_N=N, E_K=K, ctx=ctx, priority_mode=("matmul", "warp"), atoms=(atom,))
    rows = [r for r in rows if r["WM"] * r["WN"] != 1]
    return rows, WARP_KNOBS, "order"  # "order" → already sorted, rank == index


def evaluate_golden(
    M: int, N: int, K: int, dtype: str, golden_knobs: dict, ctx: Context, sm_count: int = DEFAULT_SM_COUNT
) -> tuple[dict, int | None, int]:
    """Score a matmul shape's full enumeration and return ``(pick, rank, pool)``:
    the heuristic's argmax pick, the recorded golden's 0-based rank in the
    heuristic order (``None`` if the golden isn't in the enumeration — pin / dtype
    mismatch), and the enumeration size. The rank — not whether the #1 pick equals
    the golden — is the metric that matters: it's where the tuner's patience budget
    has to reach. Returns ``({}, None, 0)`` if nothing enumerates."""
    rows, match_keys, score_key = _enumerate(M, N, K, dtype, ctx)
    if not rows:
        return {}, None, 0
    want = tuple(golden_knobs.get(k) for k in match_keys)
    gidx = next((i for i, r in enumerate(rows) if tuple(r.get(k) for k in match_keys) == want), None)
    if score_key == "order":  # warp: enumeration already priority-sorted
        return rows[0], gidx, len(rows)
    scores = [score_matmul_thread(r, M, N, K, sm_count) for r in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank = sum(1 for s in scores if s > scores[gidx]) if gidx is not None else None
    return rows[best], rank, len(rows)


def pick_matmul(M: int, N: int, K: int, dtype: str, ctx: Context, sm_count: int = DEFAULT_SM_COUNT) -> dict:
    """Best knob row for an ``(M, K) @ (K, N)`` matmul under the hardcoded
    heuristic — no prior, no measurements. Thin wrapper over :func:`evaluate_golden`
    (no golden to match). Returns ``{}`` if nothing enumerates."""
    return evaluate_golden(M, N, K, dtype, {}, ctx, sm_count)[0]
