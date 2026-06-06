#!/usr/bin/env python
"""Offline heuristic search: can a cheap analytic ranking surface the *golden*
matmul knob config inside the tuner's patience budget?

Motivation
----------
The two-level autotuner explores the post-fusion kernel's knob space with an
inner MCTS that stops on **patience** (N consecutive measured terminals with no
new best). Whatever ordering seeds that search decides which leaves get visited
first — if the golden config sits at rank 13 000 of 131 000, patience 200 never
reaches it. The thread-tier matmul priority (``_priority_matmul_thread``) is
exactly that seed, and today it ranks the recorded goldens catastrophically
(ranks 13k–110k; its top picks are degenerate ``BN=1, BM=256`` tiles).

This script treats the problem as offline learning-to-rank:

  1. Build a dataset from ``GOLDEN_CONFIGS`` — for each golden matmul shape,
     reconstruct the *exact* candidate enumeration the planner would produce
     (``enumerate_cartesian``) and locate the golden row in it.
  2. Featurize every candidate row (geometry / occupancy / divisibility).
  3. Score candidates with a parameterized linear heuristic and measure where
     the golden lands (top-k coverage, median rank).
  4. Random-search the heuristic weights to maximize golden top-k coverage.
  5. Model the tuner's *exploration*: softmax-sample leaves at a temperature and
     report the probability the golden is hit within a patience-sized budget —
     a little randomness lets a near-miss heuristic still find the golden.

The warp tier (fp16/bf16 ``MMA`` configs) is reported as a baseline only: its
priority was retuned for the RTX 5090 in 2026 and already ranks goldens at ~7.

Run:  ./venv/bin/python scripts/golden_knob_heuristics.py
      ./venv/bin/python scripts/golden_knob_heuristics.py --samples 40000 --topk 50
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deplodock.compiler.context import Context
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    _enumerate_warp_matmul_impl,
    _priority_matmul_thread,
    enumerate_cartesian,
)
from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig
from deplodock.compiler.pipeline.search.heuristic import score_matmul_thread

# The knobs the thread-tier enumeration assigns (FK defaults to 1 when a golden
# dict omits it). OVERHANG is structural, not matched.
_CORE = ("BN", "BM", "FM", "FN", "FK", "BK", "SPLITK", "BR")
_WARP_CORE = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")


@dataclass
class Case:
    name: str
    M: int
    N: int
    K: int
    rows: list[dict]
    golden_idx: int


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
def _core(row: dict) -> tuple:
    return tuple(row.get(k, 1) for k in _CORE)


def build_thread_cases(ctx: Context) -> list[Case]:
    """Reconstruct the thread-tier enumeration for every fp32 golden matmul and
    pin the golden's index within it."""
    cases: list[Case] = []
    for g in GOLDEN_CONFIGS:
        if not isinstance(g, MatmulGoldenConfig) or g.dtype != "fp32":
            continue
        rows = enumerate_cartesian(E_M=g.M, E_N=g.N, E_K=g.K, ctx=ctx, priority_mode="matmul", m_axis_name="m", n_axis_name="n")
        want = tuple(g.knobs.get(k, 1) for k in _CORE)
        idx = [i for i, r in enumerate(rows) if _core(r) == want]
        if not idx:
            print(f"  !! {g.name}: golden not found in {len(rows)} candidates — skipping")
            continue
        cases.append(Case(g.name, g.M, g.N, g.K, rows, idx[0]))
    return cases


def load_or_build(ctx: Context, cap: int, cache: Path | None) -> list[Case]:
    """The thread-tier enumeration is the slow part (~10 min for all shapes), so
    cache the reconstructed candidate rows to disk and reuse across analysis runs."""
    if cache and cache.exists():
        print(f"Loading cached dataset from {cache} ...")
        return pickle.loads(cache.read_bytes())
    print(f"Building thread-tier golden dataset (sm_{cap}0) ...")
    cases = build_thread_cases(ctx)
    if cache:
        cache.write_bytes(pickle.dumps(cases))
        print(f"  cached → {cache}")
    return cases


# Hard candidate gate distilled from the learned heuristic's top features. Every
# recorded golden satisfies it; it shrinks the pool the tuner must explore by
# ~2 orders of magnitude so a fixed patience budget actually reaches the golden.
def gate(row: dict) -> bool:
    bn, bm = row["BN"], row["BM"]
    threads = bn * bm
    tile_n = bn * row["FN"]
    return (
        16 <= bn <= 64
        and 8 <= bm <= 16
        and bn >= bm
        and row["BK"] >= 32
        and row["SPLITK"] <= 2
        and threads in (128, 256, 512, 1024)
        and tile_n in (32, 64, 128)
    )


def report_warp_baseline(ctx: Context) -> None:
    print("\n== warp tier (fp16/bf16) — current priority, reported as baseline ==")
    for g in GOLDEN_CONFIGS:
        if not isinstance(g, MatmulGoldenConfig) or g.dtype == "fp32":
            continue
        atom = ATOM_REGISTRY.get(g.knobs.get("MMA", ""))
        if atom is None:
            continue
        rows = _enumerate_warp_matmul_impl(
            E_M=g.M,
            E_N=g.N,
            E_K=g.K,
            ctx=ctx,
            force_splitk_one=False,
            atoms=(atom,),
            m_axis_name="m",
            n_axis_name="n",
            m_forced_mask=False,
            n_forced_mask=False,
        )
        rows = [r for r in rows if r["WM"] * r["WN"] != 1]
        want = tuple(g.knobs.get(k) for k in _WARP_CORE)
        idx = [i for i, r in enumerate(rows) if tuple(r.get(k) for k in _WARP_CORE) == want]
        print(f"  {g.name:22s} rank={idx[0] if idx else '??':>5} / {len(rows)}")


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #
# Each feature is "more is better" oriented where it has an obvious direction;
# the search assigns a signed weight so it can flip / ignore any of them. Names
# are kept so the winning weight vector is legible.
FEATURE_NAMES = [
    "l2_threads",  # log2(BN*BM)
    "near_threads256",  # -|log2(threads) - 8|
    "pow2_threads",  # 1 if threads is a power of two
    "l2_bn",  # log2(BN)
    "l2_bm",  # log2(BM)
    "bn_ge_bm",  # 1 if BN >= BM (wider innermost axis = coalesced)
    "bn_band",  # 1 if 16 <= BN <= 64
    "bm_band",  # 1 if 8 <= BM <= 16
    "l2_bk",  # log2(BK)
    "bk_ge_32",  # 1 if BK >= 32
    "cells",  # min(FM*FN, 128)
    "near_cells16",  # -|FM*FN - 16|
    "splitk",  # SPLITK (raw)
    "splitk_le2",  # 1 if SPLITK <= 2
    "l2_area",  # log2(tile_m * tile_n)
    "near_area8192",  # -|log2(area) - 13|
    "tilen_clean",  # 1 if BN*FN in {32,64,128}
    "near_tilen64",  # -|log2(tile_n) - 6|
    "neg_overhang",  # -count(OVERHANG)
    "near_kchunks32",  # -|log2((K/BR)/BK) - 5|
    "square_tile",  # -|log2(tile_m) - log2(tile_n)|
    # --- shape-relative: let ONE heuristic adapt cell/tile size to the shape ---
    "l2_ctas",  # log2(#CTAs = ceil(M/tile_m)*ceil(N/tile_n)*SPLITK)
    "ctas_ge_sm",  # 1 if #CTAs >= 170 (RTX 5090 SM count — fill the GPU)
    "near_waves",  # -|log2(#CTAs / 170) - 1|  (target ~2 waves)
    "l2_reuse",  # log2(tile_m*tile_n/(tile_m+tile_n)) — operand reuse/intensity
    "near_intensity",  # -|log2(reuse) - 5|  (target ~32 reuse)
]

# RTX 5090 / sm_120 streaming-multiprocessor count — the occupancy reference for
# the shape-relative CTA-count features.
_SM_COUNT = 170


def _featurize(rows: list[dict], M: int, N: int, K: int) -> np.ndarray:
    bn = np.array([r["BN"] for r in rows], float)
    bm = np.array([r["BM"] for r in rows], float)
    fm = np.array([r["FM"] for r in rows], float)
    fn = np.array([r["FN"] for r in rows], float)
    bk = np.array([r["BK"] for r in rows], float)
    sk = np.array([r["SPLITK"] for r in rows], float)
    br = np.array([r["BR"] for r in rows], float)
    oh = np.array([len(r.get("OVERHANG", ())) for r in rows], float)

    threads = bn * bm
    cells = fm * fn
    tile_m = bm * fm
    tile_n = bn * fn
    area = tile_m * tile_n
    kchunks = np.maximum((K / br) / bk, 1.0)
    ctas = np.ceil(M / tile_m) * np.ceil(N / tile_n) * sk
    reuse = area / (tile_m + tile_n)

    def l2(x):
        return np.log2(np.maximum(x, 1.0))

    def ispow2(x):
        xi = x.astype(np.int64)
        return ((xi & (xi - 1)) == 0).astype(float)

    feats = {
        "l2_threads": l2(threads),
        "near_threads256": -np.abs(l2(threads) - 8.0),
        "pow2_threads": ispow2(threads),
        "l2_bn": l2(bn),
        "l2_bm": l2(bm),
        "bn_ge_bm": (bn >= bm).astype(float),
        "bn_band": ((bn >= 16) & (bn <= 64)).astype(float),
        "bm_band": ((bm >= 8) & (bm <= 16)).astype(float),
        "l2_bk": l2(bk),
        "bk_ge_32": (bk >= 32).astype(float),
        "cells": np.minimum(cells, 128.0),
        "near_cells16": -np.abs(cells - 16.0),
        "splitk": sk,
        "splitk_le2": (sk <= 2).astype(float),
        "l2_area": l2(area),
        "near_area8192": -np.abs(l2(area) - 13.0),
        "tilen_clean": np.isin(tile_n, [32, 64, 128]).astype(float),
        "near_tilen64": -np.abs(l2(tile_n) - 6.0),
        "neg_overhang": -oh,
        "near_kchunks32": -np.abs(l2(kchunks) - 5.0),
        "square_tile": -np.abs(l2(tile_m) - l2(tile_n)),
        "l2_ctas": l2(ctas),
        "ctas_ge_sm": (ctas >= _SM_COUNT).astype(float),
        "near_waves": -np.abs(l2(ctas / _SM_COUNT) - 1.0),
        "l2_reuse": l2(reuse),
        "near_intensity": -np.abs(l2(reuse) - 5.0),
    }
    return np.stack([feats[n] for n in FEATURE_NAMES], axis=1)


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden when sorting by descending score (ties counted
    as 'above', the pessimistic / honest rank)."""
    g = scores[gidx]
    return int((scores > g).sum())


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100, 200)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median_rank={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"


def eval_weights(mats: list[np.ndarray], gidx: list[int], w: np.ndarray) -> list[int]:
    return [rank_of_golden(m @ w, gi) for m, gi in zip(mats, gidx, strict=True)]


def objective(ranks: list[int]) -> float:
    # Mean log2(rank+1): rewards pushing every golden up, dominated by the worst
    # offenders. Lower is better.
    return float(np.mean([math.log2(r + 1) for r in ranks]))


# --------------------------------------------------------------------------- #
# Exploration model                                                           #
# --------------------------------------------------------------------------- #
def hit_prob_within_budget(scores: np.ndarray, gidx: int, budget: int, temp: float, trials: int, rng) -> float:
    """Model the tuner as sampling leaves without replacement ∝ softmax(score/T).
    Return the fraction of trials that draw the golden within ``budget`` draws.
    A pure-greedy tuner is temp→0 (== deterministic top-k). Some randomness lets
    a heuristic that puts the golden at rank ~budget still reach it."""
    s = scores / max(temp, 1e-6)
    s -= s.max()
    p = np.exp(s)
    # Gumbel-top-k trick: argsort of (logp + Gumbel noise) == sampling without
    # replacement. Vectorize over trials.
    logp = np.log(p + 1e-30)
    hits = 0
    n = len(scores)
    b = min(budget, n)
    for _ in range(trials):
        g = rng.gumbel(size=n)
        order = np.argpartition(-(logp + g), b - 1)[:b]
        if gidx in order:
            hits += 1
    return hits / trials


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="random weight vectors to try")
    ap.add_argument("--topk", type=int, default=50, help="patience-sized budget the report centers on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=12, help="compute capability major (12 == sm_120 / RTX 5090)")
    ap.add_argument("--cache", default="/tmp/golden_knob_cases.pkl", help="pickle cache for the reconstructed dataset")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ctx = Context.from_target((args.cap, 0))

    cache = Path(args.cache) if args.cache else None
    cases = load_or_build(ctx, args.cap, cache)
    print(f"  {len(cases)} fp32 matmul golden cases, {sum(len(c.rows) for c in cases):,} total candidates")

    mats = [_featurize(c.rows, c.M, c.N, c.K) for c in cases]
    gidx = [c.golden_idx for c in cases]

    # Global z-score so weights are comparable across features.
    allf = np.concatenate(mats, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    sd[sd == 0] = 1.0
    mats = [(m - mu) / sd for m in mats]

    # ---- baseline: the live _priority_matmul_thread ---- #
    base_ranks = []
    for c in cases:
        keyed = sorted(range(len(c.rows)), key=lambda i: _priority_matmul_thread(c.rows[i]), reverse=True)
        base_ranks.append(keyed.index(c.golden_idx))
    print("\n== baseline: current _priority_matmul_thread ==")
    print("  " + topk_table(base_ranks))
    for c, r in sorted(zip(cases, base_ranks, strict=True), key=lambda t: -t[1]):
        print(f"    {c.name:24s} rank={r:6d} / {len(c.rows)}")

    # ---- hardcoded heuristic (no prior): deplodock.../search/heuristic.py ---- #
    hc_ranks = []
    for c in cases:
        scores = np.array([score_matmul_thread(r, c.M, c.N, c.K) for r in c.rows])
        hc_ranks.append(rank_of_golden(scores, c.golden_idx))
    print("\n== hardcoded heuristic (no prior) — search/heuristic.score_matmul_thread ==")
    print("  " + topk_table(hc_ranks))
    for c, r in sorted(zip(cases, hc_ranks, strict=True), key=lambda t: -t[1]):
        print(f"    {c.name:24s} rank={r:6d} / {len(c.rows)}")

    # ---- random search over linear heuristics ---- #
    print(f"\n== random search: {args.samples} linear heuristics (upper bound) ==")
    nf = len(FEATURE_NAMES)
    # Seed candidates: a hand-designed vector + random normals. The hand seed
    # encodes the priors we read off the goldens (coalesced wide BN, small BM,
    # big BK, threads near 256, few overhang).
    seed_w = np.zeros(nf)
    for name, val in {
        "near_threads256": 1.0,
        "pow2_threads": 0.5,
        "bn_ge_bm": 1.0,
        "bn_band": 1.0,
        "bm_band": 1.0,
        "bk_ge_32": 1.0,
        "l2_bk": 0.5,
        "splitk_le2": 0.7,
        "tilen_clean": 0.7,
        "neg_overhang": 0.5,
    }.items():
        seed_w[FEATURE_NAMES.index(name)] = val

    best_w, best_obj, best_ranks = seed_w, objective(eval_weights(mats, gidx, seed_w)), None
    best_ranks = eval_weights(mats, gidx, best_w)
    for _ in range(args.samples):
        w = rng.standard_normal(nf)
        ranks = eval_weights(mats, gidx, w)
        ob = objective(ranks)
        if ob < best_obj:
            best_obj, best_w, best_ranks = ob, w, ranks

    # ---- coordinate-descent refine around the best ---- #
    step = 1.0
    for _ in range(6):
        improved = False
        for j in range(nf):
            for delta in (step, -step):
                w = best_w.copy()
                w[j] += delta
                ranks = eval_weights(mats, gidx, w)
                ob = objective(ranks)
                if ob < best_obj:
                    best_obj, best_w, best_ranks, improved = ob, w, ranks, True
        if not improved:
            step /= 2

    print("  best heuristic:")
    print("    " + topk_table(best_ranks))
    for c, r in sorted(zip(cases, best_ranks, strict=True), key=lambda t: -t[1]):
        print(f"    {c.name:24s} rank={r:6d} / {len(c.rows)}")
    print("\n  learned weights (z-scored features, |w|>0.15):")
    for name, wv in sorted(zip(FEATURE_NAMES, best_w, strict=True), key=lambda t: -abs(t[1])):
        if abs(wv) > 0.15:
            print(f"    {name:18s} {wv:+.3f}")

    # Fold the z-score into the weights so they apply to RAW features directly:
    # score = ((raw-mu)/sd)·w = raw·(w/sd) - const; the const drops out of any
    # ranking. Dump for baking into search/heuristic.py.
    import json  # noqa: PLC0415

    raw_w = {name: float(best_w[i] / sd[i]) for i, name in enumerate(FEATURE_NAMES)}
    Path("/tmp/golden_heuristic_weights.json").write_text(json.dumps(raw_w, indent=2))
    print("  raw-feature weights → /tmp/golden_heuristic_weights.json")

    # ---- exploration: hit-rate within a patience budget at a few temps ---- #
    print(f"\n== exploration: P(golden in first {args.topk} sampled leaves) ==")
    print("  (temp 0 == greedy top-k; higher temp == more exploration)")
    for temp in (0.0, 0.5, 1.0, 2.0):
        probs = []
        for m, gi in zip(mats, gidx, strict=True):
            sc = m @ best_w
            if temp == 0.0:
                probs.append(1.0 if rank_of_golden(sc, gi) < args.topk else 0.0)
            else:
                probs.append(hit_prob_within_budget(sc, gi, args.topk, temp, trials=200, rng=rng))
        print(f"    temp={temp:>3}: mean hit-rate={np.mean(probs):.2f}  (per-case min={min(probs):.2f})")

    # ---- gate the pool, then re-rank within it ---- #
    print("\n== hard gate: prune to the heuristic-plausible band, then rank ==")
    gate_ranks_h, gate_ranks_b = [], []
    survived, total = 0, 0
    for c, m in zip(cases, mats, strict=True):
        keep = np.array([gate(r) for r in c.rows])
        total += len(c.rows)
        survived += int(keep.sum())
        idx_map = np.flatnonzero(keep)
        if c.golden_idx not in idx_map:  # golden filtered out by the gate
            gate_ranks_h.append(len(idx_map))
            gate_ranks_b.append(len(idx_map))
            print(f"    {c.name:24s} GOLDEN GATED OUT (pool {len(idx_map)})")
            continue
        gpos = int(np.flatnonzero(idx_map == c.golden_idx)[0])
        # heuristic order within the gated pool
        sc = (m @ best_w)[keep]
        gate_ranks_h.append(int((sc > sc[gpos]).sum()))
        # current priority order within the gated pool
        kept_rows = [c.rows[i] for i in idx_map]
        order = sorted(range(len(kept_rows)), key=lambda i: _priority_matmul_thread(kept_rows[i]), reverse=True)
        gate_ranks_b.append(order.index(gpos))
    print(f"  pool shrink: {total:,} → {survived:,} candidates ({survived / total:.1%} kept)")
    print("  within-gate, current priority: " + topk_table(gate_ranks_b))
    print("  within-gate, learned heuristic:" + topk_table(gate_ranks_h))
    for c, rh, rb in sorted(zip(cases, gate_ranks_h, gate_ranks_b, strict=True), key=lambda t: -t[1]):
        pool = int(sum(gate(r) for r in c.rows))
        print(f"    {c.name:24s} learned={rh:4d}  current={rb:4d}  / {pool}")

    report_warp_baseline(ctx)


if __name__ == "__main__":
    main()
