#!/usr/bin/env python
"""Offline fit of the :class:`AnalyticPrior` weights (``_W_A``) over ``knob_features``.

Motivation
----------
The two-level autotuner explores the post-fusion kernel's knob space with an inner
MCTS that stops on **patience** (N consecutive measured terminals with no new
best). Cold (no learned data) that search is ranked by the :class:`AnalyticPrior` —
a fixed linear model over the engineered ``D_*`` geometry / occupancy features
:func:`knob.knob_features` produces. If the golden config sits at rank 800 of 2400,
patience never reaches it. This script fits the linear weights so the golden lands
near the top.

It treats the problem as offline learning-to-rank, over the SINGLE featurization
(``knob.knob_features`` — no parallel feature set), covering **every kernel
regime**: fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise.

  1. For every golden (matmul thread / warp, reduce, pointwise), reconstruct the
     planner's exact candidate enumeration for that mode and locate the golden row.
     fp16/bf16 (warp) goldens enumerate BOTH tiers — scalar + warp — since a real
     fp16 matmul does, so the fit ranks the warp golden against the scalar tile it
     competes with (the warp-first default greedy's flatten no longer gets from
     enumeration order).
  2. Featurize every candidate via ``knob.knob_features`` (the ``D_*`` engineered
     features plus ``MMA_tier``, the warp/scalar tier discriminator — the ``S_*`` /
     ``H_*`` shape/regime features are constant within a shape, so they drop out of
     a within-shape ranking).
  3. Score candidates with a parameterized linear model and measure the golden's
     rank (top-k coverage, median rank).
  4. Random-search + coordinate-descent the weights to minimize mean ``log2(rank+1)``
     across all goldens (both tiers jointly — the ``D_*`` features are tier-aware,
     so one weight vector serves both).
  5. Print the winning weights as the ``_W_A`` dict to paste into
     ``compiler/pipeline/search/prior/analytic.py``.

Run:  ./venv/bin/python scripts/golden_knob_heuristics.py
      ./venv/bin/python scripts/golden_knob_heuristics.py --samples 40000
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from deplodock.compiler.context import Context
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    _enumerate_warp_matmul_impl,
    enumerate_cartesian,
)
from deplodock.compiler.pipeline.search.golden import (
    GOLDEN_CONFIGS,
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
)

# Knobs each enumeration mode assigns — the projection a golden is matched on.
_THREAD_KEYS = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
_WARP_KEYS = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")
_REDUCE_KEYS = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR", "FK")
_POINTWISE_KEYS = ("BN", "BM", "FM", "FN", "SPLITK")


def _base(ctx: Context, M: int, N: int, K: int, *, dynamic: bool = False) -> dict:
    """The shape / regime features the planner stamps and the prior featurizes —
    merged into each candidate row before ``knob_features`` so the occupancy terms
    (``#CTAs``, waves) and tier-aware BK targets fire. A dynamic (symbolic-M)
    golden mirrors the 992 stamp: the symbolic axis is EXCLUDED from the free-dim
    product and counted in ``S_ext_n_symbolic_axis`` — the same features the
    deployed featurization computes, and the flag the ``AnalyticPrior`` selects
    its dynamic weight set on."""
    free = float(N) if dynamic else float(M * N)
    base = {**ctx.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        base["S_ext_n_symbolic_axis"] = 1.0
    return base


def build_cases() -> list[tuple[str, str, int, list[dict[str, float]]]]:
    """Reconstruct each matmul golden's candidate enumeration (both tiers), pin the
    golden's index, and featurize every row. Returns ``(name, tier, golden_idx,
    feats)`` where ``tier`` is ``"thread"``/``"warp"`` and ``feats`` is the per-row
    ``D_*`` (+ ``MMA_tier``) feature dict.

    Each golden is reconstructed under its OWN card's context
    (``Context.from_target(cap, gpu_name=…)``, mirroring ``Sample.from_golden``):
    the multi-GPU golden set spans cards that differ in compute capability AND in
    SM count at the same cap (RTX 5090 = 170 vs RTX PRO 6000 = 188 vs RTX 4090 =
    128 SMs, the latter at sm_89), so both the candidate enumeration (cp.async /
    TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the
    recording card's regime — not one global cap — for the rank objective to match
    the deployed per-card featurization."""
    cases = []
    for g in GOLDEN_CONFIGS:
        ctx = Context.from_target(tuple(g.compute_cap), gpu_name=g.gpu_name)
        if isinstance(g, ReduceGoldenConfig):
            # The reduce's free axis (the ``M`` rows) maps to the planner's N axis
            # (the tuned ``FN`` register tile sweeps it), so enumerate E_M=1, E_N=M.
            base = _base(ctx, 1, g.M, g.K)
            rows = enumerate_cartesian(E_M=1, E_N=g.M, E_K=g.K, ctx=ctx, priority_mode="reduce", n_axis_name="n")
            keys, tier = _REDUCE_KEYS, "reduce"
        elif isinstance(g, PointwiseGoldenConfig):
            base = _base(ctx, g.M, g.N, 1)
            rows = enumerate_cartesian(E_M=g.M, E_N=g.N, E_K=1, ctx=ctx, priority_mode="pointwise", m_axis_name="m", n_axis_name="n")
            keys, tier = _POINTWISE_KEYS, "pointwise"
        elif isinstance(g, MatmulGoldenConfig) and g.dtype == "fp32":
            base = _base(ctx, g.M, g.N, g.K, dynamic=bool(g.dynamic))
            rows = enumerate_cartesian(E_M=g.M, E_N=g.N, E_K=g.K, ctx=ctx, priority_mode="matmul", m_axis_name="m", n_axis_name="n")
            keys, tier = _THREAD_KEYS, ("dyn" if g.dynamic else "thread")
        elif isinstance(g, MatmulGoldenConfig):
            base = _base(ctx, g.M, g.N, g.K)
            atom = ATOM_REGISTRY.get(g.knobs.get("MMA", ""))
            if atom is None:
                continue
            warp_rows = _enumerate_warp_matmul_impl(
                E_M=g.M, E_N=g.N, E_K=g.K, ctx=ctx, force_splitk_one=False,
                atoms=(atom,), m_axis_name="m", n_axis_name="n", m_forced_mask=False, n_forced_mask=False,
            )  # fmt: skip
            warp_rows = [r for r in warp_rows if r["WM"] * r["WN"] != 1]
            # Include the SCALAR tier in the candidate pool: a real fp16 matmul
            # enumerates both tiers as leaves under one fork tree, so greedy's
            # flattened pick ranks warp vs scalar directly. Enumerating warp-only
            # here hid that competition, so the fit never learned to prefer tensor
            # cores — the cold analytic ranked the scalar tile first for fp16. With
            # both tiers in the pool the rank objective penalizes "scalar outranks
            # the warp golden", so the joint fit learns the warp preference (via
            # ``MMA_tier``, kept below).
            scalar_rows = enumerate_cartesian(E_M=g.M, E_N=g.N, E_K=g.K, ctx=ctx, priority_mode="matmul", m_axis_name="m", n_axis_name="n")
            rows = list(scalar_rows) + warp_rows
            keys, tier = _WARP_KEYS, "warp"
        else:
            continue
        want = tuple(g.knobs.get(k) for k in keys)
        gidx = next((i for i, r in enumerate(rows) if tuple(r.get(k) for k in keys) == want), None)
        if gidx is None:
            print(f"  !! {g.name}: golden not in {len(rows)} candidates — skipping")
            continue
        # Keep ``D_*`` geometry/occupancy plus ``MMA_tier`` (the warp/scalar tier
        # discriminator) — the only non-``D_`` feature the tier choice rides on.
        # It's constant 0 within a scalar-only shape (fp32 / reduce / pointwise),
        # so it drops out of those within-shape rankings and only fires where both
        # tiers compete (fp16 / bf16).
        feats = [{k: v for k, v in knob.knob_features({**base, **r}).items() if k.startswith("D_") or k == "MMA_tier"} for r in rows]
        cases.append((g.name, tier, gidx, feats))
    return cases


def _matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden by descending score (ties count as 'above')."""
    return int((scores > scores[gidx]).sum())


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"


def eval_weights(mats: list[np.ndarray], gidx: list[int], w: np.ndarray) -> list[int]:
    return [rank_of_golden(m @ w, gi) for m, gi in zip(mats, gidx, strict=True)]


def objective(ranks: list[int], weights: list[float]) -> float:
    # Weighted mean log2(rank+1): rewards pushing every golden up, dominated by the
    # worst offenders. Per-case ``weights`` balance the tiers (fp16 warp is only
    # ~7/32 cases, so it'd be drowned out unweighted). Lower is better.
    vals = [w * math.log2(r + 1) for r, w in zip(ranks, weights, strict=True)]
    return float(sum(vals) / sum(weights))


def _fit(cases, names, sd_ref, *, seed_w, rng, samples):
    """Random-search + coordinate-descent one weight vector over ``cases``.
    Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by
    ``sd_ref`` (``ones`` for a raw-weight seed, the previous fit's ``sd`` to
    chain fits) and is re-scaled into this pool's z-space. Returns
    ``(best_w, best_ranks, mu, sd)`` in this pool's z-space."""
    mats = [_matrix(feats, names) for _, _, _, feats in cases]
    gidx = [gi for _, _, gi, _ in cases]
    tier_n = {t: sum(1 for _, ct, _, _ in cases if ct == t) for _, t, _, _ in cases}
    cw = [1.0 / tier_n[t] for _, t, _, _ in cases]

    # Z-score over this fit's candidate pool so weights are comparable across features.
    allf = np.concatenate(mats, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    sd[sd == 0] = 1.0
    matsz = [(m - mu) / sd for m in mats]

    best_w = seed_w * sd / sd_ref  # re-scale the seed into this pool's z-space
    best_ranks = eval_weights(matsz, gidx, best_w)
    best_obj = objective(best_ranks, cw)
    print("  seed: " + topk_table(best_ranks))

    for _ in range(samples):
        w = rng.standard_normal(len(names))
        ranks = eval_weights(matsz, gidx, w)
        ob = objective(ranks, cw)
        if ob < best_obj:
            best_obj, best_w, best_ranks = ob, w, ranks

    # Coordinate-descent refine around the best.
    step = 1.0
    for _ in range(8):
        improved = False
        for j in range(len(names)):
            for delta in (step, -step):
                w = best_w.copy()
                w[j] += delta
                ranks = eval_weights(matsz, gidx, w)
                ob = objective(ranks, cw)
                if ob < best_obj:
                    best_obj, best_w, best_ranks, improved = ob, w, ranks, True
        if not improved:
            step /= 2

    print("  best: " + topk_table(best_ranks))
    for (name, tier, _, _), r in sorted(zip(cases, best_ranks, strict=True), key=lambda t: -t[1]):
        print(f"    {name:32s} [{tier:6s}] rank={r:5d}")
    return best_w, best_ranks, mu, sd


def _print_weights(var: str, names, best_w, sd) -> None:
    # Fold the z-score into the weights so they apply to RAW features directly:
    # score = ((raw-mu)/sd)·w = raw·(w/sd) - const; the const drops out of ranking.
    raw_w = {name: float(best_w[i] / sd[i]) for i, name in enumerate(names) if abs(best_w[i] / sd[i]) > 1e-4}
    print(f"\n== {var} (paste into search/prior/analytic.py) ==")
    print(f"{var}: dict[str, float] = {{")
    for name, wv in sorted(raw_w.items(), key=lambda t: -abs(t[1])):
        print(f"    {name!r}: {wv},")
    print("}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="random weight vectors to try")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Building golden dataset (each golden under its own card's context) ...")
    cases = build_cases()
    names = sorted({k for _, _, _, feats in cases for f in feats for k in f})
    static_cases = [c for c in cases if c[1] != "dyn"]
    dyn_cases = [c for c in cases if c[1] == "dyn"]
    print(f"  {len(static_cases)} static + {len(dyn_cases)} dynamic golden cases, {len(names)} D_* features")

    # Seed: the current _W_A so each search starts from today's weights and can
    # only improve on the live ranking (the dyn fit seeds from the STATIC result
    # — the masked tier shares most of the geometry priors and diverges where the
    # boundary guard / occupancy differences demand).
    from deplodock.compiler.pipeline.search.prior.analytic import _W_A  # noqa: PLC0415

    seed_raw = np.array([_W_A.get(n, 0.0) for n in names])

    print("\n== static fit (thread / warp / reduce / pointwise tiers) ==")
    static_w, _, _, static_sd = _fit(static_cases, names, np.ones(len(names)), seed_w=seed_raw, rng=rng, samples=args.samples)
    _print_weights("_W_A", names, static_w, static_sd)

    if dyn_cases:
        print("\n== dynamic fit (symbolic-axis masked-tile goldens) ==")
        dyn_w, _, _, dyn_sd = _fit(dyn_cases, names, static_sd, seed_w=static_w, rng=rng, samples=args.samples)
        _print_weights("_W_A_DYN", names, dyn_w, dyn_sd)


if __name__ == "__main__":
    main()
