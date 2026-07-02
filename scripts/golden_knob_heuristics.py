#!/usr/bin/env python
"""Offline fit of the :class:`AnalyticPrior` weights (``_W_A``) over ``knob_features``.

Motivation
----------
The two-level autotuner explores the post-fusion kernel's knob space with an inner
MCTS that stops on **patience** (N consecutive measured terminals with no new
best). Cold (no learned data) that search is ranked by the :class:`AnalyticPrior` —
a fixed linear model over the engineered ``D_*`` geometry / occupancy features
:func:`features.knob_features` produces. If the golden config sits at rank 800 of 2400,
patience never reaches it. This script fits the linear weights so the golden lands
near the top.

It treats the problem as offline learning-to-rank, over the SINGLE featurization
(``features.knob_features`` — no parallel feature set), covering **every kernel
regime**: fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise.

  1. For every golden (matmul thread / warp / dyn, reduce, pointwise), reconstruct
     the planner's exact candidate enumeration for that mode and locate the golden
     row. Matmul goldens reuse ``analytic._enumerate`` — the SAME gate-narrowed pool
     ``eval analytic`` and the greedy deploy rank over (fp32 → thread tier, fp16/bf16
     → the warp tier alone; the block-DAG rework moved the scalar↔warp choice to a
     structural fork, so a warp golden ranks within the warp tier, not against
     scalar rows). Reduce / pointwise trace the snippet to an ``IterDag`` and compose
     the cooperative-reduce / MAP ``_moves`` offers.
  2. Featurize every candidate via ``features.knob_features`` (the ``D_*`` engineered
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

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.analytic import _enumerate
from emmy.compiler.pipeline.search.golden import (
    GOLDEN_CONFIGS,
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
)


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


def _dag_from_snippet(snippet: str, ctx: Context):
    """Trace a golden's torch snippet and lower it through ``LOOP_PASSES`` (option-0
    greedy resolve, no GPU) to the first ``LoopOp``'s ``IterDag`` — the shape the
    reduce / pointwise enumeration offers are composed over. Mirrors
    ``analytic._matmul_dag`` but regime-agnostic: ``torch.sum`` / ``torch.relu`` have
    no dedicated frontend op, so the dag comes from the real trace, not a hand-built
    graph. Returns ``None`` if nothing lowers."""
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag  # noqa: PLC0415

    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import Fork  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    def _option0(fp):
        o = fp.options[0]
        while isinstance(o, Fork) and not o.is_leaf:
            o = o.expand()[0]
        return o

    graph = graph_from_code(snippet)[0]
    terminal, _ = Run(pipeline=Pipeline.build(LOOP_PASSES), ctx=ctx).resolve(graph, _option0)
    loops = [n.op for n in terminal.nodes.values() if isinstance(n.op, LoopOp)]
    return iter_dag(loops[0]) if loops else None


def _reduce_rows(dag) -> list[dict]:
    """Cooperative-reduce candidate knob rows: the cartesian of the carrier's
    cooperative ``(bk, fk, br)`` K-partitions (``coop_reduce_offers``) × the free-row
    register tile (``reduce_reg_offers``), over the free-axis THREAD tile
    (``coop_free_threads``). The native rows speak the axis-named schedule codecs (``TILE@<axis>`` +
    ``REDUCE@<axis>``); a 1-D reduce (no outer M axis) leaves the M slot degenerate,
    which the schema-agnostic ``tile_signature`` matches against the golden's
    ``BM``/``FM`` = 1."""
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _moves  # noqa: PLC0415

    budget = _moves.Budget()
    thread = _moves.coop_free_threads(dag)  # free-axis THREAD tile (par)
    rows = []
    for bk, fk, br in _moves.coop_reduce_offers(dag):
        red = _moves.coop_reduce_knobs(dag, (bk, fk, br))  # native REDUCE@<primary>
        for reg in _moves.reduce_reg_offers(dag, budget, fk):
            rows.append({**_moves.free_split_knobs(dag, thread, reg), **red})  # par + swept reg
    return rows


def _pointwise_rows(dag) -> list[dict]:
    """Pointwise (MAP, no reduce) candidate knob rows: the cartesian of the free
    thread tile (``thread_offers``) × the register tile (``map_reg_offers``). The native
    rows carry only the free-axis tile (no ``REDUCE@`` — a MAP nest has no contraction);
    ``tile_signature``'s degenerate reduce decomposition matches the golden's
    ``BK=FK=SPLITK=BR=1``."""
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _moves  # noqa: PLC0415

    budget = _moves.Budget()
    rows = []
    for thread in _moves.thread_offers(dag, budget):
        thr = _moves.thread_knobs(dag, thread)  # par-only SPLIT
        for reg in _moves.map_reg_offers(dag, budget):
            rows.append({**thr, **_moves.reg_knobs(dag, thr, reg)})  # complete SPLIT (par×reg)
    return rows


def build_cases() -> list[tuple[str, str, int, list[dict[str, float]]]]:
    """Reconstruct each golden's candidate enumeration, pin the golden's index, and
    featurize every row. Returns ``(name, tier, golden_idx, feats)`` where ``tier``
    is ``"thread"`` / ``"warp"`` / ``"dyn"`` / ``"reduce"`` / ``"pointwise"`` and
    ``feats`` is the per-row ``D_*`` (+ ``MMA_tier``) feature dict.

    Matmul goldens enumerate via ``analytic._enumerate`` — the SAME gate-narrowed
    pool ``eval analytic`` and the greedy deploy rank over (fp32 → thread tier,
    fp16/bf16 → warp tier; the block-DAG rework moved the scalar↔warp choice to a
    structural fork, so a real fp16 matmul ranks within the warp tier alone, no
    scalar rows in the pool). A dynamic (``.dynM``) golden enumerates its hint-sized
    static twin's pool and featurizes with the symbolic-axis stamp (its own weight
    set). Reduce / pointwise goldens trace their snippet to an ``IterDag`` and
    compose the cooperative-reduce / MAP ``_moves`` offers.

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
            # (the tuned ``FN`` register tile sweeps it): trace E_M=1, E_N=M, E_K=K.
            base = _base(ctx, 1, g.M, g.K)
            dag = _dag_from_snippet(g.snippet(), ctx)
            rows, tier = (_reduce_rows(dag) if dag is not None else []), "reduce"
        elif isinstance(g, PointwiseGoldenConfig):
            base = _base(ctx, g.M, g.N, 1)
            dag = _dag_from_snippet(g.snippet(), ctx)
            rows, tier = (_pointwise_rows(dag) if dag is not None else []), "pointwise"
        elif isinstance(g, MatmulGoldenConfig):
            dyn = bool(g.dynamic)
            base = _base(ctx, g.M, g.N, g.K, dynamic=dyn)
            rows, _ = _enumerate(g.M, g.N, g.K, g.dtype, ctx)
            tier = "dyn" if dyn else ("thread" if g.dtype == "fp32" else "warp")
        else:
            continue
        if not rows:
            print(f"  !! {g.name}: nothing enumerated — skipping")
            continue
        # Match the legacy-recorded golden against the native candidate rows by
        # schema-agnostic structural signature (free-axis slots + reduce decomp +
        # atom) — the candidates speak native ``MOVE@element``, the golden YAML legacy
        # GEMM-letters, so a per-key tuple compare never lines up.
        want = features.tile_signature(g.knobs)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
        if gidx is None:
            print(f"  !! {g.name}: golden not in {len(rows)} candidates — skipping")
            continue
        # Keep ``D_*`` geometry/occupancy plus ``MMA_tier`` (the warp/scalar tier
        # discriminator, where the featurization still emits it) — the ``S_*`` /
        # ``H_*`` shape/regime features are constant within a shape, so they drop out
        # of a within-shape ranking.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if k.startswith("D_") or k == "MMA_tier"} for r in rows]
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
    from emmy.compiler.pipeline.search.prior.analytic import _W_A  # noqa: PLC0415

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
