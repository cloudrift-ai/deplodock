"""Offline diagnostics for the learned tuning prior — "can the prior actually
reach the best configs?", with no GPU.

``deplodock tune`` with no model / ``--code`` refits the prior on its persisted
reservoir dataset and prints :func:`report`: a dataset summary, per-op **argmax
reachability** (does the prior's argmax over an op's *measured* configs recover
the measured-best leaf?), an in-sample ranking calibration, and coverage of the
golden matmul configs. Reachability is the optimistic upper bound on greedy
quality — if the prior can't even rank an op's own benched leaves to the top,
greedy (which descends fork-by-fork) can't either.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def _op_sig(knobs: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in knobs.items() if k.startswith("S_")))


def _n_tunable(knobs: dict) -> int:
    return sum(1 for k in knobs if not k.startswith(("S_", "H_")))


def group_by_op(dataset: list[tuple[dict, float]]) -> dict[tuple, list[tuple[dict, float]]]:
    g: dict[tuple, list] = defaultdict(list)
    for k, v in dataset:
        g[_op_sig(k)].append((k, v))
    return dict(g)


def _op_label(sig: tuple) -> str:
    d = dict(sig)
    if d.get("S_n_mma", 0) > 0:
        kind = "matmul"
    elif d.get("S_reduce_add", 0) or d.get("S_reduce_max", 0):
        kind = "reduce"
    else:
        kind = "pointwise"
    free, red = int(d.get("S_ext_free_max", 0)), int(d.get("S_ext_reduce_max", 0))
    return f"{kind:9} free={free}" + (f" red={red}" if red else "")


def reachability(prior, groups: dict) -> list[tuple]:
    """Per op: ``(label, best_us, argmax_us, ratio, n_leaf_configs)`` — the prior's
    argmax over the op's measured *leaf* configs vs the measured best (ratio = the
    argmax config's latency / the best's; 1.0 = recovers the optimum)."""
    out = []
    for sig, grp in groups.items():
        kmax = max(_n_tunable(k) for k, _ in grp)
        leaves = [(k, v) for k, v in grp if _n_tunable(k) == kmax]
        if len(leaves) < 2:
            continue
        best_lab = max(v for _, v in leaves)
        pick_lab = max(leaves, key=lambda t: prior.mean_score(t[0]))[1]
        out.append((_op_label(sig), math.exp(-best_lab), math.exp(-pick_lab), math.exp(best_lab - pick_lab), len(leaves)))
    return out


def _calibration(prior, groups: dict) -> float | None:
    """Median per-op Spearman ρ between the prior's prediction and the measured
    reward over each op's leaf configs."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    rhos = []
    for grp in groups.values():
        kmax = max(_n_tunable(k) for k, _ in grp)
        leaves = [(k, v) for k, v in grp if _n_tunable(k) == kmax]
        if len(leaves) < 3:
            continue
        rho = spearmanr([prior.mean_score(k) for k, _ in leaves], [v for _, v in leaves]).statistic
        if not math.isnan(rho):
            rhos.append(rho)
    return statistics.median(rhos) if rhos else None


def _golden_coverage(groups: dict) -> tuple[int, int]:
    """How many golden matmul shapes have measured data in the dataset (matched by
    the op's free-dim product + reduce extent from its ``S_*`` features)."""
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS  # noqa: PLC0415

    have = set()
    for sig in groups:
        d = dict(sig)
        if d.get("S_n_mma", 0) > 0:
            have.add((int(d.get("S_ext_free_prod", 0)), int(d.get("S_ext_reduce_max", 0))))
    covered = sum(1 for g in GOLDEN_CONFIGS if (getattr(g, "M", 0) * getattr(g, "N", 0), getattr(g, "K", 0)) in have)
    return covered, len(GOLDEN_CONFIGS)


def golden_prior_eval(prior, kernel_filter: str | None = None) -> str:
    """Per golden matmul, the golden's **rank under the prior** over the shape's
    full (gated) enumeration — the realistic "would greedy-with-prior pick the
    golden?" test (greedy argmaxes the prior across the enumeration, not just the
    benched leaves that :func:`reachability` scores). The ``S_*`` shape features
    come from the dataset's matching op group (so only shapes with tuned data are
    scored); ``H_*`` is the deployable compile regime (``Context.features``,
    ``H_opt=3``) the greedy ``compile`` / ``run`` actually queries with.
    ``kernel_filter`` restricts to golden configs whose name contains it."""
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.pipeline.search import heuristic  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: PLC0415

    # Index op groups by (free-dim product, reduce extent, is-matmul) so each
    # golden shape maps to the S_* signature it was tuned under.
    index: dict[tuple, dict] = {}
    for sig in group_by_op(prior._dataset):
        d = dict(sig)
        key = (int(d.get("S_ext_free_prod", 0)), int(d.get("S_ext_reduce_max", 0)), d.get("S_n_mma", 0) > 0)
        index.setdefault(key, {k: v for k, v in d.items() if k.startswith("S_")})

    thread = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
    warp = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")
    rows, lines = [], ["[prior] golden selection — golden's rank under the prior over the full gated enumeration:"]
    for g in GOLDEN_CONFIGS:
        if not isinstance(g, MatmulGoldenConfig):
            continue
        if kernel_filter and kernel_filter not in g.name:
            continue
        s_feats = index.get((g.M * g.N, g.K, g.dtype != "fp32"))
        if s_feats is None:
            continue
        base = {**Context.from_target(g.compute_cap).features(), **s_feats}
        gold = {k: v for k, v in g.knobs.items() if k in (thread if g.dtype == "fp32" else warp)}
        _, rank, pool = heuristic.evaluate_golden(
            g.M, g.N, g.K, g.dtype, gold, Context.from_target(g.compute_cap), scorer=lambda r, b=base: prior.mean_score({**b, **r})
        )
        if rank is not None:
            rows.append((g.name, rank, pool))
    for name, rank, pool in sorted(rows, key=lambda t: -t[1]):
        lines.append(f"    {name:26}  rank {rank:5}/{pool}")
    if rows:
        ranks = [r for _, r, _ in rows]
        n = len(ranks)
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in (1, 10, 25, 50))
        lines.append(f"  median rank={sorted(ranks)[n // 2]}  {cov}  (over {n} golden shapes with tuned data)")
    else:
        lines.append("  no golden shapes have tuned data in the dataset yet")
    return "\n".join(lines)


def report(prior) -> str:
    """The full offline diagnostics block for a (re)fit prior."""
    dataset = prior._dataset
    groups = group_by_op(dataset)
    lines = [f"[prior] dataset: {len(dataset)} rows, {len(groups)} op-structures, fitted={prior.fitted}"]
    if not prior.fitted:
        lines.append("  no model — dataset below min_rows; run `deplodock tune <model>` to gather more")
        return "\n".join(lines)

    rr = reachability(prior, groups)
    if rr:
        ratios = [r[3] for r in rr]
        lines.append("[prior] argmax reachability — does the prior's argmax recover each op's measured best?")
        agg = f"mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x  worst {max(ratios):.2f}x"
        lines.append(f"  {agg}   (1.00 = optimum)")
        for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
            flag = "  <-- misses best" if ratio > 1.2 else ""
            lines.append(f"    {label:26}  best {best_us:8.2f}us  argmax {pick_us:8.2f}us  ({ratio:.2f}x, {n} configs){flag}")
    calib = _calibration(prior, groups)
    if calib is not None:
        lines.append(f"[prior] ranking calibration (median per-op Spearman): {calib:+.2f}")

    covered, total = _golden_coverage(groups)
    lines.append(f"[prior] golden coverage: {covered}/{total} golden matmul shapes have data in the dataset")
    if covered == 0:
        lines.append("  none yet — tune the golden shapes (`scripts/find_golden_configs.py` / matmul snippets) to validate against them")
    return "\n".join(lines)
