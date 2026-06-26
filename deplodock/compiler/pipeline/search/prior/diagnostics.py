"""Offline diagnostics for the learned tuning prior — "can the prior actually
reach the best configs?", with no GPU.

``deplodock tune`` with no model / ``--code`` refits the prior on its persisted
reservoir dataset and prints :func:`report`: a dataset summary, per-op **pick
reachability** (does the prior's predicted-fastest config over an op's *measured*
configs recover the measured-best leaf?), an in-sample ranking calibration, and
coverage of the golden matmul configs. Reachability is the optimistic upper bound
on greedy quality — if the prior can't even rank an op's own benched leaves to the
top, greedy (which descends fork-by-fork) can't either.
"""

from __future__ import annotations

import math
import statistics

from deplodock.compiler.pipeline.search.data import Dataset, ShapeKey


def _n_tunable(knobs: dict) -> int:
    return sum(1 for k in knobs if not k.startswith(("S_", "H_")))


def _matmul_sig(d: dict) -> bool:
    """Histogram heuristic for "this op group is a matmul": a product feeding a
    reduce-add over ≥2 distinct inputs. ``S_n_mma`` is NOT usable as the marker:
    the stamp pass runs at fusion end, before the tile tier emits ``Mma`` stmts,
    so it is 0.0 on every stamped row — gating on it made golden coverage
    permanently empty and dropped every fp16 golden from the rank/deploy joins
    (see ``ShapeKey.from_s_features``)."""
    return bool(d.get("S_reduce_add", 0) and d.get("S_pw_multiply", 0) and d.get("S_n_distinct_input", 0) >= 2)


def _op_label(sig: tuple) -> str:
    d = dict(sig)
    if _matmul_sig(d):
        kind = "matmul"
    elif d.get("S_reduce_add", 0) or d.get("S_reduce_max", 0):
        kind = "reduce"
    else:
        kind = "pointwise"
    free, red = int(d.get("S_ext_free_max", 0)), int(d.get("S_ext_reduce_max", 0))
    return f"{kind:9} free={free}" + (f" red={red}" if red else "")


def reachability(prior, groups: dict) -> list[tuple]:
    """Per op: ``(label, best_us, pick_us, ratio, n_leaf_configs)`` — the config the
    prior predicts fastest (``mean_score`` argmin) over the op's measured *leaf*
    configs vs the measured best (ratio = the picked config's latency / the best's;
    1.0 = recovers the optimum). ``groups`` maps an ``S_*`` signature to its
    :class:`Sample`s (from :meth:`Dataset.group_by_op`)."""
    out = []
    for sig, grp in groups.items():
        kmax = max(_n_tunable(s.all_knobs()) for s in grp)
        leaves = [s for s in grp if _n_tunable(s.all_knobs()) == kmax]
        if len(leaves) < 2:
            continue
        best_us = min(s.latency_us for s in leaves)  # labels are latency µs — lower is better
        pick_us = min(leaves, key=lambda s: prior.mean_score(s.all_knobs())).latency_us
        out.append((_op_label(sig), best_us, pick_us, pick_us / best_us, len(leaves)))
    return out


def _calibration(prior, groups: dict) -> float | None:
    """Median per-op Spearman ρ between the prior's predicted latency and the
    measured latency over each op's leaf configs (both µs → ρ near ``+1`` when
    well-calibrated)."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    rhos = []
    for grp in groups.values():
        kmax = max(_n_tunable(s.all_knobs()) for s in grp)
        leaves = [s for s in grp if _n_tunable(s.all_knobs()) == kmax]
        if len(leaves) < 3:
            continue
        rho = spearmanr([prior.mean_score(s.all_knobs()) for s in leaves], [s.latency_us for s in leaves]).statistic
        if not math.isnan(rho):
            rhos.append(rho)
    return statistics.median(rhos) if rhos else None


def _golden_coverage(groups: dict) -> tuple[int, int]:
    """How many golden matmul **shapes** have measured data in the dataset, matched by
    :class:`ShapeKey` (free-dim product, reduce extent, dtype flag — so an fp32
    square and its ``.fp16`` twin are counted separately). Counts *distinct shape
    keys*, not per-config rows, so multiple knob sets for one shape — and the same
    shape recurring across per-GPU golden files (``ShapeKey`` is GPU-blind) — count
    once."""
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: PLC0415

    have = set()
    for sig in groups:
        d = dict(sig)
        if _matmul_sig(d):
            have.add(ShapeKey.from_s_features(d))
    golden_keys = {g.shape_key() for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)}
    covered = sum(1 for k in golden_keys if k in have)
    return covered, len(golden_keys)


def golden_prior_eval(prior, kernel_filter: str | None = None) -> str:
    """Per golden matmul, the golden's **rank under the prior** over the shape's
    full (gated) enumeration — the realistic "would greedy-with-prior pick the
    golden?" test (greedy picks the prior's predicted-fastest config across the
    enumeration, not just the benched leaves that :func:`reachability` scores).
    The ``S_*`` shape features
    come from the dataset's matching op group (so only shapes with tuned data are
    scored); ``H_*`` is the deployable compile regime (``Context.features``,
    ``H_opt=3``) the greedy ``compile`` / ``run`` actually queries with.
    ``kernel_filter`` restricts to golden configs whose name contains it.

    The rank is a **model diagnostic, not a deploy prediction**: the enumeration
    rows carry only the planner's tunables, while the rows the model trained on
    (and the leaves greedy scores late in the descent) also carry decision /
    transport stamps (``TMA``, ``PIPELINE_STAGES``, …) — absent keys are NaN
    ("not decided") to CatBoost, so the two surfaces can disagree (the 2026-06-12
    sweep's finding 6). Deploy reality is ``Prior.pick`` (measured -O3 evidence
    first); the faithful deploy check is ``eval golden``'s real greedy compile."""
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.pipeline.search import analytic  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.golden import MatmulGoldenConfig, goldens_for_live_gpu  # noqa: PLC0415

    GOLDEN_CONFIGS = goldens_for_live_gpu()  # live card only — see golden_deploy_perf

    # Index the matmul op groups by ShapeKey (free-dim product, reduce extent,
    # dtype flag — both sides built by the ShapeKey constructors, so the fp32/fp16
    # twins never merge) so each golden shape maps to the S_* signature it was
    # tuned under.
    index: dict[ShapeKey, dict] = {}
    for sig in Dataset.from_prior(prior).group_by_op():
        d = dict(sig)
        if not _matmul_sig(d):
            continue
        index.setdefault(ShapeKey.from_s_features(d), {k: v for k, v in d.items() if k.startswith("S_")})

    thread = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
    warp = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")
    rows, skipped = [], []
    lines = ["[prior] golden selection — golden's rank under the prior over the full gated enumeration:"]
    for g in GOLDEN_CONFIGS:
        if not isinstance(g, MatmulGoldenConfig):
            continue
        if kernel_filter and kernel_filter not in g.name:
            continue
        s_feats = index.get(g.shape_key())
        if s_feats is None:
            # A silent skip here hid the fp16 lockout in the 2026-06-12 sweep —
            # every unjoinable shape gets a per-shape line instead.
            skipped.append((g.name, "no tuned rows for this shape in the prior dataset"))
            continue
        base = {**Context.from_target(g.compute_cap).features(), **s_feats}
        gold = {k: v for k, v in g.knobs.items() if k in (thread if g.dtype == "fp32" else warp)}
        # ``evaluate_golden`` ranks by descending score; the prior predicts latency
        # (lower = better), so negate to rank the predicted-fastest config first.
        _, rank, pool = analytic.evaluate_golden(
            g.M, g.N, g.K, g.dtype, gold, Context.from_target(g.compute_cap), scorer=lambda r, b=base: -prior.mean_score({**b, **r})
        )
        if rank is None:
            skipped.append((g.name, f"recorded knobs not in the enumeration ({pool} rows) — pin/dtype mismatch?"))
            continue
        rows.append((g.name, rank, pool))
    for name, rank, pool in sorted(rows, key=lambda t: -t[1]):
        lines.append(f"    {name:26}  rank {rank:5}/{pool}")
    for name, why in skipped:
        lines.append(f"    {name:26}  SKIPPED: {why}")
    if rows:
        ranks = [r for _, r, _ in rows]
        n = len(ranks)
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in (1, 10, 25, 50))
        lines.append(f"  median rank={sorted(ranks)[n // 2]}  {cov}  (over {n} golden shapes with tuned data)")
    elif not skipped:
        lines.append("  no golden shapes have tuned data in the dataset yet")
    return "\n".join(lines)


def golden_deploy_perf(prior, kernel_filter: str | None = None) -> dict[str, float]:
    """Per golden shape, ``pick_us / golden_us`` — the deployable (-O3) latency of the
    prior's predicted-best **measured** config over the golden's recorded latency, read
    from the prior's reservoir with **no re-bench**.

    Tuning re-benches every winner at -O3 (``H_opt=3``) and feeds it to the prior, so
    each tuned shape's best config has a deployable row in the reservoir. For each
    golden shape we take the op group's ``H_opt=3`` rows, pick the one ``Prior.pick``
    deploys (measured -O3 evidence first, model argmin otherwise — the same selection
    greedy ``compile`` / ``run`` make), and divide its measured latency by the golden's
    recorded ``deplodock_us`` (also -O3 → same regime, so the ratio is a real
    deployable speed comparison; <1.0 = the prior's pick is faster than golden). Shapes
    with no -O3 reservoir row are omitted (the caller renders ``—``). The reservoir is
    used rather than the raw ``perf`` table because only it carries the ``H_*`` regime
    columns needed to isolate the deployable measurements.

    Goldens are scoped to the live card (:func:`goldens_for_live_gpu`) so a multi-GPU
    goldens dir doesn't make a name's per-card entries collide on the GPU-blind
    ``ShapeKey`` (e.g. RTX 5090 / RTX PRO 6000 both ``(12, 0)``)."""
    from deplodock.compiler.pipeline.search.golden import MatmulGoldenConfig, goldens_for_live_gpu  # noqa: PLC0415

    GOLDEN_CONFIGS = goldens_for_live_gpu()

    # Deployable (-O3) measured rows per matmul op group, indexed by ShapeKey.
    # An fp32 square and its ``.fp16`` twin share (free_prod, reduce), so the key's
    # dtype flag is what keeps them apart — ``ShapeKey.from_s_features`` derives it
    # from ``S_dtype_f32`` (see its docstring for why ``S_n_mma`` can't be the key).
    index: dict[ShapeKey, list] = {}
    for sig, samples in Dataset.from_prior(prior).group_by_op().items():
        d = dict(sig)
        if not _matmul_sig(d):
            continue
        o3 = [s for s in samples if int(s.all_knobs().get("H_opt", 0)) == 3]
        if not o3:
            continue
        index.setdefault(ShapeKey.from_s_features(d), []).extend(o3)

    out: dict[str, float] = {}
    for g in GOLDEN_CONFIGS:
        if not isinstance(g, MatmulGoldenConfig) or not g.deplodock_us:
            continue
        if kernel_filter and kernel_filter not in g.name:
            continue
        leaves = index.get(g.shape_key())
        if not leaves:
            continue
        best_i, _ = prior.pick([s.all_knobs() for s in leaves])
        out[g.name] = leaves[best_i].latency_us / g.deplodock_us
    return out


def report(prior) -> str:
    """The full offline diagnostics block for a (re)fit prior."""
    dataset = prior._dataset
    groups = Dataset.from_prior(prior).group_by_op()
    lines = [f"[prior] dataset: {len(dataset)} rows, {len(groups)} op-structures, fitted={prior.fitted}"]
    if not prior.fitted:
        lines.append("  no model — dataset below min_rows; run `deplodock tune <model>` to gather more")
        return "\n".join(lines)

    rr = reachability(prior, groups)
    if rr:
        ratios = [r[3] for r in rr]
        lines.append("[prior] pick reachability — does the prior's predicted-fastest config recover each op's measured best?")
        agg = f"mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x  worst {max(ratios):.2f}x"
        lines.append(f"  {agg}   (1.00 = optimum)")
        for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
            flag = "  <-- misses best" if ratio > 1.2 else ""
            lines.append(f"    {label:26}  best {best_us:8.2f}us  pick {pick_us:8.2f}us  ({ratio:.2f}x, {n} configs){flag}")
    calib = _calibration(prior, groups)
    if calib is not None:
        lines.append(f"[prior] ranking calibration (median per-op Spearman): {calib:+.2f}")

    covered, total = _golden_coverage(groups)
    lines.append(f"[prior] golden coverage: {covered}/{total} golden matmul shapes have data in the dataset")
    if covered == 0:
        lines.append("  none yet — tune the golden shapes (`deplodock tune --golden NAME`) to validate against them")
    return "\n".join(lines)


def node_sibling_ranking(prior, nodes) -> tuple | None:
    """Fork-level ranking over the autotune ``node`` store — the search-faithful
    metric: group nodes by ``parent_key`` (each group = one fork's children, the
    partial configs the prior actually ranks during ``_select``) and measure whether
    the prior orders them by predicted latency the way their value-of-position labels
    (``value_us`` = best latency reachable below) imply.

    Returns ``(n_forks, top1_rate, median_spearman, n_children)`` — ``top1_rate`` is
    the fraction of forks whose predicted-best child is a true-best child (its
    ``value_us`` equals the fork's minimum); ``median_spearman`` is over forks with
    ≥3 children (``None`` if none). Returns ``None`` when there are no multi-child
    forks. Unlike :func:`reachability` it does NOT leaf-filter — siblings are already
    one fork level (branches and leaves alike), and the prior is scored on each
    node's full feature dict, exactly as ``mcts._select`` queries it."""
    from collections import defaultdict  # noqa: PLC0415

    from scipy.stats import spearmanr  # noqa: PLC0415

    by_parent: dict = defaultdict(list)
    for n in nodes:
        if n.parent_key is not None:
            by_parent[n.parent_key].append(n)

    top1, rhos, n_forks, n_children = 0, [], 0, 0
    for sibs in by_parent.values():
        if len(sibs) < 2:
            continue
        n_forks += 1
        n_children += len(sibs)
        scored = [(prior.mean_score(s.features), s.value_us) for s in sibs]  # (predicted, measured)
        pred_best_value = min(scored, key=lambda t: t[0])[1]  # value_us of the predicted-fastest child
        if pred_best_value <= min(v for _, v in scored) + 1e-12:  # is it a true-best child? (measured ties OK)
            top1 += 1
        if len(sibs) >= 3:
            rho = spearmanr([p for p, _ in scored], [v for _, v in scored]).statistic
            if not math.isnan(rho):
                rhos.append(rho)
    if n_forks == 0:
        return None
    return (n_forks, top1 / n_forks, statistics.median(rhos) if rhos else None, n_children)


def _node_op_label(features: dict) -> str:
    """The op label for a single ``node`` row — derived from its ``S_*`` features, the
    same labels :func:`reachability` rows carry, so ``--kernel`` filters the node store
    by op kind/shape (e.g. ``matmul`` / ``reduce`` / ``free=512``). All nodes of one op
    share these features, so filtering keeps/drops a whole op atomically — parent and
    children never split."""
    return _op_label(tuple(sorted((k, v) for k, v in features.items() if k.startswith("S_"))))


def node_report(prior, nodes, *, kernel_filter: str | None = None) -> str:
    """The ``eval prior --dataset nodes`` block: the fork sibling-ranking (the
    metric unique to this dataset) plus leaf reachability / calibration reused on the
    persistent, deduped node store. ``nodes`` is a list of :class:`db.NodeRow` from
    :meth:`SearchDB.iter_nodes`; ``kernel_filter`` keeps only nodes whose op label
    contains the substring (``--kernel``)."""
    total = len(nodes)
    if kernel_filter:
        nodes = [n for n in nodes if kernel_filter in _node_op_label(n.features)]
    suffix = f" matching --kernel {kernel_filter!r}" if kernel_filter else ""
    lines = [f"[prior] node store: {len(nodes)} nodes{suffix}, fitted={prior.fitted}"]
    if not nodes:
        if kernel_filter and total:
            lines.append(f"  no nodes match --kernel {kernel_filter!r} ({total} in store) — try an op label like 'matmul' / 'reduce'")
        else:
            lines.append("  empty — run `deplodock tune <model>` to populate the node table")
        return "\n".join(lines)
    if not prior.fitted:
        lines.append("  no fitted prior — the cold AnalyticPrior ranks by D_* geometry only; run `deplodock tune`")

    sib = node_sibling_ranking(prior, nodes)
    if sib is None:
        lines.append("[prior] fork sibling-ranking: no multi-child forks recorded")
    else:
        n_forks, top1, rho, n_children = sib
        rho_txt = f"{rho:+.2f}" if rho is not None else "n/a"
        lines.append("[prior] fork sibling-ranking — does the prior rank each fork's children by value-of-position?")
        lines.append(f"  top-1 {top1:.2f}  (predicted-best child is a true-best child)   median per-fork Spearman {rho_txt}")
        lines.append(f"  over {n_forks} forks ({n_children} children)")

    groups = Dataset.from_node_rows(nodes).group_by_op()
    rr = reachability(prior, groups)
    if rr:
        ratios = [r[3] for r in rr]
        lines.append("[prior] leaf reachability over node store — does the prior recover each op's best leaf?")
        agg = f"mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x  worst {max(ratios):.2f}x"
        lines.append(f"  {agg}   (1.00 = optimum)")
        for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
            flag = "  <-- misses best" if ratio > 1.2 else ""
            lines.append(f"    {label:26}  best {best_us:8.2f}us  pick {pick_us:8.2f}us  ({ratio:.2f}x, {n} configs){flag}")
    calib = _calibration(prior, groups)
    if calib is not None:
        lines.append(f"[prior] leaf ranking calibration (median per-op Spearman): {calib:+.2f}")
    return "\n".join(lines)
