# Analytic prior evaluation — node store (RTX 4090 + RTX 5090)

## Provenance

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Commit | `bc5ae6dae0a241c95189dacc6a89eb80b27e9896` (`bc5ae6da`) — "Fold the node merge into remote_node_tune.py so the run completes in one step" |
| Branch | `feature/collect-node-data` |
| **Prior evaluated** | **cold `AnalyticPrior`** (hand-coded; `fitted=False`) — no `~/.cache/emmy/prior.json` present, so no learned `CatBoostPrior` was loaded |
| Tune DB | `~/.cache/emmy/autotune.db` (40 MB) — `node` table only (merged via `scripts/merge_node_db.py`) |
| Dataset | search-tree `node` store: 23,971 nodes across 2 cards (RTX 4090: 12,742 · RTX 5090: 11,229), 29 op_sigs each |
| Cards' node data tuned on | CloudRift RTX 4090 + RTX 5090, `emmy tune --dataset golden` (29/29 shapes each), -O1 ranking + -O3 rebench |

Reproduce:

```bash
emmy eval prior --dataset nodes      # per-card AnalyticPrior scoring on the node store
emmy eval analytic                   # AnalyticPrior vs recorded goldens (golden-based; ignores the node DB)
```

## `eval prior --dataset nodes` — AnalyticPrior scored on the collected node data (per card)

| Metric | RTX 4090 | RTX 5090 | Meaning | Direction |
|---|---|---|---|---|
| Fork sibling-ranking — top-1 | 0.390 | 0.690 | Fraction of forks where the prior's predicted-best child is a true-best child (min `value_us`) | bigger = better |
| Fork sibling-ranking — median per-fork Spearman | +0.51 | +0.90 | Rank corr. of predicted vs value-of-position order, over forks with ≥3 children; median | bigger = better (0 = random) |
| Leaf reachability — median | 1.43× | 2.19× | `pick_latency / best_latency` per op (deployed config vs fastest measured leaf), median | smaller = better (1.00 = optimum) |
| Leaf reachability — mean / worst | 1.80× / 8.16× | 2.34× / 6.06× | same ratio, mean and worst-case | smaller = better |
| Leaf calibration — median per-op Spearman | −0.03 | +0.14 | Rank corr. of the prior's scores vs actual measured leaf latencies, per op; median | bigger = better (0 = random) |
| (context) forks / children | 269 / 2782 | 267 / 2307 | dataset sizes (not quality) | — |
| (context) ops | 28 | 29 | dataset sizes (not quality) | — |

## Fork sibling-ranking — random baseline (top-1 depends on sibling count)

top-1 is an average over forks with very uneven child counts, so the honest baseline is `mean(1/n_children)`, **not** 0.5.
There are **no ties** in either card (every fork has exactly one best child), so random `= mean(n_best/n_children) =
mean(1/n_children)`.

| | RTX 4090 | RTX 5090 |
|---|---|---|
| Prior top-1 | 0.390 | 0.690 |
| **Random baseline** (`mean 1/n_children`) | **0.379** | **0.418** |
| **Lift over random** | **1.03×** | **1.65×** |
| children/fork: mean / median / max | 10.3 / 2 / 116 | 8.6 / 2 / 94 |
| fully-tied forks | 0% | 0% |

**Read:** on the **4090 the cold prior ≈ random** (1.03× — essentially no skill at picking the best sibling); on the
**5090 it has real skill** (1.65× over random). The raw 0.39-vs-0.69 gap understates this — against the n-dependent
baselines (0.38 vs 0.42), the 4090 is at chance while the 5090 clearly beats it.

The `median per-fork Spearman` baseline is **0** for any sibling count (random permutation), so +0.51 / +0.90 are above
random — but note it is measured only on ≥3-child forks, while top-1 covers all ≥2-child forks. The 4090's decent
Spearman (+0.51) with random-level top-1 means the prior gets larger forks' overall   ordering partly right but fumbles the
single argmin (and the many 2-child forks, excluded from Spearman, pull top-1 to chance).

## `eval analytic` — AnalyticPrior vs recorded goldens (golden-based; does NOT read the node DB)

```
analytic golden rank — median=46  top1=4/16  top10=4/16  top25=4/16  top50=9/16  top100=13/16
```

| Metric | Value | Meaning | Direction |
|---|---|---|---|
| golden rank — median | 46 | position of the recorded golden in the prior's full enumeration (1 = prior's top pick) | smaller = better |
| top1 / top10 / top25 / top50 / top100 | 4/16 · 4/16 · 4/16 · 9/16 · 13/16 | goldens (of 16 ranked) within the prior's top-N | bigger = better |

Note: only the 16 fp16-square goldens received a numeric rank; the fp32 matmul goldens showed rank `?` (the cold prior
could not locate that golden config in its enumeration).

## Takeaways

- The cold `AnalyticPrior` is weak as expected: leaf calibration ≈ 0 on both cards, and on the **4090 its fork top-1 is
  no better than random**. This quantifies the headroom for a *learned* prior fit on this data.
- Clear same-arch / different-card signal: the prior orders the 5090's choices well (1.65× random, Spearman +0.90) yet
  lands further from the 5090's absolute optimum (leaf reachability median 2.19× vs the 4090's 1.43×) — exactly the
  divergence the GPU-keyed cross-hardware node store was built to expose.
- Next step (not done here): to fit + evaluate the learned `CatBoostPrior` on this data, the prior's reservoir
  (`prior.json`) is needed — only the `node` table was merged back, so an offline `emmy tune` refit has no local
  training data yet.
