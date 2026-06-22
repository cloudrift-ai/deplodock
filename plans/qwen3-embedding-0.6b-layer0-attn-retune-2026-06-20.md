# Qwen3-Embedding-0.6B — layer-0 attention re-tune (2026-06-20)

**Status:** complete. Clean dynamic (symbolic `seq_len`, masked-tile) autotune of layer 0 on an RTX 5090, re-run to
re-measure the attention kernels after the post-baseline compiler work landed — most relevantly
**[#263 online-softmax flash attention](https://github.com/cloudrift-ai/deplodock/pull/263)** (the baseline report
called fused flash the "entire e2e gap" and future work), plus **#260** (masked-K MMA P@V), **#261** (ReduceCarrier),
**#264** (atomic-free reduction). Baseline: `plans/qwen3-embedding-0.6b-layer0-tune-findings.md` (2026-06-18).

**Headline:** the layer-0 e2e is **unchanged from the baseline** — deplodock ~195–218 µs vs eager 223 (≈par to 1.15×)
and **~1.3–1.45× behind torch.compile (150 µs)**. Three things actually moved, none at the whole-program level: (1)
**flash #263 cannot reach this model** (knob off by default + GQA + scalar-tier + masked SDPA — root-caused in code,
Finding 1); (2) **tune `bench_fail` went 5 → 0** (#260 cleared the masked-K validate rejections); (3) **the baseline's
"softmax is the bottleneck" was a reproducer-total artifact** — in the *real deployed* layer the softmax reduce is
**13.6 µs / 7.1 %** with **9 K** bank conflicts (baseline NCU: 20.7 µs, **107 K** conflicts — improved), and the gap to
torch.compile is now **attention fragmentation** (4 SDPA kernels ≈40 µs vs tcompile's 1 fused flash ≈25 µs) **plus
shared-load bank conflicts in the MMA matmuls** (QK^T 983 K, MLP up-proj 1.47 M, q/k-norm 688 K conflicts; cutlass ref
= 0). A latent regression also surfaced: **prior pick-reachability collapsed** (mean 1.27× → 4.99×, worst 2.13× → 51×),
but it bites the standalone reproducers / a serving deploy, not the fused-layer greedy pick (Finding 4).

**Date:** 2026-06-20. **GPU:** NVIDIA GeForce RTX 5090 (sm_120), driver 580.159.03. **ncu:** 2025.3.1 (worked; no
permission gate).

**Run command**

```bash
# DEPLODOCK_TUNE_DB / DEPLODOCK_PRIOR_FILE → _tune/tune-attn-qwen3-emb-l0/dynamic.{db,prior.json}
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench \
    --dump-dir _tune/tune-attn-qwen3-emb-l0/dump
```

**Run stats:** 8 fused terminals in **6056.6 s (~101 min)**; 3151 perf rows; prior 7831 benches (warmup 61 / post
7770), post-Spearman **+0.99**; **0 `bench_fail`** (baseline: 5). Benched at **seq_len = 512** (the `DEFAULT_SEQ_HINT`;
`--seq-len` only sizes trace tensors).

**Number-family disclaimer (do not mix).** Every µs from a `--bench` table or an `eval` `-O3 us` column is **-O3,
deployable, CUDA-graph captured**. Tune-DB latencies (`eval variants` `us`, `pick: rank R/N`) are **-Xcicc -O1** — a
ranking signal only (reduction/attention kernels run 1.5–3× slower at -O1). The NCU `dur (ns)` are **uncaptured**
(ncu launches the eager forward too) — read NCU's *counters* (occ %, conflicts, regs), not its durations, against the
captured bench.

---

## Bench results

### Layer-0 dynamic end-to-end (seq_len = 512 hint; torch inputs tiled to match)

| Backend       | tune `--bench` | clean `run --bench` | vs eager      |
|---------------|----------------|---------------------|---------------|
| Eager PyTorch | 223            | 224                 | 1.00×         |
| torch.compile | 150            | 150                 | 1.49×         |
| **Deplodock** | **218**        | **195**             | **1.02–1.15×** |

The two `--bench` invocations bracket the deplodock number (run-to-run variance in scalar-tier picks). Either way:
≈par with eager, **~1.3–1.45× behind torch.compile** — statistically the same as the baseline (216 µs / 1.03×).

### Real deployed per-launch kernel table (clean `run --layer 0 --bench`, sorted by %)

This is the **authoritative** breakdown — the fused layer's actual kernels, NOT the standalone reproducers (which
over-count; see Finding 2). TOTAL 192.3 µs, whole-program e2e 194.5 µs.

| kernel                            | layer op                       | µs   | %     | tier   | occ  |
|-----------------------------------|--------------------------------|------|-------|--------|------|
| `k_linear_mean_reduce_63573a_xn`  | q/k-norm producer (scalar)     | 30.5 | 15.9% | scalar | 100% |
| `k_mean_linear_reduce_ca9764`     | RMSNorm + proj                 | 26.2 | 13.6% | scalar | 100% |
| `k_linear_mean_reduce_05d34c`     | q/k-norm: 2×Linear + Mean      | 23.0 | 12.0% | MMA    | 67%  |
| `k_mean_linear_reduce_34d0f3`     | RMSNorm + proj                 | 21.1 | 11.0% | scalar | 100% |
| `k_mean_d65726`                   | RMSNorm (Mean)                 | 20.4 | 10.6% | scalar | 100% |
| `k_linear_reduce_716194` (×2)     | o_proj / down                  | 15.3+1.9 | 8.9% | MMA  | —    |
| `k_sdpa_reduce_77b0f0`            | **rotary+mask+softmax**        | 13.6 | 7.1%  | MMA    | —    |
| `k_sdpa_linear_reduce_0045cd`     | **P@V**                        | 10.1 | 5.3%  | MMA    | 42%  |
| `k_linear_reduce_f94dd0`          | linear                         | 9.3  | 4.9%  | MMA    | —    |
| `k_linear_sdpa_reduce_6bb11e`     | **q_proj + QK^T scores**       | 9.4  | 4.9%  | MMA    | 17%  |
| `k_sdpa_linear_reduce_0045cd_xn`  | P@V producer (scalar)          | 6.9  | 3.6%  | scalar | 100% |
| `k_linear_0837e7`                 | linear + activation            | 2.6  | 1.3%  | MMA    | —    |

**The attention is ~20 % of the layer (≈40 µs: QK^T 9.4 + softmax 13.6 + P@V-xn 6.9 + P@V 10.1), not the dominant
cost.** The layer is dominated by the RMSNorm / q-k-norm / MLP reduce kernels (`k_mean_*`, `k_linear_mean_reduce_*`) —
but those already **beat eager and torch.compile** per-kernel (see below), so they are *not* the gap to tcompile.

### Per-kernel -O3 reproducer table (`62_kernel_bench.json`, sorted by deplodock µs)

> **Read with Finding 2.** These are *reproducer totals* — each SDPA reproducer re-lowers the full QK^T+softmax+P@V
> standalone, so its µs is a multi-kernel sum, NOT a single deployed kernel. The deployed kernel costs are the table
> above. The matmuls are fast; the reproducer total is dominated by re-materialized softmax + scalar split producers.

| kernel                        | layer op                        | eager | tcompile | deplodock | vs eager                       |
|-------------------------------|---------------------------------|-------|----------|-----------|--------------------------------|
| `k_sdpa_reduce_042770`        | softmax (reproducer = 5 kernels)| 156   | 25       | **156**   | 1.00× (reproducer artifact)    |
| `k_linear_sdpa_reduce_6bb11e` | QK^T  (reproducer = 4 kernels)  | 45    | 43       | **84**    | 0.54× (reproducer artifact)    |
| `k_sdpa_linear_reduce_0045cd` | P@V   (reproducer = 4 kernels)  | 34    | 34       | **59**    | 0.57× (reproducer artifact)    |
| `k_linear_mean_reduce_05d34c` | q/k-norm: 2×Linear + Mean       | 121   | 58       | 51        | 2.36× ✓                        |
| `k_linear_0837e7`             | linear + activation             | 25    | 25       | 26        | 0.98× ≈                        |
| `k_mean_linear_reduce_34d0f3` | RMSNorm + proj                  | 106   | 20       | 18        | 5.78× ✓                        |
| `k_linear_reduce_f94dd0`      | linear                          | 16    | 16       | 17        | 0.98× ≈                        |
| `k_mean_linear_reduce_ca9764` | RMSNorm + proj                  | 77    | 14       | 12        | 6.28× ✓                        |
| `k_linear_reduce_716194`      | linear                          | 10    | 10       | 10        | 0.99× ≈                        |
| `k_mean_d65726`               | RMSNorm (Mean)                  | 65    | 4        | 2         | 34.1× ✓                        |

---

## Finding 1 — flash attention (#263) does not reach this model (root cause in code; not a perf regression)

**Why it matters.** The baseline report (Finding 3) pinned the entire gap to torch.compile on the absence of a fused
flash kernel and called it "future work." #263 landed online-softmax flash — but it is gated out of this model three
ways, so the re-tune confirms it changes nothing here.

**Evidence (code, not measurement).**

- `compiler/pipeline/passes/loop/fusion/025_recognize_flash.py:127` — `if not flash_enabled(): raise RuleSkipped`.
  `flash_enabled()` (`_flash.py:122`) returns False unless `DEPLODOCK_FLASH=1`; **the knob defaults off**, so a
  default tune never explores the flash fork. (Confirmed: `grep -ci flash tune.log` → 0.)
- `_flash.py:132` `flash_shape_eligible` returns False on `has_mask`, on symbolic/mismatched batch-head dims (**GQA**),
  and on symbolic head_dim. Qwen3-Embedding-0.6B is **GQA: `num_attention_heads=16`, `num_key_value_heads=8`**
  (`config.json`), and its layer-0 SDPA carries a mask — either disqualifies it.
- #263 is **scalar-tier** (the docstring: "Online-softmax flash attention (scalar tier)"). The deployed softmax reduce
  here is already MMA-tier at 13.6 µs; a scalar flash nest at seq=512×16-heads would not beat it.

**Fix (priority: MEDIUM, this is the real perf headroom).** Extend the flash recognizer to **masked + GQA** SDPA and
lift it to the **warp/MMA tier**, then enable it by default for eligible shapes. Until then the layer keeps 4 separate
SDPA kernels (Finding 3). This is a compiler feature, not a tuning knob — no amount of tuning closes it.

## Finding 2 — the per-kernel reproducer table over-counts the SDPA kernels by 4–12× (read the deployed table)

**Symptom.** The `62_kernel_bench.json` table attributes 156 µs to `k_sdpa_reduce`, 84 µs to QK^T, 59 µs to P@V — which
reads as "attention is slow." It is not: in the deployed layer those same ops are **13.6 / 9.4 / 10.1 µs**.

**Evidence (`run --ir <reproducer>.torch.json --bench` decomposition).** Each sliced reproducer re-lowers the *whole*
SDPA with no surrounding graph to fuse into, so it explodes into 4–5 kernels — and the standalone re-lowering deploys
**degenerate scalar split-producers that the fused layer never emits**:

- **softmax reproducer (155.7 µs) = 5 kernels**, of which two scalar producers `k_sdpa_reduce_77b0f0_xna` + `_xnb` are
  **61.3 + 61.2 = 122.5 µs (79 %) at block=8, grid=131072, occupancy 0 %** — pure artifact. The real MMA reduces are
  13.7 + 10.9 µs. *In the deployed layer there are no `_xna`/`_xnb`; the softmax is one 13.6 µs kernel.*
- **QK^T reproducer (83.8 µs) = 4 kernels**: the QK^T matmul `k_linear_sdpa_reduce_6bb11e` is **10.7 µs**; the rest is
  re-materialized softmax (25.2 + 34.9 µs) + an `_xn` producer (11.5 µs).
- **P@V reproducer (59.4 µs) = 4 kernels**: the P@V matmul `k_sdpa_linear_reduce_2fafc0` is **11.1 µs**; the rest is
  softmax 25.4 + producer 10.6 + V-proj 10.5 µs.

**Conclusion.** Every matmul is fast and competitive (QK^T 9.4–10.7, P@V 10.1–11.1, all MMA-tier). The reproducer total
is dominated by re-decomposed softmax + scalar producers that exist *only* because the slice has nothing to fuse with.
**This is baseline Finding 2, still unfixed.** Read the deployed per-launch table; never the reproducer total keyed by
one name.

**Fix (priority: MEDIUM — tooling).** `62_kernel_bench.json` / `kernels.html` should carry the sub-kernel breakdown
(and flag rows where the reproducer re-lowers to >1 kernel) that the `run --ir` knob table already prints. This single
gap cost the most manual drilling in both this run and the baseline.

## Finding 3 — the gap to torch.compile is attention fragmentation + MMA-matmul bank conflicts

**Symptom.** Deplodock 195 vs torch.compile 150 (~45 µs). The softmax bank-conflict problem the baseline flagged
(Finding 3: 107 K conflicts) **improved** — the deployed softmax `k_sdpa_reduce` is now 8 µs with **9 K** conflicts.
The cost re-localized.

**Evidence (NCU compare, deployed layer, `ncu-layer/61_ncu_metrics.json`).** Counters (durations are uncaptured —
ignore):

| side | kernel                              | dur(ns) | occ% | sm% | dram% | fma% | ld.cnflct     | regs |
|------|-------------------------------------|---------|------|-----|-------|------|---------------|------|
| dep  | `k_linear_86a525` (MLP up-proj)     | 23,552  | 4.4  | 7.7 | 21.3  | 0.9  | **1,474,560** | 32   |
| dep  | `k_linear_sdpa_reduce` (QK^T)       | 13,600  | 4.0  | 8.8 | 27.1  | 0.9  | **983,040**   | 34   |
| dep  | `k_linear_mean_reduce_ad3a48` (qkn) | 16,352  | 10.3 | 9.4 | 32.3  | 0.7  | **688,128**   | 38   |
| dep  | `k_sdpa_reduce` (softmax)           | 8,032   | 4.2  | 0.3 | 2.0   | 0.5  | 9,216         | 52   |
| ref  | `cutlass_80_wmma_tensorop_f16` mm   | 37,248  | 8.4  | 7.1 | 48.3  | 0.4  | **0**         | 80   |
| ref  | `cutlass_80_wmma_tensorop_f16` mm   | 26,624  | 2.5  | 4.9 | 25.1  | 0.1  | 4,116         | 112  |

Two compounding causes:

1. **Fragmentation.** torch.compile fuses the whole attention into one flash kernel (≈25 µs); deplodock spreads it
   across QK^T + softmax + P@V-producer + P@V (≈40 µs). ~15 µs of the gap is the round-trips. Closing it = Finding 1
   (masked/GQA MMA flash).
2. **Codegen quality — `ldmatrix` smem bank conflicts.** Deplodock's MMA matmuls run at **0.7–1.5 M shared-load bank
   conflicts and 4–10 % occupancy** (SM 7–9 %, FMA <1 % — conflict/latency-bound, not compute-bound); the cuBLAS/
   cutlass references hit the **same shapes with 0 conflicts** (swizzled smem) at comparable occupancy. The deplodock
   kernels still win or tie on wall-time because they are well-fused, but the conflicts are pure headroom.

**Fix (priority: HIGH for #2 — broad leverage).** A conflict-free (swizzled / XOR-permuted) smem layout for the MMA
`ldmatrix` staging — this is the same root issue the baseline saw on the softmax, now visible on every MMA matmul, and
it touches the QK^T, MLP, and q/k-norm kernels (the layer's biggest absolute µs). `PAD_SMEM` / `PERMUTE_LANES` are
no-ops here (the `+1` pad only fires "when at least one Source benefits," `tile/070_pad_smem`; the conflicts are in the
fixed `ldmatrix` path the pad doesn't touch). Needs a real swizzle, not the pad fork.

## Finding 4 — prior pick-reachability regressed (latent: reproducers / serving, not the fused-layer deploy)

**Symptom.** `eval prior --dataset db`: **mean 4.99×, median 1.28×, worst 51.29×** — vs the baseline's mean 1.27× /
median 1.16× / worst 2.13× (Finding 4). The worst miss `matmul free=128 best 19.5 µs pick 1000 µs (51×)` is the
softmax MMA reduce. `eval variants` shows the standalone SDPA reproducers deploying far from the measured best:

```
k_sdpa_reduce_042770        pick rank 128/363, 8.17x of best   (deploys scalar where rank-1 is mma_m16n8k16_f16)
k_sdpa_linear_reduce_0045cd pick rank 105/418, 7.56x of best   (deploys a slow MMA: WM8/WN1 vs rank-1 WM2/WN2)
k_sdpa_linear_reduce_..._xn pick rank 156/168, 5.82x of best
k_linear_sdpa_reduce_6bb11e pick rank   3/128, 1.08x of best   (HEALTHY — only the QK^T is fine)
```

**Tell.** Every rank-1 config shows `-O3 us = —` (never -O3 re-benched) while the deployed slow picks *do* carry an -O3
number. The prior's evidence-pick prefers measured -O3 reservoir rows, and -O3 evidence exists only for the slower
configs that compiled fast at -O3 — the fast unrolled MMA configs (the reason tune compiles at -O1) appear not to have
been -O3 re-benched, so evidence-pick can't see them. `eval knobs` ranks the responsible knobs `SPLIT_CONE` 45×,
`SPLITK` 15.5×, `FM` (p90 **134×**), `MMA` 10.6× — all concentrated on the symbolic SDPA split-cone producers
(`S_dtype_f16`, `S_ext_n_symbolic_axis`, `SPLIT_CONE` are the top-regret feature groups).

**Scope (important).** This is **latent**: the *fused-layer greedy deploy* still picks healthy MMA configs (the
deployed table above is all `mma_m16n8k16_f16` at sane occupancy). The bad picks show up in (a) the standalone
reproducers and (b) — by analogy to the baseline's serving crash — a serving/whole-model graph that traces the SDPA
differently. It did not move the layer e2e, but it is a real reachability regression to fix before the serving A/B is
re-attempted.

**Fix (priority: MEDIUM).** Ensure the -O3 tolerance-band re-bench actually covers the rank-1 -O1 MMA configs for the
SDPA reduces (so evidence-pick has -O3 truth for them), and improve the `FM`/`SPLITK`/`SPLIT_CONE` features in the
prior. Confirm per kernel: re-tune the offending reproducer with 2–4× patience (no `--clean`); if rank-1 becomes the
pick it is prior/evidence, not codegen.

## Finding 5 — masked-K bench failures cleared (#260)

The tune logged **0 `bench_fail`** (baseline: 5, all "compile stage exceeded 2.0 s budget" on the masked split P@V
producer). #260's masked-K MMA staging fixes (the `Source.dtype` validate-sizing bug et al.) removed the rejections —
the masked-K MMA configs now enumerate and bench cleanly. No wasted search slots this run.

---

## Repro / artifacts

Work dir: `_tune/tune-attn-qwen3-emb-l0/` (under gitignored `_tune/`).

- Tune log: `tune.log`. Dump: `dump/` (`07_lowering_cuda.kernels/*.torch.json` reproducers, `62_kernel_bench.json`,
  `kernels.html`). Tune DB / prior: `dynamic.db`, `dynamic.prior.json`.
- Drills: `drill_k_sdpa_reduce_042770.log`, `drill_k_linear_sdpa_reduce_6bb11e.log`,
  `drill_k_sdpa_linear_reduce_0045cd.log`. Deployed layer: `run_layer_e2e.log`. NCU:
  `ncu-layer/61_ncu_metrics.{csv,json}`.
- Triage: `variants_{softmax,qkt,pv}.txt`, `prior_reach.txt`, `knobs.txt`.

```bash
# point env at this run's DB/prior
export DEPLODOCK_TUNE_DB=$PWD/_tune/tune-attn-qwen3-emb-l0/dynamic.db
export DEPLODOCK_PRIOR_FILE=$PWD/_tune/tune-attn-qwen3-emb-l0/dynamic.prior.json
D=_tune/tune-attn-qwen3-emb-l0/dump/07_lowering_cuda.kernels

deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench            # deployed kernel table (F2/F3)
deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench --profile  # NCU compare (Finding 3)
deplodock run --ir $D/k_sdpa_reduce_042770.torch.json --bench                              # reproducer decomposition (F2)
deplodock eval prior --dataset db                                                          # reachability regression (F4)
deplodock eval variants --kernel k_sdpa_reduce                                             # pick rank 128/363 (F4)
grep -ci flash _tune/tune-attn-qwen3-emb-l0/tune.log                                       # → 0, flash never fired (F1)
```

---

## Workflow notes

- **Reproducer over-counting is still the single biggest "data only after manual drilling" item (baseline Finding 2,
  unfixed).** The `62_kernel_bench.json` softmax row says 156 µs; the deployed kernel is 13.6 µs. It took a
  `run --ir … --bench` per SDPA kernel **plus** a full-layer `run --bench` to establish that the 122 µs of scalar
  block=8/occ-0 % producers are a reproducer artifact, not a deployed cost. *Proposal:* flag/decompose multi-kernel
  reproducer rows in the machine-readable output and the HTML, as the baseline already proposed. Until then, the
  per-kernel `--bench` table is actively misleading for SDPA — lead any attention analysis with the deployed
  per-launch table.
- **`eval variants` should surface the -O3 coverage gap.** The `-O3 us = —` on every rank-1 row vs a number on the
  deployed pick is the entire Finding 4 story, but it took cross-referencing three `eval` views to see it. *Proposal:*
  an `eval variants` column or note: "rank-1 has no -O3 re-bench; evidence-pick can't select it."
- **Tune wall (~101 min for one layer) matches the baseline** — still ~5× the skill's "~10–20 min" estimate. The cost
  is the SP-MCTS structural forks (8 terminals, each re-running the full per-kernel inner search). A `--max-terminals`
  / wall cap remains the obvious "quick look" lever.
- **The `[cuda] kernel 'k_mean_d65726' still pending after 0.20s` lines** spammed the `run` stdout (one pair per replay)
  — harmless polling noise, but it buried the bench table. *Proposal:* demote to `-v`.
- **No GPU flakiness this run** — single 5090, GPU free throughout, NCU ran without the perf-counter permission gate,
  0 `bench_fail`. The baseline's masked-K crash class is gone (#260).
</content>
</invoke>
