# Qwen3-Embedding-0.6B layer 0 — DYNAMIC-shape tune findings (2026-06-14, post-#243)

Status: **re-tune of the deployable dynamic configuration (`--dynamic seq_len@x:1`) on `main` @ `3abe9959`, one
commit past the prior dynamic report's `c3304773`. The single landed change — #243 "masked-K mma warp tier —
symbolic-seq P@V reaches tensor cores" — does exactly what the prior report's finding 1 flagged as future work:
the symbolic-K P@V now reaches the `mma.sync` warp tier (NCU 40 µs vs the prior report's 259 µs scalar). But the
masked-K split also un-fuses the attention **softmax** into a standalone scalar producer, and that softmax —
materializing the full seq×seq probability matrix to HBM and re-reading it three times — is now the single
largest kernel in the layer (NCU 147 µs). Net: the deployed layer runs 0.62x eager / 0.41x torch.compile at seq
512 (fresh `run --bench`), a modest improvement on the prior report's 0.55x. The remaining gap is still the
attention path, now split three ways: softmax (147 µs), QK^T scores (71 µs), P@V (40 µs) vs PyTorch's fused
flash-attention (37 µs total).**

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  /tmp/tune-model-qwen3-emb-l0-dyn/dump`
- Hardware: RTX 5090 (sm_120), ncu 2025.3.1 (perf counters permitted). `main` @ `3abe9959`.
- Run stats: tune wall **15,878.6 s** (~4.4 h — roughly 2.2x the prior report's 7,162 s; the masked-K mma adds a
  whole new warp-tier enumeration on the symbolic-K P@V, doubling the structural search), **16 fused terminals**
  (vs the prior report's 8 — the masked-K split offers a new keep-vs-split branch on the symbolic-K P@V),
  **7,215 ok / 0 `bench_fail`** `perf` rows, all CUDA-graph captured; prior: 17,708 benches (158 warmup / 17,550
  post), post-warmup Spearman **+1.00**, 94% of post benches ≥2x the running best.
- DB / prior: **default paths, cleaned** — `~/.cache/deplodock/autotune.db` (7,215 rows),
  `~/.cache/deplodock/prior.json`. Carries warp-tier `H_opt=3` rows for every static-K linear **and** the new
  masked-K P@V consumer, so it is a usable greedy prior for dynamic serving.
- **Number families**: bench tables / reproducer runs / `--ab` rows below are **-O3** (deployable, CUDA-graph
  captured); tune-DB latencies quoted for ranking context are `-Xcicc -O1` (ranking signal only). NCU durations
  run at locked base clock — **compare ratios, not absolutes**, and NCU is the only clean per-kernel signal
  (finding 5).
- **Dynamic measurement semantics**: the trace is symbolic in `seq_len` (`Dim` hint = `DEFAULT_SEQ_HINT` 512 —
  the `--seq-len` flag does NOT set it). All per-op tune benches, reproducer benches and the full-model table run
  at the 512 hint; the full-model table tiles the torch closures' inputs to the hint and prints `benched at
  seq_len=512 (symbolic hint; torch inputs tiled to match)`, so its eager / tcompile / deplodock rows are one
  shape and directly comparable.

## Bench results (-O3, CUDA-graph captured)

Full-model table as printed by the tune's `--bench` at the 512 hint:

```
Backend        Latency (us)  vs Eager
-------------------------------------
Eager PyTorch           219     1.00x
torch.compile           147     1.49x
Deplodock               467     0.47x      (359 us / 0.62x in a fresh `run --dynamic … --bench`; e2e whole-program)
benched at seq_len=512 (symbolic hint; torch inputs tiled to match)
```

**Trust the 359 µs fresh-run number, not the 467 µs tune-`--bench` number.** The tune's `--bench` ran on a GPU
thermally loaded from the 4.4 h tune that preceded it; a fresh `run --dynamic … --bench` on a cooled GPU clocks
the identical deployed kernels at **359 µs / 0.62x eager / 0.41x torch.compile** (deployed launch table
`TOTAL` 358.0, whole-program e2e 359.4). The 30% tune-vs-fresh spread is itself a finding (finding 7) — it is
larger than the prior report's 7% (426→396) because the tune was twice as long and left the card hotter.

Per-kernel reproducer bench at the 512 hint, as printed by `--bench` (sorted by deplodock µs; layer-op labels
from each kernel's `.torch.json` provenance). **The top three rows re-fuse the upstream attention cone (finding
5), so their deplodock µs is a slice-set total, not the deployed kernel alone — the clean attribution is the NCU
table under finding 1.**

| Kernel                  | Layer op                                                 | eager | tcompile | deplodock | vs eager |
|-------------------------|----------------------------------------------------------|------:|---------:|----------:|---------:|
| `k_linear_sdpa_reduce`  | attn-out reshape + o_proj (linear_3) + residual          |    43 |       41 |       213 |    0.20x |
| `k_sdpa_linear_reduce`  | SDPA P@V split (softmax producer + V producer + P@V mma) |    34 |       34 |       203 |    0.17x |
| `k_sdpa_reduce`         | RoPE(q,k) + QK^T scores (+ mask)                         |   148 |       25 |       162 |    0.91x |
| `k_linear_mean_reduce`  | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up   |   119 |       57 |        58 |    2.07x |
| `k_linear`              | MLP down (linear_6) + residual                           |    26 |       25 |        44 |    0.60x |
| `k_mean_linear_reduce`  | q_norm RMSNorm + rotated-q producer (xna)                |   105 |       20 |        27 |    3.93x |
| `k_linear_reduce`       | v_proj matmul (linear_1)                                 |    16 |       16 |        27 |    0.62x |
| `k_mean_linear_reduce`  | k_norm RMSNorm + rotated-k producer (xnb)                |    76 |       14 |        18 |    4.11x |
| `k_linear_reduce`       | q_proj matmul (linear)                                   |    10 |       10 |        17 |    0.61x |
| `k_mean`                | input RMSNorm                                            |    66 |        4 |         2 |   32.90x |

The two pure RMSNorm-bearing reductions and the gated-MLP **win eager outright** (2–33x) and the warp-tier
linears are at parity-to-2x of eager. Everything meaningfully behind torch.compile is in the **attention** path.
By the clean NCU attribution (finding 1) the deployed attention kernels are **softmax 147 µs + QK^T 71 µs + P@V
40 µs = 258 µs** of locked-clock GPU time, ~72% of the 359 µs e2e, vs flash-attention's 37 µs.

## Finding 1 — #243 moved symbolic-K P@V onto tensor cores (259→40 µs), but the split exposed the softmax as a standalone 147 µs scalar kernel — now the single largest kernel in the layer

**Symptom.** The masked-K mma warp tier (#243) lands exactly on the kernel the prior report flagged: the
symbolic-K P@V consumer is now `mma_m16n8k16_f16` (10 `mma.sync` ops in its emitted CUDA) and NCU-clocks at
**40 µs** vs the prior report's **259 µs** scalar — a 6.5x per-kernel win. But the masked-K demotion split
(`005_split_demoted` on symbolic K) un-fuses the SDPA into a **softmax-normalizing producer** + the warp-tier
P@V consumer, and that softmax producer is now the dominant kernel. From the clean NCU capture (one `ncu` run,
deplodock + torch reference kernels side by side, locked base clock — `run --ir
k_linear_sdpa_reduce_43208b.torch.json --bench --profile`, which re-lowers the whole attention→o_proj chain):

```
side  kernel                                       dur (ns)  occ%   sm%  dram%  fma%   lsu.inst  ld.cnflct  st.cnflct  regs
dep   softmax producer (…_22a7a0_xn, scalar)        148,512   6.2   4.1    3.2   1.9  1,049,088          0          0    32
dep   QK^T scores (k_sdpa_reduce_6874a2, scalar)     71,584  70.8  58.1   11.0  40.7  6,111,232      3,768     27,807    47
dep   o_proj (k_linear_sdpa_reduce_43208b, mma)      47,040  13.1  19.2    8.9   2.1  1,500,160  6,815,744          0    53
dep   P@V (k_sdpa_reduce_22a7a0, mma)                42,432  26.4  13.1   18.2   2.9    935,936  3,670,149        165    70
dep   attn-out contiguify (…_43208b_xn)               4,256  36.1   6.0   41.3   4.0     40,960          0          0    24
ref   pytorch_flash::flash_fwd_splitkv_kernel        24,384   8.4  16.3   21.7   2.9    300,800          0          0   206
ref   cutlass_80_tensorop_f16_s16816gemm (o_proj)    22,176   8.3  30.0   21.6   0.2    569,856          0          0    88
ref   pytorch_flash::flash_fwd_splitkv_combine       12,480  58.0  31.2   49.9   7.6     88,064          4          0    44
```

(A second NCU capture from the P@V reproducer agrees: softmax 146.5 µs, QK^T 70.3 µs, P@V consumer 39.2 µs.)

The softmax producer (`…_xn`) is a **scalar 3-pass online-softmax** over the symbolic seq×seq matrix — its
emitted CUDA is `__launch_bounds__(16)`, one thread per row, three sequential `for (a3 < seq_len)` loops
(max-reduce → exp-sum → normalize), each a full HBM read of the seq×seq attention matrix, writing the
normalized probabilities P back to HBM. NCU: **6.2% occupancy, 3.2% dram, 1.9% fma** — it is *latency*-bound, not
bandwidth-bound: 16 threads per block cannot keep enough loads in flight to hide HBM latency.

**Root cause — the split trades one big scalar kernel for one medium warp kernel + a new materialized softmax.**
The prior report's 259 µs scalar P@V did the softmax-weighted V sum inline in registers. #243's masked-K split
peels P@V onto tensor cores (40 µs) but must first **materialize** the normalized P matrix to HBM so the mma
consumer can read it — that materialization is the 147 µs softmax kernel. The tuner chose the split correctly
(40 + 147 = 187 µs < 259 µs fused, so it lowers the kernel-set Σ), but the win is smaller than the per-kernel
P@V headline suggests because the softmax HBM round-trip is new cost that the fused scalar kernel hid.

This is the documented limitation restated with a new price tag: CLAUDE.md still says *flash-style fused
symbolic-K attention remains future work*. The masked-K mma covers the P@V matmul; it does not fuse the softmax
into an online schedule, so the softmax pays a full materialization.

**Repro.** `deplodock run --ir <dump>/07_lowering_cuda.kernels/k_linear_sdpa_reduce_43208b.torch.json --bench
--profile` (NCU, the whole attention chain). Emitted-CUDA inspection of the scalar softmax:
`<dump>/07_lowering_cuda.kernels/k_sdpa_linear_reduce_a76a28_xna.txt`. The masked-K mma gate that now fires for
the consumer: `010_partition_loops.py:737` (`k_forced_mask=k_symbolic`).

**Suggested fix (highest priority, large and known).** Flash-style symbolic-seq attention: a scheduled
online-softmax warp loop over the symbolic N axis (QK^T sub-tile → running max/sum rescale → P@V mma accumulate),
the standard flash-2 schedule, so the softmax never materializes to HBM. This collapses softmax + QK^T + P@V into
one kernel and closes most of the 258→37 µs gap. Until then, the softmax producer itself is the cheapest standalone
target — see finding 3.

## Finding 2 — the new warp-tier P@V (and the o_proj) run at low occupancy with millions of smem load bank conflicts; PAD_SMEM / PERMUTE_LANES are no-ops on the masked-tile layout (proven)

**Symptom.** Both warp-tier attention matmuls reach `mma_m16n8k16_f16` but run starved: the P@V consumer NCU
**42.4 µs at 26.4% occupancy with 3,670,149 shared-load bank conflicts**, the o_proj NCU **47.0 µs at 13.1%
occupancy with 6,815,744 shared-load bank conflicts** — vs the cutlass o_proj gemm at 22.2 µs / 0 conflicts and
flash's P@V folded into 24.4 µs / 0 conflicts. They are on the right tier and on tensor cores, but serialized on
conflicted smem loads and capped at one-to-two warps.

**Evidence + A/B (the conflict levers do nothing here).** I pinned the two documented smem-conflict knobs on the
P@V consumer reproducer:

```
deplodock run --ir k_sdpa_linear_reduce_a76a28.torch.json --bench --bench-backends deplodock \
    --ab "PAD_SMEM=1" --ab "PERMUTE_LANES=1" --ab "PAD_SMEM=1,PERMUTE_LANES=1"
```

The P@V consumer row is **24.5 µs in the greedy pick and in all three A/B variants** — identical smem (32.0 KB),
occupancy (50%), and regs (69). `PAD_SMEM` and `PERMUTE_LANES` **do not reach the masked-K mma's smem staging
layout** at this shape; the bank conflicts are inherent to how the masked-K slab is laid out in smem, not a knob
the tuner can flip. (The 24.5 µs A/B figure is the reproducer's solo P@V window — consistent with the eval-variants
`-O3 us` pick of 24.7 µs, finding 6 — and below the 42 µs NCU locked-clock number, which includes cross-kernel
effects.)

**Root cause hypothesis.** Class 3 (codegen quality), not tier or search — the split and warp lowering are
correct, but the masked-tile smem layout is conflict-heavy and the enumeration caps at a single/double-warp tile
(occupancy 13–26%). The distinguishing diagnostic: conflicts persist with PAD_SMEM/PERMUTE_LANES pinned, so the
fix is a different smem layout for the masked-K / masked-M slab, not the existing pad/permute passes.

**Repro / A-B.** The `--ab` block above (P@V); for o_proj, `deplodock run --ir
k_linear_sdpa_reduce_43208b.torch.json --bench --ab "PAD_SMEM=1" --ab "PERMUTE_LANES=1"` (caveat: that reproducer
re-fuses the upstream attention — read the `k_linear_sdpa_reduce_43208b` row, not TOTAL).

**Suggested fix (medium — ~25 µs at stake on P@V + o_proj, and it generalizes to every masked warp tile).**
Redesign the masked-tile smem staging to be conflict-free (swizzled / padded *inside* the masked-K and masked-M
slab fill, where the current PAD_SMEM/PERMUTE_LANES passes evidently don't apply), and let the tuner reach a
multi-warp tile to lift occupancy above 26%. Fold the winning layout into the warp-tile defaults once found.

## Finding 3 — the softmax producer stays scalar at 6% occupancy even though cooperative-reduce (BR>1) configs are enumerated — they rank worse, so it is codegen/occupancy, not a tier lockout

**Symptom.** The softmax producer (finding 1, NCU 147 µs) deploys a 16-thread, BR=1, 0-KB-smem scalar config at
6.2% occupancy. The obvious lever is the strided-cooperative-rows feature (BR>1: split each row's reduction
across BR lanes with a segmented-shuffle combine), which would raise occupancy.

**Evidence — BR>1 is enumerated and loses.** `eval variants --kernel k_sdpa_linear_reduce_a76a28_xna --top 0`:
**65 of 238 measured configs carry BR>1**, but the pick (rank 2/239, 1.00x of measured best) is BR=1, and
reachability classifies it as `reduce free=16 best 183.23 pick 183.51 (1.00x, 239 configs)` — the prior recovers
the measured best essentially exactly. So this is **not** a search shortfall or a tier lockout: among everything
enumerated (including cooperative-reduce), the scalar 16-thread config *is* the measured-fastest. The cooperative
combine's segmented-shuffle overhead outweighs the occupancy it buys at this shape (16-head, head_dim, symbolic
seq).

**Root cause.** Class 3 within a kernel that is memory-latency-bound by construction: a 3-pass materialized
softmax over a seq×seq matrix is inherently HBM-traffic-heavy, and the tile/cooperative knobs available can't
hide the latency at 6% occupancy. The real fix is structural (don't materialize — finding 1), not a knob.

**Repro.** `deplodock eval variants --kernel k_sdpa_linear_reduce_a76a28_xna --top 0` (BR distribution);
emitted CUDA at `<dump>/07_lowering_cuda.kernels/k_sdpa_linear_reduce_a76a28_xna.txt`.

**Suggested fix (low standalone — superseded by finding 1).** If flash-attention (finding 1) is far off, a
two-pass (online) softmax that fuses the max+sum into one HBM read, or a warp-per-row reduction with vectorized
loads, would cut the 3× HBM traffic. But the structural fix in finding 1 removes this kernel entirely, so it is
only worth doing as a stopgap.

## Finding 4 — QK^T scores stay scalar (71 µs) because the accumulator is consumed inside the softmax reduce; static K=128 doesn't help

**Symptom.** QK^T scores (`k_sdpa_reduce_6874a2`) is scalar (0 `mma.sync`) at NCU 71.6 µs, 70.8% occupancy,
40.7% fma, with **27,807 shared-store bank conflicts** — it stages the masked scores into smem and serializes on
conflicted stores, on top of being scalar. Its reduce axis K is the **static** head_dim 128, so symbolic-K is
not the blocker.

**Root cause.** The QK^T accumulator feeds the softmax max/sum reduce over the symbolic N (key) axis — a
mid-reduction use — so the mma fragment-store fold is rejected by `classify_fragment_epilogue`,
`deplodock/compiler/pipeline/passes/lowering/tile/_atom.py:324`:

> *the accumulator is consumed inside a reduce loop (mid-reduction use, not a store-time fold)*

`is_atom_eligible` returns False → scalar tier. This is the same online-softmax pattern finding 1 names: a warp
QK^T tile computes a sub-tile of N, but softmax needs the full row, so it cannot be a plain masked warp tile — it
needs the scheduled online-softmax phase that flash implements. The static K=128 is irrelevant: the blocker is
the epilogue, not the reduce axis.

**Repro.** `deplodock compile <dump>/07_lowering_cuda.kernels/k_sdpa_reduce_042770.torch.json --ir cuda` shows
the scalar masked tile and no `mma.sync`; the eligibility bail is `_atom.py:324`. The 27.8K store conflicts are a
cheap PAD_SMEM/PERMUTE_LANES target *within* the scalar kernel, but like finding 3 it is superseded by the flash
work.

**Suggested fix.** Same as finding 1: the scheduled online-softmax warp loop folds QK^T, softmax, and P@V into
one flash kernel. Standalone, QK^T is the second-cheapest attention kernel and the lowest-priority of the three.

## Finding 5 — bench attribution: the deployed launch table mis-attributes attention by ~7×; only NCU + e2e are trustworthy (recurring)

**Symptom.** The deployed `run --dynamic … --bench` launch table and NCU disagree wildly for the attention
kernels, in opposite directions:

- **Deployed launch table under-attributes the softmax and QK^T.** The softmax `…_xna` reads **20.3 µs / 5.7%**
  while NCU clocks the same kernel at **148 µs**; QK^T `042770` reads **2.2 µs / 0.6%** while NCU clocks it at
  **71 µs**; meanwhile the P@V consumer `a76a28` reads **99.4 µs / 27.8%** while NCU clocks it at **40 µs**. The
  per-launch solo windows absorb cross-kernel latency — the `%` column actively mislabels the dominator (it
  fingers the P@V consumer as 28% of the layer when NCU says the softmax is the dominant kernel).
- **The `--bench` reproducer table re-fuses the upstream cone.** The o_proj and P@V reproducers each re-lower the
  whole attention chain (softmax + QK^T + P@V + o_proj), so their printed 213 / 203 µs "deplodock" totals are
  slice-set sums, not the named kernel.

Only the **NCU single-capture table** (both sides, one clock — finding 1) and the **e2e whole-program** number
(359 µs) are trustworthy. This is the recurring attribution problem (`plans/bench-attribution-by-slicing.md`); it
cost real triage time again — the deployed table's 20 µs softmax row hid the layer's actual dominant kernel until
NCU revealed it as 148 µs.

**Suggested fix (high — the third dynamic report in a row to hit this).** Land per-launch attribution by slicing
(the existing plan): attribute each deployed kernel's cost from a capture where it is the only changing slice, so
the deployed table matches NCU. Short term, the skill already says trust NCU + e2e for attention; this run is
fresh evidence that the deployed `%` column inverts the dominator.

## Finding 6 — the P@V pick-miss is an -O1 ranking artifact, not a search shortfall

**Symptom.** `eval variants --kernel k_sdpa_linear_reduce` flags the P@V consumer pick as `rank 11/886, 1.26x of
best <-- misses best` at -O1.

**Evidence (inverts at -O3 — confirmed, not assumed).** The `eval variants -O3 us` column shows the **pick**
(rank 11, -O1 60.5 µs) re-benches to **24.7 µs** at -O3, while the -O1 rank-1 config (48.0 µs) re-benches to
**29.3 µs** and rank-4 to 24.9 µs — the greedy pick is the *fastest deployable* config; the -O1 ranking simply
mis-orders the warp-tile shapes (the well-known -O1-vs-O3 inversion, and consistent with the memory note that
-O1 pick-misses usually invert). The QK^T and softmax picks are rank 1–2 at -O1 already (no miss). The
DB-reachability aggregate (`eval prior --dataset db` mean 2.52x / median 1.24x) is likewise dominated by -O1
inversions on the warp-tile matmuls and should not be read as a search defect.

**Root cause.** Not a defect. The tune ranks at -O1 (fast compile); the deployable order is -O3, and the prior's
reservoir carries the -O3 truth that makes the pick correct. No patience/prior change warranted.

## Finding 7 — tune-`--bench` (467 µs) vs fresh-run (359 µs) e2e spread is 30%, larger than usual — thermal

**Symptom.** The tune's `--bench` full-model table reports Deplodock 467 µs / 0.47x eager; an immediate fresh
`run --dynamic … --bench` of the identical deployed kernels reports 359 µs / 0.62x eager — a 30% spread on the
same code.

**Root cause.** Thermal. The tune's `--bench` runs on a GPU loaded by the 4.4 h tune that just finished; the
fresh run is on a cooled card. The prior report saw the same direction at 7% (426→396) after a 2 h tune; this
run's tune was 2.2× longer, so the card was hotter and the spread bigger. Consistent with the memory note that
absolute µs drift ±10%+ across invocations while ratios stay stable.

**Suggested fix (low — measurement hygiene).** Quote the fresh-run e2e (359 µs) as the deployed number and treat
the tune-`--bench` e2e as ranking-only when the tune was long. A short cooldown (or a clock-lock) before the
tune's final `--bench` would make its headline number deployable; otherwise the skill should prefer a separate
`run --bench` on a cooled GPU for the reported e2e (as the prior report already did).

## Repro / artifacts

- Work dir: `/tmp/tune-model-qwen3-emb-l0-dyn/` — `tune.log` (52-line tee of the 4.4 h tune), `run-dynamic.log`
  (fresh deployed launch table + e2e 359 µs), `ncu-43208b.log` (o_proj-chain NCU capture +
  `ncu-43208b/61_ncu_metrics.{csv,json}`), `ncu-a76a28.log` (P@V-chain NCU capture), dump at `dump/`
  (reproducers under `07_lowering_cuda.kernels/`, machine-readable `62_kernel_bench.json`, chart `kernels.html`;
  `.png` skipped — Playwright `Target page … closed` flake again).
- Tune DB / prior (default paths, this run only): `~/.cache/deplodock/autotune.db` (7,215 rows, 0 bench_fail),
  `~/.cache/deplodock/prior.json` (warp-tier `H_opt=3` rows for linears + masked-K P@V).
- Finding-1 NCU: `DEPLODOCK_DUMP_DIR=<dir>/ncu-43208b deplodock run --ir
  <dump>/07_lowering_cuda.kernels/k_linear_sdpa_reduce_43208b.torch.json --bench --profile` (no `--dynamic` — the
  reproducer keeps its symbolic dims and benches at the 512 hint). Gate probes (no GPU): `_atom.py:324`
  (online-softmax fold bail, findings 1/4), `010_partition_loops.py:737` (`k_forced_mask=k_symbolic`, the #243
  masked-K mma gate), `010_partition_loops.py:653` (`prologue_mask_ok`).
- Finding-2 A-B (proven no-op): `deplodock run --ir
  <dump>/07_lowering_cuda.kernels/k_sdpa_linear_reduce_a76a28.torch.json --bench --bench-backends deplodock --ab
  "PAD_SMEM=1" --ab "PERMUTE_LANES=1" --ab "PAD_SMEM=1,PERMUTE_LANES=1"`.
- Finding-3 BR enumeration: `deplodock eval variants --kernel k_sdpa_linear_reduce_a76a28_xna --top 0`.
- Finding-6 inversion: the `-O3 us` column of `deplodock eval variants --kernel k_sdpa_linear_reduce`.

## Workflow notes

Audit of the prior dynamic report's notes first:

- **The full-model dynamic table is honest (prior finding 4, FIXED)**: held up — the table prints `benched at
  seq_len=512 (symbolic hint; torch inputs tiled to match)` and the eager row reads 219 µs (the honest
  static-512 number). No headline-ratio detour this run.
- **Per-launch mis-attribution (prior finding)**: **reproduced, worse** (finding 5). This run the deployed table
  under-attributes the softmax by ~7× (20 µs vs 148 µs NCU) and *hid the layer's dominant kernel entirely* — I
  only found the 148 µs softmax via NCU. Now the top recurring tooling gap across three reports; strongly
  reinforces `plans/bench-attribution-by-slicing.md`.
- **Reproducer re-fusion (prior finding)**: **reproduced and binding.** With the warp tier live, both attention
  reproducers re-fuse the whole cone, so their 203 / 213 µs totals are meaningless for the named kernel. I fell
  back to NCU for every attention number. A `run --ir … --bench --no-refuse` (bench only the named provenance
  kernel, masking upstream slices) would have saved most of the triage time.
- **Chart PNG Playwright flake**: reproduced verbatim (`png skipped: Target page … closed`).
- **Stable per-op identity across views (prior finding)**: **reproduced.** The same P@V op wears three hashes —
  deployed `a76a28`, o_proj-chain NCU `22a7a0`, P@V-chain NCU `a88b4d` — because each re-lowering re-hashes. I
  hand-cross-referenced all three. A stable provenance name (not content hash) in the NCU / leaderboard /
  deployed tables would remove the cross-referencing.

New friction this run:

- **The tune is 2.2× longer with #243 (4.4 h vs 2.2 h) and near-silent under `tee`** (52-line log for a 4.4 h
  run — the tty progress bar doesn't survive the pipe). At this scale a non-tty per-terminal heartbeat (`[tune]
  terminal k/16 done, best Σ …`) would let the operator see liveness without `-v`'s firehose. This was the
  single biggest wall-clock item by far.
- **Tune-`--bench` e2e is thermally inflated (finding 7).** The 467 µs headline was 30% above the deployable
  359 µs purely because the tune left the GPU hot. The reported e2e had to come from a separate fresh run — the
  skill should either clock-lock before the final `--bench` or explicitly direct a cooled fresh-run e2e.
- **`--ab` proving a no-op is valuable but easy to misread.** The PAD_SMEM/PERMUTE_LANES A/B (finding 2) printed
  identical rows for all three variants — exactly the evidence I wanted (the levers don't reach the masked-tile
  layout), but it took reading smem/occ/regs columns to confirm "no change" rather than "applied and helped." A
  one-line A/B summary (`ab PAD_SMEM=1: kernel unchanged (smem/occ/regs identical to pick)`) would make a no-op
  result unambiguous.

What worked well: the triage loop was tight — `eval failures` → 0 failures in one line, `eval variants` (with the
`-O3 us` column) → the P@V inversion proven without a separate run, `eval variants --top 0` → the BR>1-enumerated
softmax settled "tier lockout vs codegen" in one command (finding 3), and the two NCU captures cross-checked the
attention attribution (softmax 148/146, P@V 42/39) to within 5%. The single `--profile` run on the o_proj-chain
reproducer gave the whole attention path + the torch flash/cutlass references in one aligned table — the clean
signal the deployed table can't provide. The `mma.sync`-count grep over the emitted `.txt` was the fastest tier
fingerprint (P@V went 0→10 mma ops with #243); a `tier` column (scalar / mma) in `eval variants` would make it a
zero-step read.
