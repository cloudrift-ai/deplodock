# Qwen3-Embedding-0.6B layer 0 — DYNAMIC-shape tune findings (2026-06-14)

Status: **post-M9 re-tune of the deployable dynamic configuration (`--dynamic seq_len@x:1`). The symbolic-axis
masked warp tier and the demoted-matmul split — both `future work` in the prior dynamic report — have landed in
`main`, and the picture is transformed: every static-K linear now reaches the `mma.sync` warp tier, the structural
split offers (8 fused terminals vs 1 before), and the deployed layer runs 0.55x eager / 0.37x torch.compile at
seq_len 512 — up from the prior report's 0.10x eager.** The remaining gap is now concentrated almost entirely in the
**attention** path: QK^T+softmax and P@V stay scalar-tier by design (online-softmax fragment-store + symbolic-K
reduce), losing ~9x to PyTorch's fused flash-attention. That single known limitation is ~80% of what is left.

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  /tmp/tune-model-qwen3-emb-l0-dyn/dump`
- Hardware: RTX 5090 (sm_120), ncu 2025.3.1 (perf counters permitted). `main` @ `c3304773`.
- Run stats: tune wall **7,162 s** (~2.0 h — the warp tier + structural splits are back, so the cost is back near
  the static-tune scale, ~10x the prior all-scalar dynamic tune's 730 s), **8 fused terminals** (vs 1), **4,170 ok /
  1 `bench_fail`** `perf` rows, all CUDA-graph captured; prior: 7,866 benches (158 warmup / 7,708 post), post-warmup
  Spearman **+1.00**, 95% of post benches ≥2x the running best.
- DB / prior: **default paths, cleaned** — `~/.cache/deplodock/autotune.db`, `~/.cache/deplodock/prior.json`. Unlike
  the prior report, this prior now contains warp-tier `H_opt=3` rows for every static-K linear, so it is a usable
  greedy prior for dynamic serving (its picks are rank-1-deployable on the kernels that matter — see finding 5).
- **Number families**: bench tables / reproducer runs / `--ab` rows below are **-O3** (deployable, CUDA-graph
  captured); tune-DB latencies quoted for ranking context are `-Xcicc -O1` (ranking signal only). NCU durations run
  at locked base clock — **compare ratios, not absolutes**, and NCU is the only clean per-kernel signal (finding 4).
- **Dynamic measurement semantics**: the trace is symbolic in `seq_len` (`Dim` hint = `DEFAULT_SEQ_HINT` 512 — the
  `--seq-len` flag does NOT set it). All per-op tune benches, reproducer benches and the full-model table run at the
  512 hint; the full-model table now tiles the torch closures' inputs to the hint and prints `benched at
  seq_len=512 (symbolic hint; torch inputs tiled to match)`, so its eager / tcompile / deplodock rows are one shape
  and directly comparable (the prior report's finding 4 — the shape-mismatched headline — is fixed).

## Bench results (-O3, CUDA-graph captured)

Full-model table as printed by the tune's `--bench` (now a valid like-for-like comparison at the 512 hint):

```
Backend        Latency (us)  vs Eager
-------------------------------------
Eager PyTorch           219     1.00x
torch.compile           147     1.49x
Deplodock               426     0.51x      (396 us in a fresh `run --dynamic … --bench`; e2e whole-program)
benched at seq_len=512 (symbolic hint; torch inputs tiled to match)
```

So the deployed dynamic layer is **0.51–0.55x eager / 0.35–0.37x torch.compile** at seq 512 — a ~5x improvement on
the prior report's 0.10x eager, entirely from the warp tier + split now firing for the static-K linears.

Per-kernel reproducer bench at the 512 hint, as printed by `--bench` (sorted by deplodock µs; layer-op labels from
each kernel's `.torch.json` provenance). **Read the attention rows (top three) with the finding-4 caveat: those
reproducers re-fuse the upstream attention cone, so their deplodock µs is a slice-set total, not the deployed kernel
alone.** The clean per-kernel attribution is the NCU table under finding 1.

| Kernel                        | Layer op                                                | eager | tcompile | deplodock | vs eager |
|-------------------------------|---------------------------------------------------------|------:|---------:|----------:|---------:|
| `k_sdpa_linear_reduce_a76a28` | SDPA P@V (attn-probs @ V, + in-kernel V producer)       |    34 |       34 |       240 |    0.14x |
| `k_linear_sdpa_reduce_43208b` | attn-out reshape + o_proj (linear_3) + residual         |    39 |       37 |       181 |    0.22x |
| `k_sdpa_reduce_77b0f0`        | RoPE(q,k) + QK^T + softmax                              |   149 |       25 |       155 |    0.96x |
| `k_linear_mean_reduce_05d34c` | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up  |   119 |       57 |        60 |    2.00x |
| `k_linear_0837e7`             | MLP down (linear_6) + residual                          |    27 |       27 |        44 |    0.61x |
| `k_mean_linear_reduce_67f4cf` | q_norm RMSNorm + rotated-q producer (xna)               |   105 |       20 |        27 |    3.93x |
| `k_linear_reduce_716194`      | v_proj matmul                                           |    17 |       16 |        27 |    0.63x |
| `k_mean_linear_reduce_5ceb87` | k_norm RMSNorm + rotated-k producer (xnb)               |    76 |       14 |        18 |    4.11x |
| `k_linear_reduce_f94dd0`      | q_proj matmul                                           |    10 |       10 |        17 |    0.61x |
| `k_mean_d65726`               | input RMSNorm                                           |    66 |        4 |         2 |   34.16x |

The two pure RMSNorm-bearing reductions and the gated-MLP **win eager outright** (2–34x) and the warp-tier linears
are at parity-to-2x of eager. Everything still meaningfully behind torch.compile is in the **attention** path
(`a76a28`, `43208b`, `77b0f0`) — which by the clean NCU attribution (finding 1/4) is ~320 µs of locked-clock GPU
time vs flash-attention's 36 µs.

## Finding 1 — symbolic-seq attention stays scalar; QK^T+softmax and P@V lose ~9x to fused flash-attention (the dominant remaining gap)

**Symptom.** The two attention matmuls are the only kernels that never reach the warp tier and the only ones still
far behind torch.compile. From the clean NCU capture (one `ncu` run, deplodock + torch reference kernels side by
side, locked base clock — `run --ir k_linear_sdpa_reduce_43208b.torch.json --bench --profile`, which re-lowers the
whole attention→o_proj chain):

```
side  kernel                                          dur (ns)  occ%   sm%  dram%  fma%   lsu.inst  ld.cnflct  regs
dep   P@V  (k_sdpa_reduce_22a7a0, scalar)              258,784  24.5  20.8    2.1  14.1  5,251,072          0    40
dep   QK^T+softmax (k_sdpa_reduce_6874a2, scalar)       60,608  55.7  58.3    7.3  45.4  5,529,600  4,217,783    64
dep   o_proj  (k_linear_sdpa_reduce_43208b, mma)        49,696  12.2  22.3    8.4   2.4  1,762,304  6,815,748    54
dep   o_proj xn producer (…_43208b_xn)                   4,000  32.3   6.5   30.1   3.7     40,960          0    24
ref   pytorch_flash::flash_fwd_splitkv_kernel            24,992   8.3  16.2   20.7   3.0    300,800          0   206
ref   pytorch_flash::flash_fwd_splitkv_combine_kernel    11,328  64.7  34.0   42.6   8.3     88,064          0    44
ref   cutlass_80_tensorop_f16_s16816gemm (o_proj)        22,368   8.3  29.8   16.0   0.2    569,856          0    88
```

Deplodock spends **QK^T 60.6 µs + P@V 258.8 µs = 319 µs** of locked-clock time on the same attention torch fuses
into **flash 25.0 µs + combine 11.3 µs = 36.3 µs** — a ~9x gap, and ~80% of everything left. Both deplodock
attention kernels are saturated on the **scalar FMA pipe** (P@V 14% fma at 24% occ; QK^T 45% fma) with no tensor-core
use; P@V runs `BM=16, BN=8, FM=1, FN=16` — a masked thread tile, one M-element per thread, 0 KB smem.

**Root cause — two distinct by-design gates, both correct today:**

1. **P@V is symbolic-K.** Its reduce axis is the key/value sequence length (`seq_len`, symbolic). The warp tier
   requires a static reduce — `010_partition_loops.py:685` gates the entire MMA enumeration on `not k_symbolic`. So
   P@V gets the symbolic-K masked **thread** tile at `FM=FN=1` (`prologue_mask_ok = (not prologue) or k_symbolic`,
   `010_partition_loops.py:645`), which is the documented deployment for this class.
2. **QK^T+softmax has a static K (head_dim 128) but an online-softmax epilogue.** The QK^T accumulator is consumed
   *inside* the softmax reduce over the symbolic N (key) axis, so the mma fragment-store fold is rejected by the
   first negative rule in `classify_fragment_epilogue` —
   `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py:254`:

   > *the accumulator is consumed inside a reduce loop (a mid-reduction use, e.g. online softmax rescale — needs a
   > scheduled phase, not a store fold)*

   `is_atom_eligible` returns False → scalar tier. This is exactly the flash-attention pattern: a warp QK^T tile
   computes a sub-tile of N, but softmax needs the full row, so it cannot be a plain masked warp tile — it needs the
   scheduled online-softmax phase that flash implements.

Both are the documented limitation: CLAUDE.md — *"Symbolic-K warp tier (flash-style attention) is future work."*
This report's contribution is the price tag: ~283 µs of the deployed layer (the e2e delta between 0.37x and a
hypothetical flash-parity attention) and the entire distance to torch.compile.

**Repro.** `deplodock run --ir <dump>/04_lowering_cuda.kernels/k_sdpa_linear_reduce_a76a28.torch.json --bench
--profile` (P@V) and `… k_sdpa_reduce_77b0f0.torch.json --bench --profile` (QK^T). Compile-only gate probe (no GPU):
`deplodock compile <dump>/…/k_sdpa_reduce_77b0f0.torch.json --ir cuda` shows the scalar masked tile and no
`mma.sync`; the eligibility bail is `_atom.py:254`, the symbolic-K gate is `010_partition_loops.py:685`.

**Suggested fix (highest priority, large and known).** Flash-style symbolic-seq attention: a scheduled online-softmax
warp loop over the symbolic N axis (QK^T sub-tile → running max/sum rescale → P@V accumulate), the standard flash-2
schedule, behind the `_atom.py:254` gate. This is the single change that would close most of the remaining gap to
torch.compile on every transformer layer in the dynamic configuration.

## Finding 2 — the o_proj split reaches the warp tier but runs at 12% occupancy with 6.8M smem load bank conflicts (~2.4x cuBLAS)

**Symptom.** The o_proj (`k_linear_sdpa_reduce_43208b`) **does** reach the `mma_m16n8k16_f16` warp tier — the
demoted-matmul split that was finding 2's `future work` in the prior report now offers (the dump has a populated
`05_lowering_tile__005_split_demoted.rules.txt`; the attn-out collapse materializes into an `…_xn` contiguifying
producer at 4 µs + a clean warp-tier consumer). But the consumer is **NCU 49.7 µs vs cuBLAS's 22.4 µs — 2.2x** — and
the counters show why: **12.2% occupancy** (one warp, `regs=54`, 16 KB smem) and **6.8M shared-load bank
conflicts**. It is on the right tier and on tensor cores, but starved on occupancy and serialized on smem loads.

**Evidence.** NCU rows above: `k_linear_sdpa_reduce_43208b` — occ 12.2%, `ld.cnflct` 6,815,748, sm 22.3%, vs the
cutlass gemm at occ 8.3% / 0 conflicts / sm 29.8%. cuBLAS runs lower occupancy too but with zero bank conflicts and
2x the SM throughput; the deplodock kernel's conflicts are the distinguishing defect. The deployed pick is `BM·BN`
tile with `WM/WN` warp shape, `STAGE=11`, `PAD_SMEM=False`, `PERMUTE_LANES=False`.

**Root cause hypothesis.** Class 3 (codegen quality), not a tier or search problem — the split and warp lowering are
correct, the masked-M warp tile's smem staging layout is conflict-heavy at this shape and the single-warp tile caps
occupancy. The distinguishing diagnostic between "smem layout" and "occupancy" is the conflict count vs the warp
count: 6.8M conflicts at one warp says **both** levers are live.

**Repro / A-B.** `deplodock run --ir <dump>/…/k_linear_sdpa_reduce_43208b.torch.json --bench --ab
"PAD_SMEM=1" --ab "PERMUTE_LANES=1"` (smem-conflict levers), and pin a larger warp tile to lift occupancy. Caveat:
this reproducer **re-fuses the upstream attention** (finding 4), so its headline total is dominated by P@V — read
the `k_linear_sdpa_reduce_43208b` row, not TOTAL.

**Suggested fix (medium — ~25 µs at stake, and it generalizes to every o_proj/down-proj).** Try `PAD_SMEM` /
`PERMUTE_LANES` in the warp-tile smem layout for the masked-M consumer, and let the tuner reach a multi-warp tile
(the enumeration currently picks a single-warp config here). If the conflicts drop and occupancy rises, fold the
winning knobs into the analytic/learned prior's warp-tile defaults.

## Finding 3 — the QK^T scalar kernel has 4.2M smem load bank conflicts (a lever inside the by-design scalar tier)

**Symptom.** Even granting finding 1 (QK^T is scalar by design until flash lands), the scalar kernel
`k_sdpa_reduce_6874a2` shows **4,217,783 shared-load bank conflicts** at 42.9 KB smem (NCU row above) — it stages
operands into smem and then serializes on conflicted loads, on top of being scalar.

**Root cause.** Class 3 within a class-2-by-design kernel: the smem layout of the scalar masked QK^T tile is
conflict-heavy. Independent of the flash work, reducing these conflicts is a cheap partial win while the kernel
remains scalar.

**Repro / A-B.** `deplodock run --ir <dump>/…/k_sdpa_reduce_77b0f0.torch.json --bench --ab "PAD_SMEM=1" --ab
"PERMUTE_LANES=1"`.

**Suggested fix (low — superseded by finding 1, but cheap).** PAD_SMEM / PERMUTE_LANES on the scalar QK^T smem
stage. Only worth doing if flash-attention (finding 1) is far off; otherwise finding 1 replaces this kernel entirely.

## Finding 4 — bench attribution: neither emitted table gives clean deployed per-kernel numbers; only NCU + e2e are trustworthy

**Symptom.** The two per-kernel views disagree wildly for the attention kernels, and both are wrong in opposite
directions:

- **The `--bench` reproducer table re-fuses the upstream cone.** The o_proj reproducer (`43208b`) re-lowers to four
  kernels — P@V (109 µs) + QK^T (36 µs) + xn (1.4 µs) + the actual o_proj consumer (**30.9 µs**) — so its printed
  "181 µs deplodock" is a slice-set total, and the o_proj itself is 31 µs. The QK^T reproducer (`77b0f0`) likewise
  re-fuses to 154 µs of which the QK^T core (`042770`) is 119 µs.
- **The deployed launch table mis-attributes per-launch.** In `run --dynamic … --bench`, the o_proj's `…_xn`
  producer reads **160.1 µs / 40.5%** while NCU clocks it at 4 µs, and P@V (`a76a28`) reads **2.1 µs** while NCU
  clocks the same work at 259 µs. The per-launch solo windows absorb cross-kernel latency — the `%` column actively
  mislabels the dominator (it fingered the `_xn` copy as 40% of the layer).

Only the **NCU single-capture table** (both sides, one clock — finding 1) and the **e2e whole-program** number (426
/ 396 µs) are trustworthy. This is the recurring attribution problem (`plans/bench-attribution-by-slicing.md`); it
cost real triage time again here (the deployed table's 160 µs `_xn` row sent me to NCU to discover it is a 4 µs
copy).

**Suggested fix (high — it is the third dynamic report in a row to hit this).** Land per-launch attribution by
slicing (the existing plan): attribute each deployed kernel's cost from a capture where it is the only changing
slice, so the deployed table matches NCU. Short term, the skill should say: *for the attention kernels, trust NCU
and the e2e number, not the per-launch `%` column or the re-fusing reproducer total.*

## Finding 5 — the pick-misses are -O1 ranking artifacts, not search shortfalls

**Symptom.** `eval variants` flags two dominators as `<-- misses best`: P@V (`a76a28`) pick is rank 25/543 (1.33x of
best at -O1) and QK^T (`77b0f0`) pick is rank 16/39 (1.41x). `eval prior --dataset db` reachability is mean 1.17x /
median 1.06x / worst 2.17x — all -O1.

**Evidence (both invert / are unlowerable at -O3 — confirmed, not assumed):**

- **P@V inverts at -O3.** The `eval variants -O3 us` column shows the **pick** re-benches to **181.3 µs** while the
  -O1 rank-1 config re-benches to **256.4 µs** — the greedy pick is the *fastest deployable* config; the -O1 ranking
  simply mis-orders them (the well-known -O1-vs-O3 inversion).
- **QK^T's -O1 rank-1 is unlowerable.** Pinning it (`--ab "BM=8,BN=16,…,RING=3,…"`) fails to compile:
  `DEPLODOCK_RING=3 pinned but cannot fire: BUFFER_COUNT=3 not promotable (smem cap 101376 B …)`. The "1.41x of
  best" baseline is a config that cannot deploy; the greedy `RING=2` pick is the best that lowers.

**Root cause.** Not a defect. The tune ranks at -O1 (fast compile); the deployable order is -O3, and the prior's
reservoir already carries the -O3 truth that makes the pick correct. No patience/prior change is warranted.

## Finding 6 — one bench failure: a compile-budget timeout on a scalar q_norm config

**Symptom.** `eval failures`: exactly **1 `bench_fail` row** (of 4,171) — `k_mean_linear_reduce_67f4cf` (q_norm +
rotated-q producer), error `compile stage exceeded 2.0s budget (3.21s) — variant marked bench_fail`, shared knobs
`BR=4, FM=8, SPLIT_CONE=True, MMA=0` (a scalar strided-cooperative config). The kernel's final deployed pick is fine
(`BR=32`, NCU-healthy at 100% occ).

**Root cause.** A single scalar config whose unrolled body tripped the 2 s nvcc compile budget by 1.2 s — one wasted
search slot, no effect on the deployed pick. Not worth action at 1/4171; noted for completeness.

## Repro / artifacts

- Work dir: `/tmp/tune-model-qwen3-emb-l0-dyn/` — `tune.log`, `run-dynamic.log` (deployed launch table + e2e),
  `ncu-43208b.log` (the clean per-kernel NCU capture + `ncu-43208b/61_ncu_metrics.{csv,json}`), dump at `dump/`
  (reproducers under `04_lowering_cuda.kernels/`, machine-readable `62_kernel_bench.json`, chart `kernels.html`;
  `.png` skipped — Playwright flake again).
- Tune DB / prior (default paths, this run only): `~/.cache/deplodock/autotune.db` (4,171 rows),
  `~/.cache/deplodock/prior.json` (warp-tier `H_opt=3` rows present).
- Finding-1 NCU: `DEPLODOCK_DUMP_DIR=<dir> deplodock run --ir
  <dump>/04_lowering_cuda.kernels/k_linear_sdpa_reduce_43208b.torch.json --bench --profile` (no `--dynamic` — the
  reproducer keeps its symbolic dims and benches at the 512 hint). Gate probes (no GPU): `_atom.py:254` (online-
  softmax fold bail), `010_partition_loops.py:685` (symbolic-K MMA gate), `010_partition_loops.py:645`
  (`prologue_mask_ok`).
- Finding-2/3 A-B: `deplodock run --ir <kname>.torch.json --bench --ab "PAD_SMEM=1" --ab "PERMUTE_LANES=1"`.
- Finding-5 inversion: the `-O3 us` column of `deplodock eval variants --kernel k_sdpa_linear_reduce`, and the
  `--ab` RING=3 failure on the `k_sdpa_reduce_77b0f0` reproducer.

## Workflow notes

Audit of the prior dynamic report's notes first:

- **The full-model dynamic table is wrong (prior finding 4)**: **FIXED and held up.** The table now prints `benched
  at seq_len=512 (symbolic hint; torch inputs tiled to match)` and the eager row reads 219 µs (the honest static-512
  number), not the seq-32 96 µs. This removed the entire first-half-hour detour the prior report described — the
  headline ratio was trustworthy on first read this time.
- **The skill's measurement-semantics note**: the `--seq-len`-does-not-set-the-hint clarification is now correct in
  the skill text; no friction.
- **Per-launch mis-attribution (prior finding)**: **reproduced, worse than ever** (finding 4): the deployed table's
  `_xn` row at 160 µs / 40.5% vs its 4 µs NCU truth, and P@V at 2.1 µs vs 259 µs. At this magnitude the `%` column
  is not just noisy but inverts the dominator ranking. Strongly reinforces `plans/bench-attribution-by-slicing.md` —
  this is now the top recurring tooling gap across three reports.
- **Reproducer re-fusion (prior finding)**: **reproduced and now the binding measurement problem.** With the warp
  tier live, the o_proj reproducer re-fuses the whole attention cone, so its 181 µs total is meaningless for the
  31 µs o_proj. I had to fall back to NCU for every attention/o_proj number. A `run --ir … --bench --no-refuse` (or
  a flag that benches only the named provenance kernel, masking upstream slices) would have saved most of the
  triage time this report.
- **Chart PNG Playwright flake**: reproduced verbatim (`png skipped: Target page … closed`).

New friction this run:

- **The 2.0 h tune is mostly silent under `tee`** (53-line log for a 2-hour run — the tty progress bar doesn't
  survive the pipe). At static-tune scale (multi-hour) this matters: a non-tty per-terminal heartbeat line (`[tune]
  terminal k/8 done, best Σ …`) would let the operator see liveness without `-v`'s firehose.
- **No single command gives clean deployed per-kernel cost.** Assembling the real attention attribution took a
  `--profile` NCU run + manual cross-reference of three kernel-name sets (deployed `042770`/`a76a28`, reproducer
  `22a7a0`/`6874a2`, leaderboard hashes) — the hashes move per re-lowering, so the same op wears three names across
  the views. A stable per-op identity in the NCU/leaderboard/deployed tables (provenance name, not content hash)
  would remove the hand cross-referencing.
- **What worked well**: the triage loop was tight — `eval failures` → 1 trivial timeout, `eval variants` (with the
  `-O3 us` column) → the P@V inversion proven without a separate run, `eval prior --dataset db` → reachability clean,
  and the `005_split_demoted.rules.txt` *presence* (vs its absence last time) confirmed the split now offers at a
  glance. The `--ab` row catching the unlowerable RING=3 rank-1 config (finding 5) settled "search shortfall vs
  artifact" in one command. The `mma.sync`-count grep over the emitted `.txt` was the fastest tier fingerprint — a
  `tier` column in `eval variants` (scalar / mma) would make it a zero-step read.
