# Qwen3-Embedding-0.6B layer 0 — tune findings (2026-06-10)

Status: **clean tune completes; deplodock loses end-to-end (0.70x eager, 0.33x torch.compile), two scalar-locked
kernels carry 63% of the total.** The first tune attempt crashed ~35 min in on a compiler contract bug (finding 1) —
fixed on this branch (`fix/atom-cell-ab-classify`, commit `41c871a6`), then re-run clean. Findings 2 and 3 (the two
scalar-locked dominators: gate+up 51 µs → 12.7 µs, o_proj 25 µs → 6.2 µs) are also fixed on this branch — see their
sections. With both landed, a greedy `run --layer 0 --bench` under the tune-trained prior deploys the splits
everywhere and the full layer goes **138 µs (0.70x eager) → 48 µs (2.04x eager)** — parity with torch.compile's
45 µs, from 3x behind.

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --clean --bench --dump-dir <dir>/dump`
- Hardware: RTX 5090 (sm_120), ncu 2025.3.1 (perf counters permitted)
- Run stats: tune wall 2379 s (~40 min), 3353 benched variants, 2059 ok `perf` rows, **0 `bench_fail`**, 4 structural
  (fused) terminals; prior post-warmup Spearman calibration +0.99
- **Number families**: every latency below says which family it is. The `--bench` tables are `-Xcicc -O3`
  (deployable, CUDA-graph captured). Tune-DB latencies quoted for ranking context are `-Xcicc -O1` — ranking signal
  only; on these kernels the O1/O3 gap runs 10x (`k_linear_mean_reduce`) to 137x (`k_linear_reshape_…_38c877`), and
  finding 2 shows an outright **rank inversion** between the two families.

## Bench results (-O3, CUDA-graph captured)

Full model (layer 0, seq_len 32):

```
Backend        Latency (us)  vs Eager
-------------------------------------
Eager PyTorch            96     1.00x
torch.compile            45     2.14x
Deplodock               138     0.70x
```

Per-kernel (sorted by deplodock µs; layer-op labels read off each kernel's `.torch.json` provenance reproducer):

| Kernel                                        | Layer op                                              | eager | tcompile | deplodock | vs eager |
|-----------------------------------------------|-------------------------------------------------------|------:|---------:|----------:|---------:|
| `k_linear_mean_reduce_23ab9c`                 | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up |   70 |       14 |        51 |    1.37x |
| `k_linear_reshape_transpose_sdpa_reduce_38c877` | attn-out reshape + o_proj (linear_3) + residual      |    16 |       14 |        25 |    0.65x |
| `k_sdpa_transpose_reshape_linear_reduce_0a1109` | SDPA attn@V (+ `_xn` V-slab producer)                |    12 |       12 |        10 |    1.19x |
| `k_linear_reduce_86a525`                      | MLP down (linear_6) + residual                         |     8 |        8 |         9 |    0.91x |
| `k_sdpa_transpose_unsqueeze_cat_slice_reduce_39b7dc` | RoPE (q,k) + QK^T + softmax (+ `_xna`/`_xnb`)   |    84 |        8 |         6 |   13.71x |
| `k_reshape_linear_mean_reduce_359b55`         | q_norm RMSNorm (+ q_proj slice)                        |    60 |        8 |         5 |   10.84x |
| `k_reshape_linear_mean_reduce_61f49f`         | k_norm RMSNorm (+ k_proj slice)                        |    61 |        8 |         5 |   12.43x |
| `k_linear_reduce_735349`                      | q_proj matmul                                          |     6 |        6 |         5 |    1.31x |
| `k_linear_reduce_bcc194`                      | v_proj matmul                                          |     6 |        6 |         4 |    1.54x |
| `k_mean_b6c7b1`                               | input RMSNorm                                          |    55 |        4 |         1 |   40.20x |

Two kernels dominate: `k_linear_mean_reduce_23ab9c` (51 µs, 42% of the 121 µs per-kernel Σ) and
`k_linear_reshape_…_38c877` (25 µs, 21%) — together 63%, and both are **scalar-tier locked** (findings 2–3). The
reduction/pointwise kernels (RMSNorms, RoPE+softmax) all beat eager 10–40x and edge out torch.compile; the plain
projections and the attn@V matmul (post-fix, finding 1) deploy on tensor cores and sit at ~parity.

## Finding 1 — AtomTile render crash aborted the whole tune (fixed on this branch)

**Symptom.** The first clean tune ran ~35 min (1904 variants, 0 bench fails), then died with a one-line abort and no
traceback:

```
[tune] aborted: AtomTile must be consumed by the MMA materializer (kernel/010_split_register_axes MMA arm)
before render — reached render with axes=('a5', 'a6')
```

**Evidence.** Re-running with `_tune_one` wrapped to print the traceback placed the raise inside the *inner per-op
search* (`two_level.py:315 → pipeline.py:604 → cuda/010_lower_kernelop.py:71 → AtomTile.render`), on
`k_sdpa_transpose_reshape_linear_reduce_0a1109` (the SDPA attn@V matmul) under the structural-fork trajectory
`SPLIT_CONE=1 + MMA=mma_m16n8k16_f16 + STAGE=11`. Instrumenting the cell tagger caught the exact failure:

```
[debug-011] _classify_ab FAILED k_name='a7' a_found=True b_found=False
  load A: scaled_dot_product_attention_reduce__xn  index=(a0, a1*32 + a3*16, a7*16)        ← K last: classified
  load B: linear_2_reduce                          index=(0, a7*16, 0, a0/2*128 + a2*32 + a4*8)  ← K dim 1 of 4
```

**Root cause** — three stacked defects:

1. The mma eligibility gate (`tile/_atom.py`) only required *exactly one operand K-in-last*, while the cell tagger
   (`tile/011_lower_atom_cell.py::_classify_ab`) required B to be K-in-**first**. The cone split's 4-D V slab puts K
   in dim 1 of 4 — gate admits, tagger can't classify → the cell never gets its `Mma` tag,
   `kernel/005_lower_atom_tile` raises `RuleSkipped` ("expected operand Loads + Mma"), and the orphan `AtomTile` hits
   the render guard (`ir/tile/ir.py:1024`).
2. Rewrite-time exceptions in the inner search have no per-variant containment (bench errors become `bench_fail`
   rows; a raise inside `Candidate.try_rewrite` kills the whole tune).
3. `commands/tune.py:288`'s `except RuntimeError` (the bench-watchdog catch) swallowed the traceback —
   `NotImplementedError` is a `RuntimeError` subclass.

**Fix (this branch, `41c871a6`).** One shared classifier `tile/_atom.classify_matmul_operands` used by both the gate
and the tagger (mirror by construction), extending the historical first/last rule with a positional fallback (single
K dim after every other var-carrying dim ⇒ A, before every one ⇒ B; transposed-B stays unclassifiable);
`NotImplementedError` re-raised out of the watchdog catch; render-guard message names the real consumer. Four
regression tests in `tests/compiler/passes/test_partition_planner_mma.py` (classifier unit, gate acceptance,
end-to-end `CUDA_PASSES` render of the failing shape). Defect 2 (per-variant containment of rewrite errors) is left
open — worth its own change.

**Result.** The tune completes, and the unlocked kernel set wins: attn@V deploys with 3 `mma.sync` sites at 10 µs vs
12/12 µs eager/tcompile (run 1's scalar picks for the same op ranked ~573 µs at -O1).

## Finding 2 — MLP gate+up kernel is scalar-locked: 51 µs vs torch.compile's 14 µs (fixed on this branch)

**Symptom.** The biggest kernel (42% of Σ) fuses post-attn RMSNorm + gate (linear_4) + up (linear_5) + SiLU·up into
one kernel and runs 3.6x behind torch.compile. All 54 tune-DB variants are scalar (no `MMA` column in
`eval variants --kernel 23ab9c` — the tensor-core tier was never *enumerated*): class-2 tier lockout.

**NCU compare** (reproducer run `--bench --profile`; deplodock vs the torch reference in the same capture):

```
side  kernel                                       dur (ns)  occ%   sm%  dram%  fma%   lsu.inst  regs
dep   k_linear_mean_reduce_23ab9c                    73,376  18.8  55.0    9.8  28.0  6,491,520    96
ref   cutlass_80_wmma_tensorop_f16_s161616...        18,048   8.4   7.5   40.2   0.4    221,952    80
```

Scalar FMA pipe at 28%, 6.49M load-store instructions vs cuBLAS's 222K (29x), occupancy 18.8% at 96 regs — while the
tensor-op reference is properly DRAM-bound (40.2%) at 18 µs.

**Root cause** — two stacked gates, read off the fused loop IR (`compile <reproducer> --ir loop`):

```
for a3 in 0..1024:                          # shared K loop, BOTH matmuls
    in4 = load add_5[0, a0, a3]
    v5 = in4*v4;  in5 = load w_norm[a3];  v6 = in5*v5     # normed-x cone for gate
    in6 = load linear_4_wt[a3, a2];  v7 = in6*v6;  acc1 += v7
    v8 = in4*v4;  v9 = in5*v8                              # SAME cone, duplicated SSA, for up
    in7 = load linear_5_wt[a3, a2];  v10 = in7*v9;  acc2 += v10
```

1. The `SPLIT_CONE` structural fork — whose docstring names "the gated-MLP norm prologue" as a target — is **never
   offered**: `_split_demoted.py:128-131` requires every accum to share one cone root set, but the normed-x cone is
   emitted as two SSA-distinct duplicate chains (`v6` vs `v9`), so `set(roots) != set(cone_args)` bails
   ("accums with different cones"). No `SPLIT_CONE` stamp appears on any of the kernel's 54 variants (absent = never
   offered).
2. Even if split, the consumer cell stays mma-ineligible: the dual-accum cell (3 Loads, 2 Assigns, 2 Accums) fails
   the pure-cell count gate at `_atom.py` ("Pure matmul cell: 2 Loads, 1 Assign (the multiply), 1 Accum") — the gate
   admits exactly one matmul per K loop.

Note the weights load as `linear_4_wt[a3, a2]` = `[K, N]` (the trace pre-transposes Linear weights), so this is
*not* the transposed-B limitation.

**Within-tier search is fine — and exposes an -O1/-O3 rank inversion.** `eval variants` flags the deployed pick at
rank 15/54, 1.22x of best (tune-ranking -O1). A live `-O3` A/B of the DB rank-1 config refutes it:

```
deplodock run --ir <dump>/…/k_linear_mean_reduce_23ab9c.torch.json --bench --bench-backends deplodock \
    --ab "BM=8,BN=64,BK=64,FM=2,FN=1,RING=3,STAGE=111"
# greedy pick (BM=8 BN=16 BK=64 FM=1 FN=2):  50.2 µs
# ab = DB -O1 rank-1:                       129.7 µs   ← 2.6x SLOWER at -O3
```

The -O1 ranking inverts on this kernel; the deployed pick is the right scalar pick. The entire 51 → 14 µs gap is the
tier lockout, not search.

**Suggested fix (high priority, largest single win available).** (a) Value-number the cone roots in
`_split_demoted._classify_cut`'s operand scan so structurally identical duplicated chains count as one shared cone
(or CSE the duplicate before the offer); (b) extend the mma cell gate + `011`/`005` emit to the dual-accum
shared-A/two-B cell (two C fragments, one A fragment per K step), or have the split offer also cut gate and up into
separate single-matmul kernels. (b) is the larger change but is what torch.compile effectively does.

**Fix (this branch).** (a) + the separate-kernels arm of (b), both inside `_split_demoted.try_split_demoted`: the
K-cell is value-numbered so the SSA-duplicated norm chains count as ONE cone class sharing a single `xn`
materialization, and a multi-accum K loop extracts each accum's matmul into its own clean gemm producer
(`__mm0`/`__mm1`, written at `[rows, N]` f32 — the accumulator's own precision) with the consumer rebuilt as the
pointwise combine (each K loop replaced by Loads re-reading the `mm_i` buffers under the accums' SSA names; the
SiLU·up epilogue untouched). Each gemm is then the canonical pure cell, so the existing mma gate/tagger/emit apply
unchanged. A third stacked gate surfaced on the real reproducer: the cone-dtype vote included the prologue's f32
`mean_count`/`eps` scalar constants beside the f16 cell leaves ("mixed-dtype cone leaves" bail) — the vote now
covers cell leaves only (prologue/lead loads need only resolve). Everything stays under the existing binary
`SPLIT_CONE` offer; greedy cold still keeps the fused kernel. Result on the 23ab9c reproducer (RTX 5090, -O3,
CUDA-graph captured): `xn` 2.0 µs + 2 × ~5 µs `mma_m16n8k16_f16` gemms + 0.8 µs combine = **12.7 µs vs the deployed
51 µs (4.0x)** — ahead of torch.compile's 14 µs. Regression tests in `tests/compiler/passes/test_split_demoted.py`
(`test_gated_mlp_*`, stray-stmt bail).

## Finding 3 — o_proj kernel scalar-locked by a collapsed-reshape K operand (fixed on this branch)

**Symptom.** `k_linear_reshape_…_38c877` (attn-out reshape/transpose + o_proj + residual) runs 25 µs vs eager 16 µs
(cuBLAS) / tcompile 14 µs — the only kernel losing to *eager*. All 32 DB variants scalar (`FK` register-tile rows);
pick rank 1/32 — search did its job within the tier.

**Root cause.** The matmul's A operand is the attention output read through the *collapsed* reshape/transpose index
(from `01_frontend_decomposition`):

```
in1 = load linear_3_wt[a2, a1]                                            # B = [K, N] — fine
in2 = load scaled_dot_product_attention_reduce[0, ((a0*2048 + a2)/128) % 16, a0, 0, (a0*2048 + a2) % 128]
      # A: K (a2) appears in TWO index dims — the head dim and the element dim
```

The mma gate rejects any operand whose K spreads across more than one index dim (`_atom.py`, the
`len(k_dims) > 1` check — mirroring `020_stage_inputs`, which can only stage a single-K-dim slab), so the warp tier
is never enumerated. And no structural escape exists: both operands are plain `Load`s, so
`try_split_demoted` classifies the cell as "pure cell (not demoted)" (`_split_demoted.py:126-127`) and offers
nothing — the cone split only materializes *computed* operands, not collapsed-*layout* ones.

**Suggested fix (medium-high).** Extend the structural-fork vocabulary with a layout-materialization offer: when a
matmul operand is a plain Load whose K folds across dims, offer an `xn`-style producer that writes it contiguized as
`[M, K]` (the same machinery the cone split already has for computed operands — `_split_demoted` builds exactly such
producers, K second-to-last). The tuner then prices scalar-fused vs split+mma, per the established two-level design.

**Fix (this branch).** Exactly the suggested offer, expressed inside the existing per-operand rule: a plain-Load
multiply operand whose K spans more than one index dim (the `020_stage_inputs` single-K-dim-slab mirror — a
single-K-dim Load is already stageable and stays put) classifies as a **degenerate cone whose only member is the
Load itself**, so the entire existing machinery — cone slicing, escape check, xn producer build, consumer
replacement, fragment assembly — applies unchanged; the producer comes out as the contiguizing copy
`xn[rows…, K] = load attn[collapsed]`. Composes with the finding-2 multi-accum extraction for free. Result on the
38c877 reproducer (RTX 5090, -O3, graph-captured): the o_proj op runs **0.7 µs copy + 5.5 µs `mma_m16n8k16_f16`
gemm = 6.2 µs vs the deployed 25 µs (4.0x)** — now ahead of eager's 16 µs and tcompile's 14 µs; the whole
reproducer slice (SDPA producers included, all splits pinned) benches 11.4 µs vs eager 14. Regression tests in
`tests/compiler/passes/test_split_demoted.py` (`test_layout_split_*`, single-K-dim negative).

## Finding 4 — kernel-name extraction mangles MMA kernels in the tune DB (triage blind spot)

`eval variants` lists `dpl_ldmatrix_x4 — 26 measured configs` and `mbarrier_init — 1` as if they were kernels, while
the three deployed tensor-core linears (q/v/down: `k_linear_reduce_735349/bcc194/86a525`) have **no rows at all**.
`KERNEL_NAME_RE = re.compile(r"void\s+(\w+)\s*\(")` (`search/data/sample.py:36`) takes the *first* `void name(` in
`cuda_op.pretty` — for any MMA/TMA kernel that's a `__device__` helper (`dpl_ldmatrix_x4`, `mbarrier_init`), not the
`__global__ k_*` entry. Effects: several distinct kernels collapse into one leaderboard bucket (the run-1 DB had 446
rows under `dpl_ldmatrix_x4`), their per-kernel pick-reachability is unverifiable, and `--kernel k_linear_reduce`
filters find nothing. Fix: anchor the regex on `__global__` (tolerating `__launch_bounds__`), e.g.
`__global__\s+void(?:\s+__launch_bounds__\([^)]*\))?\s+(\w+)\s*\(`. Cheap, high triage value.

## Finding 5 — `eval prior --dataset db` crashes (reachability aggregate unusable)

```
File "deplodock/compiler/pipeline/search/prior/diagnostics.py", line 45, in reachability
    kmax = max(_n_tunable(s.all_knobs()) for s in grp)
AttributeError: 'tuple' object has no attribute 'all_knobs'
```

`reachability` iterates group values that are `(key, samples)` tuples (or a dict-items view) where it expects
`Sample`s. The aggregate pick-reachability view was unusable this run (the per-kernel `eval variants` ranks stood
in). Fix: unwrap the grouping in `diagnostics.reachability` (and add a smoke test that runs it on a 2-row DB).

## Finding 6 — the end-to-end gap is bigger than the per-kernel gaps (note, no deep dive)

Full-model deplodock is 138 µs but the per-kernel Σ is 121 µs (~12% inter-kernel overhead under CUDA-graph capture).
More telling: even if every losing kernel matched torch.compile's per-op time, the Σ would be ~72 µs — still 1.6x
behind tcompile's 45 µs end-to-end. torch.compile wins partly on kernel *granularity* (e.g. one flash-attention
kernel vs deplodock's 4-kernel SDPA decomposition, norm folded into neighbors). That's a fusion/kernel-set question
beyond any single finding here; the two-level structural search is the right home for it once findings 2–3 land.

## Repro / artifacts

- Work dir: `/tmp/tune-findings-qwen3-emb-l0/` — run 1 (crash): `tune.log`, `retune-debug2.log` (instrumented
  classify failure); run 2 (clean): `run2/tune.log`, dump at `run2/dump/` (per-kernel reproducers under
  `07_lowering_cuda.kernels/`, machine-readable bench in `62_kernel_bench.json`, chart `kernels.html`), NCU CSV/JSON
  in `61_ncu_metrics.{csv,json}`.
- Finding-1 regression (no GPU needed):
  `./venv/bin/pytest tests/compiler/passes/test_partition_planner_mma.py -k middle_dim_k -q`
- Finding-2 evidence (no GPU): `deplodock compile <dump>/07_lowering_cuda.kernels/k_linear_mean_reduce_23ab9c.torch.json
  --ir loop` (the duplicated cone), then the A/B + NCU commands quoted in the finding.
- Finding-3 evidence (no GPU): the same kernel's `--ir loop` / `01_frontend_decomposition.txt` shows the two-dim K
  index on the A load.

## Workflow notes

- **Swallowed traceback cost ~45 min**: the crash printed one line; pinning the op took two full re-runs with
  hand-rolled monkeypatch drivers (the per-`Pipeline.build` re-exec of rule modules defeats normal monkeypatching —
  patches must be re-applied via a `Pass.load` hook). Fixed at the CLI layer on this branch (re-raise
  `NotImplementedError`); the remaining gap is per-variant containment of rewrite-time errors (finding 1, defect 2) —
  a raise in one variant's lowering should cost one `bench_fail`-style row, not the tune.
- **`eval variants` blind spot** (finding 4): the two biggest *new* kernel sets this run (the cone-split MMA linears)
  were invisible under their real names. Fix the name regex; consider also surfacing "kernels in the deployed
  assembly with zero DB rows" as an explicit warning line in `eval variants`.
- **`eval prior --dataset db` crash** (finding 5): the one aggregate reachability view was down this run.
- **-O1 ranking can invert**: finding 2's DB rank-1 is 2.6x slower than the deployed pick at -O3, and 38c877's O1/O3
  scale gap is 137x. The `-O3 us` column in `eval variants` was empty for every row of both dominators — the
  `DEPLODOCK_O3_TOL` re-bench band evidently didn't cover them; worth checking why (it should have re-benched
  everything within 10% of the running -O1 best).
- **Reproducer re-fusion**: `compile <38c877-reproducer>.torch.json --ir loop` re-fuses the slice into a *different*
  kernel set (3 SDPA kernels) than the deployed one — fine for op-level evidence, but it means the isolated
  reproducer can't NCU-profile the *deployed* fused kernel for SDPA-adjacent slices. The `tune --bench` per-kernel
  table avoids this (it re-lowers with tuned forks); a `--profile` flag on `tune --bench`'s per-kernel pass would
  close the gap.
- **Chart PNG flake**: `kernels.html` written, `.png` failed with a Playwright "browser closed" error — harmless but
  noisy; the HTML is the artifact that matters.
- **Log ordering**: in `run2/tune.log` the "[tune] full-model bench" header printed ~20 lines before its table
  (stdout/stderr interleave) — cosmetic, but it briefly read as an empty bench.
- The previous findings report's workflow notes can't be diffed — the executed report was deleted from `plans/`
  (commit `fecb0cb4` "delete executed plan"); only its structure survives via the tune-findings skill. If these
  reports are meant to chain (each one checks the last one's notes), they need to stay in-tree or move somewhere
  durable.
