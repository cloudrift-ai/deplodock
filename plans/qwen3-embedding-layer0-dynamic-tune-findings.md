# Qwen3-Embedding-0.6B layer 0 — DYNAMIC-shape tune findings (2026-06-11)

Status: **first clean tune of the deployable dynamic configuration (`--dynamic seq_len@x:1`): the symbolic-axis
masked-tile kernels run, are accurate, and tune cleanly — but every matmul-bearing kernel is locked to the scalar
tier, leaving the deployed program ~10x behind eager / ~15x behind torch.compile at seq_len 512.** This is the
dynamic counterpart to today's static report (`plans/qwen3-embedding-layer0-tune-findings.md`, same model/layer/GPU,
which reached 0.90x torch.compile) — the gap between the two configurations is entirely the four symbolic-axis gates
documented below, three of which are known-by-design (the "M9" milestone) and now have a measured price. A fifth
finding is a bench-harness validity defect: the dynamic full-model table compares deplodock at the 512 hint against
eager at the seq-32 trace inputs, so its printed "0.03x eager" headline is shape-mismatched and wrong.

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  /tmp/tune-findings-qwen3-emb-l0-dyn/dump`
- Hardware: RTX 5090 (sm_120), driver 580.159.03, ncu 2025.3.1 (perf counters permitted)
- Run stats: tune wall **730.0 s** (~20 min with the -O3 `--bench`; 11x faster than the same day's static clean tune,
  see finding 2), **1 fused terminal** (vs 16 static), 657 `perf` rows / 657 ok / **0 `bench_fail`**, all CUDA-graph
  captured; prior: 752 benches (91 warmup / 661 post), post-warmup Spearman +0.88, 97% of post-warmup benches ≥2x the
  running best
- DB / prior: **default paths, cleaned** — `~/.cache/deplodock/autotune.db` (657 rows, this run only) and
  `~/.cache/deplodock/prior.json` now hold exclusively this run's symbolic scalar-tier data. Finding 5 covers what
  that means for future `compile` / `run` use.
- **Number families**: bench tables / reproducer runs / `--ab` rows below are -O3 (deployable, CUDA-graph captured);
  tune-DB latencies quoted for ranking context are `-Xcicc -O1` (ranking signal only). NCU durations run at locked
  base clock — compare ratios, not absolutes.
- **Dynamic measurement semantics**: the trace is symbolic in `seq_len` (`Dim` hint = `DEFAULT_SEQ_HINT` 512,
  `deplodock/compiler/dim.py:49,106` — the `--seq-len` flag does NOT set it; trace example tensors stay seq 32). All
  per-op tune benches and per-kernel reproducer benches run at the 512 hint. The full-model table mixes shapes —
  finding 4.

## Bench results (-O3, CUDA-graph captured)

Full-model table as printed by the tune's `--bench` — **invalid comparison, kept for the record** (deplodock side at
hint 512, torch sides at trace seq 32; see finding 4):

```
Backend        Latency (us)  vs Eager
-------------------------------------
Eager PyTorch            96     1.00x   <- seq 32
torch.compile            45     2.14x   <- seq 32
Deplodock              3229     0.03x   <- seq 512 (hint-sized synthetic inputs)
```

Honest like-for-like at seq_len 512, assembled from this work dir's runs (`run --seq-len 512 --bench
--bench-backends eager,tcompile,deplodock` for the torch rows; the dynamic deployed program re-measured by
`run --dynamic seq_len@x:1 --bench`, whose whole-program e2e footer is the deplodock number):

```
Backend                       Latency (us)  vs Eager
----------------------------------------------------
Eager PyTorch (512)                    217     1.00x
torch.compile (512)                    146     1.48x
Deplodock dynamic (hint 512)          2233     0.10x   (3229 in the tune-process bench — see workflow notes)
```

Per-kernel reproducer bench at the 512 hint (sorted by deplodock µs; layer-op labels from each kernel's
`.torch.json` provenance; machine-readable rows in `62_kernel_bench.json`):

| Kernel                         | Layer op                                               |  eager |  tcompile |  deplodock |  vs tcompile |
|--------------------------------|--------------------------------------------------------|-------:|----------:|-----------:|-------------:|
| `k_sdpa_reduce_77b0f0`         | RoPE (q,k) + QK^T + softmax                            |  147.6 |      24.4 |      732.0 |        0.03x |
| `k_linear_mean_reduce_63573a`  | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up |  118.5 |      56.8 |      671.4 |        0.08x |
| `k_linear_sdpa_reduce_43208b`  | attn-out reshape + o_proj (linear_3) + residual        |   38.9 |      36.9 |      553.6 |        0.07x |
| `k_sdpa_linear_reduce_a76a28`  | SDPA attn@V (+ in-kernel V producer)                   |   28.7 |      28.7 |      298.7 |        0.10x |
| `k_linear_0837e7`              | MLP down (linear_6) + residual                         |   22.5 |      22.8 |      235.3 |        0.10x |
| `k_mean_linear_reduce_79cfb6`  | q_norm RMSNorm (+ q_proj slice)                        |  104.5 |      20.5 |       75.9 |        0.27x |
| `k_linear_reduce_f94dd0`       | q_proj matmul                                          |   14.7 |      14.4 |       69.2 |        0.21x |
| `k_mean_linear_reduce_b1a761`  | k_norm RMSNorm (+ k_proj slice)                        |   75.8 |      14.2 |       49.1 |        0.29x |
| `k_linear_reduce_716194`       | v_proj matmul                                          |   10.2 |      10.2 |       43.9 |        0.23x |
| `k_mean_d65726`                | input RMSNorm                                          |   64.4 |       4.1 |        2.4 |        1.71x |
| **Σ**                          |                                                        |  625.8 |     233.1 |     2731.6 |       0.085x |

Only the pure input-RMSNorm kernel wins. The five matmul/SDPA dominators (`77b0f0`, `63573a`, `43208b`, `a76a28`,
`0837e7`) carry 2,491 µs — 91% of the deplodock total — and all five lose 10–30x to torch.compile. Nothing here is a
search shortfall: prior pick-reachability over the DB is mean 1.04x / median 1.00x / worst 1.18x, the worst
pick-misses are immaterial next to the tier gap, and there were zero bench failures. The whole gap is what the
lowering tier *enumerates* on a symbolic graph — findings 1–3.

## Finding 1 — the warp/MMA tier is gated off for any symbolic axis (~2,400 µs of the 2,731 µs at stake)

> **Status: addressed** on `feature/symbolic-axis-parity` — masked warp tiles for symbolic M
> (`tile: masked warp tier for symbolic M (M9)`) and N (`+ runtime ldm`); untuned pinned A/Bs on the dump
> reproducers: down-proj 235 → 59.8 µs, q_proj 79 → 40.5 µs.

**Symptom.** No kernel's `eval variants` leaderboard contains an `MMA` / `WM` / `WN` column at all — the tensor-core
tier was never *enumerated* for any of the 10 kernels (657 measured configs, all scalar). In today's static run the
same matmuls deployed `MMA=mma_m16n8k16_f16` warp tiles.

**Evidence.**

- The gate, `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py:660`:

  ```python
  if mma_on and graph is not None and not prologue and not m_symbolic and not n_symbolic and not k_symbolic:
  ```

  with the comment above it: "Warp-tier MMA: only when the MMA knob is enabled …, no prologue (M9 extension),
  **static M/N/K**, and the per-kind eligibility predicate fires." Every linear in this layer has M = `seq_len`
  (symbolic) → the whole `ATOM_REGISTRY` is skipped. The masked-tile machinery itself stops at the scalar tier:
  `_enumeration.py:679` — `m_forced_mask: bool,  # noqa: ARG001 — symbolic-axis masking for warp tier lands in M9`.
- NCU compare on the cleanest case, the down-proj matmul (`run --ir …k_linear_0837e7.torch.json --bench --profile`;
  locked base clock):

  ```
  side  kernel                                       dur (ns)  occ%   sm%  dram%  fma%    lsu.inst  regs
  dep   k_linear_0837e7                               360,064  78.4  54.8    1.8  57.8  31,535,104    40
  ref   cutlass_80_tensorop_f16_s16816gemm…            31,808   8.3  31.0   16.9   0.6     954,880    96
  ```

  The deplodock kernel is healthy *for its tier* — 78% occupancy, 55% SM — and saturated on the **scalar FMA pipe**
  (57.8% fma, 1.8% DRAM, 33x cutlass's LSU instruction count). It is compute-bound on the wrong pipe; no knob in the
  enumerated space can fix that. The same signature repeats on the masked QK^T+softmax kernel (`042770` in the
  77b0f0 reproducer profile: 8.2% SM, 0.1% DRAM, 69.3M LSU inst, 10.8M smem load bank conflicts).
- Scale check: the scalar register-tile down-proj benches 235 µs vs torch's cublas 22.5 µs at 512; q/v_proj and the
  fused o_proj/gated-MLP kernels show the same 4–15x.

**Root cause.** Known limitation, correct-by-design today: warp-tier masking for symbolic axes is the planned M9
milestone (both citations above name it). This report's contribution is the price tag: ~2,400 µs vs torch.compile on
one layer at seq 512 — the difference between the static config's 0.90x tcompile and the dynamic config's 0.07x.

**Suggested fix (the M9 milestone, highest priority of everything here).** Extend `is_atom_eligible` + the masked
overhang path to admit a symbolic M axis: the M9 design point is that the mask only gates *rows of the output tile*
(MMA fragments compute garbage rows that are simply not stored), so a hint-sized warp tile with an `if (row <
seq_len)` store guard is semantically the same masked tile the scalar tier already emits. K stays static for every
linear in this model (1024/2048/3072), so the K-side `ldmatrix`/cp.async machinery is untouched.

## Finding 2 — structural splits never offer on symbolic graphs (1 fused terminal; gated-MLP and o_proj stay fused-scalar)

> **Status: addressed** on `feature/symbolic-axis-parity` (`tile: admit symbolic row axes in the demoted-matmul
> split`) — the gated-MLP reproducer splits and benches 207 µs vs the 671 µs fused-scalar baseline (untuned).

**Symptom.** `[tune] done: 1 fused terminal(s)` vs 16 terminals in the static tune. The dump has **no
`005_split_demoted` rules file** — the pass made zero offers — so the gated-MLP (`63573a`, 671 µs) and o_proj
(`43208b`, 554 µs) kernels tuned only in their fused-scalar forms, the exact shapes whose splits won the static run.

**Evidence.** The cut classifier bails on symbolic extents before any offer is built —
`deplodock/compiler/pipeline/passes/lowering/tile/_split_demoted.py:408-411`:

```python
if not k_loop.axis.extent.is_static or not outer_n.axis.extent.is_static:
    return None
if any(not lp.axis.extent.is_static for lp in rows):
    return None
```

The row axes here are `seq_len` → the second check fires for every candidate (the module docstring lists "symbolic
extents" among the conservative bails, `_split_demoted.py:60`). A side effect worth recording: the clean dynamic tune
is **11x cheaper** than the static one (730 s vs 8,243 s) precisely because the structural tree collapsed to one
terminal and the MMA inner spaces never enumerated.

**Root cause.** Conservative well-formedness bail, consistent with the module's design ("the checks here are the
cut's own WELL-FORMEDNESS conditions"). But once finding 1's M9 work lands, this becomes the binding constraint for
the two biggest kernels: a symbolic-M gemm can't reach the warp tier *anyway* unless the split machinery also learns
symbolic rows (the split exists to hand the mma cell gate a clean single-matmul K loop).

**Suggested fix (do together with M9).** The cut itself never tiles the row axes — it extracts the K-loop matmul and
re-emits the rows verbatim — so admitting symbolic `rows` extents looks mechanical (the `xn` materialization buffer
gets a symbolic leading dim, which the masked-tile path already knows how to allocate at the hint). Keep the static-K
/ static-N checks.

## Finding 3 — SDPA-prologue and cooperative-reduce kernels keep the symbolic axis fully degenerate (~450 µs)

> **Status: partially addressed** on `feature/symbolic-axis-parity` — symbolic-K prologue matmuls (P@V) get
> masked thread tiles at `FM=FN=1`; static-K prologue and cooperative-reduce kernels stay degenerate by design
> (their staged pipelines can't coexist with the per-row guard — deployment path is the finding-2 split;
> strided-cooperative symbolic axes remain the M5+ follow-up).
>
> **Strided-cooperative follow-up: addressed** on `feature/strided-coop-symbolic-rows` — the v1 `BR>1 ⇒
> BN=BM=1` constraint is lifted for cooperative-reduce kernels: static free axes thread-bind alongside the
> BR lanes (2D CTA, segmented warp-shuffle combine; 2D rows clip BR to powers of two ≤ warp_size), so the
> q/k-norm class deploys e.g. `BN=8×BR=32` 256-thread CTAs at 100% occupancy instead of 8-thread ones. A
> scoped clean tune of the `79cfb6` reproducer (159 rows, 0 bench_fail) picks a 2D row rank-1: 1.8 µs -O3
> vs 2.1 µs for the best v1 config. Note the slice totals quoted above were matmul-dominated; post-M9 the
> matmul half runs as a masked MMA kernel, so the norm-kernel remainder is ~2 µs-scale, not ~90 µs.

**Symptom.** attn@V (`a76a28`, 299 µs vs tcompile 28.7) deploys `BM=1, BN=32` — one output element per thread on the
M side, 0 KB smem, 8,192 blocks of 32 threads at 50% occ. The q/k-norm kernels (`79cfb6` 75.9 µs, `b1a761` 49.1 µs vs
tcompile 20.5 / 14.2) deploy `BM=1, BN=1, BR=8` — 8-thread blocks, 0% occupancy. These aren't pick-misses (all three
picks are rank 1 of their leaderboards); the degenerate shapes are the *whole* enumerated space.

**Evidence.** `010_partition_loops.py:624-633` — masking is disabled when a fused prologue is present:

```python
# A fused prologue (SDPA P@V: softmax max/sum) carries a per-M-row reduction whose accumulators must
# reset per register cell. Masking a symbolic M/N axis admits FM/FN > 1, register-tiling the row and
# sharing one accumulator across cells — wrong. So a prologue matmul keeps symbolic axes degenerate
# (E=1, no mask): correct via the symbolic grid, one output element per thread.
mask_ok = not prologue
```

and the cooperative-reduce analogue at `010_partition_loops.py:730-732` ("a symbolic free axis stays … symbolic grid,
just not register-tiled"). NCU on the degenerate attn@V shape (the `6e4bd6` kernel inside the 77b0f0 reproducer
profile): 52.9% SM / 52.6% scalar-fma / 2.1% DRAM, zero smem — again saturated on the scalar pipe by construction.

**Root cause.** Correctness bail, documented in-code: a masked register tile would share one per-row accumulator
across register cells. By design until cooperative work over symbolic axes lands (`_enumeration.py:520` — "cooperative
work over symbolic axes is M5+ follow-up").

**Suggested fix (medium).** Distinct from M9 and cheaper: the degenerate path needs *thread cooperation along the
static axes*, not masking — e.g. let `BR`-style strided-thread reduction co-exist with a symbolic grid axis (the
planner comment at `010_partition_loops.py:585-590` already names "strided cooperative threads" as the perf
follow-up). The q/k-norm kernels alone are ~90 µs vs tcompile.

## Finding 4 — the dynamic full-model bench table is shape-mismatched (deplodock@512-hint vs eager@32) — harness defect

> **Status: addressed** on `feature/dynamic-fullmodel-bench-hint` — `_hint_sized_inputs` tiles the torch closures'
> example inputs out to each `Dim`'s hint (in both `run --bench` and the worker's `bench_full_model_real`), and the
> table prints a `benched at seq_len=512 (symbolic hint)` note. Verified on the finding's repro: the dynamic layer-0
> run's eager row now reads 217 µs (the honest static-512 number), not 96.

**Symptom.** The tune's full-model table prints eager 96 µs / tcompile 45 µs — these match today's *static seq-32*
report (99 / 46) — while a direct static-512 run measures eager at 217 µs. The deplodock row (3229 µs tune-bench,
2233 µs in a fresh `run --dynamic … --bench`) is consistent with the Σ of the 512-hint per-kernel slices (2,731 µs),
not with seq-32 work. The printed "0.03x eager" compares different shapes.

**Evidence.** `commands/run.py::bench_full_model_real` benches the torch closures on the bound trace inputs
(`cuda_args`, seq 32) while the deplodock side goes through `_bench_interleaved` →
`backend.benchmark(compiled_graph, …)` (`run.py:1585`) with **no inputs** — and the backend benches symbolic graphs
on hint-sized (512) synthetic inputs when none are supplied. The helper's own docstring concedes the input split
("deplodock's bench uses synthetic activations vs torch's bound inputs, so only latency is comparable here",
`run.py:1093-1096`) — which is harmless for static graphs (same shape either way) and wrong for symbolic ones.

**Repro.** `deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench` — compare its eager row
(≈96 µs) against `deplodock run … --layer 0 --seq-len 512 --bench` (≈217 µs).

**Suggested fix (high — it invalidates the headline number of every dynamic findings run).** In
`bench_full_model_real`, when the compiled graph is symbolic, build the torch closures on hint-sized inputs (resize /
re-trace the example tensors to the hint) so both sides run the hint shape; print `benched at seq_len=<hint>
(symbolic)` in the table header either way. Until then, the per-kernel reproducer table is the only valid dynamic
comparison the tune emits.

## Finding 5 — the freshly built default prior is scalar-only; static greedy compiles now mispick (affects intended future use)

**Symptom.** This run's `--clean` rebuilt the **default** DB + prior (the configuration requested for future
`compile` / `bench` use) from 661 post-warmup benches that contain *zero* warp-tier rows (finding 1). A static
seq-512 greedy run under this prior deploys MMA kernels with visibly bad configs — e.g. `k_linear_mean_reduce_f1b55d`
at **4% occupancy** (141.6 µs), and the q/k-norm kernels at 8-thread degenerate configs (97.5 / 107.8 µs) — landing
at 832 µs e2e, 0.26x eager, where this morning's static-tuned deploy reached 0.90x tcompile at seq 32.

**Evidence.** `run-static512-tc.log` in the work dir (the greedy launch table shows the `mma_m16n8k16_f16` rows the
dynamic tune never trained on). The `CatBoostPrior` is global; once trained it overrides the `AnalyticPrior`
everywhere, so its warp-tier predictions here are pure extrapolation from scalar features.

**Root cause.** Not a defect — the prior is faithfully trained on what was benched. It's a deployment-hygiene
consequence of `--clean` + a symbolic-only tune.

**Suggested action (for this box, before relying on greedy static compiles).** Accumulate static rows into the same
DB/prior — e.g. `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --bench` (no `--clean`; ~2.3 h on this box) or
the cheaper `deplodock tune --dataset golden` sweep — so the learned prior sees warp-tier `H_opt=3` rows again. For
purely dynamic serving work, the prior is already the right one (its picks are rank-1 on 7 of 10 kernels here).

## Repro / artifacts

- Work dir: `/tmp/tune-findings-qwen3-emb-l0-dyn/` — `tune.log`, `run-dynamic.log` (deployed launch table + e2e
  footer), `run-static512.log`, `run-static512-tc.log` (the eager/tcompile@512 rows), `ncu-77b0f0.log` /
  `ncu-0837e7.log` (+ raw CSV/JSON under `ncu-*/61_ncu_metrics.{csv,json}`), dump at `dump/` (reproducers under
  `07_lowering_cuda.kernels/`, machine-readable bench `62_kernel_bench.json`, chart `kernels.html`; `.png` skipped —
  Playwright flake again).
- Tune DB / prior (default paths, this run only): `~/.cache/deplodock/autotune.db` (657 rows),
  `~/.cache/deplodock/prior.json`.
- Finding-1 NCU: `DEPLODOCK_DUMP_DIR=<dir> deplodock run --ir
  <dump>/07_lowering_cuda.kernels/k_linear_0837e7.torch.json --bench --profile` (no `--dynamic` — the reproducer
  keeps its symbolic dims and benches at the 512 hint).
- Finding-4 repro: the two `run` commands quoted in the finding.
- Compile-only gate probes (no GPU): `deplodock compile <dump>/07_lowering_cuda.kernels/k_linear_0837e7.torch.json
  --ir cuda` shows the scalar masked tile (`if (a1 * 32 + a3 * 2 < seq_len)` guard, global-memory inner loop, no
  `mma.sync`); grep the gates at `010_partition_loops.py:660`, `_split_demoted.py:408-411`,
  `010_partition_loops.py:633`.

## Workflow notes

Audit of today's static report's notes first:

- **Reports deleted from `plans/`**: not hit this time — the static report was still untracked in `plans/`, and this
  report deliberately took a `-dynamic-` name instead of overwriting it. The durable-home suggestion
  (`docs/findings/`) stands; two same-day reports for one model/layer make the collision concrete.
- **Silent tune under `tee`**: reproduced (49-line log for a 12-minute tune), but mattered less at this scale. The
  non-tty heartbeat suggestion stands for static-scale (2 h+) runs.
- **Per-launch mis-attribution**: reproduced at a far worse scale than the static report's 5.1-vs-0.8 µs example: the
  deployed dynamic table shows `k_mean_linear_reduce_b1a761` at **574.9 µs** where its own reproducer benches 49.1 µs,
  and the twin `716194` launches read 2.6 vs 43.9 µs for identical work. At this magnitude the per-launch `%` column
  actively misleads triage (it fingered the wrong dominator until the reproducer table corrected it). Strengthens the
  case for `plans/bench-attribution-by-slicing.md`.
- **Reproducer re-fusion**: reproduced — the `77b0f0` slice re-fuses into two kernels (`6e4bd6` + `042770`) under
  `run --ir`, so its 732 µs reproducer number is a slice-set total, not the deployed kernel alone. Same top tooling
  gap as before.
- **Chart PNG Playwright flake**: reproduced verbatim.

New friction this run:

- **The full-model dynamic table is wrong** (finding 4) — it cost the first half-hour of triage: the printed 0.03x
  headline sent me hunting for a 30x regression before grid arithmetic showed the eager row was seq-32. Until the
  harness fix lands, this skill should instruct: *for dynamic runs, ignore the full-model table; assemble the e2e
  comparison from a static `--seq-len <hint>` run plus the dynamic run's e2e footer* (as done here).
- **The skill doc's measurement-semantics note is wrong on one point**: it says all tune/bench measurements run at
  the hint — true for per-op tunes and reproducers, false for the full-model table (mixed 32/512, finding 4) — and
  implies `--seq-len` influences the hint (it doesn't; `dim.py:106` hard-codes `DEFAULT_SEQ_HINT=512`, `--seq-len`
  only sizes trace example tensors). Fix the skill text alongside the harness fix.
- **Two e2e numbers for one deployed program** (3229 µs in the tune-process bench vs 2233 µs in a fresh `run`, same
  prior, same picks-by-rank): a 45% swing between two captured whole-program measurements of nominally the same
  program. Possibly tune-process state (post-tune memory pools) vs fresh-process; `deplodock compare` can't diff
  these because only the tune wrote a dump. Worth a look when the bench-attribution refactor lands; the reproducer
  table was the stable signal as predicted by the static report's variance caveat.
- **What worked well**: the triage loop was three commands this time (`eval failures` → 0, `eval prior --dataset db`
  → reachability clean, `eval variants` → no MMA column anywhere = the entire root cause), and the rules dumps
  (`05_lowering_tile__*.rules.txt`, plus the *absence* of a `005_split_demoted` file) pointed straight at the two
  gates. The `eval variants` knob columns doubling as a tier fingerprint is exactly the kind of view the earlier
  reports asked for.
