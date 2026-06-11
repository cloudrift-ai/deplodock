---
name: tune-findings
description: Use this skill when the user asks to "analyze tune findings for <model>", "tune and analyze model X", "why is deplodock slower than eager / torch.compile on X", "do a per-kernel performance analysis", "drill into kernel performance", "profile the compiled kernels with NCU", or otherwise wants a clean autotune of a model (or one layer), an end-to-end + per-kernel bench against PyTorch eager and torch.compile, a root-cause analysis of every underperforming kernel (search shortfall, tier/optimization lockout, codegen quality, bench failures), and a findings report saved to plans/. Modeled on plans/qwen3-embedding-layer0-tune-findings.md.
version: 0.1.0
---

# Tune a model and produce a per-kernel findings report

A "tune findings" pass answers: after a clean autotune, where does deplodock stand vs eager PyTorch and
torch.compile on this model, which kernels lose, and *why* — with each "why" pinned to a failure class, evidence
(NCU counters, tune-DB rows, emitted CUDA), and a repro command. The deliverable is a report in `plans/`, in the
shape of `plans/qwen3-embedding-layer0-tune-findings.md` (read it first — it is the reference for tone, structure,
and evidence density).

Reuse the existing CLI for everything: `tune` / `run` / `compile` / `eval` already cover tuning, benching, NCU
profiling, knob pinning, and prior diagnostics. Do not write ad-hoc bench scripts.

## Prerequisites

- A CUDA GPU. Check `ncu` is on PATH and perf counters are permitted (`ncu --version`; a later
  `ERR_NVGPUCTRPERM` means the NVIDIA perf-counter permission gate — note it in the report and skip NCU rather
  than fighting it).
- Scope: default to **one layer** (`--layer 0`) unless the user asks for the whole model — a single-layer clean
  tune + bench is ~10–20 min; whole-model is much longer. Quick ungated default model:
  `Qwen/Qwen3-Embedding-0.6B`.
- Work dir: pick a fresh dir (e.g. `/tmp/tune-findings-<slug>/`) holding the dump dir and tee'd logs — the
  report quotes both.

## Step 1 — clean tune + -O3 bench

```bash
deplodock tune <model> --layer 0 --clean --bench --dump-dir <dir>/dump 2>&1 | tee <dir>/tune.log
```

One command does the whole measurement pass: `--clean` nukes the tune DB + prior + cubin caches (a *clean* tune —
no warm-prior carryover), the tune itself fills the DB with -O1 ranking rows and trains the prior, and `--bench`
re-benches the tuned result at **-O3** (deployable, CUDA-graph captured): the full-model
eager / torch.compile / deplodock table, the per-kernel table (each kernel re-lowered with its tuned forks,
benched from its `.torch.json` reproducer against eager + tcompile), and `<dump>/kernels.html`.

Record the run stats for the report header — wall time, benched-variant count, ok vs `bench_fail`:

```bash
sqlite3 ~/.cache/deplodock/autotune.db "SELECT status, COUNT(*) FROM perf GROUP BY status"
```

(Honor `DEPLODOCK_TUNE_DB` if set. No `sqlite3` binary → `./venv/bin/python -c "import sqlite3; ..."`.)

**CRITICAL: never mix the two number families.** Tune-DB latencies are `-Xcicc -O1` (ranking signal only —
reduction/attention kernels run 1.5–3× slower than -O3); the `--bench` tables are -O3 and deployable. Every
number in the report says which it is; comparisons across the two families are findings-invalidating errors.

## Step 2 — headline tables

- Copy both `--bench` tables into the report. Sort the per-kernel table by deplodock latency and add a
  **Layer op** column: each `<dump>/*_lowering_cuda.kernels/<kname>.torch.json` (and its `.txt` summary) holds
  the original torch ops the kernel realizes — read them to label rows (kernels are op-named: `k_rms_norm`,
  `k_sdpa_reduce`, …; same-named kernels disambiguate by hash suffix).
- Name the dominating kernels: cumulative µs until ~80% of the deplodock total. Those get the deep dives;
  kernels already beating eager and tcompile get at most one line.

## Step 3 — triage every underperforming kernel

For each kernel meaningfully behind eager or tcompile, assign one (or more) of four failure classes. The class
determines the drill-down:

1. **Search shortfall** (patience / prior pick-reachability): the tune DB holds a faster measured variant than
   the pick the prior deploys. Diagnostics: `deplodock eval prior --dataset db [--kernel SUBSTR]` (does the
   prior recover each op's measured-best leaf?) and `deplodock eval knobs --kernel SUBSTR` (per-knob regret).
   Raw view: `SELECT latency_us_median, knobs FROM perf WHERE status='ok' ORDER BY latency_us_median LIMIT 10`
   filtered to the op via the knobs/kernel-name columns.
2. **Tier / optimization lockout**: the deployed pick is scalar-tier (`MMA=0`, no warp tile) where a
   tensor-core mapping should exist, or it lacks TMA / cp.async / an epilogue fold. Evidence: the knob columns
   in the kernel table + the dump's CUDA source. Find the gate that fires in
   `deplodock/compiler/pipeline/passes/lowering/tile/` (eligibility in `010_partition_loops.py` / `_atom.py`,
   structural offers in `005_split_demoted.py`, transport in `050_use_tma.py`) and say *why* it fires. If a
   structural fork can unlock the tier, A/B it by pinning the decision knob (e.g. `DEPLODOCK_SPLIT_CONE=1`).
3. **Codegen quality**: right tier, still slow vs cuBLAS / tcompile. NCU compare (step 4) + read the emitted
   CUDA (`deplodock compile <reproducer>.torch.json --ir cuda`, knobs pinned to the deployed pick).
4. **Bench failures**: `SELECT knobs FROM perf WHERE status='bench_fail'` — cluster the failing rows by shared
   knob (e.g. all have `TMA: true`), match against tune-log error lines (`cuTensorMapEncodeTiled`,
   `HungKernelError`), and build a compile-only repro with `DEPLODOCK_KNOBS` pinning. Failed benches are wasted
   search slots even when the kernel's final pick is fine.

## Step 4 — drill down per kernel

- **Isolated reproducer** — accuracy + 3-way bench for just that op (re-lowered, so the tuned forks apply):

  ```bash
  deplodock run --ir <dump>/*_lowering_cuda.kernels/<kname>.torch.json --bench \
      --bench-backends eager,tcompile,deplodock
  ```

- **NCU** — append `--profile` to the command above. It re-launches the run under `ncu` with the curated metric
  set (`commands/run.py::_NCU_METRICS`: occupancy, SM/DRAM/FMA throughput, smem bank conflicts,
  registers/thread); `--target-processes all` captures the torch/cuBLAS kernels in the same CSV, so deplodock's
  rows sit beside the library kernel for a direct counter-by-counter compare. With a dump dir set, the parsed
  output lands in `<dump>/61_ncu_metrics.{csv,json}`. Typical reads: low `sm__warps_active` → occupancy /
  register pressure (check `launch__registers_per_thread`); high LSU inst count + bank conflicts → smem layout
  (try `PAD_SMEM` / `PERMUTE_LANES`); high `dram__throughput` with low `sm__throughput` → memory-bound, tiling
  or fusion issue; low `sm__pipe_fma` on a matmul → not on tensor cores (back to class 2).
- **Knob A/B** — pin any DB variant's knobs and re-bench or inspect source:

  ```bash
  DEPLODOCK_KNOBS="BM=...,BN=...,BK=...,MMA=1,..." deplodock run --ir <kname>.torch.json --bench
  DEPLODOCK_KNOBS="..." deplodock compile <kname>.torch.json --ir cuda   # source only, no GPU needed
  ```

  Structural decision knobs (the `SPLIT_*`-style names stamped into `op.knobs`) pin via `DEPLODOCK_<KNOB>=...`.
- **Code reading** — every class-2/3 finding cites the responsible rule or gate as `file:line`; quote the gate
  condition, don't paraphrase it from memory.
- **Search-shortfall confirmation** — re-tune just the reproducer with more patience (no `--clean`, accumulate):

  ```bash
  deplodock tune <dump>/*_lowering_cuda.kernels/<kname>.torch.json --patience <2-4x default>
  ```

  If the better variant becomes the pick, the finding is "patience/prior", not codegen.

## Step 5 — write the report

Save to `plans/<model-slug>-layer<N>-tune-findings.md` (drop `-layer<N>` for whole-model). Mirror the reference
doc's structure:

- **Header**: status line, the exact run command, date, run stats (wall time, benched variants, ok/`bench_fail`
  counts), and the -O3-vs--O1 disclaimer ("numbers below are the `--bench` -O3 re-bench; tune-DB latencies
  quoted for ranking context are -O1").
- **Bench results**: both tables (full model + per-kernel with the Layer-op column), then one sentence naming
  the dominating kernels and their combined µs.
- **One `## Finding N — <title>` per root cause**, ordered by µs at stake. Each finding carries: symptom (the
  numbers), evidence (NCU counters, DB rows, emitted-CUDA observations), root cause or the best hypothesis with
  the distinguishing diagnostic (cite the gate as `file:line`), a repro command (reproducer path and/or pinned
  knobs), and a suggested fix with priority.
- **Repro / artifacts** tail: log + dump locations, the useful SQL, and copy-pasteable pinned-knob repro blocks
  (compile-only repros that need no GPU are the most valuable).
- Wrap to ~120 chars (repo-wide markdown rule); tables may overflow.

## Notes

- A finding must be actionable: it either names the code gate responsible or states the diagnostic that
  separates the competing hypotheses. "Kernel X is slow" with no class is not a finding.
- Variance caveat: scalar-tier tuned picks swing run-to-run (prior pick-reachability), so per-kernel reproducer
  rows — not the full-model total — are the stable before/after signal. Re-run the reproducer bench once before
  reporting a surprising number.
- Known-limitation findings (e.g. a warp-tier gate that is correct-by-design) still go in the report when the
  kernel is a big slice of the total — recorded with the µs at stake, like the reference doc's finding 5.
