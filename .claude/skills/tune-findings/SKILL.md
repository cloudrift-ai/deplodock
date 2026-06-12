---
name: tune-findings
description: Use this skill when the user asks to "analyze tune findings for <model>", "tune and analyze model X", "why is deplodock slower than eager / torch.compile on X", "do a per-kernel performance analysis", "drill into kernel performance", "profile the compiled kernels with NCU", or otherwise wants a clean autotune of a model (or one layer), an end-to-end + per-kernel bench against PyTorch eager and torch.compile, a root-cause analysis of every underperforming kernel (search shortfall, tier/optimization lockout, codegen quality, bench failures), and a findings report saved to plans/. Modeled on plans/qwen3-embedding-layer0-tune-findings.md.
version: 0.3.0
---

# Tune a model and produce a per-kernel findings report

A "tune findings" pass answers: after a clean autotune, where does deplodock stand vs eager PyTorch and
torch.compile on this model, which kernels lose, and *why* — with each "why" pinned to a failure class, evidence
(NCU counters, tune-DB rows, emitted CUDA), and a repro command. The deliverable is a report in `plans/`, in the
shape of `plans/qwen3-embedding-layer0-tune-findings.md` (read it first — it is the reference for tone, structure,
and evidence density).

Reuse the existing CLI for everything: `tune` / `run` / `compile` / `eval` / `compare` already cover tuning,
benching, the variant leaderboard, failure forensics, knob A/Bs, NCU comparison, and run-to-run diffing. Do not
write ad-hoc bench scripts or hand-written SQL.

## Prerequisites

- A CUDA GPU. Check `ncu` is on PATH and perf counters are permitted (`ncu --version`; a later
  `ERR_NVGPUCTRPERM` means the NVIDIA perf-counter permission gate — note it in the report and skip NCU rather
  than fighting it).
- Scope: default to **one layer with dynamic shapes** (`--layer 0 --dynamic seq_len@x:1`) — symbolic-`seq_len`
  masked-tile kernels are the deployable artifact, and a single-layer clean tune + bench stays ~10–20 min. The
  per-layer dynamic trace goes through `trace.huggingface.build_layer_wrapper` (in-graph sliced rotary); its one
  trace input is named `x`, hence the single spec. Whole-model (much longer) only when the user asks — its
  dynamic specs are `--dynamic seq_len@input_ids:1 --dynamic seq_len@attention_mask:2 --dynamic
  seq_len@attention_mask:3 --dynamic seq_len@position_ids:1`. Quick ungated default model:
  `Qwen/Qwen3-Embedding-0.6B`.
- Static (no `--dynamic`) only if the user explicitly wants shape-specialised kernels; say so in the report
  (specialised kernels, no masked-tile guards — not the deployable configuration).
- Work dir: pick a fresh dir (e.g. `/tmp/tune-findings-<slug>/`) holding the dump dir and tee'd logs — the
  report quotes both, and `deplodock compare` diffs dump dirs across runs.

## Step 1 — clean tune + -O3 bench

```bash
deplodock tune <model> --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir <dir>/dump 2>&1 | tee <dir>/tune.log
```

(Whole-model scope: drop `--layer 0` and use the four whole-model `--dynamic` specs from Prerequisites.)

One command does the whole measurement pass: `--clean` nukes the tune DB + prior + cubin caches (a *clean* tune —
no warm-prior carryover), the tune itself fills the DB with -O1 ranking rows and trains the prior, and `--bench`
re-benches the tuned result at **-O3** (deployable, CUDA-graph captured): the full-model
eager / torch.compile / deplodock table, the per-kernel table (each kernel re-lowered with its tuned forks,
benched from its `.torch.json` reproducer against eager + tcompile), `<dump>/kernels.html`, and the
machine-readable `<dump>/62_kernel_bench.json` that `deplodock compare` diffs later.

Record the run stats for the report header — wall time, benched-variant count, and the ok / `bench_fail` split
(`eval failures` prints both counts; honor `DEPLODOCK_TUNE_DB` if set).

**CRITICAL: never mix the two number families.** Tune-DB latencies are `-Xcicc -O1` (ranking signal only —
reduction/attention kernels run 1.5–3× slower than -O3); the `--bench` tables are -O3 and deployable. Every
number in the report says which it is; comparisons across the two families are findings-invalidating errors.

**Dynamic-run measurement semantics**: with `--dynamic`, all tune/bench measurements run the symbolic kernels at
the `Dim` hint (`DEFAULT_SEQ_HINT=512`) — record "benched at seq_len=512 (symbolic)" in the report header. The
masked-tile boundary guards (`if (coord < seq_len)`) are part of the measured cost; that overhead vs a
shape-specialised kernel is itself a legitimate finding, not noise.

## Step 2 — headline tables

- Copy both `--bench` tables into the report. Sort the per-kernel table by deplodock latency and add a
  **Layer op** column: each `<dump>/*_lowering_cuda.kernels/<kname>.torch.json` (and its `.txt` summary) holds
  the original torch ops the kernel realizes — read them to label rows (kernels are op-named: `k_rms_norm`,
  `k_sdpa_reduce`, …; same-named kernels disambiguate by hash suffix).
- Name the dominating kernels: cumulative µs until ~80% of the deplodock total. Those get the deep dives;
  kernels already beating eager and tcompile get at most one line.

## Step 3 — triage every underperforming kernel

Two commands produce most of the triage evidence:

```bash
deplodock eval variants --kernel <SUBSTR>   # per-kernel leaderboard: measured configs fastest-first,
                                            # the prior's deployed pick marked + ranked, -O3 re-bench column
deplodock eval failures                     # bench_fail rows clustered by (kernel, error) + shared knobs
```

For each kernel meaningfully behind eager or tcompile, assign one (or more) of four failure classes:

1. **Search shortfall** (patience / prior pick-reachability): `eval variants` shows the pick ranked far from the
   measured best (`pick: rank R/N, X.XXx of best … <-- misses best`). Confirm with
   `deplodock eval prior --dataset db` (aggregate reachability) and `deplodock eval knobs` (per-knob regret).
   The rank-1 row's knobs are your A/B pin for step 4.
2. **Tier / optimization lockout**: every row in the `eval variants` leaderboard is scalar-tier (`MMA=0`, no
   warp tile) — the tensor-core variants were never *enumerated*, so an eligibility gate fired. Find it in
   `deplodock/compiler/pipeline/passes/lowering/tile/` (eligibility in `010_partition_loops.py` / `_atom.py`,
   structural offers in `005_split_demoted.py`, transport in `050_use_tma.py`) and say *why* it fires. If a
   structural fork can unlock the tier, A/B it by pinning the decision knob (e.g. `DEPLODOCK_SPLIT_CONE=1`).
   (MMA rows present but slow → class 3, not class 2.)
3. **Codegen quality**: right tier, still slow vs cuBLAS / tcompile. NCU compare (step 4) + read the emitted
   CUDA (`deplodock compile <reproducer>.torch.json --ir cuda`, knobs pinned via `DEPLODOCK_KNOBS`).
4. **Bench failures**: `eval failures` gives each cluster's count, error text (recorded in the DB's `error`
   column — no log grepping), and the knob assignments shared by every failing row (e.g. all have `TMA=1`).
   Build a compile-only repro with `DEPLODOCK_KNOBS` pinning. Failed benches are wasted search slots even when
   the kernel's final pick is fine.

## Step 4 — drill down per kernel

- **Isolated reproducer** — accuracy + 3-way bench for just that op (re-lowered, so the tuned forks apply):

  ```bash
  deplodock run --ir <dump>/*_lowering_cuda.kernels/<kname>.torch.json --bench \
      --bench-backends eager,tcompile,deplodock
  ```

  Dumped reproducers already carry the symbolic dims from a dynamic trace — do **not** pass `--dynamic` with
  `--ir` (rejected: the trace is already complete); the re-bench runs the masked-tile kernel at the hint.

- **Knob A/B in one run** — pin any leaderboard variant beside the greedy pick with `--ab` (repeatable; the
  `DEPLODOCK_KNOBS` grammar). Each config re-lowers fresh and prints as an `ab KNOBS` row in the kernel table,
  knob diffs red:

  ```bash
  deplodock run --ir <kname>.torch.json --bench --ab "BM=16,BN=64,BK=32" --ab "MMA=1,WM=32,WN=64"
  ```

  For source-only inspection (no GPU): `DEPLODOCK_KNOBS="..." deplodock compile <kname>.torch.json --ir cuda`.
- **NCU** — append `--profile` to the reproducer run. It re-launches under `ncu` (curated metric set,
  `commands/run.py::_NCU_METRICS`) and prints the **`ncu compare` table**: the deplodock kernel and the
  torch/cuBLAS reference kernels side by side — duration, occupancy, SM/DRAM/FMA throughput, LSU inst count,
  smem bank conflicts, regs/thread in one aligned view. With a dump dir set, the raw CSV + parsed JSON also land
  in `61_ncu_metrics.{csv,json}`. Typical reads: low `occ%` → register pressure (check `regs`); high `lsu.inst`
  + bank conflicts → smem layout (try `PAD_SMEM` / `PERMUTE_LANES`); high `dram%` with low `sm%` →
  memory-bound, tiling or fusion issue; low `fma%` on a matmul → not on tensor cores (back to class 2).
- **Code reading** — every class-2/3 finding cites the responsible rule or gate as `file:line`; quote the gate
  condition, don't paraphrase it from memory.
- **Search-shortfall confirmation** — re-tune just the reproducer with more patience (no `--clean`, accumulate):

  ```bash
  deplodock tune <dump>/*_lowering_cuda.kernels/<kname>.torch.json --patience <2-4x default>
  ```

  If the better variant becomes the pick, the finding is "patience/prior", not codegen.
- **Before/after across runs** — after any fix or re-tune, diff the two dump dirs instead of eyeballing logs:

  ```bash
  deplodock compare <dir-before>/dump <dir-after>/dump
  ```

  Matched kernels print A/B ratios (re-tuned hashes pair as `old -> new`); one-side-only kernels flag kernel-set
  changes (structural fork / fusion differences).

## Step 5 — write the report

Save to `plans/<model-slug>-layer<N>-tune-findings.md` (drop `-layer<N>` for whole-model). Mirror the reference
doc's structure:

- **Header**: status line, the exact run command, date, run stats (wall time, benched variants, ok/`bench_fail`
  counts), the -O3-vs--O1 disclaimer ("numbers below are the `--bench` -O3 re-bench; tune-DB latencies
  quoted for ranking context are -O1"), and whether the run was dynamic (symbolic `seq_len`, benched at the
  512 hint — the default) or static (shape-specialised).
- **Bench results**: both tables (full model + per-kernel with the Layer-op column), then one sentence naming
  the dominating kernels and their combined µs.
- **One `## Finding N — <title>` per root cause**, ordered by µs at stake. Each finding carries: symptom (the
  numbers), evidence (NCU compare rows, `eval variants` rank lines, `eval failures` clusters, emitted-CUDA
  observations), root cause or the best hypothesis with the distinguishing diagnostic (cite the gate as
  `file:line`), a repro command (reproducer path, `--ab` specs, and/or pinned knobs), and a suggested fix with
  priority.
- **Repro / artifacts** tail: log + dump locations and copy-pasteable repro blocks (compile-only repros that
  need no GPU are the most valuable).
- Wrap to ~120 chars (repo-wide markdown rule); tables may overflow.

## Step 6 — workflow retrospective (always include)

End the report with a `## Workflow notes` section — suggestions for improving this analysis loop itself, written
for whoever maintains the deplodock CLI and this skill. While working, note every point of friction as it
happens (don't reconstruct from memory at the end); then report, with concrete numbers where possible:

- **Slow steps**: anything that dominated wall-clock (a tune phase, an NCU pass, repeated re-lowering) — what
  took how long, and whether a flag, cache, or narrower scope could cut it.
- **Many-step detours**: any answer that took several tool calls / commands / file reads to assemble by hand —
  that's a missing command or a missing column in an existing view (this is exactly how `eval variants`,
  `eval failures`, `run --ab`, the `ncu compare` table, and `deplodock compare` came to exist).
- **Flakiness**: commands that needed retries, GPU-state hiccups (hung kernels, dirty contexts), run-to-run
  variance that forced re-benching — say which step and how often.
- **Output friction**: tables that were hard to read or paste into the report, missing units/labels, data that
  existed only in a log or HTML but not machine-readable, names that had to be cross-referenced by hand.
- For each item: one line of symptom + one line of proposed improvement (new flag, new `eval` view, doc fix,
  skill-step change). If a previous findings report's workflow notes exist, say which of those got fixed and
  whether the fix held up.

## Notes

- A finding must be actionable: it either names the code gate responsible or states the diagnostic that
  separates the competing hypotheses. "Kernel X is slow" with no class is not a finding.
- Variance caveat: scalar-tier tuned picks swing run-to-run (prior pick-reachability), so per-kernel reproducer
  rows — not the full-model total — are the stable before/after signal. Re-run the reproducer bench once before
  reporting a surprising number; `deplodock compare` makes the cross-run check one command.
- Known-limitation findings (e.g. a warp-tier gate that is correct-by-design) still go in the report when the
  kernel is a big slice of the total — recorded with the µs at stake, like the reference doc's finding 5.
