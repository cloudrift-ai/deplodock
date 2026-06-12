---
name: tune-golden
description: Use this skill when the user asks to "tune the goldens", "update the golden configs", "re-tune the goldens", "run the golden sweep", "refresh golden matmul configs", "evaluate goldens and update", "tune/seed the dynamic goldens", or otherwise wants to re-tune the GOLDEN_CONFIGS matmul shapes (static and dynamic/.dynM symbolic-axis entries alike), A/B the greedy pick against the recorded golden, and record genuine improvements into the per-GPU golden YAML. Tunes the whole golden dataset, benches greedy-vs-golden per shape with `deplodock tune` / `run --bench`, categorizes (better → replace, same → add, worse → leave), edits the goldens YAML by hand, and writes a findings report to plans/ — unlike tune-model, the target config is known here, so the report analyzes the analytic/learned prior's expectation against it (rank, per-knob misses, recommendations) plus workflow notes.
version: 0.3.0
---

# Evaluate and update the golden matmul configs

The golden set (`deplodock/compiler/pipeline/search/goldens/<gpu>.yaml`, e.g. `rtx5090_sm120.yaml`) is the
deployable-latency ground truth: per shape, the best-known knobs + the deplodock-vs-cuBLAS latencies. This skill
re-tunes every shape, checks whether the current deployable greedy pick beats the recorded golden, records the real
wins, and writes a findings report to `plans/` — which shapes the search/prior couldn't reach, why, what to do about
it, and how this workflow itself could improve.

What sets this apart from the `tune-model` skill: here the **target config is known** (the recorded golden), so the
analysis can evaluate the *expectation* directly — where the golden ranks under the analytic and learned priors, which
knobs the greedy pick misses, and whether the search ever measured it. Use the `deplodock` CLI for all of it (`tune`,
`run --bench --golden/--ab`, the `eval` views) — no ad-hoc bench scripts or hand-written SQL.

Requires a CUDA GPU. The whole sweep is ~30 min (23 shapes). Goldens are **hand-maintained YAML** — never dump them with
PyYAML (it destroys the flow-style `{BN: 16, ...}` knob dicts and key order); edit the YAML text directly.

## Step 1 — Tune the whole golden dataset

```
deplodock tune --dataset golden --clean
```

This tunes every golden shape in one in-process loop (`handle_tune` over `_tune_targets` — the same codepath as a
single-shape tune, differing only in the dataset it builds), sharing one tune DB / bench worker / prior. `--clean`
clears the DB + prior once up front, then accumulates across shapes. Narrow with `--kernel SUBSTR` (e.g.
`--kernel square` or `--kernel q_proj`) to re-tune a subset. This trains the prior; it does not update goldens.

## Step 2 — A/B the greedy pick vs the recorded golden, per shape

For each shape, run the deployable comparison:

```
deplodock run --bench --golden NAME
```

The kernel table shows the **greedy pick** row (what `run`/`compile` deploy) and one `golden NAME` row per recorded
config, **all benched live this run** — a real A/B. Knob columns are in the header; rows carry positional values.
Compare the greedy `us` against the *best* (min) golden row's `us`.

## Step 3 — Categorize each shape

Using the live A/B (greedy vs best golden row, same run):

- **better** — greedy >3% faster → **replace** the shape's golden(s) with the greedy config.
- **same** — within 3% AND different knobs → **add** the greedy config as an extra entry alongside the existing one(s).
- **same, same knobs** — greedy reproduces a golden → nothing to do.
- **worse** — greedy >3% slower → leave the golden untouched (the prior couldn't reach it).

## Step 4 — Confirm wins above the noise floor (critical)

The `golden NAME` row is a *live re-bench*, and it swings ~10–13% run-to-run for the smaller shapes. So a marginal
"better" (<~13%) can be the golden benching slow this run, not a real greedy win. Before recording any win:

- Re-run `deplodock run --bench --golden NAME` once or twice; only record wins that reproduce clearly above the ~10–13%
  noise band. Treat <~5% deltas as noise (no change).

## Step 5 — Edit the goldens YAML

Open the per-GPU file (`goldens/<gpu>.yaml`). Each entry is:

```yaml
  - kernel: matmul
    name: qwen3_06b.q_proj.s128
    M: 128
    N: 2048
    K: 1024
    knobs: {BN: 16, BM: 8, FM: 4, FN: 2, FK: 1, BK: 64, SPLITK: 1, BR: 1, STAGE: '11', RING: 3}
    deplodock_us: 18.4
    cublas_us: 20.9
```

When recording a config:

- **`deplodock_us`** — the greedy pick's `us` from the **-O3** A/B (`run --bench`), not the -O1 tune ranking pass.
- **`cublas_us`** — reuse the shape's existing value. cuBLAS is the config-independent torch reference (true-fp32 SGEMM
  for fp32 shapes, HGEMM for `*.fp16`), so it doesn't change when our knobs change. The derived `ratio` / `golden` flag
  recompute from the two latencies.
- **`knobs`** — record only the **search knobs** the table shows (BM, BN, BK, BR, FM, FN, FK, WM, WN, SPLITK, RING,
  STAGE, MMA, WARPSPEC, OVERHANG). Drop the transport/codegen control flags the planner re-derives (GROUP_M, TMA,
  PIPELINE_STAGES, ASYNC_COPY, PAD_SMEM, HOIST_COMPUTE, NOATOMIC, VECTORIZE_LOADS, PERMUTE_LANES, INTERLEAVE_LOADS) and
  the `S_*` / `H_*` feature columns.
- A shape with the **same knobs** as an existing entry but a faster `us` is a codegen win — just lower that entry's
  `deplodock_us`, don't add a duplicate.

### Pruning

Keep multiple entries per shape **only** when they're at parity (within ~3%). When a replace makes an old entry
strictly slower (>3%), **delete** the old entry — don't keep superseded-slower alternates.

## Dynamic (`.dynM`) goldens

A matmul golden may mark its M axis **symbolic** — the kernel is then a masked-tile artifact (ceil-div grid +
boundary guard, one cached kernel for any runtime seq_len), tuned and benched at the `Dim` hint. In the YAML this is
the `dynamic:` block; `M` doubles as the hint, and the name carries a `.dynM` suffix:

```yaml
  - kernel: matmul
    name: qwen3_06b.q_proj.s512.dynM
    M: 512            # the Dim hint — MUST equal DEFAULT_SEQ_HINT (512) today, see below
    N: 2048
    K: 1024
    dynamic: {seq_len: {input: x0, axis: 0}}
    knobs: {BN: 32, BM: 8, FM: 8, FN: 4, FK: 1, BK: 64, SPLITK: 1, BR: 1, STAGE: '11', RING: 2}
    deplodock_us: 52.0
    cublas_us: 53.5
```

**`M` must equal `DEFAULT_SEQ_HINT` (512)** — the pipeline tiles/benches a symbolic axis at the *global* Dim hint,
not at the traced size, so an M=1024 "dynamic golden" would silently be measured at 512 and duplicate the
(N, K, hint-512) shape (seen live in the 2026-06-12 seeding: a 1024³ symbolic-M trace produced the exact
`kv_proj.s512.dynM` kernel). The schema rejects other values until per-Dim hints are plumbed.

Everything in steps 1–7 applies unchanged — the spec is **part of the config**, so `tune --dataset golden`,
`tune --golden NAME`, and `run --bench --golden NAME` all apply it to the (re-)trace automatically. Never pass a CLI
`--dynamic` next to `--golden` (it's rejected). Specifics:

- A dynamic golden is a **separate shape** from its static twin (different deployment artifact, own variant space,
  own `ShapeKey` — `is_dyn` keeps them apart in every eval/diagnostics join). Never merge or cross-compare their
  entries; the static `qwen3_06b.q_proj.s512` and `…s512.dynM` rows coexist.
- The A/B benches **at the hint** (the table prints a `benched at seq_len=… (symbolic hint)` note); `cublas_us` is
  the hint-shaped torch reference, so the ratio is apples-to-apples *at the hint*. Knob comparisons work exactly as
  for static shapes.
- **Seeding a new dynamic shape** (no recorded entry yet): tune + bench via the snippet + an explicit spec —
  `deplodock tune -c "<matmul snippet>" --dynamic seq_len@x0:0` (accumulate into the existing DB/prior — no
  `--clean`), then `deplodock run --bench -c "<snippet>" --dynamic seq_len@x0:0`; record the greedy kernel's -O3
  `us` as `deplodock_us`, the same run's eager row as `cublas_us`, and the table's search knobs. `x0` is the
  snippet's lhs `(M,K)`; only the M axis may be symbolic (symbolic K is future work and the schema rejects it).
- `eval prior --dataset golden` prints a per-shape `SKIPPED: no tuned rows` line for any golden (dynamic or static)
  whose shape has no tuned data — a `.dynM` entry skipping there means the symbolic shape was never tuned, not that
  the join failed.
- The cold `AnalyticPrior` ranks `.dynM` shapes under a dedicated masked-tier weight set (`_W_A_DYN`, selected
  on the stamped `S_ext_n_symbolic_axis`), fit by `scripts/golden_knob_heuristics.py` over the recorded dynamic
  goldens (2026-06-12 refit: median cold rank 0). Re-run the script after recording new `.dynM` goldens — it
  prints both `_W_A` and `_W_A_DYN` to paste into `search/prior/analytic.py`.

## Step 6 — Validate

```
deplodock/../venv/bin/pytest tests/compiler/test_golden_configs.py -q
```

This loads every YAML and checks the schema + derived properties. Spot-check a few entries:

```
./venv/bin/python -c "from deplodock.compiler.pipeline.search.golden import goldens_by_name; \
print(goldens_by_name('square.512'))"
```

## Step 7 — write the findings report

Save to `plans/golden-sweep-<gpu>-findings.md` (e.g. `golden-sweep-rtx5090-findings.md`). Mirror the tone and evidence
density of the `tune-model` reports (`plans/*-tune-findings.md`; executed ones are deleted — recover the latest via
`git log --diff-filter=D --name-only -- 'plans/*findings*'` + `git show <commit>^:<path>`). Note friction and findings
**as they happen** during steps 1–6 — don't reconstruct from memory at the end.

- **Header**: date, GPU, the exact sweep command, wall time, and the category tally (N replaced / N added / N
  unchanged / N worse).
- **Per-shape outcome table**: shape name, greedy µs, best-golden µs, ratio, category. All numbers from the -O3
  `run --bench` A/B, never the -O1 tune DB.
- **One `## Finding N — <title>` per prior shortfall**, ordered by how far the pick lands from the golden. The
  "worse" shapes (greedy >3% slower) are the findings: the search/prior failed to reach a config it was literally
  shown. For each, gather the evidence:
  - `deplodock eval golden --kernel <SUBSTR>` — the per-knob `found/golden` diff: *which* knobs the greedy pick got
    wrong.
  - `deplodock eval analytic --kernel <SUBSTR>` — the golden's rank under the cold `AnalyticPrior` over the full
    enumeration. A deep rank here is the "poor analytic heuristic on this config" signal: the hand-coded weights
    misprice this shape, and patience can't reach it cold.
  - `deplodock eval prior --dataset golden --kernel <SUBSTR>` — the rank under the *learned* prior + the `vs gold`
    -O3 perf column. Deep analytic rank but shallow learned rank → the heuristic is the problem, not the search.
  - `deplodock eval variants --kernel <SUBSTR>` — was the golden config ever *measured* this sweep (reachability),
    and where the deployed pick ranks among measured variants.
  - **Before calling a knob mismatch a search shortfall**, check the `-O3 us` column in `eval variants` or run one
    `run --bench --ab "<golden knobs>"` A/B: -O1 ranking flags often invert at -O3, so an apparent pick miss can be
    the -O1/-O3 gap, not the prior.
- **Each finding ends with a recommendation**, with priority: refit the analytic weights
  (`scripts/golden_knob_heuristics.py`) when a shape family is systematically mispriced, a missing `D_*` engineered
  feature when the prior can't see what distinguishes the golden, a patience bump when the golden ranks shallow but
  the search stops early, or an enumeration/eligibility gate (cite `file:line`) when the golden's config was never
  offered at all.
- **End with `## Workflow notes`** — a retrospective on this loop itself, for whoever maintains the CLI and this
  skill: slow steps (what dominated the ~30 min and whether a flag/cache/narrower `--kernel` could cut it),
  many-step detours (answers assembled by hand across several commands — each is a missing `eval` view or column),
  flakiness (the noise-floor re-runs of step 4: which shapes, how many re-runs), and output friction (data only in
  logs, hand cross-referencing). One line of symptom + one line of proposed improvement each. If a previous sweep's
  report exists, say which of its workflow notes got fixed and whether the fix held.
- Wrap to ~120 chars (repo-wide markdown rule); tables may overflow.

## Notes

- The isolated `scripts/find_golden_configs.py` regen uses a cold per-shape search that lands well below the warm-prior
  greedy pick, so it can't reproduce these wins — that's why this workflow records from `run --bench` rather than that
  script.
- Commit the YAML edit with a message listing each shape's before→after and why (better / same-diff), per the repo's
  contribution rules (feature branch, `make lint`, `make test`).
