---
name: update-goldens
description: Use this skill when the user asks to "update the golden configs", "re-tune the goldens", "run the golden sweep", "refresh golden matmul configs", "evaluate goldens and update", or otherwise wants to re-tune the GOLDEN_CONFIGS matmul shapes, A/B the greedy pick against the recorded golden, and record genuine improvements into the per-GPU golden YAML. Tunes the whole golden dataset, benches greedy-vs-golden per shape with `deplodock tune` / `run --bench`, categorizes (better → replace, same → add, worse → leave), and edits the goldens YAML by hand.
version: 0.1.0
---

# Evaluate and update the golden matmul configs

The golden set (`deplodock/compiler/pipeline/search/goldens/<gpu>.yaml`, e.g. `rtx5090_sm120.yaml`) is the
deployable-latency ground truth: per shape, the best-known knobs + the deplodock-vs-cuBLAS latencies. This skill
re-tunes every shape, checks whether the current deployable greedy pick beats the recorded golden, and records the real
wins.

Requires a CUDA GPU. The whole sweep is ~30 min (23 shapes). Goldens are **hand-maintained YAML** — never dump them with
PyYAML (it destroys the flow-style `{BN: 16, ...}` knob dicts and key order); edit the YAML text directly.

## Step 1 — Tune the whole golden dataset

```
deplodock tune --dataset golden --clean
```

This fans out one `deplodock tune --golden NAME` subprocess per shape (isolated so one bad shape can't abort the sweep),
`--clean` on the first so the learned prior is rebuilt fresh then accumulated across the set. Narrow with
`--kernel SUBSTR` (e.g. `--kernel square` or `--kernel q_proj`) to re-tune a subset. This trains the prior; it does not
update goldens.

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

## Step 6 — Validate

```
deplodock/../venv/bin/pytest tests/compiler/test_golden_configs.py -q
```

This loads every YAML and checks the schema + derived properties. Spot-check a few entries:

```
./venv/bin/python -c "from deplodock.compiler.pipeline.search.golden import goldens_by_name; \
print(goldens_by_name('square.512'))"
```

## Notes

- The isolated `scripts/find_golden_configs.py` regen uses a cold per-shape search that lands well below the warm-prior
  greedy pick, so it can't reproduce these wins — that's why this workflow records from `run --bench` rather than that
  script.
- Commit the YAML edit with a message listing each shape's before→after and why (better / same-diff), per the repo's
  contribution rules (feature branch, `make lint`, `make test`).
