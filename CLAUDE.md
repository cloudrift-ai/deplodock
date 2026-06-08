# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deplodock is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

The `README.md` is intentionally short — example-driven, no narrative. For details, consult the ARCHITECTURE.md files:

- **CLI usage** (deploy local/ssh/cloud, bench, teardown, vm, hardware-aware deploy, fixed-host mode, experiments, CI workflow) → [`deplodock/commands/ARCHITECTURE.md`](deplodock/commands/ARCHITECTURE.md)
- **Recipe format** (matrices/cross/zip combinators, variant filtering, deep merge, named fields, extra_args validation, command recipes, aggregate, docker_options, driver/cuda pinning, SGLang) → [`deplodock/recipe/ARCHITECTURE.md`](deplodock/recipe/ARCHITECTURE.md)
- **Compiler** (Graph IR dialects, passes, backends) → [`deplodock/compiler/ARCHITECTURE.md`](deplodock/compiler/ARCHITECTURE.md) and child docs

When the user asks about a CLI flag, recipe field, or matrix combinator, read the relevant ARCHITECTURE.md before answering — they hold details that are no longer in the README.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `DEPLODOCK_DUMP_DIR` environment variable (optional) — when set, all compiler stages dump intermediate artifacts (graphs, CUDA kernels, execution plans) to this directory for debugging. Per kernel, the dump also writes a `<kname>.torch.json` reproducer — the original PyTorch ops that kernel implements (sliced by op provenance), with an `i/N` coverage header (full vs partial) — runnable via `deplodock run --ir <kname>.torch.json --bench` to reproduce accuracy / latency vs torch for that op. Kernels are named after the ops they realize (`k_rms_norm`, `k_sdpa_reduce`)
- `DEPLODOCK_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path (`~/.cache/deplodock/autotune.db`). `deplodock tune` reads from / writes to this path. NOTE: the greedy DB→fork replay (`_best_fork`) that let `compile` / `run` pick a previously-tuned variant was **removed** in the learned-prior work; `compile` / `run` now pick forks from the global `Prior` — a `FallbackPrior` (`search/prior/`) that ranks by the learned `CatBoostPrior` once trained and the hand-coded `AnalyticPrior` cold (the `mean_score` argmin — lowest predicted latency; option-0 only if no prior loads at all) — not the DB. The learned half is a separate JSON checkpoint (`DEPLODOCK_PRIOR_FILE` → `~/.cache/deplodock/prior.json`); `tune` writes it, `compile` / `run` read it (the `AnalyticPrior` is fixed code, no file).

All `DEPLODOCK_*` config env vars (the two above plus `DEPLODOCK_NVCC_FLAGS`, `DEPLODOCK_DEBUG`, `DEPLODOCK_KNOBS`,
`DEPLODOCK_TUNE_PATIENCE`, `DEPLODOCK_TUNE_EPS`, `DEPLODOCK_O3_TOL`, `DEPLODOCK_BENCH_BACKENDS`, `DEPLODOCK_CUBIN_CACHE`,
`DEPLODOCK_NO_NVCC`, `DEPLODOCK_GPU_LOCK`,
…) are read and written through a single module, `deplodock/config.py` — the sole owner of `os.environ` for these vars.
CLI `--flag` overrides (e.g. `--nvcc-flags`) resolve via `config.set_nvcc_flags` inside the library, not in the command
layer, so programmatic callers and tests get the same precedence. The dynamic `DEPLODOCK_<KNOB>` namespace is owned by
`compiler/pipeline/knob.py` (which borrows `config.knob_var` / `config.knob_raw`); provider/secret vars stay with
`deplodock/redact.py`.

## Running Tests

```bash
make test
```

Or for a specific test file:

```bash
./venv/bin/pytest tests/test_recipe.py -v
```

When running a large subset (e.g. `tests/compiler/`), pass the same xdist flags `make test` uses to parallelize:

```bash
./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup
```

`-n auto` spawns one worker per core; `--dist=loadgroup` keeps tests sharing an `xdist_group` (e.g. CUDA context) on the
same worker.

## CLI Commands

- `deplodock deploy local ...` — deploy locally via docker compose
- `deplodock deploy ssh ...` — deploy to remote server via SSH
- `deplodock deploy cloud ...` — provision a cloud VM and deploy via SSH
- `deplodock bench recipes/* ...` — deploy + benchmark + teardown on cloud VMs (recipe dirs as positional args)
- `deplodock bench recipes/* --filter "KEY=PATTERN"` — run only variants matching the filter (fnmatch glob, repeatable, AND logic)
- `deplodock bench experiments/...` — run an experiment (results stored in the experiment dir)
- `deplodock teardown <run_dir>` — clean up VMs left running by `bench --no-teardown`
- `deplodock vm create gpu --gpu NAME --gpu-count N [--provider X] [--authorized-key PATH ...]` — create a VM by GPU name (orchestrator: retries, candidate fallback, orphan cleanup). `--authorized-key` (repeatable, also on `deploy cloud`) installs extra SSH public keys in the VM's authorized_keys alongside `--ssh-key`'s own `.pub`
- `deplodock vm create gcp ...` — create a specific GCP GPU VM (single-shot manual)
- `deplodock vm create cloudrift ...` — create a specific CloudRift GPU VM (single-shot manual)
- `deplodock vm delete gcp ...` — delete a GCP GPU VM
- `deplodock vm delete cloudrift ...` — delete a CloudRift GPU VM
- `deplodock pull <model>` — download a HuggingFace model to local cache
- `deplodock trace <model> [--layer N] [--seq-len N]` — trace a transformer layer (or the whole model if `--layer` is omitted) to Graph IR (JSON). Whole-model tracing patches HF's dynamic causal-mask construction via `trace.huggingface.build_full_model_wrapper`.
- `deplodock trace --code "EXPR"` — trace an inline `nn.Module` expression (last stmt must be a call, e.g. `"torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))"`)
- `deplodock compile <model_or_ir> [--layer N] [--seq-len N] [--dump-dir DIR] [--target sm_NN]` — run `decomposition → optimization → fusion` and save the fused `Graph[LoopOp]` (auto-pulls + traces if given a model ID; omit `--layer` for whole-model). `--target sm_NN` (e.g. `sm_80`, `sm_90`, `sm_120`) overrides the live device's compute capability so passes that gate on hardware features (TMA, cp.async) take the target's path.
- `deplodock compile --code "EXPR" [--ir STAGE]` — trace + compile an inline `nn.Module` expression in one step (same grammar as `trace --code`; last stmt must be a call)
- `deplodock compile <ir_file> --ir {torch|tensor|loop|kernel|cuda}` — print the requested IR stage to stdout. `loop` renders fused `LoopOp` bodies (post decomposition+optimization+fusion); `kernel` renders the per-kernel AST (post LoopOp→KernelOp lowering); `cuda` renders the per-kernel CUDA source (post KernelOp→CudaOp lowering).
- `deplodock compile ... --dynamic NAME@INPUT:AXIS` (repeatable) — make axis `AXIS` of the traced input named `INPUT` symbolic. Forwards to `torch.export(..., dynamic_shapes={INPUT: {AXIS: Dim(NAME)}})`; torch's SymInt propagation threads `Dim(NAME)` through every downstream FX tensor. The compiled CUDA kernel signature gains an `int <NAME>` runtime arg per dim; the launch resolves NAME from input array shapes (one cached kernel runs at any seq_len). A symbolic free axis is tiled for its `Dim` hint (`DEFAULT_SEQ_HINT=512`) and emitted as a **masked tile**: a ceil-div grid over the symbolic extent plus an `if (coord < NAME)` boundary guard, so the hint-sized tile shape stays correct at any runtime size (`tune` benches and `compile` picks the hint-sized variant; the backend benches symbolic graphs at the hint when no inputs are supplied). Cooperative-reduce (RMSNorm/softmax) and SDPA-prologue matmuls keep the symbolic axis degenerate — masking would register-tile a per-row reduction whose accumulator can't be shared across cells. Multiple specs sharing a NAME use the same `torch.export.Dim` instance (required so torch recognises e.g. `input_ids:1` and `attention_mask:2` as the same symbol). Examples: `--code` form `--dynamic seq_len@x:1`; whole HF model `--dynamic seq_len@input_ids:1 --dynamic seq_len@attention_mask:2 --dynamic seq_len@attention_mask:3 --dynamic seq_len@position_ids:1` (the whole-model wrapper switches to `dynamic=True` so mask + position_ids flow in as arg-positions). `--seq-len` only sizes the example tensors handed to `torch.export.export` (defaults to 32); skip it unless you want larger trace inputs. Per-layer trace specialises `cos/sin` and currently rejects `--dynamic`. Same flag accepted by `tune` and `run --code`. See `plans/dynamic-shapes.md`.
- `deplodock tune` (no model / `--code`) — **offline mode**: refit the global learned prior on its persisted reservoir
  dataset (no GPU, no benching) and print diagnostics — per-op **pick reachability** (does the prior's predicted-fastest
  config recover each op's measured-best config?), median ranking calibration (Spearman), and golden-matmul coverage. Use it to see
  whether the prior can actually reach the best configs it's been tuned on (`compiler/pipeline/search/prior/diagnostics.py`).
- `deplodock tune <model_or_ir|--code EXPR> [--patience N] [--ucb-c C] [--explore-eps E] [--bench] [-q]` — **two-level** autotune (see
  `compiler/pipeline/search/two_level.py`): an outer SP-MCTS over fusion forks (graph-changing `frontend`+`loop` passes;
  today deterministic → one terminal) whose reward is `Σ best-per-op time` from an inner search that tunes each
  post-fusion kernel **independently** in its own single-node slice (`lowering` passes only, so `Σ_k n_k` benches not
  the product). Per-op results key structurally (`op_cache_key`) so they transfer/share. The inner search runs for
  **every** op on every pass — never skipped on prior effort; replay is cheap, not gated: each benched terminal hits the
  per-variant `perf` cache, so an identical re-run (same prior) replays every variant with no GPU bench, while the
  ever-changing global prior can steer the same-patience search down a new trajectory and bench only the genuinely-new
  variants it surfaces (the old `op_effort` skip-already-tuned gate, which suppressed that re-exploration, is gone).
  Persists `perf` / `lowering` / inventory rows to the SQLite cache (path from `DEPLODOCK_TUNE_DB` or
  `~/.cache/deplodock/autotune.db`). The inner MCTS (PUCT over the global learned `CatBoostPrior`) stops on
  patience (N consecutive measured terminals without a new best). `--clean` nukes the tuning DB + cubin/kernel caches
  first. **tune compiles kernels at `-Xcicc -O1`** (fast nvcc compile — dodges a cicc/LLVM blowup on big unrolled
  register-tile kernels, up to ~200×) — but **-O1 is NOT runtime-optimal**: reduction/attention kernels can run 1.5–3×
  slower than -O3, so tuned latencies are a *ranking* signal, not deployable numbers (re-bench the winner with
  `--bench` below, or `run --bench`). To keep the **learned prior** deployable anyway, the engine **re-benches at
  `-Xcicc -O3`** every config **within `DEPLODOCK_O3_TOL` (default 10%) of the best -O1 so far** — not just a strict new
  global-best — and feeds each as an extra training row tagged `H_opt=3` (so `compile` / `run`, which run at -O3, rank by
  the deployable numbers — the -O1 sweep alone ties configs that differ at -O3, e.g. a reduction's `FK` or a warp tile's
  `WARPSPEC`). The tolerance band gives the prior an -O3 truth sample for every near-best contender, not only the winner;
  each config is re-benched at most once. See `plans/golden-sweep-report.md`.
  Override the opt level / flags with `--nvcc-flags "…"` (e.g. `-Xcicc -O3`); the
  flags are folded into the cubin cache key and the `perf` context key, so -O1-tuned and -O3 rows never clobber.
  On default verbosity (tty) a live single-line **progress bar** tracks completed/total tuned op leaves with a
  `<kernel> <current us> (best <best us>) <knobs>` tail — the current latency is fixed-width and the knobs sit
  last, so the prefix stays put as the per-variant latency changes (no flicker); `-v` shows the per-`[tune]` INFO
  lines instead, `-q` is quiet (errors only, no bar — the final summary still prints). `--bench` re-benches the tuned
  winner at **-O3** (deployable, not the -O1 ranking pass): the full model **against the real torch module** (eager /
  `torch.compile` / Deplodock, end-to-end) and each kernel via its provenance `.torch.json` reproducer (re-lowered so
  the tuned forks are picked) vs eager / `torch.compile` / Deplodock, then prints both comparison tables. The
  full-model bench is skipped when the input is an `--ir` JSON file (no module available); the per-kernel table still
  runs. `--bench-backends` defaults to `eager,tcompile,deplodock` (overrides the `run` default that drops tcompile —
  the ~0.8 s JIT is worth paying for the deployable comparison). `--warmup`/`--iters`/`--seed` mirror `run`. When a
  dump dir is set (`--dump-dir`/`DEPLODOCK_DUMP_DIR`) it also writes an HTML per-kernel chart to
  `<dump-dir>/kernels.html` (+ best-effort `.png`).
- `deplodock tune --dataset golden [--kernel SUBSTR] [--clean] [...]` — tune **every golden shape** (the built-in
  equivalent of looping `--golden NAME` over `GOLDEN_CONFIGS`; `--kernel SUBSTR` narrows by name). Single-shape and
  golden-set tune go through the **same** codepath: `handle_tune` builds a list of `(label, code, input)` targets via
  `_tune_targets` — the **only** place the two diverge (one target from `--code`/positional/`--golden NAME`, or the
  whole golden set from `--dataset golden`) — then loops, calling the shared `_tune_one` per target. The loop runs
  **in-process** with one shared tune DB, one bench worker, and the in-memory learned prior (no per-shape re-import):
  benching is already subprocess-isolated (`_tune_backend` sets `bench_wall_timeout_s` → each variant runs in a
  SIGKILL-able `_bench_worker`), so a wedged kernel dies with its worker and the parent stays clean shape-to-shape. A
  saturated-queue `RuntimeError` (dirty parent stream) aborts the remaining sweep (per-op bests are already in the DB; a
  re-run resumes). `--clean` clears the DB + prior **once** up front, then accumulates across shapes; `--bench`
  re-benches each tuned shape at -O3 (works per target — `os._exit` only fires at process end). `--dataset db` is
  rejected (DB rows have no shape to tune). Reuses the shared `--dataset` vocabulary from `commands/dataset_args.py`;
  drives the `update-goldens` skill's tune step.
- `deplodock run <model> [--layer N] [--seq-len N] [--bench] [--target sm_NN]` — trace + compile + execute a whole HuggingFace model (or one `--layer`) on the CUDA backend, check accuracy vs eager, and (with `--bench`) print a latency table comparing eager PyTorch / `torch.compile` / Deplodock end-to-end against the real torch module. Same positional / `--layer` / `--seq-len` grammar as `compile` / `tune`. NOTE: greedy `run` / `compile` pick forks from the global `Prior` (`FallbackPrior`: learned `CatBoostPrior` once trained, else the cold `AnalyticPrior`; `mean_score` argmin — lowest predicted latency), not the DB — so `run` numbers reflect the prior trained by a prior `tune` (and a sensible analytic cold pick before any `tune`). A `.json` positional behaves like `--ir`.
- `deplodock run --golden NAME [--bench]` — run the named golden config (shorthand for `--code <its snippet>`, same flag as `tune --golden`; unknown NAME lists the names). With `--bench` each recorded golden for the kernel's shape is **compiled with its knobs pinned and benched live this run** (`_bench_golden_variants`), then printed as a row labeled `golden NAME` in the Kernel column (its own measured µs / grid / block / smem / regs / occ; `%` column `--`, since it's not part of the deplodock TOTAL) right beneath the matching greedy-pick kernel — a real A/B, not the stored number. The knob columns are aligned across rows and colored like `deplodock eval` (shared `commands/table` — the knob name is the column header, cells carry the value only): a golden cell is red where it differs from the greedy pick. A golden NAME may map to **multiple** configs (one shape can carry several knob sets, e.g. a newly found faster variant beside the old); each is benched and shown (each re-traces a fresh graph — a frontend graph can't be re-compiled in place).
- `deplodock run --code "EXPR" [--bench] [--warmup N] [--iters N] [--target sm_NN]` — compile + execute an inline `nn.Module`/torch expression on the CUDA backend, check accuracy vs eager, and (with `--bench`) print a latency table comparing eager PyTorch / `torch.compile` / Deplodock. Same `--code` grammar as `compile --code`. `--target sm_NN` overrides the live device's compute capability (same flag as `compile`), so feature-gated passes take the target's path while the kernel still runs on the live GPU — e.g. `--target sm_80` lowers a matmul through the cp.async transport and `--target sm_70` through plain sync staging, both runnable on a newer card, which makes the TMA / cp.async / double-buffer rungs A/B-benchable on one GPU.
- `deplodock run --ir <file.json> [--bench]` — load a JSON IR dump (any stage), finish lowering, execute on random seeded inputs. For a **frontend-dialect** graph (e.g. a dumped `<kname>.torch.json` reproducer) it also builds a real-torch reference (`compiler/backend/torch_ref.py`) and prints the same accuracy check + eager / `torch.compile` / Deplodock table as `--code`; non-frontend IR (loop/tile/…) benches deplodock-only.
- `deplodock inspect <ir_file>` — display graph IR summary (op counts, inputs, outputs)
The `eval` subcommands share a `--dataset {golden,db}` vocabulary (`commands/dataset_args.py`): `golden` reads the
recorded `GOLDEN_CONFIGS`, `db` reads the tune DB's measured `perf` rows. Both flow through one read-view —
`compiler/pipeline/search/data/` (`Sample` / `Dataset` / `ShapeKey`) — which also backs the prior `fit` featurization
and the diagnostics grouping, so golden filtering, the DB join, and `knob_features` live in one place. Source is
orthogonal to analysis: a degenerate combo (e.g. `eval knobs --dataset golden`) fails fast with a specific message.

- `deplodock eval knobs [--dataset db] [--db PATH] [--min-variants N] [--kernel SUBSTR]` — knob-impact analysis from the
  autotune DB (`--dataset db`, the default; `--dataset golden` is rejected — goldens carry no kernel C identity): the
  registered knob schema, then (with a tune DB) per-knob regret + a knob-interaction matrix sorted by geomean impact
  (joins `perf` with `cuda_op` via `Dataset.from_db().group_by_kernel_name()`) — drives Fork-tree knob ordering.
- `deplodock eval analytic [--dataset golden] [--kernel SUBSTR]` — evaluate the cold-start **`AnalyticPrior`** (the
  hand-coded linear model over `knob.knob_features` that replaced `score_matmul_thread` / the `_priority_matmul_*`
  enumeration sort; the cold half
  of the ONE ranking path — see `compiler/pipeline/search/prior/`) on each `GOLDEN_CONFIGS` shape: the golden's **rank**
  under the prior over the shape's full enumeration (no GPU, no learned data, no measurements; the metric the tuner's
  patience must reach) + per-knob `found/golden` (mismatches in red), summarized as median + top-k. The
  `search/analytic.py` module is now just the golden-eval glue (`evaluate_golden` / `pick_matmul`) around the prior
  (`eval analytic` shows the matmul goldens; the prior also ranks the cooperative-reduce / pointwise goldens). Weights fit
  offline by `scripts/golden_knob_heuristics.py` (jointly over every kernel regime — matmul fp32/fp16, reduce, pointwise — tier-balanced).
- `deplodock eval prior [--prior PATH] [--dataset {golden,db}] [--db PATH] [--kernel SUBSTR] [--features]` — evaluate the
  learned `CatBoostPrior`. Default `--dataset golden`: the golden's rank under the prior over the full enumeration, then
  the greedy pipeline pick vs golden (per-knob `found/golden`). `--dataset db` instead reports the prior's pick
  **reachability** over the tune DB's *measured* variants (does the prior recover each op's measured-best leaf?) — the
  orthogonal counterpart to the golden views, reusing `diagnostics.reachability` over `Dataset.from_db().group_by_op()`.
  Reads the prior JSON (`DEPLODOCK_PRIOR_FILE` or `--prior`; option-0 when none loaded). `--features` (golden mode) also
  prints the exact regressor input per golden config (`knob.knob_features`: `S_*` structural/shape + `H_*` regime +
  tuning knobs; the shape enters only as coarse `S_ext_*` products/maxes, so the occupancy/CTA/reuse terms the prior
  needs are added as engineered `D_*` features). The golden `S_*` here is the full histogram (the shape's snippet is
  compiled and cached via `data.Sample.from_golden(compile_s_feats=True)`), matching what a DB-trained prior saw.
- `deplodock eval golden [--prior PATH] [--dataset golden] [--kernel SUBSTR] [--features]` — the greedy pipeline pick vs
  recorded golden, per config (the actionable "did the pipeline reproduce the golden knobs?" table only — no analytic-rank or
  rank-under-prior diagnostics; use `eval analytic` / `eval prior` for those). The view to watch while iteratively
  tuning golden shapes. `--features` still prepends the per-config regressor feature vector.
- `deplodock tune --golden NAME [--clean]` — tune the named golden config (shorthand for `--code <its snippet>`), so
  the learned prior can be built up one shape at a time: `tune --golden square.512 --clean`, then `eval golden`, then
  `tune --golden square.1024` (no `--clean`, to accumulate), then `eval golden` again. An unknown NAME lists the names.
- Quick test model (ungated, Llama arch): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- GPU benchmark model (ungated, 0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Block benchmark script: `python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32`
- Per-kernel chart: `python scripts/bench_model_kernels.py --model Qwen/Qwen3-Embedding-0.6B --layer 0` — compiles with a dump, benches each prov-named kernel from its `.torch.json` reproducer (eager / `torch.compile` overlaid where the kernel is torch-runnable — including linear/attention, whose transposed weights are matched via `load_ops`-replayed constants), and renders a per-kernel latency bar chart via `deplodock.visualize`. `--tune` autotunes each kernel first.

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips `perf`-marked tests; see `tests/perf/ARCHITECTURE.md`)
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`deplodock bench recipes/*`)
- `make bench-kernels` — run per-kernel perf comparison vs PyTorch (`tests/perf/`, requires CUDA)
- `make clean` — remove venv and generated files

## Documentation Conventions

**Wrap every `.md` file in the repo to ~120 characters.** This includes `README.md`, every `ARCHITECTURE.md`, every file
under `plans/`, every file under `docs/`, and any other markdown anywhere in the tree. Do NOT wrap at 70–80 characters —
that is the default markdown habit, and it is wrong for this repo. Aim for lines in the 90–120 range.

Table rows, ASCII diagrams, and long URLs may overflow past 120 if wrapping would hurt readability — that's the only
acceptable reason to go wider. Python code stays under 140 chars (Ruff-enforced).

## Contribution Instructions

IMPORTANT: You MUST follow ALL of these steps for EVERY code change. Do NOT skip any step.

### Writing code

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`) — NEVER commit directly to `main`
2. Write code following guidelines in `STYLE.md`, `README.md` and `ARCHITECTURE.md` files in respective folders
3. Add tests if reasonable (in `tests/` following `tests/ARCHITECTURE.md` guidelines)

### Before committing (MANDATORY — do NOT skip these)

You MUST complete ALL of the following checks before every commit. These are not optional:

4. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
5. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md` and compare
6. **Update `CLAUDE.md`** if general instructions are no longer accurate — READ this file and compare
7. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant `ARCHITECTURE.md` and compare
8. **Run tests**: `make test` — fix any failures before proceeding
9. **Run linter**: `make lint` — if it fails, run `make format` and re-check

### Submitting

10. Push and open a PR

# Behavioral Guidelines:

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
