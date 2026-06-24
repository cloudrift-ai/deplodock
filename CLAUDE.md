# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deplodock is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

The `README.md` is intentionally short — example-driven, no narrative. For details, consult the ARCHITECTURE.md files:

- **CLI usage** (deploy local/ssh/cloud, bench, teardown, vm, hardware-aware deploy, fixed-host mode, experiments, CI workflow) → [`deplodock/commands/ARCHITECTURE.md`](deplodock/commands/ARCHITECTURE.md)
- **Serving** (vLLM out-of-tree embedding plugin — deplodock-compiled kernels behind vLLM's `/v1/embeddings`; `serving` extra) → [`deplodock/serving/ARCHITECTURE.md`](deplodock/serving/ARCHITECTURE.md)
- **Recipe format** (matrices/cross/zip combinators, variant filtering, deep merge, named fields, extra_args validation, command recipes, aggregate, docker_options, driver/cuda pinning, SGLang) → [`deplodock/recipe/ARCHITECTURE.md`](deplodock/recipe/ARCHITECTURE.md)
- **Compiler** (Graph IR dialects, passes, backends) → [`deplodock/compiler/ARCHITECTURE.md`](deplodock/compiler/ARCHITECTURE.md) and child docs
- **Tile lowering** (LoopOp → TileOp: `enumeration` builds the block-DAG `TileGraph` + searches the `Schedule`,
  `assembly` assembles the tower; **purely algebraic moveset — no shape specializations**, dispatch on the carrier
  algebra `MAP`/`SEMIRING`/`MONOID`/`TWISTED_MONOID`) →
  [`deplodock/compiler/pipeline/passes/lowering/tile/ARCHITECTURE.md`](deplodock/compiler/pipeline/passes/lowering/tile/ARCHITECTURE.md)

When the user asks about a CLI flag, recipe field, or matrix combinator, read the relevant ARCHITECTURE.md before answering — they hold details that are no longer in the README.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `DEPLODOCK_DUMP_DIR` environment variable (optional) — when set, all compiler stages dump intermediate artifacts (graphs, CUDA kernels, execution plans) to this directory for debugging. Per kernel, the dump also writes a `<kname>.torch.json` reproducer — the original PyTorch ops that kernel implements (sliced by op provenance), with an `i/N` coverage header (full vs partial) — runnable via `deplodock run --ir <kname>.torch.json --bench` to reproduce accuracy / latency vs torch for that op. Kernels are named after the ops they realize (`k_rms_norm`, `k_sdpa_reduce`)
- `DEPLODOCK_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path (`~/.cache/deplodock/autotune.db`). `deplodock tune` reads from / writes to this path. NOTE: the greedy DB→fork replay (`_best_fork`) that let `compile` / `run` pick a previously-tuned variant was **removed** in the learned-prior work; `compile` / `run` now pick forks from the global `Prior` — a `FallbackPrior` (`search/prior/`) via `Prior.pick`: measured -O3 reservoir evidence first (`evidence_pick` — a config the tune proved fastest at `H_opt=3` beats any unmeasured extrapolation; the evidence ships inside the prior checkpoint, so greedy stays prior-only), the model argmin otherwise (the learned `CatBoostPrior` once trained, the hand-coded `AnalyticPrior` cold — a separate `_W_A_DYN` weight set ranks symbolic-axis masked-tile kernels; option-0 only if no prior loads at all) — not the DB. The learned half is a separate JSON checkpoint (`DEPLODOCK_PRIOR_FILE` → `~/.cache/deplodock/prior.json`); `tune` writes it, `compile` / `run` read it (the `AnalyticPrior` is fixed code, no file).

All `DEPLODOCK_*` config env vars (the two above plus `DEPLODOCK_NVCC_FLAGS`, `DEPLODOCK_DEBUG`, `DEPLODOCK_KNOBS`,
`DEPLODOCK_TUNE_PATIENCE`, `DEPLODOCK_TUNE_EPS`, `DEPLODOCK_O3_TOL`, `DEPLODOCK_ANALYTIC_TILT`, `DEPLODOCK_BENCH_BACKENDS`,
`DEPLODOCK_CUBIN_CACHE`, `DEPLODOCK_NO_NVCC`, `DEPLODOCK_GPU_LOCK`, `DEPLODOCK_SERVING_STATIC` (static batched serving))
are read and written through a single module, `deplodock/config.py` — the sole owner of `os.environ` for these vars.
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
- **Timing metrics**: `deploy` and `bench` time each rental/deploy phase (vm/remote provision, image pull, model
  download, model load + CUDA-graph warmup, smoke test, benchmark, teardown) via `deplodock/timing.py::PhaseTimer`.
  `bench` stores them in each task's `.json` (`"timing"` key) + `.txt` (`=== Timing ===` section) and prints a `TIMING`
  summary table; standalone `deploy` commands print the breakdown (display-only). See
  [`deplodock/commands/ARCHITECTURE.md`](deplodock/commands/ARCHITECTURE.md) → Timing metrics.
- `deplodock teardown <run_dir>` — clean up VMs left running by `bench --no-teardown`
- `deplodock vm create gpu --gpu NAME --gpu-count N [--provider X] [--provisioning-model {FLEX_START,SPOT,STANDARD}] [--authorized-key PATH ...]` — create a VM by GPU name (orchestrator: retries, candidate fallback, orphan cleanup). `--provisioning-model` overrides the per-GPU hardware-table default (`STANDARD` = on-demand) while keeping the orchestrator's full config.yaml/SSH treatment; GCP-only, ignored for CloudRift. `--authorized-key` (repeatable, also on `deploy cloud`) installs extra SSH public keys in the VM's authorized_keys alongside `--ssh-key`'s own `.pub`
- `deplodock vm create gcp ...` — create a specific GCP GPU VM (single-shot manual)
- `deplodock vm create cloudrift ...` — create a specific CloudRift GPU VM (single-shot manual)
- `deplodock vm delete gcp ...` — delete a GCP GPU VM
- `deplodock vm delete cloudrift ...` — delete a CloudRift GPU VM
- `deplodock serve <model> [--stock] [--bench] [vllm flags...]` — serve an embedding model via `vllm serve` with the
  deplodock plugin flags baked in (`--runner pooling --enforce-eager --hf-overrides …DeplodockEmbedModel…`, default
  `--max-model-len 4096`); unrecognized flags forward to vLLM (after a literal `--`, verbatim). `--stock` drops the
  plugin for the raw-vLLM baseline (same max-model-len → apples-to-apples A/B in two invocations). `--bench` makes it
  a one-shot benchmark: start server → wait `/health` → `vllm bench serve --backend openai-embeddings`
  (`--max-concurrency`/`--num-prompts`/`--random-input-len`/`--bench-seed`) → print results → shut down. Needs the
  `serving` extra.
- `deplodock pull <model>` — download a HuggingFace model to local cache
- `deplodock trace <model> [--layer N] [--seq-len N]` — trace a transformer layer (or the whole model if `--layer` is omitted) to Graph IR (JSON). Whole-model tracing patches HF's dynamic causal-mask construction via `trace.huggingface.build_full_model_wrapper`.
- `deplodock trace --code "EXPR"` — trace an inline `nn.Module` expression (last stmt must be a call, e.g. `"torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))"`)
- `deplodock compile <model_or_ir> [--layer N] [--seq-len N] [--dump-dir DIR] [--target sm_NN]` — run `decomposition → optimization → fusion` and save the fused `Graph[LoopOp]` (auto-pulls + traces if given a model ID; omit `--layer` for whole-model). `--target sm_NN` (e.g. `sm_80`, `sm_90`, `sm_120`) overrides the live device's compute capability so passes that gate on hardware features (TMA, cp.async) take the target's path.
- `deplodock compile --code "EXPR" [--ir STAGE]` — trace + compile an inline `nn.Module` expression in one step (same grammar as `trace --code`; last stmt must be a call)
- `deplodock compile <ir_file> --ir {torch|tensor|loop|kernel|cuda}` — print the requested IR stage to stdout. `loop` renders fused `LoopOp` bodies (post decomposition+optimization+fusion); `kernel` renders the per-kernel AST (post LoopOp→KernelOp lowering); `cuda` renders the per-kernel CUDA source (post KernelOp→CudaOp lowering).
- `deplodock compile ... --dynamic NAME@INPUT:AXIS` (repeatable) — make axis `AXIS` of the traced input named `INPUT` symbolic. Forwards to `torch.export(..., dynamic_shapes={INPUT: {AXIS: Dim(NAME)}})`; torch's SymInt propagation threads `Dim(NAME)` through every downstream FX tensor. The compiled CUDA kernel signature gains an `int <NAME>` runtime arg per dim; the launch resolves NAME from input array shapes (one cached kernel runs at any seq_len). A symbolic free axis is tiled for its `Dim` hint (`DEFAULT_SEQ_HINT=512`) and emitted as a **masked tile**: a ceil-div grid over the symbolic extent plus an `if (coord < NAME)` boundary guard, so the hint-sized tile shape stays correct at any runtime size (`tune` benches and `compile` picks the hint-sized variant; the backend benches symbolic graphs at the hint when no inputs are supplied). `run` / `tune` `--bench`'s full-model table also tiles the torch closures' example inputs out to the hint (`commands/run._hint_sized_inputs`) so eager / `torch.compile` / Deplodock all bench one shape, and the table prints a `benched at seq_len=… (symbolic hint)` note. Dumped per-kernel `.torch.json` reproducers keep the symbolic dims: `run --ir` / `tune --bench`'s per-kernel pass bench them at the hint against a hint-shaped torch reference — no `--dynamic` flag needed (and `--ir` rejects it). Symbolic M/N axes reach the **warp/MMA tensor-core tier** as masked mma.sync tiles (per-element store guards, clamped slab fills / gmem-direct loads, runtime ldm), and the structural splits (`005_split_demoted`) offer on symbolic-row **and symbolic-N** graphs (the rotary QK^T's symbolic-N key cone materializes canonically into a warp-tier consumer; the dynamic o_proj's collapsed attn-out splits into a contiguizing `xn` producer + warp-tier consumer; a symbolic dim var in a collapsed-reshape stride is a legitimate read, not an unmodeled-scope bail). A symbolic **K** (reduce) axis now ALSO reaches the warp tier as a **masked-K mma tile**: the K is tiled at the hint (`ceil(seq_len/(BK·atom_k))` `K_o` serial steps) and the final partial slab is **zero-filled in smem** — `_stage_expand` clamps the K gmem index for a safe read and zeroes the loaded value where the K coord `>= seq_len` (`(k < seq_len) ? v : 0`), so the mma accumulates zero past the runtime extent (a clamped *duplicate*, the M/N edge-clamp, would corrupt the reduction). A masked-K bundle is pinned to the **SYNC transport** (cp.async / ring buffers can't ternary a copied value — `040_use_ring_buffers` declines the ring); the `005_split_demoted` cut now **offers on symbolic K** too, so the SDPA P@V demotion un-fuses into a softmax-normalizing `xn` producer + a clean symbolic-K consumer that reaches the tensor-core tier (matching its static twin). The SDPA-prologue **fused** P@V (before that split) still keeps the symbolic axis degenerate at `FM=FN=1`; static-K prologue kernels (fused gated-MLP) and cooperative-reduce kernels also keep the symbolic axis degenerate — their staged pipelines can't coexist with the per-row guard (their deployment path is the split). Cooperative-reduce kernels regain CTA parallelism via **strided-cooperative rows**: static free axes thread-bind alongside the `BR` cooperative lanes (BN·BM > 1 with BR > 1; the combine is a segmented warp shuffle over each row's BR lanes, so those rows' BR clips to powers of two ≤ warp_size), so e.g. a symbolic-seq per-head q/k-norm deploys a `BN×BR` CTA instead of an 8-thread degenerate one. The masked-K mma tier (above) covers both a clean symbolic-K matmul and the **batched** P@V split-consumer (16 heads): `classify_matmul_operands` recognizes a B operand `xnb[head, k, n]` whose K dim sits after a leading batch axis (one var dim — the N output — follows K). Only the SDPA-prologue **fused** P@V (before the split) stays degenerate; flash-style fused symbolic-K attention remains future work. Multiple specs sharing a NAME use the same `torch.export.Dim` instance (required so torch recognises e.g. `input_ids:1` and `attention_mask:2` as the same symbol). Examples: `--code` form `--dynamic seq_len@x:1`; whole HF model `--dynamic seq_len@input_ids:1 --dynamic seq_len@attention_mask:2 --dynamic seq_len@attention_mask:3 --dynamic seq_len@position_ids:1` (the whole-model wrapper switches to `dynamic=True` so mask + position_ids flow in as arg-positions); per-layer (`--layer N`) the single spec `--dynamic seq_len@x:1` (the layer is traced through `trace.huggingface.build_layer_wrapper`, which slices precomputed rotary cos/sin in-graph instead of passing trace-seq-len-specialised `(cos, sin)` kwargs). `--seq-len` only sizes the example tensors handed to `torch.export.export` (defaults to 32); skip it unless you want larger trace inputs. Same flag accepted by `tune` and `run --code`. A **golden config** may record a symbolic M axis itself (YAML `dynamic: {seq_len: {input: x0, axis: 0}}` on a matmul entry; `M` doubles as the hint and must equal `DEFAULT_SEQ_HINT` (512) — the pipeline tiles a symbolic axis at the global hint regardless of trace size, so the schema rejects other values until per-Dim hints are plumbed; `.dynM` name suffix by convention — a dynamic golden is a separate deployment artifact from its static twin, own knobs + latency, never merged): `tune --golden NAME` / `tune --dataset golden` / `run --golden NAME` apply the recorded spec to every (re-)trace automatically, and a CLI `--dynamic` next to `--golden` / `--dataset golden` is rejected (the spec is part of the config, the same way `--ir` rejects it). See `plans/dynamic-shapes.md` and `plans/dynamic-shape-goldens.md`.
- `deplodock tune` (no model / `--code`) — **offline mode**: refit the global learned prior on its persisted reservoir
  dataset (no GPU, no benching) and print diagnostics — per-op **pick reachability** (does the prior's predicted-fastest
  config recover each op's measured-best config?), median ranking calibration (Spearman), and golden-matmul coverage. Use it to see
  whether the prior can actually reach the best configs it's been tuned on (`compiler/pipeline/search/prior/diagnostics.py`).
- `deplodock tune <model_or_ir|--code EXPR> [--patience N] [--ucb-c C] [--explore-eps E] [--gpus N | --devices 0,1,2] [--bench] [-q]` — **two-level** autotune (see
  `compiler/pipeline/search/two_level.py`): an outer SP-MCTS over structural forks (the graph-changing
  `frontend`+`loop` passes plus the pre-partition head of `lowering/tile`, where `005_split_demoted`'s keep-vs-split
  offer branches the outer tree — one terminal per kernel set; identical offer sites replay the trajectory's
  first decision, read off the graph via `Op.source` + the stamped decision knobs; a graph with no structural
  offers yields one terminal) whose reward is
  `Σ best-per-op time` from an inner search that tunes each post-fusion kernel **independently** in its own
  single-node slice (post-partition `lowering` forks only, so `Σ_k n_k` benches not the product). Greedy `compile` /
  `run` deploy a structural option when the *trained* prior prices its kernel-set Σ cheaper (cold compiles never
  change kernel sets) — see `plans/structural-forks-in-two-level.md`. Per-op results key structurally
  (`op_cache_key`) so they transfer/share. `--gpus N` / `--devices 0,1,2` fan the inner per-kernel search out
  across GPUs — one in-flight bench per device-pinned `_AsyncBenchWorker` on a single asyncio event loop
  (`two_level._inner_reward_async`); single-thread asyncio touches the shared DB / prior only between bench
  `await`s, so no locks, and the default single-GPU path is the byte-identical one-slot serial case. Parallelism
  is bounded by the unique-kernel count; devices must be homogeneous (one perf key per tune). The inner search runs for
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
  `-Xcicc -O3`** every config **within `DEPLODOCK_O3_TOL` (default 15%) of the best -O1 so far** — not just a strict new
  global-best — and feeds each as an extra training row tagged `H_opt=3` (so `compile` / `run`, which run at -O3, rank by
  the deployable numbers — the -O1 sweep alone ties configs that differ at -O3, e.g. a reduction's `FK` or a warp tile's
  `WARPSPEC`). The tolerance band gives the prior an -O3 truth sample for every near-best contender, not only the winner;
  each config is re-benched at most once. See `plans/golden-sweep-report.md`.
  Override the opt level / flags with `--nvcc-flags "…"` (e.g. `-Xcicc -O3`); the
  flags are folded into the cubin cache key and the `perf` context key, so -O1-tuned and -O3 rows never clobber.
  All tune/bench timings are **CUDA-graph-captured** (pure GPU time, no per-launch dispatch gaps); each `perf` row
  records its measurement mode in a `captured` column, and on write a captured measurement supersedes a wall-semantics
  one for the same key (never the reverse) — old rows keep serving replay/prior training and upgrade in place.
  Recorded goldens keep their original numbers until the next re-record.
  On default verbosity (tty) a live single-line **progress bar** tracks completed/total tuned op leaves with a
  `<kernel> <current us> (best <best us>) <knobs>` tail — the current latency is fixed-width and the knobs sit
  last, so the prefix stays put as the per-variant latency changes (no flicker); `-v` shows the per-`[tune]` INFO
  lines instead, `-q` is quiet (errors only, no bar — the final summary still prints). `--bench` re-benches the tuned
  winner at **-O3** (deployable, not the -O1 ranking pass): the full model **against the real torch module** (eager /
  `torch.compile` / Deplodock, end-to-end) and each kernel via its provenance `.torch.json` reproducer (re-lowered so
  the tuned forks are picked) vs eager / `torch.compile` / Deplodock, then prints both comparison tables. The
  full-model table and the per-kernel rows are timed under **CUDA graph capture** (pure GPU time — the op-by-op torch
  replay is otherwise dispatch-bound, and per-launch dispatch inflates small kernels everywhere); a bench that fails
  capture falls back to uncaptured timing for all its backends and is flagged in a table note. The
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
  drives the `tune-golden` skill's tune step.
- `deplodock run <model> [--layer N] [--seq-len N] [--bench] [--target sm_NN]` — trace + compile + execute a whole HuggingFace model (or one `--layer`) on the CUDA backend, check accuracy vs eager, and (with `--bench`) print a latency table comparing eager PyTorch / `torch.compile` / Deplodock end-to-end against the real torch module. The Deplodock row is a **whole-program** measurement — windows around replays of one CUDA graph holding every launch, the same semantics as the captured torch rows; the kernel table's `TOTAL` (sum of per-launch solo windows, which miss cross-kernel cache effects) prints beside a `whole-program (e2e)` footer. Same positional / `--layer` / `--seq-len` grammar as `compile` / `tune`. NOTE: greedy `run` / `compile` pick forks from the global `Prior` (`FallbackPrior` via `Prior.pick`: measured -O3 reservoir evidence first, else the model argmin — learned `CatBoostPrior` once trained, cold `AnalyticPrior` otherwise), not the DB — so `run` numbers reflect the evidence + prior a previous `tune` checkpointed (and a sensible analytic cold pick before any `tune`). A `.json` positional behaves like `--ir`.
- `deplodock run --golden NAME [--bench]` — run the named golden config (shorthand for `--code <its snippet>`, same flag as `tune --golden`; unknown NAME lists the names). With `--bench` each recorded golden for the kernel's shape is **compiled with its knobs pinned and benched live this run** (`_bench_golden_variants`), then printed as a row labeled `golden NAME` in the Kernel column (its own measured µs / grid / block / smem / regs / occ; `%` column `--`, since it's not part of the deplodock TOTAL) right beneath the matching greedy-pick kernel — a real A/B, not the stored number. The knob columns are aligned across rows and colored like `deplodock eval` (shared `commands/table` — the knob name is the column header, cells carry the value only): a golden cell is red where it differs from the greedy pick. A golden NAME may map to **multiple** configs (one shape can carry several knob sets, e.g. a newly found faster variant beside the old); each is benched and shown (each re-traces a fresh graph — a frontend graph can't be re-compiled in place).
- `deplodock run ... --bench --profile` — re-launch the run under `ncu` (curated counter set,
  `commands/run.py::_NCU_METRICS`) and print an **`ncu compare`** table: the deplodock `k_*` kernels and the
  torch/cuBLAS reference kernels side by side in one aligned table (duration, occupancy, SM/DRAM/FMA throughput, LSU
  inst count, smem bank conflicts, regs/thread) — the ncu child launches the eager forward too, so the reference rows
  are in the same capture. With a dump dir the raw CSV + parsed JSON also land in `61_ncu_metrics.{csv,json}`.
  Silently skipped when `ncu` isn't on PATH; typical failure is the NVIDIA perf-counter permission gate.
- `deplodock run ... --bench --ab "K1=V1,K2=V2"` (repeatable) — bench an extra variant with these knobs pinned (the
  `DEPLODOCK_KNOBS` grammar, `compiler/pipeline/knob.py::parse_knob_spec`) and print it as a live `ab KNOBS` row in the
  kernel table beneath the greedy kernel with the matching `S_*` shape signature (knob cells red where they differ from
  the greedy pick — the same machinery as the `--golden` A/B rows, generalized to ad-hoc knob dicts, so a tune-DB
  variant from `eval variants` can be A/B'd in one process instead of one `DEPLODOCK_KNOBS=...` run per config). Works
  with `--code` / `--golden` (fresh re-trace per config) and `--ir` (fresh reload + tail re-lowering per config;
  ignored with a warning on fully-lowered cuda IR — no forks left to pin). Requires `--bench`.
- `deplodock run --code "EXPR" [--bench] [--warmup N] [--iters N] [--target sm_NN]` — compile + execute an inline `nn.Module`/torch expression on the CUDA backend, check accuracy vs eager, and (with `--bench`) print a latency table comparing eager PyTorch / `torch.compile` / Deplodock. Same `--code` grammar as `compile --code`. `--target sm_NN` overrides the live device's compute capability (same flag as `compile`), so feature-gated passes take the target's path while the kernel still runs on the live GPU — e.g. `--target sm_80` lowers a matmul through the cp.async transport and `--target sm_70` through plain sync staging, both runnable on a newer card, which makes the TMA / cp.async / double-buffer rungs A/B-benchable on one GPU.
- `deplodock run --ir <file.json> [--bench]` — load a JSON IR dump (any stage), finish lowering, execute on random seeded inputs. For a **frontend-dialect** graph (e.g. a dumped `<kname>.torch.json` reproducer) it also builds a real-torch reference (`compiler/backend/torch_ref.py`) and prints the same accuracy check + eager / `torch.compile` / Deplodock table as `--code` — timed under CUDA graph capture (pure GPU time; falls back to uncaptured timing with a printed note if capture fails); non-frontend IR (loop/tile/…) benches deplodock-only.
- `deplodock inspect <ir_file>` — display graph IR summary (op counts, inputs, outputs)
- `deplodock compare <dumpA> <dumpB> [--tol 0.10]` — diff two dump dirs' bench results: the full-model backend table
  (`60_bench_compare.json`), the per-kernel deplodock -O3 latencies (`62_kernel_bench.json`, machine-readable per-kernel
  rows `tune --bench` now writes beside `kernels.html`), and the raw per-launch times (`60_benchmark.json`) as fallback.
  Kernels match by exact provenance name first, then base name with the trailing content hash stripped (order of
  appearance), so a re-tuned kernel whose hash moved still pairs and prints as `old -> new`; one-side-only kernels are
  listed as kernel-set changes (structural fork / fusion differences). Ratios outside `--tol` color green/red. The
  before/after view for compiler changes — per-kernel rows, not the full-model total, are the stable cross-tune signal.
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
  the greedy pipeline pick vs golden (per-knob `found/golden`) with a **`vs gold`** perf column — the deployable (-O3)
  latency of the prior's predicted-best **measured** config over the golden's recorded `deplodock_us`, read from the
  prior's reservoir with **no re-bench** (`diagnostics.golden_deploy_perf`). Both sides are -O3 (tuning re-benches every
  winner at `H_opt=3`), so the ratio is a real deployable comparison (<1.0 = the prior's pick beats golden); a shape with
  no -O3 reservoir row shows `—`. The shape key (`ShapeKey` — the single golden↔measured join key) splits on
  `S_dtype_f32`, so an fp32 square and its `.fp16` twin (same free-dim product / reduce extent) don't merge, and on
  the symbolic-axis flag (`S_ext_n_symbolic_axis`), so a `.dynM` golden never merges with its static twin (its key
  mirrors the 992 stamp: symbolic axes are excluded from the extent products). `--dataset db` instead reports the prior's pick
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
- `deplodock eval variants [--dataset db] [--db PATH] [--kernel SUBSTR] [--prior PATH] [--top N]` — per-kernel
  leaderboard of the tune DB's measured variants (leaf configs only, fastest first, knob columns in the shared
  `commands/table` view), with the config the global `Prior` would deploy marked `◄` + ranked (`pick: rank R/N, X.XXx
  of best`, flagged when >1.2x — the per-kernel drill-down behind `eval prior --dataset db`'s aggregate reachability),
  the kernel's `bench_fail` count in the header, and a `-O3 us` column where the prior reservoir holds an `H_opt=3`
  re-bench for the config (the -O3 re-bench feeds only the reservoir, never a `perf` row, so DB latencies are the
  tune's -O1 ranking numbers). `--dataset golden` is rejected (goldens carry no per-variant measurements). The view
  that answers "did the search/prior reach the best measured config for this kernel, and which knobs distinguish it?"
  without hand-written SQL.
- `deplodock eval failures [--dataset db] [--db PATH] [--kernel SUBSTR]` — the tune DB's `bench_fail` rows clustered by
  `(kernel, error)`, each cluster with its row count and the tunable knob assignments shared by EVERY failing row (the
  "all 28 failures have `TMA=1`" signal). The failure text comes from the `perf` table's `error` column (recorded by
  `_bench_terminal` on bench failure, whitespace-collapsed + truncated; pre-error-column DBs migrate additively on the
  next writer open and their old rows cluster under `(no error recorded)`) — no more tune-log grepping.
- `deplodock tune --golden NAME [--clean]` — tune the named golden config (shorthand for `--code <its snippet>`), so
  the learned prior can be built up one shape at a time: `tune --golden square.512 --clean`, then `eval golden`, then
  `tune --golden square.1024` (no `--clean`, to accumulate), then `eval golden` again. An unknown NAME lists the names.
- Quick test model (ungated, Llama arch): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- GPU benchmark model (ungated, 0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Block benchmark script: `python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32`
- Per-kernel chart: `python scripts/bench_model_kernels.py --model Qwen/Qwen3-Embedding-0.6B --layer 0` — compiles with a dump, benches each prov-named kernel from its `.torch.json` reproducer (eager / `torch.compile` overlaid where the kernel is torch-runnable — including linear/attention, whose transposed weights are matched via `load_ops`-replayed constants), and renders a per-kernel latency bar chart via `deplodock.visualize`. `--tune` autotunes each kernel first.
- New-model discovery: `python scripts/new_models.py [--since YYYY-MM-DD] [--text-only] [--include-supported] [--arena] [--json]` — lists open-weight models OpenRouter hosts (catalog entries with a `hugging_face_id`), verifies each on HuggingFace, and ranks the unsupported, recently-released ones by HF `trendingScore`/downloads/likes. Keyless + read-only (OpenRouter `/api/v1/models` + HF `/api/models/{id}`); excludes families already in `recipes/` (base-model match) and models older than `--since` (default ~90 days) by default. `--arena` adds LMArena Elo/rank from the `lmarena-ai/leaderboard-dataset` HF dataset (`text`/`latest`/`overall`, ~360 models, keyless) by fuzzy name-match, and lists the open arena models it couldn't link (fuzzy misses / outside the window). Triage feed for the `benchmark-new-model` flow.

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
