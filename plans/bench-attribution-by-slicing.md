# Bench refactor: per-kernel attribution by program slicing — e2e as the only measurement primitive

Status: **proposed** (design agreed in review of PR #227; not started). Follow-up to the finding-6 work on
`fix/finding6-e2e-bench` (whole-program e2e windows, `plans/qwen3-embedding-layer0-tune-findings.md` finding 6).

## Motivation

`benchmark_program` currently measures two different things in one loop: per-launch solo windows (each kernel
replayed back-to-back in its own captured batch graph, `CompiledProgram._graphs`) and — since the finding-6 fix —
a whole-program window (every launch in program order in ONE captured graph, `_e2e_graph`). The duality exists
because per-launch sums mis-attribute and don't compose into an end-to-end number, while attribution (which kernel
dominates, per-op DB rows, the knob table) still needs per-kernel figures.

The cleaner decomposition: **benching a program end-to-end is the only measurement primitive**. Per-kernel
attribution is not a second measurement mode — it is *program construction*: slice the compiled graph into
single-kernel programs (`single_node_graph`, which the autotune sweep and the dump's per-kernel reproducers already
use) and run the same `benchmark()` on each. The window mechanics (batch calibration, CUDA-graph capture, watchdog,
budgets) apply unchanged because every sub-program goes through the same path.

Why this is better:

- **One primitive, backend-portable.** A future backend implements "bench a program"; attribution comes free
  because slicing happens at the graph level (backend-neutral). No backend ever implements per-kernel timing.
- **`CompiledProgram` loses a mode.** `_graphs` (per-launch batch graphs) and `_e2e_graph` collapse into one
  captured graph per program; `iter_once`'s per-position event bookkeeping shrinks to one window.
- **Split-K pricing gets honest for free.** A multi-kernel slice (gemm + atomic-fixup) is priced by its slice-e2e —
  inter-kernel effects included — instead of a sum of solo windows. Today's sum is the less deployment-faithful
  number (see the cuda ARCHITECTURE.md note added on the finding-6 branch).
- **Kills the mis-attribution class.** Finding 6's `mm0`/`mm1` artifact (identical work, 5.1 vs 0.8 µs solo
  windows) can't happen when each kernel is benched as its own program with its own warmup and calibration.

## Current state (what already exists)

- `single_node_graph(graph, node_id)` (`search/slice.py`) slices one node with its inputs promoted to graph inputs.
  Works on every stage incl. cuda — `CompilerDump.dump_kernels` already writes a per-CudaOp `.json` slice, and the
  two-level inner search benches per-op slices this way. The sweep IS the proposed model already.
- `benchmark_program` measures whole-program e2e automatically for multi-launch programs (finding-6 branch).
- `BenchmarkResult` is backend-neutral (`backend/base.py`): `time_ms`/`min_ms` (Σ per-launch), `per_launch`,
  `e2e_ms`/`e2e_min_ms`.

## Target design

1. `benchmark_program(graph)` → one captured graph, whole-program windows, `time_ms`/`min_ms` ARE the program time.
   `per_launch`, `e2e_ms`/`e2e_min_ms`, and the `_graphs` machinery are gone. Single-launch programs are unchanged
   by construction (their solo window already is the program time).
2. New graph-level helper (CLI/pipeline layer, NOT backend): `kernel_slices(compiled_graph) -> list[(node_id,
   CudaOp, Graph)]` — one single-node slice per kernel-bearing node, reusing `single_node_graph`.
3. `run --bench`'s kernel table: bench each slice via the normal `backend.benchmark`, render µs/% from the slice
   results. The `%` column is "share of Σ slice times", with the true program e2e printed beside it (the Σ-vs-e2e
   footer relationship from the finding-6 branch survives, just with the roles explicit).
4. The interleaved comparison (`_bench_interleaved`): emmy side = whole-program bench (one number, same as the
   torch closures). The per-iter `on_iter` interleaving is unchanged.

## Migration list (consumers of `per_launch` / the old sum semantics)

Ordered so each step lands green:

1. **`Pipeline._bench_terminal`** (`pipeline.py:~1269`): today it zips `result.per_launch` with the slice's
   CudaOp nodes to persist a per-op `perf` row per kernel (and falls back to `time_ms / N` on count mismatch).
   Under the new model a multi-kernel slice yields ONE program time. Decision needed (see Open questions): either
   (a) persist the slice total against the *tuned* op only (the fixup kernels are part of that op's variant — the
   structurally honest choice, and what the variant ranking actually consumes), or (b) sub-slice multi-kernel
   slices once more and bench each kernel separately for the DB rows. (a) is recommended; it also makes the perf
   row equal the ranking signal. NOTE: this changes recorded latencies for split-K variants — old DB rows priced
   the gemm and fixup separately; bump/annotate the perf context key if mixing matters (the `captured` column
   precedent: additive migration, never clobber).
2. **`run --bench` kernel table** (`commands/run.py::_print_kernel_stats`, `times_by_idx` at ~478): take a
   `[(node_id, op, BenchmarkResult)]` list from the slice benches instead of `bench.per_launch`. The golden/`--ab`
   A/B rows (`g_times`, ~537) migrate identically (their benches are already separate programs — only the row
   extraction changes).
3. **`60_benchmark.json`** (`pipeline/dump.py::dump_benchmark`): replace the `per_launch` array with a
   `per_kernel` array of slice results (name, time_ms, samples-derived min). Keep reading the old key in
   `commands/compare.py::_compare_per_launch` for one transition (compare matches by kernel name, so the rename is
   the only break).
4. **`tune --bench` per-kernel table / `62_kernel_bench.json`**: already slice-based (provenance reproducers);
   no semantic change, but the "Emmy" number per slice becomes the slice e2e — strictly more honest, and
   `emmy compare` across the boundary will show small shifts on multi-kernel slices. Call it out in the PR.
5. **Backend cleanup**: drop `capture_launch_graphs`/`_graphs`/`_graph_batch_sizes`, the per-position event lists,
   and `LaunchTime`/`per_launch` from `BenchmarkResult` once nothing reads them. `iter_once` becomes "replay the
   program graph once per window" + the uncaptured warmup walk.
6. **Docs**: backend + cuda ARCHITECTURE.md (the per-launch-window sections and the finding-6 e2e section merge
   into one "bench primitive" section), CLAUDE.md run/tune bullets, the finding-6 resolution note.

## What must be preserved

- **Warmup guards stay per-launch.** The uncaptured warmup iters are where the hung-kernel watchdog and the
  zero-elapsed degenerate-launch guard name a *specific* kernel. Keep the warmup walk launch-by-launch (it already
  is — capture only ever covered measurement) so a hang inside an 18-kernel program still says which kernel.
- **One warm state per comparison table.** The eager/tcompile/emmy table is untouched (one interleaved loop).
  The kernel-table slice benches run sequentially with their own warmups — fine for attribution, but the table
  should keep the program-e2e footer so the Σ-vs-e2e gap stays visible rather than implied.
- **Sweep cost profile.** The inner search benches single-node slices — identical cost before/after. No new
  capture or windows are added anywhere on the tune hot path.
- **`run_program_debug`** (per-launch buffer snapshots for accuracy debugging) is orthogonal — it walks launches,
  not timing windows. Unaffected.

## Costs / risks

- Interactive `run --bench` pays N × (alloc + warmup) for the kernel table (compile amortized by the cubin cache).
  For an 18-kernel layer at the default warmup/iters this is seconds, not minutes; acceptable. If it grows, the
  kernel table can move behind a flag or reuse smaller iters.
- A slice benches on fresh deterministic inputs, not the in-context activations. Latency is shape-driven for these
  kernels, so this is the same trade the reproducer benches already make — but it's a (minor) semantics shift for
  the kernel table, worth one line in the table header.
- Slices of mid-program kernels must be constructible: `single_node_graph` promotes intermediate buffers to inputs
  and `_allocate` fills them deterministically — already exercised by the sweep and dump paths, but the kernel
  table will hit every kernel of every model benched, so expect edge cases (e.g. TMA descriptor prebuild on sliced
  inputs; cf. the open `run --ir` cuda-stage descriptor round-trip defect in the finding-6 findings doc).

## Open questions

1. `_bench_terminal` multi-kernel slices: option (a) one row for the tuned op vs (b) sub-slice per kernel — see
   migration item 1. Recommended (a); decide before touching the DB writer, and decide whether old split-K rows
   need a context-key bump or can coexist (they ranked under the old semantics; replay correctness matters more
   than absolute comparability).
2. Does anything still need `LaunchTime.samples` granularity (variance diagnostics) after the migration? If yes,
   slice results carry their own samples — the field moves, not disappears.
3. `ncu compare` row alignment (`--profile`): unchanged (NCU profiles real launches, not bench windows), but the
   per-kernel table it sits beside will be slice-ordered instead of launch-ordered — pick one ordering for both.

## Validation

- Unit: slice-bench of a 2-kernel chain ≈ sum of single-kernel slice benches (loose tolerance); program e2e of the
  chain ≥ max(slice times); single-launch program result identical pre/post refactor.
- The finding-6 live check, re-run: Qwen3-Embedding-0.6B layer 0 on the RTX 5090 — table standings within noise of
  eager 99 / tcompile 46 / emmy 51 µs e2e; kernel-table Σ within ~10% of the old per-launch Σ (48.7 µs), with
  per-kernel numbers now reproducible across runs (the mm0/mm1 5.1-vs-0.8 artifact gone — both gemms should read
  ~equal, cf. their NCU durations).
- A `--clean` tune of one golden shape end-to-end (sweep cost unchanged, DB rows written, prior trains).
- `emmy compare` between a pre- and post-refactor dump of the same model: full-model row stable, per-kernel
  diffs explainable by the slice semantics (call out in the PR).
