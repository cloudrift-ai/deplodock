# Per-op autotuning by isolated single-op subgraphs

## Context

`deplodock tune` runs one SP-MCTS tree over the whole graph (`search/policy/mcts.py`). Because the pipeline applies
rules to the evolving candidate sequentially, fork points **nest**: every multi-option decision is re-explored under
every choice of the earlier ones — a search space that is the **product** `∏_k n_k` of the per-op variant counts,
scored by a single aggregate reward `1/total_us` with one global patience counter. This causes the documented
starvation ([[project_mcts_exploration_limit]]): the bottleneck op dominates the aggregate and global patience
expires before deep ops (sdpa.s512 P@V) reach good tiles.

**Verified key fact** ([[autotune_no_graph_forks]]): every multi-option fork in the pipeline today is an **op
fork** — an in-place `Op` rebind (`list[TileOp]` from pad_smem / stage_inputs / hoist; the `partition_loops` `Fork`
tree → TileOps). No rule emits a multi-option `Graph` fork. An op rebind leaves the graph (data flow, shapes, every
other op) unchanged, so each op's cost is **separable**: `T = Σ_k t_k` with `t_k` depending only on op `k`'s own
variant.

**Goal:** tune each op independently in its **own single-node graph** instead of cross-producting them in one tree.
Total bench cost drops from `∏_k n_k` to `Σ_k n_k`, and each op gets its own patience.

## Design

### Tune each op as an isolated single-node graph

Separability means an op can be compiled + benched on its own. So:

1. Run the **deterministic structural passes** (`frontend/*` + `loop/*`; no multi-option forks there) on the full
   graph once → the fused `LoopOp` kernels. These are the "ops".
2. For each LoopOp node, slice a **single-node graph** (the op + `InputOp` stubs for its inputs, its output marked)
   — the same op-provenance slicing already behind the per-kernel `.torch.json` reproducers
   (`compile --dump-dir`, consumed by `scripts/bench_model_kernels.py --tune`).
3. Run a **plain, unmodified `TuningSearch`** on that single-node graph. The subgraph holds exactly one op, so the
   existing MCTS explores only that op's forks (partition × pad × stage), with `patience` as the op's budget.
4. Persist results to the shared DB and mark the op terminated.

The standalone results transfer to the full graph for free because the DB is keyed **structurally**: `op_cache_key`
is name-invariant and a pure digest of body + knobs (`search/keys.py`). The sliced LoopOp's body is identical to its
in-graph form, so its forks produce identical `TileOp`/`CudaOp` keys and `lowering` edges. The final full-graph
`Pipeline.run` resolves every kernel to its tuned best via `_best_fork` (`pipeline.py:623`) — the same path
`deplodock run` already uses to consume tuning results ([[project_tune_run_integration]]).

### "Terminated" = tuned-to-requested-patience (skip already-tuned ops)

Per op, persist the tuning effort and gate on it — this is the "skip already-tuned ops" feature, needed regardless:

```
terminated(op) ⟺ recorded_effort(ctx, op.key) >= requested_patience
recorded_effort = ∞                  if the op's search EXHAUSTED its variants
                = requested_patience if it stopped on patience      (kept as max over runs)
```

So re-running at the same patience drills nothing (idempotent); a higher patience re-tunes only under-tuned ops
(incremental deepening); a shared DB across sessions skips already-tuned ops (resumable). "Is the whole graph
tuned?" is a pure DB query: `all(terminated(op) for op in ops)` — no runtime flag.

### The meta-loop (driver, in `commands/tune.py`)

```
fused = run frontend + loop passes on the full graph          # deterministic → LoopOp kernels
for op in fused.loop_op_nodes:
    if db.terminated(ctx, op.key, requested_patience):        # pure DB check — skip already-tuned
        continue
    sub = single_node_graph(op)                                # op + InputOp stubs (provenance slice)
    search = TuningSearch(patience=requested_patience)         # plain, unmodified
    drain Pipeline.tune(sub, search=search, db=db, backend=...)
    db.record_effort(ctx, op.key, ∞ if search.stop_reason != "patience" else requested_patience)
final = Pipeline.run(graph, db=db)                             # each kernel resolves to DB-best via _best_fork
```

Order is irrelevant (separable) — one search per op, no iteration. Benching is per-kernel (one launch), and an
already-tuned op is skipped by the DB check before any work.

## What this does NOT need (dropped from earlier drafts)

No changes to `TuningSearch`, `push`, `Fork`, or `Candidate`. No second policy, no `Fork.kernel` tag, no
`kernel_name` attribution, no per-kernel `observe` widening, no `focus`/`_drill_match`/`is_op_fork`/`drilled_this_pass`.
`TuningSearch` runs unmodified on a one-op graph; the aggregate reward is already per-op there since the graph has
one op. The greedy / DB / `_best_fork` / `_bench_terminal` machinery is reused as-is.

## Milestones (single branch, commit after `make test` — [[feedback_single_branch_milestones]])

- **M1 — Tuning-effort record in the DB** + unit tests: `record_effort(ctx, op_key, effort)` /
  `effort_for(ctx, op_key)` (max-kept, `∞` sentinel for exhausted, context-keyed). The "skip already-tuned"
  primitive; independently testable.
- **M2 — Single-node graph slice** for a LoopOp node + unit test (round-trips through lowering to the same
  `op_cache_key`s as in the full graph). Reuse the existing provenance slicer if it factors cleanly.
- **M3 — Meta-loop driver** in `commands/tune.py`: structural prefix → per-op slice → standalone `TuningSearch` →
  `record_effort` → final greedy emit. `--patience` documents "per op". Per-op `[tune]` progress logging.
- **M4 — Docs** (`search/ARCHITECTURE.md` + compiler ARCHITECTURE): per-op isolated tuning, separability rationale,
  effort-gated skip, structural DB handoff to greedy.

## Files to modify

- `deplodock/compiler/pipeline/search/db.py` — `record_effort` / `effort_for` + schema (`tuning_effort` keyed by
  `(context_key, op_key)`).
- `deplodock/commands/tune.py` — meta-loop: structural prefix, per-op slice, standalone tune, effort record, final
  emit; `--patience` help.
- Single-node slice helper — reuse / factor the existing op-provenance slicer (the one behind `.torch.json`); locate
  during M2 (likely in the dump / trace slicing path).
- Docs as in M4.
- No changes to `mcts.py` / `pipeline.py` fork handling / `Fork` / `candidate.py`.

## Tests

- `test_db.py`: `record_effort` max-keeps, `∞` for exhausted, context-keyed (seq=32 effort ≠ seq=512).
- Single-node slice test: a sliced LoopOp lowers to the same `op_cache_key`s / `lowering` edges as the full graph.
- Meta-loop test on a 2-kernel synthetic: tunes to `Σ_k n_k` benches (no product), both bests in `db.lowering`,
  `Pipeline.run` picks them up; **idempotency** (re-run same patience → no work) and **deepening** (higher patience
  re-tunes only under-tuned ops).
- Keep `test_greedy_db_lookup.py`, `test_tune_accuracy.py`, `test_thunk_forks.py` green (`TuningSearch` unchanged).

## Verification (end-to-end)

- `./venv/bin/pytest tests/compiler/pipeline/search/ tests/compiler/test_tune_accuracy.py -p no:randomly`.
- Real tune of the Qwen layer / sdpa.s512: each kernel tuned standalone, deep kernels (P@V) reach good tiles, total
  benches ≈ `Σ_k n_k`, per-kernel DB medians improve vs the old aggregate run; re-run is a no-op; then
  `deplodock run --ir … --bench` / `Pipeline.run` picks up the tuned variants.
- `make test` + `make lint` before each milestone commit.

## Risks / assumptions

- **Isolation bench vs in-context.** Tuning a sliced one-op graph uses synthetic seeded inputs and a cold-ish L2
  with no neighbor launches. Sound under separability (`t_k` is shape-driven, neighbor-independent) and identical to
  the tradeoff the existing per-kernel reproducer tuning already makes. The final full-graph `Pipeline.run` still
  measures the real assembled latency.
- **Structural DB handoff** relies on the sliced op producing the same `op_cache_key` / `lowering` `child_key` as in
  the full graph (true: `op_cache_key` is a name-invariant body+knobs digest, and the sliced body is identical) and
  on `_best_fork` re-matching it — the proven tune→run path ([[project_tune_run_integration]]).
- **Effort keying** is per `(context_key, op_key)`; confirm the slice preserves input shapes so the partition
  enumeration (and thus the keys) matches the full graph. "Exhausted vs patience-stopped" reads `search.stop_reason`.
- **Structural prefix enumeration**: fusion decides kernel boundaries, so the LoopOp set must come from running the
  (deterministic) frontend + loop passes on the full graph first, not from the raw input graph.
