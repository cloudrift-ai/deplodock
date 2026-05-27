# Two-level autotuning: outer fusion MCTS + inner separable per-op search

## Context

`deplodock tune` runs **one** SP-MCTS tree over the whole graph (`search/policy/mcts.py`). Because the pipeline
applies rules sequentially, fork points **nest** — so the tree mixes two very different kinds of decision and
cross-products them:

- **op-variant forks** (tile / pad / stage choices for one kernel), and
- **graph/fusion forks** (how ops are grouped into kernels).

This product `∏` over the op-variant forks, scored by one aggregate reward with one global patience, causes the
documented starvation ([[project_mcts_exploration_limit]]): the bottleneck op dominates and patience expires before
deep ops (sdpa.s512 P@V) reach good tiles.

The two kinds of decision have opposite structure:

- **Op-variant forks are separable** ([[autotune_no_graph_forks]] — every multi-option fork today is an in-place
  `Op` rebind that leaves the graph unchanged): `T = Σ_k t_k`, each `t_k` depends only on op `k`'s own variant.
- **Fusion forks are NOT separable**: changing how ops fuse changes *which ops exist*, so it's a genuine
  whole-graph decision.

**Goal:** split the search into two levels. Factor the separable op-variant search *out* into per-op sub-searches;
keep an **outer MCTS over fusion forks** whose terminal reward is the sum of the best per-op times. Today fusion is
deterministic (no multi-option fusion forks) so the outer tree is trivial (one terminal), but this structure is
exactly what preserves the ability to **search fusion** when those forks exist.

## Design

### Outer MCTS — over fusion forks

The outer search drives only the graph-changing passes (`frontend/*` + `loop/*`, which contain any fusion forks). A
**terminal** is the state where the cursor reaches `partition_loops` (the first `lowering/tile` pass) — i.e. every
op is post-fusion and structurally final. ("Reached `partition_loops`", not "is a LoopOp": `loop/fusion/*` keeps
merging/splitting LoopOps, so a LoopOp mid-pipeline may still fuse.)

Each outer terminal is a candidate fused graph. Its **reward** is the whole-graph cost from the inner search:

```
reward(terminal) = 1 / Σ_{op ∈ terminal.kernels}  best_per_op_time(op)
```

backpropagated into the outer tree by the existing MCTS machinery, so it compares fused graphs and searches for the
best fusion.

### Inner search — separable per-op tuning

To evaluate an outer terminal, tune each of its kernels **independently in its own single-node graph**:

1. Slice the finalized kernel into a single-node graph (op + `InputOp` stubs) — the op-provenance slicing already
   behind the per-kernel `.torch.json` reproducers (`compile --dump-dir`, `scripts/bench_model_kernels.py --tune`).
2. Run a plain, unmodified `TuningSearch` on it — the subgraph holds one op, so MCTS explores only that op's forks
   (partition × pad × stage), with `patience` as the op's budget.
3. Persist per-op `perf` / `lowering` + the tuning **effort** to the DB.

Results are keyed **structurally** (`op_cache_key` = name-invariant body+knobs digest, `search/keys.py`), so they
transfer to the assembled graph unchanged AND are **shared across outer terminals**: two fusion candidates that
share an identical op reuse its tuning (DB hit), so fusion search only pays for the ops that differ.

### "Terminated" effort — skip already-tuned ops

Per op, gate the inner search on persisted effort (the "skip already-tuned" feature, needed regardless):

```
terminated(op) ⟺ recorded_effort(ctx, op.key) >= requested_patience
recorded_effort = ∞ if the op's search EXHAUSTED its variants, else requested_patience   (max over runs)
```

→ idempotent re-runs, incremental deepening at higher patience, resumable across sessions.

### Whole-graph reward

The whole-graph reward is **not** accumulated during either search incrementally — it is the **outer terminal
reward**, `Σ best-per-op time`, computed by running the inner tuning at that terminal. It is used (drives fusion
MCTS), not merely reported. Separately, after the best fusion is chosen, the assembled graph is benched once for the
**real in-context whole-graph latency**; comparing it to the `Σ` estimate is the **separability check** (a gap
exposes L2 / clock / launch coupling the isolated benches can't see).

### Driver

```
outer  = TuningSearch(patience=fusion_patience)               # over fusion forks only
struct = Pipeline.build([frontend, loop])                      # graph-changing passes; terminal at partition_loops
for fused in struct.search(outer, ctx, db):                    # each terminal = a candidate fused graph
    total = 0
    for op in fused.kernel_nodes:                              # post-fusion, finalized
        if not db.terminated(ctx, op.key, patience):
            sub = single_node_graph(op)
            drain Pipeline.tune(sub, search=TuningSearch(patience), db=db, backend=...)   # inner per-op MCTS
            db.record_effort(ctx, op.key, ∞ if exhausted else patience)
        total += db.best_per_op_time(ctx, op.key)
    outer.observe(1/total, "ok")                               # whole-graph reward → fusion search
best  = outer.best_terminal                                    # today: the single fused graph
final = Pipeline.run(graph, db=db, backend=...)                # assemble DB-best kernels; bench = real whole-graph
report(final latency vs total)                                 # separability check
```

**Today** there are no multi-option fusion forks, so `outer` yields one terminal and this reduces to "tune each op
once, sum, then assemble." The outer tree is the generalization that lets fusion forks plug in with no further
change to the inner search.

## What this needs / does NOT need

- `TuningSearch` is reused **unmodified at both levels** — outer over fusion forks (truncated `[frontend, loop]`
  pipeline, custom Σ-per-op reward), inner over one op's forks (single-node graph, built-in reward). No `push` /
  `Fork` / `Candidate` changes, no `Fork.kernel` tag, no `focus`/`is_op_fork`/`drilled_this_pass`.
- New: the per-op-reward function (slice + inner tune + sum), the DB effort/best-latency primitives, and the driver
  that iterates outer terminals computing the reward. The outer uses `Pipeline.search` directly (manual `observe`)
  rather than `Pipeline.tune`, since its terminal reward comes from inner tuning, not `_bench_terminal`.

## Milestones (single branch, commit after `make test` — [[feedback_single_branch_milestones]])

- **M1 — DB effort + per-op best-latency** in `db.py`: `record_effort`/`effort_for` (`∞` sentinel, context-keyed,
  max-kept) and a `best_per_op_time(ctx, op_key)` read. The "skip already-tuned" primitive; independently testable.
- **M2 — Single-node slice** for a finalized kernel + test (round-trips to the same `op_cache_key`s as in the full
  graph). Reuse the provenance slicer behind `.torch.json` if it factors cleanly.
- **M3 — Inner per-op reward function**: slice → standalone `TuningSearch` (effort-gated skip) → `record_effort` →
  return `Σ best-per-op time`. Test: 2-op subgraph set tunes to `Σ_k n_k` benches (no product), bests in DB.
- **M4 — Outer driver** in `commands/tune.py`: `[frontend, loop]` `Pipeline.search` with `TuningSearch`, terminal
  reward = M3, backprop; pick best fused graph; final assembled bench + separability report. (Outer trivial today.)
- **M5 — Docs** (`search/ARCHITECTURE.md` + compiler ARCHITECTURE): two-level search, separability, effort-gated
  skip, structural DB handoff/sharing.

## Files to modify

- `deplodock/compiler/pipeline/search/db.py` — effort + best-latency primitives (`(context_key, op_key)`).
- `deplodock/commands/tune.py` — two-level driver (outer fusion search + inner per-op reward + final bench).
- Single-node slice helper — reuse / factor the existing op-provenance slicer; locate during M2.
- Docs as in M5.
- No changes to `mcts.py` / `Fork` / `candidate.py` (`TuningSearch` reused at both levels).

## Tests

- `test_db.py`: `record_effort` max-keeps / `∞` exhausted / context-keyed; `best_per_op_time`.
- Single-node slice: a sliced finalized kernel lowers to the same `op_cache_key`s / `lowering` edges as the full graph.
- Inner reward on a 2-op synthetic: `Σ_k n_k` benches (no product), bests in DB, idempotency (re-run = no work),
  deepening (higher patience re-tunes only under-tuned ops); DB sharing (a shared op tuned once across two terminals).
- Outer driver: with no fusion forks, one terminal, reward = Σ per-op best; `Pipeline.run` picks the bests up.
- Keep `test_greedy_db_lookup.py`, `test_tune_accuracy.py`, `test_thunk_forks.py` green (`TuningSearch` unchanged).

## Verification (end-to-end)

- `./venv/bin/pytest tests/compiler/pipeline/search/ tests/compiler/test_tune_accuracy.py -p no:randomly`.
- Real tune of the Qwen layer / sdpa.s512: each kernel tuned standalone, deep kernels (P@V) reach good tiles, total
  benches ≈ `Σ_k n_k`, per-kernel DB medians improve vs the old aggregate run; re-run is a no-op; final assembled
  bench reports the whole-graph latency and matches `Σ` per-op (separability holds); `deplodock run --bench` agrees.
- `make test` + `make lint` before each milestone commit.

## Risks / assumptions

- **Outer is trivial today** (no multi-option fusion forks) — confirm the two-level structure adds no overhead for
  the single-terminal common case, and that it's the clean insertion point for future fusion forks.
- **Outer reward is an estimate** (`Σ` isolated per-op best). Sound under separability; validated by the final
  assembled bench. A gap is the in-context-coupling signal (future trigger for in-context re-tune).
- **Structural DB handoff + sharing** rely on the sliced/finalized op producing the same `op_cache_key` /
  `lowering` `child_key` as the full graph (true: name-invariant body+knobs digest; sliced body identical) and on
  `_best_fork` re-matching it — the proven tune→run path ([[project_tune_run_integration]]). Slice only post-fusion
  kernels (cursor past `partition_loops`), else the body — and key — differ and the handoff silently misses.
- **Effort keying** per `(context_key, op_key)`; the slice must preserve input shapes (partition enumeration depends
  on extents). "Exhausted vs patience-stopped" reads `search.stop_reason`.
