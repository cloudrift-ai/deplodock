# Deterministic resolution as a fold: `Run.resolve` + decision traces

**Status:** implemented 2026-06-10 (M1–M3 on `feature/resolve-trace-driver`) — follow-up to
`plans/structural-forks-in-two-level.md` (M3/M3b).

## Problem

Greedy compile is not a search, but it runs through the search abstraction, and every natural property of what it
actually is — a deterministic fold over the pipeline (at each fork, evaluate a pure function of `(options, op,
prior)`, take the argmin, continue) — has to be simulated with state on a policy object. The `Search` protocol
(`push`/`pop`/`observe`, token-threaded lineage) is the right abstraction for MCTS: there is genuinely a frontier to
rank, a tree to backprop, an order to decide. Greedy has none of that. Forcing it through `push`/`pop` produced the
bookkeeping inventory `policy/greedy.GreedySearch` carries today, each piece a workaround for the interface shape:

- `_pending` — a one-slot queue simulating "just return the choice" through an enqueue interface.
- `partition_scores` + the `PARTITION_RULE` name sniff inside `push()` — an out-of-band return value: a decision made
  inside an enqueue callback has no way to return anything to its caller, so the structural-pricing probe snoops the
  chosen score off a policy attribute keyed by rule name.
- `_price_memo`, `_price_structural` — the pricing probe drives a whole second pipeline (`_tile_pipeline()`) just to
  observe one decision the fold would have made anyway, then memoizes it and guards against recursing.
- `picked_structural`, `blocked` — after-the-fact feedback channels between `Pipeline.run`'s validity-retry loop and
  the policy, smuggled through constructor args and read-back attributes.

There is also a hidden cost: the generic drive spawns `LazyCandidate` siblings at every fork and the pended one
materializes via `resolve()`, which copies the whole graph. Sharing-by-snapshot exists for *sibling exploration*;
greedy drops every sibling immediately, so a whole-model compile pays one full graph copy per fork point for nothing.

The boundary that should be drawn (and that recent work kept converging on): facts about the *compilation* live in
the graph/IR — knob stamps as decisions and idempotence guards, `Op.source` as attribution, structural-decision
replay derived from both (`_replay_structural_decision`). Facts about the *process* live with the process: the MCTS
tree, frontier, visit counts are exploration state and stay in `TuningSearch`; a deterministic resolution's process
state should be a *return value* (a trace), not accumulated policy attributes.

## Design

Split the engine's one driver loop into two entry points sharing all the real machinery (matching, cursor, rule
batches, apply, structural-decision replay):

```python
# exploration (tune) — unchanged; Search remains the protocol for MCTS
for token, cand in run.drive(graph): ...

# deterministic resolution (compile, pricing probes, assembled-graph lowering)
terminal_graph, trace = run.resolve(graph, decide)
```

- **`ForkPoint`** — what `decide` sees at a multi-option rewrite: the `Match`, the raw `options` list (concrete
  `Op`/`Graph` leaves and lazy `Fork`s, exactly as `try_rewrite` returns them), the root op, and `ctx`. No
  `LazyCandidate`s: `resolve` has one live graph and applies in place — no sibling snapshots, no per-fork copies.
- **`decide: ForkPoint -> Op | Graph | Fork(leaf)`** — returns the option to apply (leaf Forks unwrap via the
  existing `_concrete_option`). The fork-tree flatten moves out of the policy into a pure helper over `Fork`s (the
  current `_leaves` works on `LazyCandidate`s only because the drive wraps everything in them).
- **`trace: list[Decision]`** — one entry per fork: `(rule_name, node_id, chosen_kind, knob_delta, score,
  n_options)`. The trace is the *only* output channel besides the terminal graph; everything that is snooped off
  `GreedySearch` attributes today becomes a trace query:
  - the structural price of a kernel = the `score` of its slice-resolve's entry at the partition fork
    (`partition_scores` and the in-`push` rule-name check disappear; the rule name is still the query key, but it
    lives at the one call site that asks the question);
  - `picked_structural` = "the trace contains a `Graph` choice";
  - compile diagnostics get a real artifact (which forks existed, what was picked, at what predicted cost) instead
    of needing `-vv` log archaeology.
- **`greedy_decide(prior, blocked, price_structural=True)`** — a factory in `policy/greedy.py` returning the decide
  function: flatten to complete leaves, skip blocked tile identities, `prior.mean_scores` argmin, and
  `_pick_structural` pricing (now calling `run.resolve` on the kernel slice and reading the trace, memoized per
  `op_cache_key` as today). This is all that remains of `GreedySearch`; the class is deleted, `Pipeline.run`'s
  docstring and the `Search` ABC stop mentioning a greedy policy. `Search.push(structural=)` stays — tune uses it.
- **`Pipeline.run`** keeps its retry semantics but expresses them as `decide` wrappers instead of side-channels:
  the validity fallback wraps `greedy_decide` with the blocked-tiles filter; the structural retirement wraps it with
  `price_structural=False`. The re-drive mechanism itself is kept deliberately: because greedy is deterministic,
  re-driving with one fork blamed replays every other choice identically — cheap non-chronological backtracking
  without graph snapshots or an undo log. With a backend, `Pipeline.run` benches the terminal via `_bench_terminal`
  once after `resolve` (today it inherits that from routing through `tune`).
- **Engine-shared pieces stay engine-shared**: `_replay_structural_decision` (identical offer sites take the same
  side) is consulted by both `drive` and `resolve`; `_is_structural_option` classification is unchanged; the rule
  batch / `is_alive` / cursor-advance semantics are one implementation used by both, never forked.

### What deliberately does NOT change

- `TuningSearch`, the MCTS tree, PUCT, the two-level outer/inner split, composed Σ rows — untouched. Tune still
  needs a real search; the protocol it has is right for it.
- The re-drive retry (see above). A snapshot-based "backtrack to the failing fork" was considered and rejected: the
  failing fork is identified only at a later pass (`KernelOp.validate` at materialize time), interleaved with other
  nodes' forks, so chronological backtracking would undo unrelated good choices; blame-the-node + deterministic
  re-drive is equivalent and simpler.
- Graph-maximalism is out of scope: the retry blocklist must outlive the graph it came from (the retry rebuilds from
  the original), and exploration state is about the process — neither belongs in the IR.

## Milestones (single branch, after PR #222 merges; commit after each `make test` passes)

1. **M1 — `Run.resolve` + `ForkPoint` + trace.** New entry point sharing the batch machinery with `drive` (extract
   the shared body rather than duplicating the loop); a `decide` that always returns option-0 reproduces today's
   no-prior compile. Inert: nothing routed through it yet. Tests: trace shape on a forked compile; in-place apply
   (no graph copies — assert terminal `is` the seeded graph object); structural replay consulted.
2. **M2 — port greedy.** `greedy_decide` factory; `Pipeline.run` and the structural-pricing probe move onto
   `resolve` (+ trace queries); the retry becomes decide-wrappers. Acceptance: kernel-set and knob picks bit-for-bit
   identical to `GreedySearch` on the no-GPU suites (cold and stub-prior paths — `test_split_demoted` greedy tests,
   `test_lowering_error_guardrail`, planner-memo tests all green unmodified); whole-model compile peak memory drops
   (no per-fork graph copies).
3. **M3 — delete `GreedySearch` + docs.** Remove the class and its dissolved state; update
   `pipeline/ARCHITECTURE.md` (Drivers + two-level sections), `policy/` docstrings, CLAUDE.md if wording drifts.
   `run_two_level_tune`'s final assembly goes through the same `Pipeline.run`, so it ports for free.

## Verification

- `make test` + `make lint` green at each milestone; the greedy behavioral pins above unmodified.
- A/B determinism check during M2 review: compile a handful of shapes (norm→linear f16, f32 matmul, an SDPA chain)
  under old and new drivers with the same prior file and diff terminal-graph digests — must be identical.
- GPU smoke: `deplodock run --code` on the norm→linear shape with a trained prior still deploys the split
  (structural pricing through the trace), and a cold run keeps one kernel.

## Relationship to other plans

- **`plans/structural-forks-in-two-level.md` M4** — independent, but M4's open question (b) (hoisted `SPLITK`
  candidates must match partition's divisor checks) points at the same deeper cut this plan stops short of: a
  first-class per-kernel enumeration API extracted from `_plan_kernel`, so "what configs can this kernel have and
  what does the prior predict" is callable without driving the partition rule at all. That would let the pricing
  probe skip pipeline-driving entirely (price = prior argmin over the enumeration) and is what `eval analytic`
  approximates for goldens today. Treat it as part of M4's planner work, not this refactor — `resolve` keeps the
  probe simple in the meantime.
- The trace is also the natural substrate for future compile explainability (`compile --explain`: per-fork options /
  pick / predicted µs) and for regression-diffing two priors' decisions without re-benching.
