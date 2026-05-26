# N-way Buffering for K-Outer Staged Matmul Kernels

## Context

Today the tile-lowering chain hardcodes a 2-deep ring buffer for K-outer staged matmul:
`040_use_ring_buffers.py` has `_BUFFER_COUNT = 2`, the pipelining schedule in `015_pipeline_k_outer.py` issues a
single-iteration prologue and uses `AsyncWait(keep=len(stages))` (which is structurally `keep = (N-1)·len(stages)`
with `N=2`), and the smem budget check is `2 × Σ slab_bytes ≤ ctx.max_dynamic_smem`. The `BufferedStage`
dataclass at `deplodock/compiler/ir/tile/ir.py:441-468` already carries `buffer_count: int` with `>= 2` enforced,
and the materializer at `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` reads
`stage.buffer_count` for both cp.async smem sizing (line 813-815) and TMA smem + mbar array sizing
(line 333-343). The TMA per-slot mbarrier path in `015` already computes `slot = k_var % buffer_count` and
`phase = (k_var / buffer_count) % 2` (line 196-200) — i.e. it is *already generalized* for arbitrary buffer
count, just never exercised with `buffer_count > 2`.

Goal: let the compiler pick `buffer_count ∈ {2, 3, 4}` per-tile, gated on smem fit and K-outer trip count, and
make the pipelining schedule's prologue length scale with `buffer_count - 1`. `DEPLODOCK_BUFFER_COMPUTE`
(line 232 of ARCHITECTURE.md) is the orthogonal "also ring-buffer the fused compute Stage's output" experimental
toggle; this plan reuses the same `_BUFFER_COUNT` for the compute slab when that toggle is set, so the two
features compose without a separate decision.

**Risk note up front.** Step 4 (the `015` pipelining rewrite) carries the riskiest perf impact: an off-by-one in
`AsyncWait(keep=...)` either makes the steady-state wait for too few in-flight loads (data race, wrong results)
or too many (no overlap, perf regression to single-buffer). The TMA path has the same risk on `MbarrierWait`
slot/phase. Both must be verified against a hand-traced 3-iter / N=3 schedule and the existing 2-iter / N=2
baseline before flipping any default.

### Design decisions

1. **Where buffer_count is chosen.** `040_use_ring_buffers.py` picks the largest `N ∈ {4, 3, 2}` whose
   `N × Σ slab_bytes ≤ ctx.max_dynamic_smem` and whose `K_outer.extent ≥ N`, no fork. Rationale:
   `Σ slab_bytes` is only known post-staging (after `020_stage_inputs` + `070_pad_smem` run); the planner sees
   pre-staging axis extents and would have to re-estimate slab sizes, double-counting the padding pass.
   Greedy-largest-that-fits is monotone (deeper buffers never hurt latency hiding once smem fits), so the
   autotune cartesian doesn't gain anything from a `BUFFERS` knob — and adding one would 3× the matmul variant
   count (today ~6 priority-survivors × 3 buffer choices = 18×) for no measured win. When/if we discover N=3 is
   *not* monotone over N=2 on some shape (e.g. occupancy cliff from smem reserving an SM-resident block),
   promote to a knob with a 2-element `BUFFERS` hint.

2. **K_outer trip count.** Require `K_outer.extent ≥ N` strictly. Fallback chain: try N=4 → N=3 → N=2 → leave
   as plain `Stage` (no buffering). At `extent == N`, the steady-state loop has `extent - (N-1) = 1` iteration,
   which is still valid — `015` already handles 1-iter steady-state via the explicit `keep=` carry comment in
   its docstring. At `extent < N`, fall through to a smaller `N`.

3. **Pipeline rewrite (`015`).** Generalize: prologue substitutes `K_outer → 0, 1, …, N-2` (N-1 stage
   issuances), steady-state body issues `K_outer → K_outer + (N-1)` while computing `K_outer`, epilogue drains
   the trailing N-1 chunks. The cp.async `AsyncWait(keep=…)` count is `keep = (N-1) × len(stages)` — leaves
   exactly the N-1 most-recently-issued chunks' commits in flight. For N=2 this is `len(stages)`, matching
   today's code and docstring. The TMA wait path already parameterizes slot/phase by `buffer_count`, so it
   requires *no change* beyond carrying the larger `buffer_count` through `stage.buffer_count`.

4. **Materializer changes.** Confirmed: the materializer at
   `100_materialize_tile.py:343` (`full_extents = (stage.buffer_count, *stage.alloc_extents)`) and line 813-815
   already use `stage.buffer_count` as the leading smem dim, so cp.async smem sizing is N-buffer ready as-is.
   The TMA mbar array at line 333-338 sizes from `group_buffer_count[gid]`, which is `max(stage.buffer_count)`
   across the group — also N-buffer ready. The smem-index in the phase position is `stage.phase`, an `Expr`,
   so any modulus works. **No materializer code changes** are required; only an assertion-style smoke test.

5. **Smem budget check.** Generalize line 89 of `040_use_ring_buffers.py` from
   `_BUFFER_COUNT * sum(...) <= smem_budget` to a loop over candidate `N` values, picking the largest fit. The
   planner's pre-staging extent doesn't see padding, so it can't make this decision soundly — keep it in `010`
   where post-staging slab bytes are visible via `Stage.smem_bytes`.

6. **TMA / cp.async compatibility.** Both `050_use_tma.py` and `060_use_async_copy.py` already propagate
   `buffer_count=s.buffer_count` when narrowing the type. No code changes needed there. One subtle concern: the
   TMA mbar array length scales linearly with `buffer_count`, and on sm_90+ each mbar is 8 B; 4-buffer with 3
   stages per group costs 96 B of smem for mbars — negligible against the 227 KB budget but worth budgeting
   alongside slabs.

7. **In-loop Accum-read gate.** Confirmed orthogonal to buffer depth — the gate exists to reject in-loop
   online-softmax-style merges that compound fp32 drift across reordered commits. Buffer depth doesn't change
   the algebra (drift comes from re-association of additions, same with N=2 or N=3). Keep the gate unchanged.

---

## Step 1 — Slab-fit picker helper in `040_use_ring_buffers.py`

**Why.** Centralize the "largest N that fits in smem and respects K_outer extent" decision so the rewrite
function reads `picked_count = _pick_buffer_count(...)` rather than branching on a hardcoded constant. Keeps
the rest of the rewrite single-path.

**Change.** Add a private `_pick_buffer_count(stages, k_outer_extent, smem_budget) -> int | None` returning
`max(N for N in (4, 3, 2) if N <= k_outer_extent and N * sum(s.smem_bytes for s in stages) <= smem_budget)`
or `None`. Note: `Stage.smem_bytes` returns the *unbuffered* slab — see `ir/tile/ir.py:397-407` (the buffered
subclass overrides with `× buffer_count`). Pass plain `Stage` instances at the gate (pre-promotion), so this
stays a one-line product.

**Files.** `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py` (~15 lines added).

**Verification.** Unit test in `tests/compiler/passes/test_double_buffer_n_way.py`: synthesize a `Stage` with
`slab_bytes = 30 KB`, set `ctx.max_dynamic_smem = 100 KB` → assert pick is 3 (3·30=90 ≤ 100, 4·30=120 > 100);
set `K_outer.extent = 2` → assert pick is 2 regardless of smem; set `extent = 1` → assert `None`.

## Step 2 — Wire picker into the rewrite gate

**Why.** Replace the hardcoded `_BUFFER_COUNT` constant and its use in the budget check + `BufferedStage`
construction with the picked value.

**Change.** In `_process_scope` (line 69-95): replace `_BUFFER_COUNT * sum(...) <= smem_budget` with
`picked = _pick_buffer_count(stages, int(s.axis.extent), smem_budget); if picked is None: continue`. Pass
`picked` into `_double_buffer(loop, buffer_count=picked)`. In `_double_buffer` (line 146-207): replace the two
`buffer_count=_BUFFER_COUNT` literals and the `Literal(_BUFFER_COUNT, "int")` in `phase` with the parameter.
Delete the `_BUFFER_COUNT` module constant.

**Files.** `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py` (~10 lines touched).

**Verification.** `make test` — existing `test_double_buffer_*` tests must still pass with no diff in their
emitted IR (smem budgets are sized so today's targets fit at N=2; if a small target happens to fit at N=3 it
will silently upgrade — re-bless that single golden if it occurs and confirm the upgrade is intentional). Add
a new test: a Stage with tiny slab + `K_outer.extent=8` + sm_90 budget → assert emitted
`BufferedStage.buffer_count == 4`.

## Step 3 — Generalize `AsyncWait(keep=…)` in the pipelining schedule

**Why.** The steady-state must leave the N-1 most-recently-issued chunks' commits in flight. The current
`keep=len(stages)` is the N=2 special case of `keep=(N-1)·len(stages)`.

**Change.** In `_pipeline` (line 161-232) of `015_pipeline_k_outer.py`: read
`buffer_count = stages[0].buffer_count` (already done at line 185 for the TMA path) before the cp.async branch;
in the cp.async branch at line 225 use `AsyncWait(keep=(buffer_count - 1) * len(stages))`. The TMA branch
(line 220) needs the same `(buffer_count - 1) * len(stages)` correction on `keep` — though the TMA
materializer ignores `keep` in favor of `phase`/`slot`, keeping `keep` honest preserves the fall-through to
cp.async semantics on materialize paths that haven't been updated.

**Files.** `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py` (~5 lines touched).

**Verification.** Hand-trace test in `tests/compiler/passes/test_pipeline_k_outer_n_way.py`: build a synthetic
K-outer loop with two `AsyncBufferedStage(buffer_count=3)`, run the rule, assert the steady-state body contains
`AsyncWait(keep=4)` (2 stages × (3-1) prologue chunks).

## Step 4 — Generalize prologue length to N-1 chunks

**Why.** For N=3 the prologue must issue chunks 0 *and* 1 (so the steady-state can read chunk i while chunks
i+1 and i+2 are in flight). Today the prologue is one chunk (substitutes `K_outer → 0`).

**Change.** In `_pipeline` (line 169-176): replace the single `sigma_first = Sigma({k_var: Literal(0)})` +
single-iteration `prologue` list with a loop emitting one prologue copy of each stage per
`p in range(buffer_count - 1)` with `sigma_p = Sigma({k_var: Literal(p, "int")})`. The main loop's extent also
drops by `buffer_count - 1` (line 226: `n_chunks - (buffer_count - 1)` instead of `n_chunks - 1`). The
epilogue's `sigma_last` substitutes `K_outer → n_chunks - 1` *and* the epilogue must drain *and re-execute*
the trailing `buffer_count - 1` chunks' reduce bodies, not just one — extend the epilogue loop to iterate over
`range(n_chunks - (buffer_count - 1), n_chunks)` substituting each `K_outer` value in turn.

**Files.** `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py` (~25 lines touched —
this is the substantive change in the pass).

**Verification.** Two-pronged. (a) Golden IR test: 3-stage, N=3, `K_outer.extent=6` → prologue has 2×3=6 stage
issuances, steady-state loops over `extent - 2 = 4` iters, epilogue computes chunks 4 and 5. Assert exact
statement counts. (b) Numerical: run an existing matmul correctness test (e.g.
`tests/integration/test_matmul_correctness.py` if it exists, else add one against a fixed BK / BN / BM combo)
with `DEPLODOCK_FORCE_BUFFERS=3` and assert max-abs err vs PyTorch ≤ existing tolerance. This is the
load-bearing correctness check for the whole plan.

## Step 5 — Env-var override for forced buffer count (debugging / A/B)

**Why.** Letting the user force `DEPLODOCK_FORCE_BUFFERS=3` (or `=2` to restore today's behavior) is the
fastest A/B comparison for the perf win. Matches the pattern of `DEPLODOCK_HOIST_COMPUTE` and other
knob-pin env vars in `pipeline/knob.py`.

**Change.** In `_pick_buffer_count` (Step 1): if `os.environ.get("DEPLODOCK_FORCE_BUFFERS")` is set, parse to
int and either (a) return it if it fits + respects extent, or (b) raise a clear `RuleSkipped` so the user
notices the requested depth was infeasible (don't silently fall back — that defeats A/B intent). Document in
`deplodock/compiler/pipeline/ARCHITECTURE.md` line ~232 alongside `DEPLODOCK_BUFFER_COMPUTE`.

**Files.** `040_use_ring_buffers.py` (~8 lines), `ARCHITECTURE.md` (~2 lines).

**Verification.** Unit test: `monkeypatch.setenv("DEPLODOCK_FORCE_BUFFERS", "3")` → assert N=3 picked; set to
"5" → assert `RuleSkipped`.

## Step 6 — Tighten `BufferedStage.__post_init__` assertion

**Why.** The class allows any `buffer_count >= 2`, but `015`'s schedule embeds the modulus expression and
`010`'s phase expression both assume a `buffer_count ∈ {2, 3, 4}` to keep the parity-flip / mbar slot
arithmetic legible. Loud-fail on unexpected values rather than silently miscompile.

**Change.** In `BufferedStage.__post_init__` (`ir/tile/ir.py:455-458`), tighten to
`if self.buffer_count not in (2, 3, 4): raise ValueError(…)`. Keep `2` in the allowed set explicitly.

**Files.** `deplodock/compiler/ir/tile/ir.py` (~3 lines).

**Verification.** Existing IR construction tests covering N=2 must pass; add a test asserting
`BufferedStage(buffer_count=5, …)` raises.

## Step 7 — Smoke benchmark + perf gate

**Why.** N=3 / N=4 is supposed to *win* on sm_90 (TMA mbarrier overhead amortizes better with deeper queues).
If it doesn't, we want to know before promoting it past the env-var-gated path.

**Change.** No code. Run `scripts/bench_matmul.sh` (or the equivalent — locate via
`find /home/dikobraz/Projects/deplodock/scripts -name "bench*"`) with `DEPLODOCK_FORCE_BUFFERS={2,3,4}` on a
representative TinyLlama-shape matmul and an SDPA-reduce kernel. Record the table in the PR description.

**Files.** None.

**Verification.** Pass criteria: at least one of N=3 or N=4 ≥ 1.05× speedup over N=2 on sm_90 for the
representative matmul. If N=3 regresses universally, scope down the plan: keep the picker but cap at N=2 by
default, ship `DEPLODOCK_FORCE_BUFFERS` as the opt-in escape hatch only, and re-investigate before making N>2
the silent default.

## Step 8 — Promote to default if perf gates pass

**Why.** Once Step 7's table shows N>2 is a win, remove the env-var requirement by letting the
"greedy-largest-that-fits" picker run by default.

**Change.** This is already how `_pick_buffer_count` works after Step 1 — the env var only *overrides* the
greedy pick. So Step 8 is just landing the commit with no env var set in CI, after a `make test` + `make lint`
clean run.

**Files.** None additional.

**Verification.** Full suite: `make test` and `make lint` clean. Spot-check three real model integration tests
(TinyLlama prefill, an SDPA test, an RMSNorm test) for max-abs-err parity within tolerance.

---

## Failure modes to watch

- **Autotune time explosion.** Avoided by *not* making `BUFFERS` a knob (per design decision 1). If we ever
  promote it to a knob, gate it behind a `KnobType.INT` with hints `(2, 3)` only (3-element max) and add a
  planner-level smem-fit prune so infeasible variants don't reach the fork. 6 base variants × 2 buffer choices
  = 12× (manageable).
- **Steady-state degenerate to 1 iter.** Today's code already handles `n_chunks - 1 == 1`; with N=3 and
  `extent == 3` we hit `n_chunks - (N-1) == 1` similarly. The "carry `keep` explicitly through unrolling"
  comment at lines 205-208 of `015` is the load-bearing invariant — preserve it verbatim.
- **TMA mbar smem cost.** N=4 with 3-stage SDPA group costs 32 × 3 = 96 B of mbar smem vs 16 × 3 = 48 B today.
  Negligible but include in the budget check if it ever matters (it doesn't on any current target).
- **`extent < 2` case.** The `is_matmul_k_outer` gate already requires `extent >= 2`; no new check needed. For
  `extent == 2` with picked N=3, the picker falls back to N=2 per Step 1.
- **Step 4 epilogue drain re-execution.** Easy to get wrong: the epilogue must re-execute the reduce body for
  chunks `n_chunks - (N-1)` through `n_chunks - 1`, one σ-substitution each, with appropriate `AsyncWait`s in
  between for the cp.async path or just mbarrier waits for TMA. Hand-trace this on paper for N=3 before
  writing the code.

## Test additions

- `tests/compiler/passes/test_double_buffer_n_way.py` — picker (Step 1), env override (Step 5), tight
  assertion (Step 6).
- `tests/compiler/passes/test_pipeline_k_outer_n_way.py` — `keep=` generalization (Step 3), prologue length +
  epilogue drain (Step 4).
- `tests/integration/test_matmul_n_buffer_correctness.py` — end-to-end correctness at N=3 and N=4 vs PyTorch
  reference, via `DEPLODOCK_FORCE_BUFFERS`.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py`
- `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py`
- `deplodock/compiler/ir/tile/ir.py`
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py`
- `deplodock/compiler/pipeline/ARCHITECTURE.md`
