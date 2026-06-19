# Online softmax → flash-attention — design plan

Bring scaled-dot-product attention into a single fused kernel that tiles over the KV (reduce) axis and never
materializes the `[S_q, S_k]` score matrix, by teaching the reduce IR a **streaming reduce** with online softmax. This
plan covers the new IR primitive, where it lives in the pipeline, and an incremental path that validates each new
capability in isolation before wiring it into the SDPA nest.

Builds directly on the `ReduceCarrier` + algebraic-op-traits refactor (PR #261): the streaming combine is a new
`ReduceCarrier` subclass that advertises `associative=True`, so `is_reduce`, axis threading, split-KV, and
cooperative-combine recognize it with no new isinstance ladders.

## Why — what the current path does, what flash changes

`frontend/decomposition/010_sdpa.py` decomposes `SdpaOp` into `K^T → (GQA broadcast) → QK^T matmul → scale → +mask →
softmax → P@V matmul`. Softmax (`100_softmax.py`) becomes `max → sub → exp → sum → div` — two sibling K-loops over the
KV axis (a max sweep, then a sum-of-exp sweep that reads the max). The deployed kernel set comes from
`lowering/tile/005_split_demoted.py`, which un-fuses P@V into a softmax-prob producer `xn[m, k]` + a clean
(masked-)K gemm consumer.

That split path **materializes the full `[S_q, S_k]` prob matrix** — the O(N²) traffic flash exists to avoid. Flash is
the opposite move: fuse QK^T + softmax + P@V into one kernel, tile the KV axis, and keep only a running `(max, sum,
output)` per query row. The win is memory traffic + the dropped score materialization; it grows with sequence length
(modest at `DEFAULT_SEQ_HINT=512`, dominant at 4–32k). Confirm the target seq-len regime justifies the work before
committing — at short seq the split path is already close.

## The target loop nest (non-causal, static seq first)

```
for h in 0..H:                          # head                          (grid)
  for mq in 0..S_q:                     # query rows, tiled BR          (grid)
    Init(m_i = -inf)                    # running row-max   [BR]
    Init(l_i = 0)                       # running denom     [BR]
    Init(O_i = 0)                       # running output    [BR, d]      (mma fragment)
    for kv in 0..S_k step BK:           # KV tiles — the STREAMING serial reduce (K_o)
      S        = Q[mq] @ K[kv]^T        # [BR, BK] scores — inner mma, reduce over d
      tile_max = rowmax(S)              # [BR]
      m_new    = max(m_i, tile_max)
      alpha    = exp(m_i - m_new)       # rescale coefficient [BR]
      P        = exp(S - m_new)         # [BR, BK]
      tile_sum = rowsum(P)              # [BR]
      l_i      = l_i * alpha + tile_sum
      O_i      = O_i * alpha + P @ V[kv]  # inner mma, reduce over BK; O_i is SCALED each step
      m_i      = m_new
    O_i = O_i / l_i                     # finalize
    write O[h, mq] = O_i
```

The combine over the triple `(m, l, O)` is **associative** — the log-sum-exp monoid. Merging two partial states
`(m_a, l_a, O_a)` and `(m_b, l_b, O_b)`:

```
m = max(m_a, m_b)
l = l_a·exp(m_a − m) + l_b·exp(m_b − m)
O = O_a·exp(m_a − m) + O_b·exp(m_b − m)        identity = (m=−inf, l=0, O=0)
```

The per-tile step above is this combine with `(m_b, l_b, O_b)` = the current tile's *local* softmax partial. Because it
is a monoid, split-KV (flash-decoding) and cooperative-combine are reachable later — exactly the property the
`ReduceCarrier` trait advertises.

## What the IR can't express today — three gates

The repo already names all three as the unhandled case:

1. **Cross-accumulator reads.** `l_i`/`O_i` updates read `alpha`, derived from `m_i` and `m_new`.
   `lowering/tile/_helpers.py::accums_independent` returns `False` for exactly this ("rejects online algorithms — online
   softmax, Welford").
2. **A mid-reduction rescale.** `_atom.py::classify_fragment_epilogue` bails when "the accumulator is consumed *inside*
   a reduce loop — e.g. online softmax rescale — needs a scheduled phase, not a store fold".
3. **A loop-carried, mutated MMA accumulator.** `O_i` survives across the `K_o` loop and is *scaled* each step. Today an
   `Mma`'s `c` fragment is init-once and written at kernel end (`011_lower_atom_cell` + kernel lowering).

The fix is **not** to relax these gates globally. It is a new `ReduceCarrier` that encapsulates the coupling, so the
loose cross-`Accum` reads never appear in the body for `accums_independent` to see, and the rescale is a scheduled phase
the lowering emits — not an epilogue fold.

## The new primitive — `FlashCombine` on `ReduceCarrier`

A tuple-valued monoid carrier, sibling of `Accum`/`Mma`:

```python
@dataclass(frozen=True)
class FlashCombine(ReduceCarrier):
    state:   tuple[str, ...]   # carried (m_i, l_i, O_i)
    partial: tuple[str, ...]   # this tile's (tile_max, tile_sum, tile_pv)
    axes:    tuple[str, ...] = ()
    # defines()       -> state      (carried SSA defs visible after the loop)
    # deps()          -> partial    (carried read is implicit, like Accum/Mma)
    # carried_names() -> state
    # partial_deps()  -> partial    (the K_o pipeline scheduler reads this)
    # associative / commutative / has_identity -> True (the LSE monoid)
```

The rescale (`alpha = exp(m_i − m_new)`, `l·alpha + …`, `O·alpha + …`) lives **inside the combine's lowering**, not as
loose body statements — so gate (1) never fires and the combine stays a single carrier the partition planner can place.
`Init` (already in the IR) seeds the triple at the `mq` scope; the final `O_i / l_i` is the monoid finalize.

### Traits live on the carrier — no `CombineKind` (done)

An earlier sketch had `ReduceCarrier.combine_op()` return an `ElementwiseImpl`, then widened to a `CombineKind` protocol
so the LSE combine — a *tuple* monoid that doesn't fit `ElementwiseImpl`'s scalar-fn / float-identity mold — could be
returned too. That object earns nothing: the reassociation gates only ever read three booleans, and reifying them forces
the `Mma.combine_op() → ElementwiseImpl("add")` fiction (an `Mma` has no scalar op; its accumulation merely *is*
additive).

So the carrier exposes the algebraic traits **directly** — `associative` / `commutative` / `has_identity` as properties
on `ReduceCarrier`:

- `Accum` forwards to its scalar `op` (a `max` Accum and a `sum` Accum differ; `self.op` is the source of truth).
- `Mma` reports the additive-fold constants (`True / True / True`).
- `FlashCombine` reports the LSE-monoid constants (`True / True / True` — associative *and* commutative, which is what
  makes split-KV legal).

`ElementwiseImpl` keeps its own traits (they still serve scalar ops, `ReduceOp` / `ScanOp`, and `Accum`'s forwarding).
The gate becomes `carrier.associative and carrier.commutative` — one hop, no protocol, no `LSECombine`. The streaming
**merge codegen** is the `FlashCombine` subclass's lowering rule (`isinstance`-keyed, like `Mma`'s), and the op-chain →
carrier **detection** lives in the trigger pass — neither needs a reified combine object. This refactor (drop
`combine_op()`, add the three trait properties) is already landed; it is the foundation the rest of this plan builds on.

## Where it lives in the pipeline

**Frontend — recognition.** Recognize attention at `SdpaOp`, *before* `010_sdpa` decomposes it (the cheapest point).
Two strategies:

- **(A) Direct lowering — recommended.** Emit the flash loop nest from `SdpaOp` into Loop IR (a dedicated lowering, or a
  flash branch of `010_sdpa`), bypassing the generic softmax decompose. Stop at **Loop IR**, not a CUDA template, so the
  existing tile/stage/MMA machinery (cp.async, TMA, ldmatrix, smem padding, the masked-K zero-fill) still applies.
- **(B) Loop-fusion rewrite — rejected as primary.** Keep the decomposition and re-associate `QK^T→softmax→P@V` into the
  streaming form at Loop IR. This asks the value-preserving fuser to *discover* online softmax — a global reassociation,
  not a local splice. Too much to ask of the generic fuser.

**Tile partition (`010_partition_loops`).** The KV reduce already splits `K → K_o·(…) + K_i + …`; `K_o` is the natural
home for the streaming-outer loop, so the tiling shape exists. New work: host the `FlashCombine` rescale at the `K_o`
boundary, model `O_i` as a loop-carried fragment, and — a real bonus — **causal masking becomes tile-skip** (drop KV
tiles fully above the diagonal, partial-mask only the diagonal tile). The partial final KV tile reuses the **masked-K
zero-fill already built** (`_stage_expand`'s `(k < seq_len) ? v : 0`), so the symbolic-K (dynamic seq_len) case the
CLAUDE.md repeatedly flags as future work falls out almost for free.

**Kernel/cuda lowering.** Loop-carried fragment lifetime + the per-`K_o`-step `O_i *= alpha` scale — the part with no
current analogue (init-once fragments today). The two inner matmuls (QK^T over `d`, P@V over `BK`) lower as `Mma` cells
nested in the `K_o` loop; the P@V `Mma`'s accumulator **is** the carried `O_i`.

## Static and dynamic shapes — one kernel, parity by construction

Flash serves a static and a dynamic (`--dynamic seq_len@…`) sequence through the **same** fused nest — not two
codepaths. The streaming reduce, the `(m, l, O)` carry, and the rescale are *identical*; only the KV (reduce) axis
treatment differs, and that difference is exactly the masked-tile machinery the compiler applies everywhere else:

- **Static seq.** The KV extent is baked; the `K_o` streaming loop has a compile-time trip count and no final guard.
- **Dynamic seq.** The KV axis is `Dim('seq_len')`, tiled at `DEFAULT_SEQ_HINT` (512) with a ceil-div `K_o` trip count
  (`(seq_len + BK − 1) / BK`) and a **masked final KV tile** — the partial last tile zero-fills past the runtime extent
  via the masked-K zero-fill already built (`_stage_expand`'s `(k < seq_len) ? v : 0`). A zero score contributes
  `exp(−inf) = 0` to the softmax and `0` to `O_i`, so the online recurrence stays bit-correct at any runtime size.

So dynamic flash is not a feature bolted on at the end; it is the static nest under the compiler's standard
symbolic-extent treatment (the same `Dim` ceil-div the partition planner already emits for any masked tile), and the
`FLASH` knob's analytic term sees the symbolic seq through its stamped `hint` (the `_extents` companion change below).
Attention's dynamic axis is one symbol, `seq_len`, that lands on **both** the query (free / M) and KV (reduce / K)
positions — so dynamic mode exercises the masked-row *and* masked-K paths together. The work that makes dynamic real is
the masked final KV tile + symbolic grid in lowering; everything above it (the carrier, the rescale, the knob) is
mode-agnostic.

## The `FLASH` knob + analytic cold-start

Flash is a **structural fork**: deploy the fused online-softmax kernel, or keep today's score-materializing path (the
`005_split_demoted` split). Declare it exactly like that pass's `SPLIT_CONE`
(`lowering/tile/005_split_demoted.py`) — a module-level `Knob("FLASH", KnobType.BOOL, hints=(True, False))` in the flash
recognition pass, auto-registered by `knob.registry()`'s module walk, stamped onto each branch's `op.knobs` with an
idempotence guard so a re-entered pass reads the decision off the graph (no `off=`, to keep the absent-vs-declined
distinction the prior trains on). The two-level outer MCTS then offers it like any other fork, and greedy
`compile` / `run` pick via `Prior.pick`.

**Cold-start opinion — the analytic prior.** No flash kernel has ever been tuned, so there is no measured data and
nothing for `scripts/golden_knob_heuristics.py` to fit — the cold pick must be a *hand-set* rule. The `AnalyticPrior`
(`search/prior/analytic.py`) is the home: it already ranks every enumerated config by a linear quality over
`knob.knob_features`, and the feature it needs is already stamped. The KV/sequence loop is the **reduce** axis, and its
static extent rides in as `S_ext_reduce_max` / `S_ext_reduce_prod` (`loop/fusion/992_stamp_structural_features.py`) — so
"how long is the sequence loop" is directly readable.

The heuristic is one gated term, kept **local to `AnalyticPrior.score`** so it neither perturbs the shared featurizer
nor forces a refit of the `_W_A` / `_W_A_DYN` weights:

```python
# AnalyticPrior.__init__ — hardcoded parameters (overridable for eval analytic sweeps)
flash_seq_threshold: float = 1024.0   # favor flash once the reduce loop is at least this long
flash_weight:        float = <W>      # how hard the cold pick leans

# AnalyticPrior.score — after  feats = knob.knob_features(knobs)
flash_on = feats.get("FLASH", 0.0)                                            # 0 / 1
long_seq = feats.get("S_ext_reduce_max", 0.0) >= self._flash_seq_threshold    # hint-valued when symbolic
quality += self._flash_weight * flash_on * (1.0 if long_seq else -1.0)        # +reward long, −penalty short
```

`flash_on · (±1)` is the **interaction** the gate needs — a plain weight on the boolean can't say "good when long, bad
when short." Above the threshold it *rewards* `FLASH=True` (raises quality → lower `exp(−scale·quality)` → better);
below it *penalizes* `True`, so the cold pick stays on the split-materialize path where flash's carried-state + rescale
overhead isn't repaid. `FLASH=False` scores zero either way (`flash_on = 0`), so the split keeps its geometry-driven
rank. Threshold and weight are hardcoded `__init__` params, **not** fit — the learned `CatBoostPrior` takes over the
moment real flash configs are tuned and the measured `H_opt=3` rows exist.

For that uniform gate to see the dynamic case, a **companion change to the shared `_extents`** is needed — it *replaces*
today's special-casing rather than adding more. Today `992` *excludes* a symbolic axis from `S_ext_reduce_*` /
`S_ext_free_*` and only counts it in `S_ext_n_symbolic_axis`, so a dynamic-`seq_len` reduce reads `reduce_max` as its
*static* axes only (the head-dim inner reduce, never the sequence — not `0`, as an earlier draft of this plan said).
Instead: **stamp the symbolic axis's `hint` into the existing magnitude feature** (a static extent and a symbolic hint
are interchangeable for occupancy / tiling reasoning), and **add an `S_ext_reduce_symbolic` flag** to carry the
static-vs-dynamic distinction the exclusion used to imply. Then `S_ext_reduce_max` is hint-valued for symbolic seq, the
FLASH gate needs no symbolic branch, and a weight on the new flag can still price a masked/dynamic reduce differently
from a same-size static one.

This edits the shared structural-feature contract, so its blast radius is real — but partly anticipated: `ShapeKey`
already reserves room for "the planned symbolic-axis flag" (`search/data/shape.py`).

- **`ShapeKey.from_matmul`** sets `free_prod = N if dynamic else M*N` to mirror the exclusion; with the hint stamped it
  becomes `M*N` in both, so static and `.dynM` twins separate **only** by `is_dyn` (already a key field) — the flag, not
  the magnitude, keeps them apart. `from_features` / `from_db` already derive `is_dyn` from `S_ext_n_symbolic_axis`, so
  the join stays consistent.
- **`tests/compiler/pipeline/search/test_data.py::test_golden_dynamic_compile_s_feats_mirror`** asserts the old
  exclusion (`S_ext_free_prod == 512.0  # N only`); update it to the hint-inclusive product.
- **`_W_A_DYN`** was fit with symbolic extents excluded; its magnitude features now carry hints, so re-run
  `scripts/golden_knob_heuristics.py` for the dynamic goldens. The `S_ext_n_symbolic_axis`-based weight-set switch is
  unaffected (still `> 0`). Doc wording that says "symbolic axes excluded, mirroring the 992 stamp" (CLAUDE.md,
  `shape.py`, `sample.py`) needs the same update.

Scope note: this steers the cold *pick* only. The knob has no consumer until Step 4 lowers the fused nest, so wire the
knob + analytic term **with** Step 4, not before — until then `FLASH=True` is an option the lowering can't realize.

## Test strategy — static/dynamic parity

Every flash test runs under **both** shape modes from one body, reusing the existing parity harness rather than forking
static and dynamic suites:

- **`shape_mode` fixture** (`tests/compiler/conftest.py`) — `@pytest.fixture(params=["static", "dynamic"])` plus
  `dyn_M(mode, seq)` (returns `Dim('seq_len')` or the int). A flash test names `shape_mode` and threads `dyn_M` into its
  builder's seq axis, exactly like `tests/compiler/test_matmul_mma_parity.py::test_static_dynamic_mma_parity`
  (`_mma_graph(shape_mode, M)`). For SDPA one `seq_len` symbol sizes the Q/K/V seq axis, so one builder covers both.
- **Shared accuracy harness.** Extend `_run_module_with_eager` (`tests/compiler/e2e/test_attention_chains.py`) to take
  the mode / a `dynamic_shapes` dict. It already traces + compiles + runs deplodock and torch eager under one GPU-lock
  window (`backend.run(..., pre_run=…)`); the only addition is passing `dynamic_shapes={"q": {2: _seq_len_dim()}, "k": …,
  "v": …}` to `trace_module` in dynamic mode (the `test_cuda_sdpa_over_symbolic_seq_len` spec) and running at several
  runtime seq_lens via `CompiledProgram.rebind` (the `test_qwen_layer_dynamic` pattern) — proving one cached kernel
  serves every size. This extension is reusable by the existing static-only `test_attention_chains.py` SDPA tests too.
- **xfail-to-green workflow.** Write the parity tests up front and mark the `dynamic` param `xfail(strict=True)` with a
  reason string until the dynamic lowering (masked final KV tile + symbolic grid) lands, then delete the marker — the
  same flip-xfails-green flow as `plans/sdpa-n-axis-detection.md`. Static parity is green from Step 4; dynamic flips at
  Step 6. `strict=True` guarantees the marker can't silently outlive the fix.
- **Golden parity.** A flash golden carries a static entry *and* a `.dynM` twin (the `MatmulGoldenConfig(dynamic=…)`
  convention, adapted — flash isn't a plain matmul, so the golden's snippet is the SDPA `--code`). `shape_key()` keeps
  the twins on separate join keys (symbolic axis excluded from the extent product, `is_dyn` set), so `eval analytic`
  ranks the static under `_W_A` and the dynamic under `_W_A_DYN` — the FLASH term's symbolic clause is exercised by the
  twin, not a bespoke case.

## Implementation status (2026-06-19)

A **scalar-tier** flash path is implemented and GPU-verified end-to-end (RTX 5090, sm_120); the tensor-core P@V tier
remains future work. What landed, and how it diverged from the original framing:

- **Step 1 (done)** — `FlashCombine` carrier (`ir/stmt/leaves.py`): the `(m, l, O)` tuple monoid, traits, `is_reduce`,
  `rewrite` threading. Unit-tested.
- **Step 2 (done)** — `FlashCombine.render` lowers the LSE rescale **directly** (not via a separate pass) into fp32
  assignments against `Init`-declared carried scalars; `LoopOp` validation learned `Init` + `FlashCombine`. Verified via
  `LoopOp.forward` (cppyy CPU JIT) vs numpy — `tests/compiler/ir/test_flash_combine_forward.py`.
- **Steps 3 + 4 are coupled — realized together at the scalar tier.** Step 3 (loop-carried *scaled* MMA fragment) cannot
  be built or verified in isolation: there is no frontend op for a scaled-accumulator matmul, the scale is **per-query-
  row** (the `m16n8k16` `c[4]` fragment maps regs→rows), and the `alpha` only exists once the softmax half of the tile is
  built. So the scalar nest was implemented first: `frontend/decomposition/008_sdpa_flash.py` recognizes `SdpaOp` (before
  `010_sdpa`) and, gated by the `FLASH` knob, emits a single fused `LoopOp` that runs **one independent streaming softmax
  per output element `(…, m, d)`** — correct (the score `s = Σ_dd Q·K` is an inner reduce nested in the KV streaming
  reduce; the nest pre-places `Init(sacc)` at the KV-body scope so it resets per step), but redundant (recomputes scores
  per `d`). The tensor-core P@V tier (Step 3's `RegScale` fragment, the real perf win) is **still future work**.
- **Steps 4–6 (done, scalar tier)** — non-causal, causal (per-element `kv ≤ m` mask), and dynamic (symbolic `seq_len`
  threaded through both the masked-row M and the symbolic reduce — one cached kernel serves every size) all GPU-verified
  vs torch SDPA at ~1–3e-7. `tests/compiler/e2e/test_flash_attention.py` (9 tests). `FLASH` is read from the
  `DEPLODOCK_FLASH=1` env pin today; the two-level `OptionFork` offer + `AnalyticPrior` cold-start term (and the
  `_extents` symbolic-hint change) are **not yet wired** — a follow-up. With `FLASH` off the default `010_sdpa` path is
  unchanged.

**Remaining:** the tensor-core P@V tier (loop-carried per-row-scaled `Mma` fragment — the real Step 3); the `FLASH`
structural-fork offer + analytic cold-start; causal tile-skip (only the per-element mask exists); GQA + explicit-mask
flash (both fall through to `010_sdpa`).

## Incremental steps (each independently verifiable)

Every step's accuracy test is written under the `shape_mode` fixture (see "Test strategy"), so static and dynamic are
one test from the start — the `dynamic` param sits `xfail(strict=True)` until Step 6 flips it. The steps below are
ordered by *implementation* dependency, not by shape mode.

- **Step 0 — done.** `ReduceCarrier` (PR #261), then traits moved onto the carrier directly (`associative` /
  `commutative` / `has_identity` properties; `combine_op()` dropped — this branch). Verify: suite green.
- **Step 1 — `FlashCombine` carrier.** Add the tuple-monoid `ReduceCarrier` subclass above (state / partial / carried
  surface + the constant traits). Verify: unit tests on the carrier surface + `is_reduce`; full compiler suite green.
- **Step 2 — streaming reduce WITHOUT matmul.** Implement the carrier's rescale lowering for a scalar target —
  **online-softmax-of-a-vector** or **Welford mean/variance** (same carried-tuple + rescale machinery, no mma nesting).
  Verify: `LoopOp.forward()` CPU reference (fusion correctness with no GPU) + `run --code` accuracy vs a torch reference
  for the snippet.
- **Step 3 — loop-carried, scaled MMA accumulator.** Teach kernel lowering to keep an `Mma` `c` fragment alive across a
  serial-outer loop and scale it per step. Verify: a matmul-with-per-step-rescale micro-kernel, accuracy + the emitted
  CUDA shows the carry living across `K_o`.
- **Step 4 — flash from `SdpaOp`, non-causal, static seq (`shape_mode=='static'` green).** Compose Steps 2–3 into the
  nest, gated by the `FLASH` structural-fork knob + its analytic cold-start term (see "The `FLASH` knob"). Verify: the
  `shape_mode` SDPA parity test passes in static mode (dynamic still xfailed); the kernel table shows **one** `k_sdpa…`
  kernel (no `xn` score materialization) when `FLASH=True`; `eval analytic` picks `FLASH=True` past the threshold and
  `False` below; `--bench` vs the split path at long seq.
- **Step 5 — causal.** Tile-skip above the diagonal + diagonal-tile partial mask. Verify: causal SDPA accuracy (static
  mode); dumped IR shows skipped KV tiles.
- **Step 6 — dynamic parity, flip green.** Add the masked final KV tile + symbolic ceil-div `K_o` grid (reusing the
  masked-K zero-fill) so the *already-written* `shape_mode=='dynamic'` tests pass; delete their `xfail` markers and the
  golden `.dynM` twin's. Verify: the `shape_mode` SDPA parity test passes in **both** modes; dynamic accuracy at several
  runtime seq_lens from one cached kernel (`CompiledProgram.rebind`); static and dynamic kernel tables match modulo the
  boundary guard; `eval analytic` ranks the `.dynM` twin under `_W_A_DYN` with `FLASH=True`.

## Open questions / risks

- **Register pressure.** `O_i` `[BR, d]` fragment + Q/K/V staging coexisting across `K_o` may force a smaller `BR` than
  a plain gemm. Expect the tuner to pick conservative `BR`; watch occupancy in `--bench --profile`.
- **Flash vs the split path.** Flash makes `005_split_demoted`'s score materialization unnecessary *for attention*. We
  keep both and offer flash as a **structural fork** (the `FLASH` knob above) so the two-level tuner can A/B
  flash-vs-split per shape — flash wins at long seq, the split may still win at short. The analytic cold-start biases
  the fork by reduce extent (see "The `FLASH` knob + analytic cold-start"); open sub-question is the exact threshold and
  weight, which only real `--bench` numbers at a range of seq lengths can calibrate.
- **Where to recognize SDPA.** `SdpaOp` pre-decomposition (strategy A) vs a loop-IR pattern. Plan assumes A.
- **Numerics.** Rescale + `exp` must run in fp32 accumulators (matching eager fp16 attention); the carried `O_i` is the
  fp32 mma fragment, cast to the write dtype only at finalize.
- **The fused prologue P@V** currently stays degenerate at `FM=FN=1` for symbolic K; flash is its proper replacement, so
  Step 6 and that existing path should converge rather than both being maintained.

## Scope

**In:** the streaming-reduce IR primitive (the `FlashCombine` carrier), and flash for SDPA — non-causal → causal →
dynamic seq_len. Both static and dynamic shapes are in scope and ship at **test parity** (one `shape_mode` test body per
case); static lands first by implementation dependency, dynamic flips its xfail at Step 6. GQA broadcast is already
handled upstream of the nest.

**Out (future):** flash-decoding / split-KV across CTAs (reachable because the combine is a monoid, but its own work),
paged-KV, sliding-window. These ride on the same carrier once the single-CTA nest lands.
