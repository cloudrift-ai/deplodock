# Tile IR rebuild

Status: **rebuild in progress — Phases 1a–1d, 2 and 3 have landed: the no-fold skeleton, the per-cell reduce (reduction
+ online softmax unified by the twist), scalar matmul (`SEMIRING`), scalar **flash attention** (the nested
`MONOID(SEMIRING)`), the high-level op tree that replaced the `build_*` body assembly, and the op tree now FLOWING as
the carried IR: `010_recognize` is the sole Loop-IR → Tile-IR boundary — it lifts EVERY kernel to a `TileOp` carrying
one op-tree `AlgebraNode` (`Map` / `Monoid` / `Semiring`) with its parallel axes on the node's `free` field and an
**empty** schedule, and `020_schedule` only moves `free` onto `grid_axes`; loop materialization is deferred entirely to
`lowering/kernel`. All recognition (flash, online softmax, normalize, lift) is consolidated in `010_recognize` (the
`_flash` / `_softmax` helpers hold the pattern matchers). Every elementwise, index-only, reduction, static-matmul, and
static-attention `e2e/` kernel — including whole transformer blocks (TinyLlama + Qwen) AND the un-fused RoPE
self-attention — is recovered and un-xfailed. Remaining: dynamic shapes, and the perf tiers (cooperative / cross-CTA
reduce, the mma/blocked/split-K `Atom`).** Branch `refactoring/tile-ir-rebuild`.

The tile IR — `deplodock/compiler/ir/tile/` (the `TileGraph` / `TileOp` / `StageBundle` / block-DAG / warp-tile
datatypes) and `deplodock/compiler/pipeline/passes/lowering/tile/` (the `enumeration` + `assembly` lowering passes) —
is being rebuilt from scratch. Everything upstream (frontend → tensor → loop IR) and downstream (kernel → cuda IR,
the backend) stays. This doc tracks the rebuild; it opens with the demolition that already landed.

## The recovery contract

`tests/compiler/e2e/` is the **only** thing the rebuild must satisfy. Every file there is black-box: it builds a
graph, runs it through `CudaBackend`, and compares the output to a numpy/torch reference (a handful also assert on
generated CUDA source strings — those are part of the contract too). None of them assert on tile-IR Python objects, so
they survive any internal redesign. The rebuild is "done" when the whole `e2e/` suite is green with an **empty** xfail
registry (below).

## Phase 0 — Demolition (done)

### What was demolished — functionality

The lowering tier that turned a fused loop kernel into executable GPU code was removed wholesale. Everything from the
loop IR onward through code generation is gone, so today nothing compiles past the loop IR; the surviving passes import
cleanly but the lowering stage is an empty hole. The specific capabilities that no longer exist:

- **Tiling of pointwise and reduce kernels** — mapping iteration axes onto grid blocks, threads, and register/serial
  loops; the per-kernel schedule search that picked those tile shapes.
- **Reductions** — re-bracketing a contraction across a serial inner loop and a cooperative thread fan-in, with the
  cross-thread combine (warp-shuffle / tree) and masked tails filled by the carrier identity.
- **Tensor-core matmul** — folding a contraction cell onto `mma.sync`, including transposed operands, masked and
  symbolic-shape edges, and fused residual / pointwise / causal-mask epilogues.
- **Attention** — streaming online-softmax (flash) attention in both a scalar and a tensor-core form; causal masking,
  grouped-query heads, and symbolic sequence length.
- **Shared-memory operand staging** — cooperative smem slabs for reused operands, plus the bulk-async (TMA / cp.async)
  double-buffered transport and its software-pipelined prologue/main/epilogue.
- **Cross-CTA partitioning** — split-K matmul, split-reduce, and split-KV (flash-decoding) across thread blocks, with
  either an atomic finalize or a deferred combine kernel.
- **Kernel fusion across the producer/consumer seam** — keeping a producer and its matmul consumer in one kernel via a
  shared smem buffer, and the alternative of cutting a demoted operand out to a gmem intermediate.
- **Autotuning of the above** — the two-level search over scheduling forks, the per-kernel structural slicing it tunes
  in, the analytic + learned priors that rank tile configurations, and the heuristic tile-shape planner.
- **Supporting lowering services** — generating CUDA source from a tiled kernel, deriving launch bounds and CTA/thread
  coordination from a kernel's placement, shared-memory bank-conflict diagnostics, the tile/kernel IR dump stages, and
  round-tripping tiled kernels through the serialized-IR format.

### Tests

Tile-IR **unit** tests — the ones that imported tile datatypes / pass internals and asserted on their structure — were
deleted outright. They are gone for good; they tested an implementation that no longer exists, and the rebuild gets a
fresh set. Backend-accuracy tests that were scattered across `tests/compiler/` and `tests/compiler/passes/` were
consolidated into `tests/compiler/e2e/` (clean files moved with `git mv`; mixed files had their accuracy tests salvaged
into new e2e files and their internal-only assertions dropped). One genuine duplicate was removed. The result is a
single accuracy-only recovery set in `e2e/`, green today.

### Guideline for future agents — deleting unit tests

While rebuilding, you will encounter tests that fail or stop collecting because they reach into tile internals. The rule:

- **A tile-IR unit test → delete it. Do not fix it, do not port it.** A test is a tile-IR unit test if it imports from
  `deplodock.compiler.ir.tile` or `deplodock.compiler.pipeline.passes.lowering.tile` and asserts on those objects
  (tile graphs/ops, stage bundles, offers, fragments, atomization, assembly, warp-specialize materialization, tile
  naming, enumeration counts, etc.) instead of running a graph and checking numerical output. These are coupled to the
  old internals by design; the new internals get new unit tests written against them, not the old ones patched.
- **An integration / accuracy test (anything in `tests/compiler/e2e/`) → never delete or weaken it to make it pass.**
  It encodes the contract. If the in-progress rebuild makes it fail, register it as an expected failure (below) and
  move on; it flips back to a hard requirement automatically once the capability is restored.

When in doubt: does the test run on the backend and compare output? Keep it (xfail if currently broken). Does it only
inspect tile Python objects? Delete it.

### Integration-test xfail mechanism

A single registry drives expected failures during the rebuild — no scattered `@pytest.mark.xfail` decorators.

- **Registry file:** `tests/xfail_registry.py`. It exports `XFAIL: dict[str, str]` mapping a **test node-id substring**
  to a one-line reason. A substring like `"test_foo.py"` xfails a whole file; a full id like
  `"test_foo.py::test_bar"` xfails one case.
- **Application:** the `pytest_collection_modifyitems` hook in `tests/conftest.py` marks every collected item whose
  `nodeid` contains a registered substring with `xfail(strict=False)`.
- **Recovery semantics:** `strict=False` means a test that starts passing again reports as **XPASS**, not a failure —
  that is the signal a capability came back. Delete its entry and it reverts to a required test. **An empty `XFAIL` dict
  means the rebuild is fully recovered.** The dict is **populated today** — the demolition registered every test that
  exercised the removed lowering tier (whole-file entries where a file fails entirely, single-nodeid entries where a
  file still has passing tests). It shrinks as the rebuild restores each capability and the tests flip to XPASS.
- **Collection-time caveat:** a file whose *module-level* import of a tile symbol breaks fails at collection, before any
  item exists to mark — pytest reports that as an error, which xfail cannot catch. The demolition handled this so
  `TILE_ENTANGLED_FILES` is **empty**: pure tile-IR unit-test files were deleted, and the integration/accuracy files
  whose imports were load-bearing had those imports guarded (`try/except ModuleNotFoundError`) so they still collect and
  their now-failing cases are registered in `XFAIL` above. A rebuild that reintroduces and then renames a tile symbol
  should repeat that pattern — guard the import, register the nodeid — rather than letting a collection error return.

## Phase 1+ — Rebuild (in progress)

The recovery loop is fixed regardless of design: tear out a slice of the old tile IR, watch the relevant `e2e/` tests go
red, register them in `XFAIL`, build the replacement, and remove registry entries as the tests flip to XPASS. The
rebuild is complete when `e2e/` is green and `XFAIL` is empty.

### Phase 1a — the skeleton (no-fold kernels) — done

The first slice rebuilds the layering end-to-end for the simplest algebraic kind — kernels that carry **no combine**
(every axis parallel, no fold). The design follows the algebra: **the schedule is separate from the combine.** One op
carries the schedule; the combine stays in the body and is read back by kind, so the later kinds extend the same
skeleton by supplying a combine rather than adding lowering code.

- **`ir/tile/` — `TileOp`** (rebuilt): a `BodyOp` carrying the *schedule* — `grid_axes`, the axes tiled onto the thread
  grid — while the body holds the leaf compute (and, later, a reduce `Loop` + `ReduceCarrier` for the fold). The algebra
  is **not stored**: `TileOp.algebra_kind` reads it back via `classify_algebra` (the project's "algebra is a derived
  cache" rule). Wired into `search/keys.py` (`dialect_of` → `"tile"`, `op_cache_key`).
- **`ir/kernel/` — `Tile`** (kernel-IR stmt): the hardware realization of the schedule — one thread per output cell. It
  emits the linear-thread decode (`_gid = blockIdx.x·blockDim.x + threadIdx.x`, the `_gid < N` guard, the per-axis index
  decode) around the body. Geometry only; static extents for now.
- **`lowering/tile/020_schedule.py`** — `LoopOp → TileOp`: dispatches on `Loop.algebra_kind`; schedules the
  no-fold kind by flattening the free-loop nest into `grid_axes` + a per-cell body. A kernel that carries a combine
  (`reduce_axis_names` non-empty) is left un-lowered (still xfailed) until its schedule is built.
- **`lowering/kernel/010_materialize.py`** — `TileOp → KernelOp`: wraps the per-cell body in `Tile`. Algebra-generic — a
  fold would ride inside the body untouched.
- **`lowering/cuda/010_lower_kernelop.py`** — `KernelOp → CudaOp`: renders the body and sizes the launch
  (`ceil(N/256)` CTAs × 256). (Was the demolition stub.)

Recovered `e2e/` capability: every elementwise + index-only (broadcast / reshape / transpose / slice / unsqueeze / cat /
gather / embedding / index-select / pow) kernel, fp32 and fp16. Their `XFAIL` entries are removed; the matmul / reduce /
softmax / attention kernels remain registered, waiting on their schedules.

### Phase 1b — the per-cell reduce (`MONOID`), unified by the twist — done

The fold kinds enter on the same skeleton: the article's framing is that a plain reduction and online softmax differ
**only by the twist** ψ (the rescale-by-max bijection), so both must share one representation and one lowering. The slice
makes that literal — normal reduction uses the *degenerate* (identity) twist, online softmax the max twist, and nothing
in the schedule or the materializer branches on which.

- **Representation — one carrier, twist extracted.** The combine is split out of the algebra: `Monoid` (`ir/stmt`) holds
  the algebra (`state` / `partial` / `identity` / `commutative`), and a separate **`Twist`** holds the ψ-conjugated
  combine as data (`merge` / `combine_states` / `state_b`). The monoid is shared; ψ lives entirely in those programs and
  is the only thing that varies — a plain reduction's identity twist (`Twist.degenerate`: componentwise
  `state_i = op_i(state_i, partial_i)`), online softmax's max-rescale, a future mma-fragment realization. Every reduce
  carrier is normalized to a `Monoid`: a scalar `Accum` → degenerate-twist monoid (`Accum.as_monoid`), an
  already-twisted `Monoid` (online softmax / flash) is kept. The combine is read off the twist directly
  (`monoid.twist.merge` / `.combine_states` / `.state_b`).
- **Recognize then schedule (two passes in `lowering/tile/`).** `010_recognize` does ALL algebra recognition, in order:
  (1) **flash attention** — a softmax-then-P@V kernel + its scaled-QK producer fuse to one flash `LoopOp` (the
  `(m, l, O)` twisted monoid); (2) **online softmax** — an adjacent `(rowmax, Σ exp)` reduce pair fuses to one
  `(m, d)` `Monoid`; (3) **normalize** — every remaining `MONOID` reduce loop's `Accum` → `Init + degenerate Monoid`
  (a `SEMIRING` contraction keeps its `Accum`, else `classify_algebra` would flip to `MONOID`). Flash precedes
  online-softmax precedes normalize — each later step consumes the `Accum`s an earlier one matches. Recognition is
  **always on** (the `FLASH` / `ONLINE_SOFTMAX` knobs were dropped); the flash builders live in `lowering/tile/_flash.py`
  (the demolished `loop/recognize/` pass is gone). `020_schedule` does the geometry: `_peel` maps the free axes
  *enclosing* the reduce onto `grid_axes` (one thread per output row) and leaves the reduce `Loop` — plus any epilogue /
  output sweep sharing its accumulator — serial in the per-cell body. `MAP` and `MONOID` schedule identically; `SEMIRING`
  is skipped (so a recognized flash kernel, whose inner Q·K dot product is a `SEMIRING` reduce, stays un-lowered until
  the matmul/attention tier — attention remains xfailed, no regression).
- **Materialize / cuda — unchanged.** The `Monoid` renders through `render_merge_program`, the reduce `Loop` through
  `Loop.render`, the seed through `Init`; the per-cell body (fold + epilogue + normalize sweep) sits inside the same
  `Tile` thread-decode the no-fold kind already used. So a plain `reduce_sum` emits `acc = acc + x` and online softmax
  emits the rescale-and-add merge through the **same** path.

Recovered `e2e/` capability: `reduce_sum` / `reduce_max` / `mean` / `keepdim`, `rmsnorm`, `softmax`, online softmax, and
the "cooperative" K=512/2048 variants (correct via the serial per-thread fold — the cooperative *schedule* is a perf
tier, added later). Registry residuals: flash attention, cross-CTA split-reduce, and the matmul/sdpa tune cases.

### Phase 1c — scalar-tier matmul (`SEMIRING`) — done

A contraction is the `SEMIRING` reduce — `reduce(+) ∘ map(⊗)` — and at the scalar tier it schedules *exactly* like a
reduction: one thread per output cell, the K axis a serial fold (`acc += a·b`). So `020_schedule` just stops skipping
`SEMIRING` — `_peel` already keeps the reduce loop serial in the per-cell body, and `Loop.render` already emits the
`Accum` fold. The `Accum` is kept (not degenerate-monoidized) so the kind stays `SEMIRING` for the future mma Atom.

Two guards added to `020_schedule`:
- **nested contraction** (a reduce loop whose body holds another reduce loop — flash's `kv` monoid over the `Q·K` dot
  product) is skipped: the streaming/attention tier isn't built. So matmul (flat) lowers; flash stays gated.
- **symbolic axis** is skipped: the scalar `Tile` decode needs static extents (a dynamic `seq_len` matmul previously hit
  the SEMIRING skip first; now it's skipped explicitly, staying un-lowered for the dynamic tier).

Recovered: static `matmul` / `linear` (+bias), fp16 matmul, and the many tier-pinned matmul *accuracy* tests
(`test_lowering_blocked_gemm`, `test_stage_scalar`, `test_matmul_mma_parity[static]`, `test_knob_pinning` matmul/gated-MLP,
the matmul `tune`/`two_level` cases) — the pinned tile/mma/staging/split-K knobs are no-ops at the scalar tier but the
output is correct, so they pass on accuracy now and will *guard* their tier once it's built. Registry residuals in those
files are the genuinely tier-needing cases (mma codegen, dynamic, split-K, attention).

### Phase 1d — scalar flash attention (nested contraction) — done

`020_schedule` no longer gates the nested contraction. Flash is a `MONOID(SEMIRING)`: the `kv` monoid (the `(m,l,O)`
LSE carrier) streaming over the inner `Q·K` `SEMIRING` dot product. At the scalar tier each reduce loop — flat or
nested — renders as a serial fold through its carrier (one thread per output cell), so the nested flash nest lowers
through the **same** generic `Tile` + `Monoid.render` + `Loop.render` path as any reduce; **no flash-specific lowering
exists**. The `build_flash_*` output (still the recognizer's, for now) is just an ordinary `LoopOp` that this generic
path materializes.

Recovered: `sdpa` / `sdpa_causal` / `sdpa_gqa`, the `test_flash_attention` static cases (causal / GQA / additive mask /
kv-tile), `test_attention_chains` self-attn + masks, and the **whole transformer blocks** (`test_block` TinyLlama +
Qwen). Residuals: the **dynamic/symbolic** flash variants (need the dynamic-shape tier) and the obsolete flash-knob-off
test.

### Phase 2 — the high-level op tree (dissolving `build_*`) — done

The compute layer is lifted from hand-assembled low-level loop stmts to a small algebraic op tree. The algebraic
vocabulary is consolidated in `ir/stmt/algebra.py` — the lift `Map`, the carrier `Monoid` + `Twist`, and the `Semiring`
contraction view — and the op tree built on it (`Reduce`, `TensorRef`, `lower`) lives in `ir/tile/ops.py`. Just two node
kinds plus an operand descriptor:

- **`Reduce`** — a fold over one axis through a `Monoid`+`Twist` carrier, with **partials nested** (a `Map`, a
  `TensorRef`, or another `Reduce`). `lower` generates its structure: an `Init` per carried state (from the carrier
  identity), the streaming `Loop`, and the carrier fold.
- **`Map`** — a pointwise body. It **subclasses `Body`** (so it carries Body's analysis helpers and has *no fields*): a
  `Map` simply *is* its loop-IR stmts — the operand `Load`s, the lift `Assign`s, an optional masking `Select`, and the
  output `Write` at the kernel root — last-binding a value name. There is nothing to lower; the stmts are the lowering.
  This folds in what would otherwise be separate nodes: the causal mask is just a `Select` stmt inside the score `Map`
  (the index predicate `kv ≤ m` lives in the index-aware tree, never in the index-free carrier), and a recovered
  fused-RoPE score the tree can't reconstruct is just a `Map` of the spliced stmts.
- **`TensorRef`** (buffer + index exprs) — the only place layout lives; a direct-load partial of a `Reduce`.

A contraction is `Reduce(+)` over a `Map` (the `×` lift); flash is `Reduce(lse)` over `(scaled Σ Q·K, V)` with the `O/l`
projection as the root `Map` — validated against `softmax(QK^T)·V`.

`_flash_loop_body` is **deleted**; `build_flash_frag` / `build_flash_recovered` no longer hand-assemble a kernel body —
each builds this op tree and calls `lower`, differing only in the score `Map` (clean `Load`s of Q/K vs. a recovered
fused-RoPE subgraph spliced in verbatim). The flash skeleton (the `(m,l,O)` streaming fold + the `O/l` projection) is
expressed exactly once, in the tree. The `build_*` functions remain only as the recognizer's pattern → graph-fragment
constructors.

Geometry stays separate: `Tile(op, placement)` maps axes to grid/thread/serial/coop/split (a later slice). Next: the
remaining perf tiers (cooperative / split / mma `Atom`) become placements/atoms on the same tree, plus the dynamic-shape
(symbolic `seq_len`) tier.

### Phase 3 — the op tree as flowing IR (`TileOp` carries it; recognition consolidated) — done

The op tree was a construction-time scaffold (built inside `build_*`, immediately lowered to a `LoopOp`). It now flows as
the carried IR, and recognition is unified in ONE rule:

- **`010_recognize` is the sole Loop-IR → Tile-IR boundary.** It lifts EVERY kernel (not just flash) to a `TileOp`
  carrying one op-tree `AlgebraNode` with an **empty** schedule. After it, nothing downstream traffics in `LoopOp`. The
  steps, in order, all in this one rule (no separate `005_flash` / softmax pass): (1) **flash** — a graph rewrite fusing
  a softmax-then-P@V kernel with its clean scaled-QK producer (`_flash.try_flash`); (2) **online softmax** —
  `_softmax._fuse`; (3) **normalize** — `Accum → degenerate Monoid` (`Semiring.match` keeps a contraction's `Accum`);
  (4) **lift** — `_peel` the free axes, then lift the per-cell compute into ONE node: a pure pointwise body is a `Map`; a
  single FLAT reduce is a self-contained `Monoid` (plain reduce / online softmax) or `Semiring` (clean contraction)
  node, with any pre/post pointwise stmts as a projection `Map` over it. A cell the lift can't cleanly factor (no reduce,
  several reduces, or a nested non-flash reduce) stays a flat `Map` of the per-cell stmts — still a valid op-tree node,
  and it lowers verbatim (this is what recovered the un-fused RoPE-attention path: a multi-reduce softmax-then-P@V kernel
  is a correct flat `Map`).
- **Free axes live on the algebra node (`Map` / `Monoid` / `Semiring` gained a `free` field).** This resolves the type
  tension that an op-tree node can't be wrapped by loop-IR `Loop` stmts: the lifted node carries its parallel axes
  directly. `010_recognize` fills `free` and leaves `TileOp.grid_axes` empty; `020_schedule` moves `free → grid_axes`
  and clears the field; `materialize` lowers a free-less node.
- **Flash-producer deferral.** Because flash fusion reads the score producer's Q/K as plain `Load`s, a node that IS a
  flash score producer is deferred (left a `LoopOp`, `_flash.is_flash_score_producer`) so step 4's lift doesn't turn it
  into a `TileOp` before its consumer fuses. A later scan re-visits it (by then consumed/removed, or lifted normally if
  no flash consumer remains).
- **Output-store glue is `Write`-presence-driven.** `materialize` adds the grid-cell store (`Write(out, grid, op.out)`)
  iff the lowered body carries no `Write` of its own — so a bare `Monoid` / `Semiring` (reduce / matmul) and a flash
  projection `Map` get glue, while a pointwise `Map` or a projection that writes through its own output sweep
  (rmsnorm / softmax) does not.
- **Contraction-lift soundness.** `Semiring.match` reconstructs operands as bare one-`Load` `Map`s, so the lift only
  emits a `Semiring` for a CLEAN contraction (`_clean_contraction`: the lift multiplies the operand loads directly); a
  contraction with per-operand preprocessing (e.g. pre-scaled inputs `Σ (x·s)·(y·s)`) lifts to a `Monoid` whose partial
  `Map` captures the full per-element compute instead. Loop-invariant prologue stmts that feed the reduce are routed
  into that partial (lowered inside the loop, like flash's scale); ones feeding only the epilogue stay after it; a stmt
  feeding both keeps the whole cell a flat `Map`.

Flash recognizes only the **clean** scaled-QK producer (Q/K as plain `Load`s); RoPE'd-QK attention is not fused, but its
**un-fused** fallback now lowers correctly via the flat-`Map` lift, so `test_block` (TinyLlama + Qwen) and the RoPE
`test_attention_chains` self-attention recovered (their `_NO_RECOVERED` xfails are removed). Re-supporting flash *fusion*
of RoPE attention (a producer-splicing score `Map`) is a perf follow-up, not a correctness gap.

This recognition is still case-by-case (flash / online-softmax / contraction patterns); the intent is to grow it toward
ONE algorithmic algebra recognizer. Remaining: the `body` property still lowers on access (for the cache key / dumps)
while `materialize` does the deployable lower + glue — a structural key off `op` directly would drop even the
property-side lower, but the payoff is small since `lower` is pure.
