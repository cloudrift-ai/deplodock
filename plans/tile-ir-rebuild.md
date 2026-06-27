# Tile IR rebuild

Status: **rebuild in progress — Phase 1a (no-fold kernel skeleton: `TileOp` schedule → kernel-IR `Tile` → `CudaOp`) and
Phase 1b (the per-cell reduce, normal reduction + online softmax unified by the twist) have landed; every elementwise,
index-only, and reduction `e2e/` kernel is recovered and un-xfailed. The remaining tiers (cooperative / cross-CTA reduce
schedules, matmul, flash attention) stay registered in the xfail registry.** Branch `refactoring/tile-ir-rebuild`.

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

### Phase 1c+ — the remaining tiers — not started

Add the cooperative / cross-CTA reduce *schedules* (warp-shuffle / smem-tree combine, split-reduce, flash-decoding —
perf, not correctness), the contraction (`SEMIRING` — mma / split-K), and streaming-monoid flash attention. Each supplies
its combine + launch geometry to the same `TileOp` / `Tile` skeleton. To be scoped here as each lands.
