# Tile IR rebuild

Status: **demolition landed — tile IR + tile/kernel lowering passes removed, the codebase imports clean, the
test suite is green through the xfail registry below. Rebuild not started.** Branch `refactoring/tile-ir-rebuild`.

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

## Phase 1+ — Rebuild (not started)

To be designed. The recovery loop is fixed regardless of design: tear out a slice of the old tile IR, watch the
relevant `e2e/` tests go red, register them in `XFAIL`, build the replacement, and remove registry entries as the tests
flip to XPASS. The rebuild is complete when `e2e/` is green and `XFAIL` is empty. Design phases (data model, lowering
moves, scheduling/search integration) will be appended here as they are scoped.
