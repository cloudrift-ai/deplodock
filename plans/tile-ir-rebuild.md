# Tile IR rebuild

Status: **demolition done, rebuild not started.** Branch `refactoring/tile-ir-rebuild`.

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
  means the rebuild is fully recovered.** The dict is empty today (tile IR is still intact, so nothing fails yet); it
  gets populated as the rebuild tears things out and emptied as each piece is restored.
- **Collection-time caveat:** a file whose *module-level* import of a tile symbol breaks fails at collection, before any
  item exists to mark — pytest reports that as an error, which xfail cannot catch. The files that keep a load-bearing
  tile import (their accuracy tests build references through tile-internal helpers) are listed in
  `TILE_ENTANGLED_FILES` in the registry for visibility. When the rebuild removes/renames those symbols, make the import
  lazy (so the file still collects and the failure becomes a markable item) and add the nodeid to `XFAIL`.

## Phase 1+ — Rebuild (not started)

To be designed. The recovery loop is fixed regardless of design: tear out a slice of the old tile IR, watch the
relevant `e2e/` tests go red, register them in `XFAIL`, build the replacement, and remove registry entries as the tests
flip to XPASS. The rebuild is complete when `e2e/` is green and `XFAIL` is empty. Design phases (data model, lowering
moves, scheduling/search integration) will be appended here as they are scoped.
