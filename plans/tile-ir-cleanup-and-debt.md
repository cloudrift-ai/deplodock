# Tile IR: cleanup, consolidation, xfail fixes, debt

Successor to the executed `tile-ir-rebuild.md` (deleted): everything left of the rebuild that is neither warp
specialization (`warp-specialization.md`) nor flash performance (`flash-enhancements.md`). The rebuild's mandate stays
in force for every item here — one hierarchical emitter, zero divergent codegen paths, zero back-compat shims, and a
phase is done only when the code it replaced no longer exists (the invariants now live in
`compiler/pipeline/passes/ARCHITECTURE.md` / `lowering/kernel/ARCHITECTURE.md`; this plan doesn't restate them).

The recovery contract is unchanged: `tests/compiler/e2e/` is black-box and MUST stay green; a capability item is done
when its xfail entry flips XPASS and is deleted.

## 1. xfail fixes (the registry is down to 4 entries)

- **`test_fused_edge.py::test_fused_map_matmul[warp-broadcast…]`** — the BROADCAST producer cone recognizes as a flat
  un-annotated `Map` (no `Reduction` node for `_schedule._demoted_warp_option` to nodify). A recognition gap, not a
  lowering gap: teach `010_recognize`'s lift to annotate the broadcast-producer contraction so the demoted-cone warp
  option can fire.
- **`test_fused_edge.py::test_fused_rmsnorm_linear[warp…]`** — the MONOID (rmsnorm) producer's cone carries a reduce,
  so it is not compute-fillable per cell. Needs the cooperative-prologue warp fusion: the cone's reduce runs once
  cooperatively (the shared-row `sync` `Stage` machinery is the seam), its result feeding the A-slab compute-fill.
- **`test_vllm_plugin_gpu.py` / `test_vllm_plugin_gen_gpu.py`** — the serving-stack e2e on the rebuilt compiler.
  Likely mostly a validation run (whole-model compile paths are recovered); fix what falls out.

## 2. Consolidation debt (atom-as-descriptor, the last deviation)

- **Placement-keyed fold move (deviation 4).** The fragment `__shfl` row landed with tensor-core flash; the full
  collapse has not: `emit_combine` (cross-thread smem), the fragment row-reduce (within-warp `__shfl`), and
  `030_split`'s cross-CTA combine are still three implementations of "fold + a data-move keyed on where the reduced
  axis sits" (within-lane = none, within-warp = shuffle, within-block = smem, cross-CTA = split finalize). Collapse to
  ONE move selector consumed by the one `_bind` pipeline.
- **Nodify the scalar per-cell contraction at recognize time** (deferred out of the rebuild's Phase 0). Today a scalar
  per-cell contraction rides a flat `Map(body=(annotated CONTRACTION loop, …))` all the way through materialize, and
  `ops.reduce_loop`'s body-scan + the `Map.out` carrier arm exist to serve it. Nodifying (with a deferred `tile`)
  retires the body-scan and the flat-contraction special form — a recognizer refactor, proven by the e2e suite.

## 3. Debt

- **Scalar-tier gmem→smem ring** (`depth ≥ 2`; the scalar staged K-loop is single-buffer today —
  `_resolve_scalar_stage` clamps to `d1`). Layer the cp.async ring / TMA mbarrier phases on the existing
  `_staged_slab_kloop` skeleton exactly as the warp tier does — a `Stage.depth` parameter, not a new path. This is
  also what makes the recorded scalar goldens (`d2+/tma/ring` on register tiles) reachable again; re-run the golden
  A/B (tune-golden flow) once it lands, and extend `search/space.stage_moves(warp=False)` past `d1`.
- **Thread the resolved `Stage` through `030_split`** — split partials are gmem-direct today ( `_splitk_option` stamps
  `STAGE=""` and the enumeration skips staged×split rows). Threading it lets split-K compose with operand staging;
  un-skip the rows in `_schedule._tile_rows` when it lands.
- **Rebuild the `find_all_bindings` bank-conflict staging oracle** (`test_bank_conflicts.py` was deleted with the old
  IR; the visualizer scripts are its consumers). Rebuild against the kernel-IR slab layouts.
- **Rebuild the offline analytic-weight fitter** (`scripts/golden_knob_heuristics.py`): its non-matmul path still
  imports the demolished `enumeration/_iterdag`; the matmul path should reuse `search/analytic._enumerate` (the live
  fork capture). Then refit `_W_A` over the restored space — the current fit has no `D_stage_*` terms and ranks the
  staged golden geometries at 365+ (fp16 gmem-direct goldens rank ~6; see `emmy eval analytic`). The refit is the
  gate for the flash-form fork (`flash-enhancements.md`).

## 4. Endgame purge (when the registry hits empty)

Delete `tests/xfail_registry.py`, the `pytest_collection_modifyitems` hook in `tests/conftest.py`,
`TILE_ENTANGLED_FILES`, and unwrap every guarded `try/except ModuleNotFoundError` tile import (e.g.
`test_lowering_error_guardrail.py`). The recovery apparatus is itself the last shim.

## Verification

Per item: the named xfail flips XPASS (delete its entry); `make test` + `make lint`; staging changes assert
bit-identity vs gmem-direct; `tile_signature` invariance for any featurization-adjacent change. Golden A/B after the
scalar ring.
