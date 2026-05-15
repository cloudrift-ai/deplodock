# Plan: give `Stage` a body, then drop derived fields

## Context

`Stage` (`deplodock/compiler/ir/tile/ir.py:198`) today declares a cooperative
load of a slab from gmem to smem via four-tuple geometry (`buf`, `origin`,
`axes`, `addressing` + optional `pad`). The materializer expands that to
a per-thread cooperative-load loop `Load(gmem); Write(smem)`.

This shape is too narrow. The motivating problem: compiling
`F.silu(gate)*up @ w` at tile-IR yields a body that recomputes the silu
chain inside every per-thread reduce iteration, including 16x redundancy
across the N-thread axis and a further 56x redundancy across N-tile blocks
that share the same K-slab. The silu chain is loop-invariant w.r.t. those
axes but `hoist_loop_invariants` can't lift it: the cone is mixed with
`a4`-thread-dependent reads inside the same reduce, and `a4` isn't a serial
Loop. Tile-IR has no other mechanism to share computed values across the
thread grid.

The fix is to let `Stage` carry the producer-side compute. Instead of
"load gate into smem," a fused stage says "load gate and up, run the silu
chain, store the product into smem." All cross-thread / cross-N-tile
redundancy collapses, since the cooperative load already covers each gmem
element exactly once per CTA.

To get there, `Stage` needs a body. Once the body exists, the three fields
that describe a single source slab (`buf`, `origin`, `addressing`) become
redundant — the same information lives inside body Loads' `input` and
`index`. Phase 1 lands the body additively; phase 2 deletes the redundant
fields and derives them on demand for the passes that still want a "single
source slab" view.

The fusion pass itself (a new `008_fuse_stage_epilogue.py` between
`007_stage_inputs` and `008_register_tile`) is out of scope for this plan
— it's the consumer that justifies the refactor. Once `Stage.body` exists,
that pass is a localized rewrite over `Tile` bodies.

## Phase 1 — add `body: Body` to `Stage` (additive)

Goal: every `Stage` instance carries a body that fully describes its
cooperative-load program. Legacy fields stay for now; trivial-body case is
indistinguishable from today's behavior.

### IR change (`deplodock/compiler/ir/tile/ir.py`)

- Add `body: Body = ()` to `Stage`. Subclasses (`BufferedStage`,
  `AsyncBufferedStage`, `TmaBufferedStage`) inherit it through the dataclass.
- Override `nested()` → `(self.body,)`, `with_bodies()` → `replace(self,
  body=bodies[0])`. `Stage` becomes block-structured, picked up by every
  generic walker (`Body.map`, the new `topo_sort_siblings`, validators).
- `has_side_effects()`: keep the default; the body's `Write` (to smem,
  not a graph output) is not a side-effect-pinning Write. Leave the
  existing scope-pinning of Stages handled by callers as it is today.
- Add a `__post_init__` that, when `body` is empty, populates it with the
  canonical `(Load(name="_v", input=self.buf, index=_origin_plus_cache(...)),
  Write(output=self.name, index=_cache_index(...), value="_v"))` synthesized
  from `(buf, origin, axes, addressing)`. This keeps every existing
  constructor working unchanged and guarantees `body` is always populated
  once the stmt is constructed.
- Validation: when both `buf`/`origin`/`addressing` and a non-trivial body
  are supplied, assert that the body's first `Load` matches them
  (input == buf, index matches origin + cache decode). This catches drift
  while both representations coexist.

### Constructor sites (touch each, keep behavior identical via auto-population)

- `pipeline/passes/lowering/tile/007_stage_inputs.py:379` — no change needed;
  `__post_init__` synthesizes the trivial body.
- `pipeline/passes/lowering/tile/010_double_buffer.py:158` — uses
  `dataclasses.replace`; body is carried through automatically.
- `pipeline/passes/lowering/tile/011_tma_copy.py:160` — same, replace().
- `pipeline/passes/lowering/tile/013_async_copy.py:79` — same, replace().
- `pipeline/passes/lowering/tile/012_split_inner_for_swizzle.py:182` —
  constructs a fresh Stage with split axes; the body's Loads still
  reference the original (pre-split) cache vars. Either: (a) re-synthesize
  the trivial body from the new geometry (works for today's case where
  012 only runs on trivial-body TMA stages); or (b) σ-substitute cache
  vars in the existing body. Start with (a) and assert the body is trivial
  here.
- `pipeline/passes/lowering/tile/014_pad_smem.py:193` — `replace(pad=...)`;
  body unchanged.
- `pipeline/passes/lowering/tile/015_pipeline_k_outer.py:158` — calls
  `stage.rewrite(σ)`. Make sure `Stage.rewrite` threads σ through `body`
  so K-outer Vars inside body Loads get substituted along with `origin`.

### Materializer change (`pipeline/passes/lowering/kernel/001_materialize_tile.py`)

- `_emit_stage` (line 648) and `emit_tma_stage` (line 197) keep their
  existing paths for now — they read the legacy fields, which still exist.
  No functional change in phase 1.
- Add an assertion at the top of both that `stage.body` is trivial (one
  Load, one Write) and that body Load's input/index match
  `(buf, origin, axes, addressing)`. This is the migration trip-wire: any
  fused stage will fail here, signaling phase 2 is needed before the
  fusion pass can land.

### Validation hook

- `TileOp.__post_init__` calls `normalize_body` which recurses through
  `nested()` — Stage bodies get the standard normalize pipeline
  (`topo_sort_siblings`, SSA rename, etc.) automatically.

### Tests (additive)

- New unit test: construct a `Stage` with no body; assert the synthesized
  body is the expected `Load + Write` pair.
- New unit test: construct a `Stage` with an explicit non-trivial body;
  assert it round-trips through `rewrite()` (σ-substitution touches both
  legacy `origin` and body Load indices consistently).
- Re-run `tests/compiler/` end-to-end — every existing Stage-emitting
  pipeline should keep passing with byte-identical kernel IR (the trivial
  body is materializer-invisible).

## Phase 2 — drop `buf` / `origin` / `addressing`, derive on demand

Goal: `Stage` carries only what isn't recoverable from the body —
`name`, `axes`, `pad`, and the transport-specific fields on subclasses.
Everything else moves to per-Load accessors that walk the body.

### IR change

- Delete `buf`, `origin`, `addressing` from `Stage`. The trivial-body
  auto-population in phase 1 goes away (callers now pass a body
  directly; no need to synthesize from removed fields).
- Add cached properties:
  - `Stage.source_loads(self) -> tuple[Load, ...]` — every `Load` in
    `body` (post-order), in body iteration order. The trivial case
    returns a 1-tuple.
  - `Stage.primary_load(self) -> Load` — `source_loads[0]`. Used by
    passes that today reach for `buf` / `origin` / `addressing` and
    only make sense for single-source stages; raises if the stage has
    multiple sources.
  - `Stage.buf(self) -> str` — `primary_load.input`. Property, not field.
  - `Stage.origin(self) -> tuple[Expr, ...]` — `primary_load.index` with
    cache axis Vars substituted to `Literal(0)`. Uses
    `_classify`'s existing logic from `007_stage_inputs.py`, lifted into
    a reusable helper in `tile/ir.py` or a sibling module.
  - `Stage.addressing(self) -> AffineAddressing | TemplateAddressing` —
    derived by running the existing classification on `primary_load.index`
    against `axes`. Same helper as above.

### Migrate field readers (the work)

Each site that currently reads `stage.buf` / `stage.origin` /
`stage.addressing` keeps reading them — they're now properties. So the
text-level change at most call sites is zero. Exceptions:

- `pipeline/passes/lowering/kernel/001_materialize_tile.py:648`
  (`_emit_stage`) — instead of `Load(gmem); Write(smem)` synthesized
  from legacy fields, walk `stage.body` and emit each body stmt
  positionally: `Load`s become per-thread gmem reads, elementwise
  `Assign`s become per-thread compute, the terminal `Write` becomes
  the smem store. Cooperative-loop scaffolding (the `StridedLoop` over
  cache axes) stays identical.
- `pipeline/passes/lowering/kernel/001_materialize_tile.py:197`
  (`emit_tma_stage`) — TMA materialization requires a single `Load` with
  `AffineAddressing` and no compute between Load and Write. Add an
  eligibility check via `len(stage.source_loads) == 1 and not _has_compute(stage.body)`;
  if non-trivial, fall back to sync materialization (cp.async / TMA
  subclasses can't carry compute in phase 2).
- `pipeline/passes/lowering/tile/011_tma_copy.py:122,222,226,239,246` —
  reads `buf`, `origin`, `addressing`. After phase 2 these are
  properties, no source change. But add the same single-load-trivial-body
  precondition: TMA promotion only applies to trivial-body stages.
- `pipeline/passes/lowering/tile/012_split_inner_for_swizzle.py:159,169` —
  the assertion that addressing is Affine becomes a property check; the
  rewrite that appends a dim to `addressing.dims` becomes a rewrite of
  the body's `Load.index` (re-classify after rewrite to confirm the
  decoded form is still affine). This is the one site where phase 2
  requires non-trivial rewrite work.
- `pipeline/passes/lowering/tile/015_pipeline_k_outer.py:158` — already
  uses `stage.rewrite(σ)`; phase 1 threaded σ through body. No change.
- `compiler/diagnostics/bank_conflicts.py:118` — reads `Stage` + `Load`
  pairs from the tile body; uses `stage.name`, `stage.axes`, `stage.pad`.
  No legacy field used. No change.

### Tests

- Unit test: construct a `Stage` with a multi-Load body; assert
  `primary_load` raises and `source_loads` returns the right tuple.
- Unit test: TMA eligibility — non-trivial body → no TMA promotion.
- Re-run the full compiler suite; the only behavior change should be in
  the migrated TMA / swizzle paths, which gain a "trivial body required"
  precondition (no production stages today are non-trivial, so existing
  tests pass).

## Out of scope (follow-up)

A new pass `pipeline/passes/lowering/tile/008_fuse_stage_epilogue.py`
that, given a tile body produced by 007, walks each consumer cone whose
SSA transitively depends only on staged buffers + staging axes, and folds
that cone into the upstream Stage's body. This is what eliminates the
silu-chain redundancy in the motivating example. It's a separate plan
because it depends on phase 2 being done (the fused stage *will* have a
non-trivial body, breaking phase-1's trip-wire assertions).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — add `body` field, properties,
  validation; both phases.
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py`
  — assertion trip-wire in phase 1; body-walking emitter in phase 2.
- `deplodock/compiler/pipeline/passes/lowering/tile/007_stage_inputs.py`
  — primary Stage constructor; touched in both phases (phase 2 starts
  passing an explicit body).
- `deplodock/compiler/pipeline/passes/lowering/tile/{010_double_buffer,
  011_tma_copy,012_split_inner_for_swizzle,013_async_copy,014_pad_smem,
  015_pipeline_k_outer}.py` — promote / rewrite Stages; phase 1 needs
  `replace()` audits, phase 2 needs the 011/012 trip-wires.
- `deplodock/compiler/ir/stmt/normalize.py` — already topo-sorts siblings
  via `nested()`; nothing to add, but Stage's new block-stmt status
  exercises that path.
- `tests/compiler/pipeline/passes/lowering/tile/` — extend existing
  per-pass tests; add the new unit tests above.

## Reused utilities

- `Body` / `Body.map` / `Body.coerce` (`deplodock/compiler/ir/stmt/body.py`)
  — body field type, recursion machinery.
- `Stmt.nested()` / `with_bodies()` / `defines()` / `deps()`
  (`deplodock/compiler/ir/stmt/base.py`) — Stage becomes a block stmt; the
  generic walkers in `normalize.py` already use these.
- `_classify` in `007_stage_inputs.py` — the existing logic that maps a
  Load's index to `(origin, addressing)`. Lift into a reusable helper for
  phase-2 property derivation; one source of truth.
- `Stage.rewrite` (existing) — Stage `rewrite` registration in
  `deplodock/compiler/ir/stmt/passes.py` already threads σ through the
  existing fields; extend to thread through `body` once it exists.

## Verification

- After phase 1:
  - `make test` — full suite green, byte-identical kernel IR for every
    Stage-emitting recipe (a recipe-level golden comparison or just
    pytest passing is enough since materialization is unchanged).
  - `deplodock compile -c "torch.matmul(F.silu(torch.randn(1,512,18944))*torch.randn(1,512,18944), torch.randn(18944,3584))" --ir loop`
    — same output as today.
  - `deplodock compile … --ir tile` — every Stage prints with the
    auto-synthesized `body=(Load(...); Write(...))`. Visual confirmation
    + new unit test.
- After phase 2:
  - `make test` — full suite green; behavior changes (TMA/swizzle
    preconditions) only trigger on stages no existing pipeline creates.
  - `make lint` clean.
  - Spot-check `deplodock compile … --ir cuda` on a TMA-capable matmul
    recipe — generated SASS unchanged (since no production stage is
    non-trivial yet).
- After follow-up fusion pass lands:
  - Recompile the silu motivating example; confirm the per-thread reduce
    body no longer contains `v0..v23` silu chain (it lives inside
    `Stage.body` instead). Confirm one fewer smem slab (no `up_smem`).
  - Benchmark vs pre-fusion: expect a measurable speedup on
    silu-gated-matmul recipes; no regression elsewhere.
