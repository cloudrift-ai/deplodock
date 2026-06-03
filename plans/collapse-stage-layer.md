# Collapse the tile-IR staging layer: delete `Stage`, fold sources + compute onto `StageBundle`

## Context

The tile-IR staging path carries **two grouping levels for the same concept**, plus an overloaded field:

- `Stage.sources: tuple[Source, ...]` — group operands filled cooperatively behind one barrier.
- `StageBundle.stages: tuple[Stage, ...]` — group stages sharing a transport policy + the consumer body.
- `Stage.compute: Body | None` — overloads a transport `Stage` to *also* be a "hoisted invariant compute" stage.

These exist for historical reasons. We established (across the preceding discussion) that:

- The shared-barrier semantics a multi-source `Stage` encodes are fully recoverable at the **bundle** level — TMA
  already proves it: `050_use_tma._split_multi_source_stages` splits multi-source into single-source stages and the
  bundle's mbarrier (`arrive_count = N`) re-groups them.
- The only obstacle to a clean bundle-level barrier was the **compute stage sitting as a peer in `stages`**, which
  forces order-dependent per-stage syncs + reliance on `110_drop_redundant_syncs`. Once compute is a distinct **bundle
  phase**, the transport stages are homogeneous and both the multi-source grouping and the `030` cone/non-cone stage
  partition become redundant.
- A single-source `Stage` would be a trivial `Source` wrapper; a multi-source `Stage` a redundant sub-group. So the
  right end state is **no `Stage` layer**: `StageBundle` holds `sources` directly + an optional compute phase + the
  consumer body.

**This is a structural refactor, not a behavioral one — output is byte-identical.** `110_drop_redundant_syncs` already
collapses to the same syncs the bundle-level emission produces. The payoff: one grouping mechanism instead of three;
deletion of `_split_multi_source_stages` and the `030` stage partition; removal of the `Body(self.stages)`
synthetic-body traversal hack; and ~12 `stage.compute is not None` / per-stage-iteration sites simplified to
per-source / per-bundle. Verified by the existing `max_diff=0` accuracy guards.

## Compute is just a `Body` — derive everything else at emit

The hoisted compute reads already-staged sibling slabs, applies a transform, and writes one derived slab. Crucially,
the compute **body is self-describing** and `030` already authors the producer↔consumer SSA binding inside it, so the
phase needs **no extra structured fields** — not an output `Source`, not an output name, not an iteration-axes list:

- **Output slab name** — lives in the body's single `Write.output`. `030` puts the *same* name on the consumer's
  rewritten `Load(input=fused_name, …)`, so the name is an SSA def-use edge between two bodies. It must be chosen when
  both ends are created (at `030`) — it cannot be deferred to emit, because the consumer body already references it —
  but it does **not** need to be stored on a field; it is already in the `Write`.
- **Iteration axes + extents** — the compute body's `Load`s name their input slabs (the cone sources, which live in
  `bundle.sources` with their `cache_axes`). The fused output shares the cone's `cache_axes` (that's how `030` derives
  `fused_cache_axes`), and the `Write.index` Vars are those same axis names. So the materializer recovers the
  cooperative-loop domain from the inputs the body reads.
- **dtype** — the `Write` value's stamped dtype (after `030_stamp_types`).

So:

```python
@dataclass(frozen=True)
class StageBundle(Stmt):
    sources: tuple[Source, ...]    # MOVED off Stage — all gmem transport operands
    body: Body                     # consumer
    compute: Body | None = None    # NEW — self-describing: Loads(sibling slabs) + Assigns + Write(fused)
    policy: StagePolicy = StagePolicy.SYNC
    buffer_count: int = 1
    phase: Expr | None = None
    pipeline_depth: int = 1
# `Stage` is DELETED.
```

Because `compute` is a plain `Body`, the standard `nested()` / `with_bodies()` traversal covers it — no `ComputePhase`
dataclass, no hand-rolled descent for a non-`Stmt` wrapper.

## Milestone 1 — move `compute` from `Stage` to `StageBundle`

Independently shippable; output byte-identical. After M1 the cone/non-cone partition is already gone (transport stays
one multi-source `Stage`; compute is a bundle `Body`).

- **`ir/tile/ir.py`**: add `StageBundle.compute: Body | None`; remove `Stage.compute` (Stage becomes sources-only) and
  its `nested`/`with_bodies`/`external_reads` compute branches.
- **`StageBundle.nested()` / `with_bodies()`**: expose `compute` alongside the consumer `body`. At M1 the member stages
  still need traversal, so keep the existing stages descent (the `Body(self.stages)` synthetic body) and add `compute`:
  e.g. `nested() -> (Body(stages), compute or Body(()), body)` with the matching 3-arity `with_bodies`. (M2 drops the
  stages dimension, leaving the clean 2-arity `(compute or Body(()), body)`.)
- **`ir/tile/passes.py`**: the `@rewrite` / `@simplify` `StageBundle` handlers descend `compute` as a `Body` (mirroring
  the existing `body` override). Remove the `Stage.compute` descent from the `Stage` handlers.
- **`tile/030_hoist_invariant_compute.py`**: keep cone *detection* (`_try_find_cone`, `_ssa_dataflow`) but in the
  True-polarity rewrite emit **one transport `Stage` holding all original sources** + `StageBundle.compute = <compute
  body>` — drop the `(non_cone, cone, compute)` 3-stage split and the `fused_source` `Source`. The compute body is the
  existing `compute_stmts` (cone Loads + Assigns + `Write(fused_name, …)`); the consumer reduce rewrite is unchanged.
  **The single transport stage's source order must equal `non_cone_sources + cone_sources`** (byte-identical risk #1).
- **`kernel/100_materialize_tile.py` `emit_bundle_producer`**: emit transport stages, then — if `bundle.compute` — emit
  the compute phase from the bundle body. Derive the compute slab name (`Write.output`), loop domain (from the
  `cache_axes` of the input sources the compute `Load`s), and dtype (`Write` value) at emit. The TMA "trailing wait
  once" filter `[m for m in bundle.stages if m.compute is None]` becomes just the transport stages. Preserve the
  first-seen `Smem`-decl order for the compute slab (byte-identical risk #2, highest on the gated-MLP shape).
- **`_stage_expand.emit_compute_stage`**: take the compute `Body` (+ the derived axes/dtype) instead of a `Stage`.
- **`kernel/020_place_inits.py`, `kernel/030_stamp_types.py`**: descend / stamp `bundle.compute` instead of the member
  compute-stage.

## Milestone 2 — collapse `sources` onto the bundle, delete `Stage`

Mechanical after M1; output byte-identical.

- **`ir/tile/ir.py`**: move `sources` onto `StageBundle`; delete `Stage`. `nested()` becomes `(compute or Body(()),
  body)`. Update `local_decls` / `external_reads` / `exprs` / `smem_bytes` to read `self.sources` directly.
- **Delete `tile/050_use_tma._split_multi_source_stages`** — the materializer + `_tma_groups` already iterate
  per-`Source` (keyed by `Source.name`), so TMA emits one descriptor per source directly; `emit_tma_stage`'s
  `assert len(stage.sources) == 1` becomes a per-source loop over `bundle.sources`.
- **Flatten per-stage → per-source** everywhere. Pattern: replace `for stage in bundle.stages: for src in
  stage.sources:` with `for src in bundle.sources:`, and **delete** any `isinstance(stmt, Stage)` arms (bare `Stage` no
  longer appears in `body.iter()` once the synthetic stages-Body is gone). Sites:
  - tile: `020_stage_inputs`, `021_hoist_staged_loads_above_mask`, `026_unify_sibling_stages`, `040_use_ring_buffers`,
    `050_use_tma` (`_stamp_source_swizzle`, `_stage_eligible`, `_promote`), `060_use_async_copy`, `070_pad_smem`.
  - kernel: `005_lower_atom_tile`, `010_split_register_axes` (delete its `Stage` branch), `030_stamp_types` (drop the
    top-level `isinstance(s, Stage)` dispatch arm), `050_vectorize_loads` (delete `Stage` branch),
    `060_permute_lane_accesses` (delete `Stage` branch), `100_materialize_tile`, `_tma_groups._add_bundle`.
  - core: `ir/tile/passes.py` (StageBundle handler descends `sources` exprs + `compute` + `body`; delete the `Stage`
    handlers), `map_staged` in `ir/tile/ir.py` (thread `bundle.sources` directly).
- **`080_pipeline_stages.py`**: still assumes exactly one `StageBundle` per `SerialTile(serial_outer)` (unchanged — we
  keep one bundle); confirm its monolithic `rewrite()` / `_issue_only` / phase-prefix logic flows through the new shape.

## Verification (run after each milestone; commit on green per the single-branch-milestones convention)

```bash
# M1
./venv/bin/pytest tests/compiler/ir/tile/test_stage_bundle.py tests/compiler/passes/test_knob_pinning.py \
  tests/compiler/passes/test_materialize_warp_specialize.py tests/compiler/passes/test_stage_inputs_block_recognition.py \
  -p no:randomly -q
# golden CUDA diff on the gated-MLP kernel (the real gate for the 030 cone-collapse): compile pre/post, diff source

# M2
./venv/bin/pytest tests/compiler/passes/ tests/compiler/ir/tile/ -p no:randomly -n auto --dist=loadgroup -q
./venv/bin/pytest tests/compiler/test_matmul_mma_tma.py tests/compiler/test_matmul_mma.py \
  tests/compiler/test_matmul_mma_staged_pipelined.py tests/compiler/test_use_tma.py -p no:randomly -q

# final
make test && make lint
```

`test_tma_mma_matches_f32_reference` (`max_diff=0`) is the backstop: a mis-ordered source list, dropped compute-body
σ-substitution, or wrong `Smem`-decl order would corrupt it.

## Risks (validated against the code)

All four originally-flagged risks were traced and come back **LOW severity**. The one previously called "the only spot
that might bite" (consumer resolution) is effectively a non-issue; the real watch-items are the two ordering risks,
which the golden-CUDA diff catches directly.

1. **Consumer resolution of the fused slab — LOW (effectively safe).** The `map_staged` in-scope slab table has exactly
   one consumer: `kernel/005_lower_atom_tile`, which uses it to resolve **mma operand** slabs
   (`smem_sources[a_load.input]`). Compute-hoist and mma are **structurally mutually exclusive**: `030`'s cone detection
   requires a non-empty `cone_assigns` set (`if not cone_assigns: continue`), and an mma reduce body is `Load a / Load b
   / Mma` — `Mma` is not an `Assign`, so no cone ever fires on an mma kernel. Hence a compute-hoist bundle never carries
   an `AtomTile`, and the fused slab is never looked up in that table. Separately, the consumer `Load(input=fused)`'s
   **dtype** does not come
   from the fused `Source`'s stamp anyway — `030_stamp_types` leaves the fused `Source` **unstamped** today (`buf` names
   no graph node → render-time fallback), and the consumer Load resolves via the `source_dtypes` / global-smem / graph /
   fallback chain. So dropping the fused `Source` removes nothing load-bearing. No mitigation required.
2. **Source order in the collapsed transport stage (M1) — LOW.** Keep the merged transport stage's `sources` equal to
   `non_cone_sources + cone_sources` so the cooperative-load emit order (hence the `110`-collapsed sync sequence) is
   unchanged. Caught by the golden-CUDA diff.
3. **`Smem`-decl first-seen order (M1) — LOW.** Emit the fused slab's `Smem` decl in its current position (after the
   transport sources). Same golden-diff guard.
4. **`passes.py` compute descent — LOW (likely a no-op today, but mandatory).** `080_pipeline_stages._try_pipeline`
   only fires on `policy in {ASYNC, TMA}`; compute-hoist bundles are emitted `SYNC` and the compute phase blocks TMA
   (`050` is all-or-nothing, rejects compute). Even if a hoist bundle were promoted to ASYNC and pipelined, the cone is
   K-invariant **by construction** (`030` requires `free_vars ⊆ cache_axes`, excluding the K index), so `080`'s
   σ-substitution (a K-offset) is a no-op on the compute body. Still, the `StageBundle` rewrite handler **must** descend
   `compute` (a one-line addition mirroring the `body` override) — forgetting it is silent, and harmless only as long as
   the K-invariance holds.

**Plan correction found during validation:** the hoist/gated test is `tests/compiler/passes/test_knob_pinning.py` (not
`tests/compiler/`). The Verification block above is fixed accordingly.

## Tests to rewrite (construct `Stage` / `StageBundle` directly — update to the new shape; none need deleting)

`tests/compiler/ir/tile/test_stage_bundle.py`, `tests/compiler/passes/test_use_tma.py`,
`tests/compiler/passes/test_warp_specialize.py`, and any `test_ring_buffer*` / `test_pad_smem*` / `test_stage_inputs*`
that build stages. `test_knob_pinning.py` (gated-MLP cone + multi-source TMA) and the `matmul_mma_*` digests are the
end-to-end guards and should pass unchanged.

## Branch / sequencing

One branch off `main` (e.g. `refactor/collapse-stage-layer`), milestone commits after each green test run. M1 is the
conceptual keystone and can ship / pause independently; M2 is the mechanical payoff. Both byte-identical.
