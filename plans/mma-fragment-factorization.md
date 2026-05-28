# MMA Fragment Factorization

## Context

Adding WMMA / `mma.sync` (sm_70+ FP16 and sm_80+ BF16) tensor-core support to deplodock is naturally framed as a new
*fragment shape* on the matmul reduce, **not** a new lowering branch alongside the existing scalar register-tile path.
Every matmul kernel — scalar or MMA — produces independent accumulator cells along output M and N, replicated by the
planner along its REGISTER (`FM`/`FN`) axes. The only differences are (a) how many threads jointly own one cell (1 for
scalar, 32 for WMMA, 128 for wgmma), and (b) what hardware instruction updates that cell per K step (`fma` for scalar,
`mma.sync` for WMMA, `wgmma` for sm_90+).

Unifying the two paths under a single **atom kind** knob — with the scalar path being just `(1, 1, 1)` — collapses what
would otherwise be two parallel planner enumerators, two parallel register-replication passes, and two parallel
materializer branches into one factorization with a per-cell dispatch on `ATOM_KIND`.

> ### ⚠️ Codebase drift since this plan was first written
>
> This plan was authored against an older IR where output-axis bindings lived as `Role` / `BIND_BLOCK` / `BIND_THREAD`
> attributes in `ir/axis.py`. **That model has been deleted.** The current state (verified 2026-05-27):
>
> - **Bindings are encoded by tile-flavor *type*.** `ir/axis.py` carries a bare `Axis` (name + `Dim` extent +
>   `source_axis` back-pointer + `real_extent`). The binding decision is the *class* of the wrapping Stmt:
>   `GridTile` (blockIdx) / `ThreadTile` (threadIdx) / `RegisterTile` (per-thread register cell) / `SerialTile`
>   (`kind ∈ {plain, serial_outer, stage_inner, pipeline}`) / `StridedTile`. All live in
>   `deplodock/compiler/ir/tile/ir.py`. There is no `Role` enum, no `BIND_*` constant, and **no `BIND_WARP` to add** in
>   `axis.py`.
> - **`Role` survives only as a planner-internal label.** `010_partition_loops.py` defines a private
>   `class Role(enum.Enum)` (`BLOCK / THREAD / REGISTER / STAGE_INNER / SERIAL_OUTER / PIPELINE`) used by `_wrap_tower`
>   to pick the flavor. Its own docstring says it "never reaches downstream passes (which discriminate on tile-flavor
>   type instead)." So `Role.ATOM`, if added, lives **in the planner module**, not `axis.py`, and must be paired with a
>   concrete tile flavor (or be knob-only — see Design decision 2).
> - **The factorization is 3-level — `BLOCK · THREAD · REGISTER` — with no GROUP/WARP tier.** Output axis split (from
>   the `010_partition_loops` docstring): `A → A_b·(T·R) + A_t·R + A_r` where `T = BN|BM` (THREAD) and `R = FN|FM`
>   (REGISTER). K split: `K → K_s·(K_o·br·bk) + K_o·(br·bk) + K_i·br + K_c` (`K_s` split-K BLOCK, `K_c` cooperative-K
>   THREAD). A warp tier between BLOCK and REGISTER is genuinely *new* structure this plan must add.
> - **`lane` / `warp` are already render-derived, not bound.** `ThreadTile.render` emits
>   `int lane = threadIdx.x & 31; int warp = threadIdx.x >> 5;` when `_body_uses_lane_warp(body)` is true. WMMA can lean
>   on this rather than inventing a binding.
> - **The register-replication pass moved and was renamed.** `006a_register_tile_planned` is now
>   `deplodock/compiler/pipeline/passes/lowering/kernel/010_split_register_axes.py` (a `TileOp → TileOp` rewrite that
>   consumes `RegisterTile` and replicates its body per cell, before `100_materialize_tile.py`). Several docstrings/
>   comments still say "006a" — stale. **This pass, not the materializer, is where the scalar per-cell `Init`/`Accum`
>   chain is produced.**
> - **The planner emits a lazy Fork tree, not a flat cartesian.** `rewrite` returns `Graph | None | TileOp | Fork |
>   list[Fork]`; `_build_fork_tree_lazy` nests `_group_level → _bmbn_level → _fmfn_level → _br_level → _leaf_forks`.
>   Each fork is an *op fork* (graph-preserving, separable — see `autotune_no_graph_forks` memory) consumed by the
>   two-level tuner. An `ATOM_KIND` dimension is a new **fork level**, not an outer cartesian loop.
> - **`Context` has `compute_capability: tuple[int, int]`** (e.g. `(8, 0)`) plus `warp_size: int = 32` — **no
>   `ctx.arch` int**. Eligibility uses `ctx.compute_capability >= (7, 0)`. `_helpers.py` re-exports
>   `compute_capability()` (a tuple) from `compiler.target`.
> - **There is no `BF16` `DataType`.** `dtype.py` defines `F32`, `F16`, `F16x2` only. `backend/cuda/dtype.py` and
>   `backend/torch_ref.py` have some bf16 spelling, but a first-class `BF16 = DataType("bf16", …, 2)` must be **added**
>   before M9 can register bf16 atom kinds.
> - **The knob module is `pipeline/knob.py` (singular).** `KnobType ∈ {INT, BOOL, BINMASK}` — **no `STR`**. Knobs are
>   declared as module-level `Knob(...)` constants *inside the owning rule* and harvested by a `sys.modules` registry
>   walk; there is no central registration file. `format_tuning_knobs` currently only drops `BOOL` markers.
> - **`permute_lane_accesses` lives in `kernel/`, not `tile/`** — `kernel/060_permute_lane_accesses.py`;
>   `pack_fp16_pairs` is `kernel/070_pack_fp16_pairs.py`. Both are `TileOp → TileOp`.
> - **There is no `001_launch_geometry` pass.** `010_partition_loops` *is* the launch-geometry + factorization
>   decision; it directly emits the `GridTile/ThreadTile/RegisterTile` tower via `_wrap_tower`. Docstrings that mention
>   `001_launch_geometry` are stale.
> - **Kernel render builds includes from signature dtypes** (`cuda_includes(sig_dtypes)`) and prepends a forward-decl
>   `_TMA_PRELUDE` when TMA descriptors are present — *because NVRTC does not ship `<cuda.h>`/`<cuda/barrier>`*. The same
>   risk applies to `<mma.h>` (see M6 + Failure modes): the intrinsic path may not be NVRTC-available and a raw-PTX
>   `mma.sync` fallback may be forced, contradicting Design decision 10.
>
> The milestones below are rewritten to match this reality. Where the original structural choice no longer maps
> cleanly, the adjustment is called out inline and any genuinely-open decision is flagged rather than invented.

### The unified factorization

For each output axis (one shown; M / N / K symmetric where applicable), the goal is to extend today's 3-level split
with a warp (GROUP) tier and a hardware ATOM tier:

```
N → N_b · (GROUP_N · CELL_N · ATOM_N) + N_g · (CELL_N · ATOM_N) + N_c · ATOM_N + N_a

       BLOCK              GROUP                  CELL              ATOM
```

- `N_b` — `GridTile` (today's `N_b`, `Role.BLOCK`).
- `N_g` — the **new** warp tier. Scalar collapses it (`GROUP_N = 1`, no warp tile); MMA makes it the per-N warp count.
  In the current model this is **not** a binding — it is either a new `WarpTile` flavor or a repurposing of the
  `ThreadTile` axes with an implicit ×32 lane dimension (see Design decision 3 — open).
- `N_c` — `RegisterTile`, replication count. Today's `FN` for scalar; the MMA cell count per warp.
  `010_split_register_axes` replicates the body per cell regardless of what the body contains.
- `N_a` — the hardware-atomic extent. For scalar `ATOM_N = 1` (dropped by `_wrap_tower`'s size-1 filter, so the scalar
  tower is structurally unchanged). For MMA `ATOM_N = 16`. The materializer never iterates it; it dispatches on
  `ATOM_KIND` and emits one scalar or fragment instruction per cell.

K side: today's `K_o · (br · bk) + K_i · br + K_c` factorization gains an inner ATOM_K extent. Scalar `bk` = today's
`bk` × 1; MMA effective K-chunk = `bk × 16` (WMMA K dim). A `FRAG_K`-style count (MMA-only) controls how many `mma.sync`
instructions fire per K_o iteration — analogous to today's `bk` role for scalar.

### What this dissolves

- Two factorizations → one. `_plan_kernel` / `_build_split_body` keep their three-way detection (matmul /
  cooperative-reduce / pointwise); the matmul branch enumerates a single fork space over
  `(BM/BN, FM/FN, BK, SPLITK, BR, ATOM_KIND)`. Scalar is `ATOM_KIND="scalar"`; MMA is e.g.
  `ATOM_KIND="wmma_m16n16k16_f16"`.
- Two register-replication passes → one. `010_split_register_axes` already walks `RegisterTile` and replicates body
  stmts; what it replicates (scalar `Init`+`Accum` vs MMA fragment decls + `MmaSync`) is decided at materialize time by
  the atom kind — not by a separate replication pass.
- Two materializer paths → one dispatch. `100_materialize_tile.py` (and/or `010_split_register_axes`, see M5) reads
  `ATOM_KIND` and dispatches. Scalar emission stays the default; MMA is a single new branch.
- The fp16 `__half2` packing pass (`kernel/070_pack_fp16_pairs.py`) is conceptually `ATOM_KIND` "f16x2 scalar" packing.
  v1 keeps it a separate post-pass and skips it on MMA kinds (M7).

### Scope guard

**Fragmentize only matmul reductions** (entry point `is_matmul_reduce` in `tile/_helpers.py`). Pointwise, softmax,
cooperative-reduce, SDPA's non-matmul reduces — none get an atom kind. For non-matmul kernels `ATOM_KIND` is implicitly
`"scalar"` and never explored.

### Risk note up front

The riskiest single step is M5 (materializer dispatch on `ATOM_KIND`): an off-by-one in the fragment lane mapping or
smem→fragment address calc silently produces wrong matmul results. WMMA accumulators are warp-shared across distributed
registers; a single thread can't sanity-check its value. Verification compares full output matrices against a PyTorch
reference at end-to-end correctness, not at the per-thread level.

A secondary risk is **NVRTC `<mma.h>` availability** — the codebase already forward-declares TMA intrinsics
(`_TMA_PRELUDE`) precisely because NVRTC omits the headers. If `wmma::*` intrinsics aren't reachable, M6 must fall back
to a raw-PTX `mma.sync` prelude, which contradicts Design decision 10 and enlarges M4/M5. Probe this *before* M4 (a
5-line NVRTC smoke compile of a `wmma::fragment` kernel) so the IR-Stmt shape is decided with the answer in hand.

A third risk is the planner fork explosion: today's matmul enumerator already produces dozens of variants per kernel.
Adding an `ATOM_KIND` fork level with 1–2 extra options up to ~doubles the per-kernel search; the two-level tuner's
patience stop and `_priority_matmul` ordering (MMA variants ranked first when eligible) are the mitigations.

## Design decisions

1. **`ATOM_KIND` lives on `LoopOp.knobs` / `TileOp.knobs`, with a registry resolving it to a full spec.** A kernel-wide
   *string* knob naming the atom kind (`"scalar"`, `"wmma_m16n16k16_f16"`, …); a module-level
   `ATOM_REGISTRY: dict[str, AtomSpec]` maps the kind to a frozen record carrying shape `(M, N, K)`, per-operand dtype
   dict (`{"a": F16, "b": F16, "c": F32}`), the hardware instruction family, and the group size (threads per cell).
   Future kinds (NVFP4/MXFP4 scaled MMA, wgmma) extend the registry rather than the knob schema. Eligibility predicates,
   the fork enumerator, the materializer dispatch, and the warp-tier launch-geometry all read from the registry.
   Resolve with `op.knobs.get("ATOM_KIND", "scalar")`.

2. **No `Role.ATOM` *binding* — ATOM is registry-resolved metadata, not an `axis.py` enum.** *(Changed from the
   original: `axis.py` no longer has a `Role` enum.)* Two viable shapes:
   - **(2a, recommended) Knob-only ATOM.** The atom extent is *not* a tile-flavor layer. The planner sizes the
     `RegisterTile` (CELL) axes and the σ-arithmetic to cover the cell grid; `ATOM_KIND` carried as a knob tells the
     materializer to emit one fragment instruction per replicated cell. The scalar path is byte-identical today (no new
     flavor, no new layer to inline). Downstream tile passes need no awareness of a new flavor.
   - **(2b) `AtomTile` flavor + planner-internal `Role.ATOM`.** Add `Role.ATOM` to the planner enum, a `_layer_kind_for`
     mapping to `"atom"`, an `AtomTile(ParallelTile)` flavor, and `_wrap_tower` handling. The scalar extent-1 ATOM layer
     is dropped by `_wrap_tower`'s existing size-1 filter (`010_partition_loops.py:517`, drops any non-BLOCK extent-1
     axis), so the scalar tower is still unchanged. But *every* tile/kernel pass would have to learn to pass `AtomTile`
     through — net new surface for marginal structural benefit now that Role is planner-internal.
   - **Recommendation: 2a.** It matches the "tile-flavor type encodes the decision; everything else is a knob"
     invariant and keeps the scalar no-op trivially true. 2b is only worth it if a later async kind (wgmma/tcgen05)
     needs the atom layer to carry issue/wait scheduling structurally — revisit then.

3. **The warp (GROUP) tier — open decision, must be resolved in M1/M3.** *(Replaces the original `BIND_WARP` decision,
   which targeted a deleted binding.)* WMMA has 32 threads jointly own one 16×16 cell, so the per-element `ThreadTile`
   tier (one thread = one output element) does not apply to MMA; instead warps are distributed over M/N. Today
   `ThreadTile.axes` product == per-CTA thread count, and both `_build_linear_tid` (`100_materialize_tile.py`) and
   `_launch_bounds_for` (`ir/kernel/render.py`) assume that. Options:
   - **(3a) New `WarpTile(ParallelTile)` flavor** whose axis-extent product is the *warp* count; `_launch_bounds_for`
     and the linear-tid math learn `threads = warps · 32`; `lane = threadIdx.x & 31` is the implicit intra-atom
     coordinate the materializer consumes.
   - **(3b) Reuse `ThreadTile`** with the convention that for MMA the thread axes are warp axes and one synthetic
     ×32 lane factor is folded into the launch-bounds/tid computation under an `ATOM_KIND != "scalar"` branch.
   - 3a is cleaner (the flavor type keeps encoding the binding tier) and slots beside a future `WgroupTile` (128-thread
     groups). 3b is less code but smuggles a non-obvious ×32 into the tid math. **Pick in M1 once the scalar no-op is
     in; default to 3a unless it bloats the no-op milestone.**

4. **Eligibility predicate.** `is_atom_eligible(kind, loop_op, ctx) -> bool`, dispatched via the registry. The
   `wmma_m16n16k16_f16` predicate checks: (a) `is_matmul_reduce` fires on at least one reduce in the body; (b) every
   K-indexed Load and the Accum target dtype is `F16`; (c) `ctx.compute_capability >= (7, 0)` (BF16 kinds require
   `>= (8, 0)`); (d) output M, N extents divisible by 16; K extent divisible by 16 (defer the `% (16·BR)` check until
   BR is picked). Lives in `tile/_helpers.py` beside `is_matmul_reduce`, or in `tile/_atom.py`.
   `is_atom_eligible("scalar", …)` is always `True`.

5. **Atom-kind candidate set, v1.** `"scalar"` always; `"wmma_m16n16k16_f16"` when eligible (the WMMA "square" shape —
   broadest arch support, simplest lane mapping). BF16 + skewed WMMA shapes land in M9. NVFP4/MXFP4/wgmma wait for their
   own plans.

6. **Scalar path is a no-op refactor.** M1 plumbs `ATOM_KIND` end-to-end with `"scalar"` as the only registered kind.
   Existing golden IR tests must pass byte-identical post-normalization — `ATOM_KIND="scalar"` is elided by
   `format_tuning_knobs`, and the emitted tower is structurally unchanged (no ATOM layer under approach 2a; an
   inlined extent-1 layer under 2b). If anything in M1 changes a golden, that's a bug, not a re-bless.

7. **`pack_fp16_pairs` interaction.** For MMA kernels there are no scalar f16 `Init`/`Accum` to pair — the C-fragment IS
   the accumulator. Skip when `ATOM_KIND != "scalar"` (M7).

8. **`permute_lane_accesses` interaction.** *(File moved to `kernel/`.)* Permutes LDS.128 indices on staged Loads to
   break bank conflicts. WMMA uses its own swizzled `load_matrix_sync` access. Skip when `ATOM_KIND != "scalar"` (M7).

9. **Other downstream passes stay unchanged.** The tile chain (`020_stage_inputs`, `030_hoist_invariant_compute`,
   `040_use_ring_buffers`, `050_use_tma`, `060_use_async_copy`, `070_pad_smem`, `080_pipeline_stages`,
   `090_mark_unroll`) operates on K_o / stage structure and doesn't touch the output-axis cell tower below the reduce —
   confirm by reading each `PATTERN` before relying on this. Under approach 2a no new flavor exists, so there is nothing
   for them to skip.

10. **MMA via `wmma::load_matrix_sync` / `mma_sync` / `store_matrix_sync` in v1, *if NVRTC ships `<mma.h>`*.** The
    intrinsic path is simplest to plumb and verify. **Gated on the M6/Failure-modes NVRTC probe** — if the header isn't
    reachable, fall back to a raw-PTX `mma.sync` + `ldmatrix` prelude (analogous to `_TMA_PRELUDE`). The IR Stmt
    abstraction (`MmaLoad` / `MmaSync` / `MmaStore`) is the same either way; only the `render()` body text differs.

---

**Prerequisite landed:** `Axis.source_axis` is in place (`ir/axis.py` — every split sub-axis back-points to its parent
via `Axis.split`, `compare=False`/`hash=False` so Var-rename invariance holds). The MMA enumerator can use it for
BLOCK·GROUP·CELL·ATOM grouping without name-suffix matching.

## M1 — Plumb `ATOM_KIND` registry through the planner as a no-op

**Why.** Establish the unified-factorization scaffolding without changing any emitted CUDA. The scalar path goes
through the new code path with `ATOM_KIND="scalar"`. If this milestone shifts any golden IR or any test result, the
scaffolding is wrong; fix before proceeding. **Also resolve Design decision 3 (warp tier 3a vs 3b) here** — the choice
affects `_wrap_tower` / `_launch_bounds_for` and is cheapest to make while everything is still scalar.

**Change.**

- New file `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py`: define `AtomSpec` (frozen dataclass:
  `shape: tuple[int, int, int]`, `operand_dtypes: Mapping[str, DataType]`, `instruction: str`, `group_size: int`) and
  `ATOM_REGISTRY: dict[str, AtomSpec]` seeded with one entry
  `"scalar" → AtomSpec((1,1,1), {"a":…,"b":…,"c":F32}, "fma", 1)`.
  Public helpers `atom_spec(kind)`, `atom_shape(kind)`, `atom_group_size(kind)`. Prefixed `_` so the rule loader skips
  it (`engine._load_rules` filters `startswith("_")`) — same convention as `_helpers.py`.
- `deplodock/compiler/pipeline/knob.py`: add `STR` to `KnobType` (parse = identity, pretty = `str()`). In
  `format_tuning_knobs`, elide `ATOM_KIND="scalar"` (default-elision — extend the existing marker-drop loop with a
  per-knob default check, or special-case the `"scalar"` value).
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py`: declare the knob as a module-level constant
  beside `BN`/`BM`/… (the registry walk harvests it): `ATOM_KIND = Knob("ATOM_KIND", KnobType.STR, hints=("scalar",),
  help="Hardware MMA atom kind (scalar = per-thread FMA register tile)")`. In `_build_split_body`'s matmul tower
  construction (around lines 1430–1475), under approach 2a stamp `knobs["ATOM_KIND"] = "scalar"` on every emitted
  variant and otherwise leave the tower untouched; under 2b also prepend `(N_a, Role.ATOM)` / `(M_a, Role.ATOM)` /
  `(K_a, Role.ATOM)` extent-1 layers to the `layers` list and the K-sigma. `TileParams` gains an `atom_kind: str =
  "scalar"` field (frozen — keeps de-dup working). `_materialize` / `_score_variant` thread it through.
- If approach 3a (recommended) is chosen: add the `WarpTile` flavor stub to `ir/tile/ir.py` now (unused at scalar) only
  if it keeps M1 small; otherwise defer the flavor to M3 and keep M1 purely knob-plumbing.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~40 lines, new)
- `deplodock/compiler/pipeline/knob.py` (~15 lines: `KnobType.STR` + default elision)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~20 lines: knob const + `TileParams`
  field + stamping)

**Verification.** `make test` byte-clean — no golden bless. Spot-check one matmul kernel's lowered CUDA dump under
`DEPLODOCK_DUMP_DIR` against a pre-M1 snapshot: identical. Confirm `ATOM_KIND` does not appear in any `perf`-context
key or rendered knob string for scalar variants (else -O1/-O3 rows and pre-M1 cache rows split spuriously).

## M2 — Per-kind eligibility predicate

**Why.** Before adding MMA kinds to the fork space, the planner must know which kernels qualify. A wrong predicate
silently disables MMA on eligible kernels (perf regression) or enables it on ineligible ones (compile error / wrong
output).

**Change.**

- `_atom.py`: add `"wmma_m16n16k16_f16" → AtomSpec((16,16,16), {"a":F16, "b":F16, "c":F32}, "wmma", 32)`. Add a
  per-entry `eligibility` callable on `AtomSpec` (or a parallel `ATOM_ELIGIBILITY` dict). Public dispatcher
  `is_atom_eligible(kind, loop_op, ctx) -> bool`.
- WMMA-F16 predicate per Design decision 4, using `ctx.compute_capability >= (7, 0)` (**not** `ctx.arch`). Reuse
  `is_matmul_reduce` from `tile/_helpers.py`; read Load/Accum dtypes off the Loop-IR body (pre-`030_stamp_types`, so
  derive from `graph.nodes[buf].output.dtype` via the `match`/`ctx` graph if the body Loads don't yet carry `.dtype`).
- Module-level `_ATOM_KINDS_V1: tuple[str, ...] = ("scalar", "wmma_m16n16k16_f16")` in priority order.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~30 lines)

**Verification.** Unit test `tests/compiler/passes/test_partition_planner_mma.py`: `Context(compute_capability=(7,5))`
+ a TinyLlama-shape matmul `LoopOp` with f16 Loads → `is_atom_eligible("wmma_m16n16k16_f16", …)` True; same with f32 →
False; `(6, 0)` (Pascal) → False; a softmax `LoopOp` → False. `is_atom_eligible("scalar", …)` always True.

## M3 — Extend the planner fork tree over `ATOM_KIND`

**Why.** Wire eligible atom-kind candidates into variant enumeration so the tuner sees scalar and MMA configs as
siblings of one op fork — consistent with the two-level tuner (`autotune_no_graph_forks`,
`project_two_level_tuning`).

**Change.** In `010_partition_loops.py`:

- `_plan_kernel`: compute the eligible kinds (`[k for k in _ATOM_KINDS_V1 if is_atom_eligible(k, loop_op, ctx)]`) once
  per matmul kernel and stash on the `_Plan` / `KernelShape`.
- `_enumerate_cartesian_impl` (the inner enumerator, ~line 1111): for each base `(bm, bn, fm, fn, bk, splitk, br)`
  variant, emit one `TileParams` per eligible kind, applying the kind's divisibility (`E_M % (bn_c·fn·atom_n) == 0`,
  etc.) and the warp-tier constraint (`group_size > 1 ⇒` BN/BM are warp counts, not per-element thread widths — wire
  through the Design-decision-3 representation). Stamp `atom_kind` on each `TileParams`.
- `_build_fork_tree_lazy` (~line 251): add an `ATOM_KIND` fork **level** (outermost or just-inside `_group_level`, so
  MMA vs scalar is an early, cheap drill decision). Mirror the existing `_group_level / _bmbn_level / _fmfn_level /
  _br_level` structure with a `_kind_level`.
- `_priority_matmul` (~line 952): rank MMA variants strictly above scalar when both are present (MMA amortizes K-loop
  overhead far better). Keep the existing thread-count-near-256 tiebreaker.
- Build the actual `WarpTile`/`ThreadTile` tower per Design decision 3 in `_build_split_body` for `atom_kind != "scalar"`.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~50 lines)
- `deplodock/compiler/ir/tile/ir.py` (~30 lines if adding `WarpTile` per 3a)

**Verification.** Unit test (same file as M2): an eligible TinyLlama matmul `LoopOp` → enumerated `TileParams` include
≥1 with `atom_kind="wmma_m16n16k16_f16"`; a non-eligible (f32) matmul → none; a pointwise kernel → no `ATOM_KIND` knob
stamped (defaults to `"scalar"`). Fork-tree shape: assert the kind level appears and MMA leaves sort before scalar.

## M4 — Kernel-IR Stmts: `MmaFragment`, `MmaLoad`, `MmaSync`, `MmaStore`

**Why.** Hardware primitives for the materializer to emit. Live in `deplodock/compiler/ir/kernel/ir.py` beside `Smem`,
`Sync`, `TreeHalve`, `CpAsyncCopy`, `TmaLoad` — these are hardware-ISA concepts, not Tile-IR scheduling.

**Change.** Four new `@dataclass(frozen=True)` `Stmt` subclasses, each implementing `pretty(indent)` and
`render(ctx: RenderCtx)` (matching the existing per-Stmt render convention — Stmts emit their own CUDA lines):

- `MmaFragment(name, role, shape: tuple[int,int,int], dtype: DataType)`. `role` is a free-form string (`"a"`/`"b"`/`"c"`;
  future kinds add `"a_scale"`, …). Renders `wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> name;`
  (role-dependent matrix tag + layout).
- `MmaLoad(frag, src_buffer, src_offset: Expr, ldm: int)`. Renders `wmma::load_matrix_sync(frag, &<buffer>[offset], ldm);`.
- `MmaSync(c_frag, a_frag, b_frag)`. Renders `wmma::mma_sync(c, a, b, c);`. Future scaled-MMA kinds (NVFP4) get a
  sibling 5-operand Stmt; async kinds get `MmaIssue`/`MmaWait`.
- `MmaStore(dst_buffer, dst_offset: Expr, frag, ldm: int, layout: Literal["row","col"])`. Renders
  `wmma::store_matrix_sync(&<buffer>[offset], frag, ldm, wmma::mem_row_major);`.

Note on `structural_key`: kernel-IR Stmts don't define per-Stmt `structural_key()` — structural keying is at `Body`
level (`Body.structural_key()` runs `normalize_body`); rename-invariance comes from the SSA renamer, not a per-Stmt
key. So **do not** add a `structural_key` method (the original plan's M4 assumed one); instead ensure the new Stmts
implement `defines()` / `deps()` / `exprs()` / `local_decls()` correctly so the existing dedup and rename machinery
treats fragment names as renameable SSA and the `src_offset` Expr participates in keying.

**Files.**

- `deplodock/compiler/ir/kernel/ir.py` (~120 lines)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~6 lines: add the four Stmts to the kernel-dialect table around line 296+)

**Verification.** `tests/compiler/ir/test_kernel_mma.py`: construct each Stmt, `render` with a stub `RenderCtx`, assert
emitted CUDA matches a golden string. SSA-rename: rename a fragment name across a `MmaSync` + `MmaStore` pair and assert
the body still round-trips.

## M5 — Materializer dispatch on `ATOM_KIND`

**Why.** The load-bearing change. When `ATOM_KIND != "scalar"`, the matmul cell emits an MMA fragment chain instead of
scalar `Init`/`Accum`. This is where correctness can silently break.

**Dispatch point — decide between two passes (both `TileOp → TileOp`/`KernelOp`):**

- The **scalar per-cell `Init`/`Accum` chain is produced in `kernel/010_split_register_axes.py`**
  (`_replicate_register_tiles`), *before* `100_materialize_tile.py`. The materializer currently passes `RegisterTile`
  through (`100_materialize_tile.py:387`) because by then 010 has consumed it.
- **Recommended:** for `ATOM_KIND != "scalar"`, have `010_split_register_axes` route to a new
  `_replicate_mma_cells` helper that, per `(register_m, register_n)` cell, emits `MmaFragment` a/b/c decls + `MmaLoad`s
  + `MmaSync` (instead of scalar replicated `Init`/`Accum`), and the C-fragment `MmaStore` after the K loop. The
  materializer (`100`) then lowers the staged smem and surrounding scaffolding as today and treats the Mma Stmts as
  opaque leaves. This keeps the "where cells are replicated" logic in one pass and the "smem/sync scaffolding" logic in
  the other — matching the current split of responsibilities.
- Alternative: keep `RegisterTile` for MMA (010 skips when `ATOM_KIND != "scalar"`) and do all MMA emission in `100`.
  Rejected — duplicates the cell-walk that 010 already does well.

**Change** (in `kernel/010_split_register_axes.py`, plus a thin read in `100`):

1. Read `kind = root.op.knobs.get("ATOM_KIND", "scalar")`; `spec = atom_spec(kind)`. When `"scalar"`, current behavior
   unchanged.
2. When `spec.instruction == "wmma"`: walk to the reduce-K `SerialTile(kind in {serial_outer, stage_inner})` and its
   containing `RegisterTile` cell tower. For each `(register_m, register_n)` cell:
   - emit a C `MmaFragment` in the prelude (before the K_o loop);
   - inside the K_o → stage_inner body emit a/b `MmaFragment` decls (operand dtypes from `spec.operand_dtypes`),
     `MmaLoad` from the staged smem slab at the warp-cooperative offset (offset uses `spec.group_size` for warp
     counting + the render-derived `lane`/`warp`), then `MmaSync(c, a, b)`;
   - after the K_o loop, `MmaStore` from each C fragment to the smem accumulator (or to gmem via `Write` if no combine).
3. Guard the MMA path behind `_MMA_ENABLED = config`-driven (`DEPLODOCK_MMA`, default on) so there's a safe off switch
   until M8 passes. Route this through `deplodock/config.py` (the sole owner of `DEPLODOCK_*` env reads, per
   CLAUDE.md), **not** a bare `os.environ.get`.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/010_split_register_axes.py` (~120 lines: `_replicate_mma_cells`
  + dispatch)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` (~10 lines: treat Mma Stmts as opaque
  leaves if any handling is needed)
- `deplodock/config.py` (~4 lines: `DEPLODOCK_MMA` accessor)

**Verification.** (a) Golden IR: a small synthetic f16 matmul (M=N=K=64) with `ATOM_KIND="wmma_m16n16k16_f16"` produces
a body with the expected `MmaFragment` decl count and `MmaSync` chain length; assert structurally against a pretty-print.
(b) **End-to-end numerical** (the real gate): an f16 matmul runs with the MMA fork picked and produces output matching
the PyTorch reference within fp16 tolerance (max-abs-err ≤ 1e-2 vs f32 reference, or ≤ existing test tolerance).

## M6 — CUDA render: `<mma.h>` include / `nvcuda` namespace (or PTX fallback)

**Why.** The new Stmts render their own lines (M4); render-time scaffolding needs the include + namespace so the kernel
compiles.

**Change.** In `deplodock/compiler/ir/kernel/render.py`, in `render_kernelop`:

- Detect MMA: walk the body once for any `MmaFragment`/`MmaLoad`/`MmaSync`/`MmaStore` (mirror the `desc_names` /
  `_TMA_PRELUDE` detection that already exists for TMA). When present, prepend `#include <mma.h>\nusing namespace
  nvcuda;\n` to the output (alongside / before the dtype-driven `cuda_includes(...)` line).
- **NVRTC availability is not assumed.** Before relying on the include, run the M6 probe (see Failure modes): compile a
  trivial `wmma::fragment` kernel via the project's NVRTC path. If it fails, emit an `_MMA_PRELUDE` of raw-PTX
  `mma.sync.aligned.*` + `ldmatrix` `__forceinline__` wrappers (the `_TMA_PRELUDE` pattern), and M4's `render()` bodies
  call those wrappers instead of `wmma::*`. This is the contingency Design decision 10 calls out.

**Files.**

- `deplodock/compiler/ir/kernel/render.py` (~15 lines, or ~80 if the PTX prelude is needed)

**Verification.** Render an `MmaSync`-bearing kernel; assert the output preamble contains the include + namespace (or
the PTX prelude). Compile via the project's NVRTC fixture and assert no compile error.

## M7 — Skip incompatible kernel passes

**Why.** `pack_fp16_pairs` and `permute_lane_accesses` don't apply to MMA kernels and would corrupt them.

**Change.**

- `kernel/070_pack_fp16_pairs.py`: at the top of `rewrite`,
  `if root.op.knobs.get("ATOM_KIND", "scalar") != "scalar": raise RuleSkipped("non-scalar atom kind; pack-half2 N/A")`.
- `kernel/060_permute_lane_accesses.py` *(note: in `kernel/`, not `tile/` as the original plan said)*: same guard. The
  check generalizes to every future MMA kind.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/070_pack_fp16_pairs.py` (~5 lines)
- `deplodock/compiler/pipeline/passes/lowering/kernel/060_permute_lane_accesses.py` (~5 lines)

**Verification.** `tests/compiler/test_register_tile_planned_mma_skip.py`: run an MMA-eligible kernel through the
pipeline with MMA on; assert via the `.rules.json` dump that both passes log `RuleSkipped` on the MMA variant.

## M8 — End-to-end correctness + perf gate

**Why.** The load-bearing question: does the MMA path produce correct output across realistic shapes, and does it win
on perf vs scalar register-tile.

**Change.** No code beyond test additions.

- `tests/compiler/test_matmul_mma.py`: parametrize over (M, N, K) — at least (64,64,64), (512,512,512),
  (4096,4096,4096) — and dtype f16 (bf16 deferred to M9). Pin the MMA fork via
  `DEPLODOCK_KNOBS="ATOM_KIND=wmma_m16n16k16_f16"` (and `DEPLODOCK_MMA=1`); assert max-abs-err vs the f32 PyTorch
  reference within fp16 tolerance.
- `tests/perf/test_matmul_mma_perf.py` (under the `perf` marker — see `tests/perf/ARCHITECTURE.md`): bench
  TinyLlama-shape matmul scalar vs MMA at **-O3** (deployable, not the -O1 tune-ranking flags); assert MMA ≥ 2× scalar
  on sm_80+. Skip if `compute_capability < (7, 0)`.
- Run `make bench-kernels-tuned` (or `deplodock tune … --bench`) and confirm the autotune DB picks the MMA variant on
  matmul kernels in a TinyLlama / Qwen layer; record the table in the PR.

**Files.**

- `tests/compiler/test_matmul_mma.py` (~100 lines)
- `tests/perf/test_matmul_mma_perf.py` (~50 lines)

**Verification.** New tests pass; bench shows the expected MMA margin (2–8× on f16, shape-dependent). If MMA *loses* on
M=N=K=64, that's expected (launch overhead dominates) — `_priority_matmul` + the tuner pick scalar there. Confirm via
the autotune DB which atom kind was picked per shape.

## M9 — Additional atom kinds (skewed WMMA, BF16)

**Why.** WMMA on sm_80+ also supports `(8,32,16)` / `(32,8,16)` for skinny matmuls (attention projections), and BF16
has the same shape menu with different operand dtype.

**Prerequisite — add a `BF16` `DataType`.** `dtype.py` has only `F32` / `F16` / `F16x2`. Add
`BF16 = DataType("bf16", np.dtype(...), 2)` and wire its CUDA spelling in `backend/cuda/dtype.py` (`cuda_name` /
`cuda_includes` → `__nv_bfloat16` + `<cuda_bf16.h>`) and `backend/torch_ref.py`. *(The original plan omitted this — it
assumed `BF16` already existed.)*

**Change.** Add `ATOM_REGISTRY` entries `"wmma_m16n16k16_bf16"`, `"wmma_m8n32k16_f16"`, `"wmma_m32n8k16_f16"` (+ bf16
skewed). Extend `_ATOM_KINDS_V1 → _ATOM_KINDS_V2`. The per-kind eligibility already gates BF16 on
`compute_capability >= (8, 0)` and skewed shapes on divisibility — no new planner checks. The materializer is
registry-driven (M5's `spec` lookup), so no codegen change beyond the entries (and the bf16 fragment dtype in
`MmaFragment.render`).

**Files.**

- `deplodock/compiler/dtype.py` (~3 lines: `BF16`)
- `deplodock/compiler/backend/cuda/dtype.py` (~6 lines: bf16 spelling/include)
- `deplodock/compiler/backend/torch_ref.py` (~3 lines: bf16 ref)
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~20 lines: entries)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~3 lines: candidate list)

**Verification.** Re-run M8's bench with the expanded set; confirm the tuner picks `"wmma_m8n32k16_f16"` on ≥1 skinny
matmul in Qwen (MLP gate/up projections) and that it beats the square shape there.

## M10 — Documentation + ARCHITECTURE.md updates

**Why.** Once the fragment factorization is the canonical model, the scalar register-tile descriptions read
incompletely. "Every matmul has an atom kind; scalar is `(1,1,1)`" should be first-class.

**Change.**

- `deplodock/compiler/pipeline/ARCHITECTURE.md`: extend the partition-planner factorization description (the
  `A → A_b·(T·R) + A_t·R + A_r` block) to `BLOCK · GROUP · CELL · ATOM`; document `ATOM_KIND` in the knob table; describe
  the `ATOM_REGISTRY` model and current entries. Update the `lowering/kernel/` order line (~line 349) to note the
  `split_register_axes` MMA-cell branch and the `pack_fp16_pairs` / `permute_lane_accesses` MMA skips.
- `deplodock/compiler/ir/ARCHITECTURE.md`: add the four new Kernel-IR Stmts to the kernel-dialect table (~line 296+).
- If approach 3a added a `WarpTile`: document it in the Tile-IR flavor list (`ir/ARCHITECTURE.md` ~line 54) and
  `ir/tile/ir.py`.
- `CLAUDE.md`: nothing needed — README stays example-driven; `make tune-kernels` already exercises the path.
- Wrap all markdown at ~120 chars (CLAUDE.md doc convention).

**Files.**

- `deplodock/compiler/pipeline/ARCHITECTURE.md` (~15 lines)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~12 lines)

**Verification.** `make lint` clean. Spot-read the changed sections.

---

## Failure modes to watch

- **NVRTC `<mma.h>` unavailable (probe first).** The codebase forward-declares TMA intrinsics because NVRTC omits
  `<cuda.h>`/`<cuda/barrier>`. `<mma.h>` may be the same. **Mitigation: a 5-line NVRTC smoke compile of a
  `wmma::fragment` kernel before M4**, so the IR-Stmt render bodies are decided knowing whether `wmma::*` or raw-PTX
  `mma.sync`+`ldmatrix` is the target. This reorders risk to the front instead of discovering it at M6.
- **Silent miscompile on fragment lane mapping.** `wmma::load_matrix_sync(frag, ptr, ldm)` expects `ptr` at the first
  element of the warp's tile and `ldm` as the leading-dimension stride *in elements*. Easy off-by-one. M8's
  end-to-end test catches it; M5's golden IR test does not.
- **Warp-tier launch geometry (Design decision 3).** `_launch_bounds_for` and `_build_linear_tid` compute per-CTA
  threads as the product of `ThreadTile` axis extents. If MMA warps are encoded as thread axes without the ×32 lane
  factor, launch bounds and the tid decode are wrong by 32×. Whichever of 3a/3b is chosen must update both sites.
- **`MmaSync` is synchronous in v1.** Hopper/Blackwell tensor cores are async (issue+wait). The registry's
  `spec.instruction == "wmma"` branch opts into synchronous semantics; future `"wgmma_*"`/`"tcgen05_*"` kinds add their
  own materializer branch with `MmaIssue`/`MmaWait`. Structure `_replicate_mma_cells` per-instruction-family so async
  kinds don't rewrite the WMMA path.
- **MMA + cooperative-K conflict.** v1 cooperative-K is `BR > 1 ⇒ BN = BM = 1`. With MMA, BN/BM are warp-count × 16, so
  `BN = BM = 1` is impossible. **Gate MMA off when BR > 1** — the MMA forks only enumerate `BR = 1`. Defer MMA +
  cooperative-K.
- **fp16 accumulator dtype.** WMMA's C-fragment is fp32 even with f16 A/B; most scalar matmuls already accumulate in
  f32, so drift should match or improve. If a test relied on a scalar f16 accumulator dtype it may need re-blessing —
  audit during M8.
- **Scalar no-op regression.** M1's whole purpose is byte-identical scalar output. If a golden busts, the cause is
  almost always either `ATOM_KIND` leaking into a rendered knob string / `perf` key (fix `format_tuning_knobs`
  elision), or — under approach 2b — an extent-1 ATOM layer not being dropped by `_wrap_tower`'s size-1 filter (verify
  the filter's non-BLOCK drop fires for the ATOM role). Approach 2a sidesteps both.

## Future extensions (out of scope)

This plan ships WMMA on sm_70–sm_89 only. Future plans extend the registry without redesigning the factorization:

- **NVFP4 / MXFP4 scaled MMA (Blackwell sm_100+).** New `ATOM_REGISTRY` entries with FP4 operand dtypes + FP8 scale
  dtype + a `scale_block_size` field; a sibling `MmaScaledSync` Stmt; a `Tmem` allocation primitive beside `Smem`;
  async `MmaIssue`/`MmaWait` threaded through `080_pipeline_stages`. Frontend: a quantized `MatmulOp` / `ScaledMatmulOp`
  carrying scale operands.
- **wgmma (Hopper sm_90).** Same shape menu minus scale operands; a 128-thread `WgroupTile` flavor beside the M3
  `WarpTile`; the async issue/wait infra NVFP4 also needs.
- **Sparse MMA (sm_80+ 2:4).** New entries with a sparsity-metadata operand; one more fragment role per cell.

**Why these slot in cleanly.** Each future kind adds (registry entry, optional new Stmt, optional new memory tier). The
planner's σ-split, the per-cell replication, the warp-tier launch geometry, the fork enumeration, and the env-pinning
workflow read from the registry — none embed v1's "WMMA is special" assumptions. The materializer's instruction-family
dispatch (`spec.instruction`) is the single extension point for new codegen paths.

## Test additions summary

- `tests/compiler/passes/test_partition_planner_mma.py` — `is_atom_eligible` (M2), fork enumeration includes MMA
  variants when eligible (M3).
- `tests/compiler/ir/test_kernel_mma.py` — new Stmt construction + render + SSA-rename round-trip (M4).
- `tests/compiler/passes/test_materialize_tile_mma.py` — golden IR for a small MMA kernel (M5).
- `tests/compiler/test_matmul_mma.py` — end-to-end f16 (M8) / bf16 (M9) correctness across shapes.
- `tests/perf/test_matmul_mma_perf.py` — perf gate, MMA vs scalar (M8).
- `tests/compiler/test_register_tile_planned_mma_skip.py` — incompatible-pass skip guards (M7).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — tile flavors (`GridTile`/`ThreadTile`/`RegisterTile`/…); new `WarpTile` if 3a.
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` — `AtomSpec`, `ATOM_REGISTRY`, `is_atom_eligible` (new).
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — factorization, fork tree, `ATOM_KIND`
  knob, planner `Role` (internal), `_wrap_tower`, `_priority_matmul`.
- `deplodock/compiler/pipeline/passes/lowering/tile/_helpers.py` — `is_matmul_reduce`, `compute_capability`.
- `deplodock/compiler/ir/kernel/ir.py` — `MmaFragment` / `MmaLoad` / `MmaSync` / `MmaStore`.
- `deplodock/compiler/pipeline/passes/lowering/kernel/010_split_register_axes.py` — per-cell replication; MMA-cell
  branch (M5 dispatch point).
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — smem/sync scaffolding; opaque Mma
  leaves.
- `deplodock/compiler/ir/kernel/render.py` — `<mma.h>` include / PTX prelude (M6).
- `deplodock/compiler/pipeline/knob.py` — `KnobType.STR` + default elision.
- `deplodock/compiler/dtype.py` — `BF16` (M9).
- `deplodock/compiler/backend/cuda/dtype.py` — bf16 CUDA spelling (M9).
- `deplodock/compiler/pipeline/passes/lowering/kernel/070_pack_fp16_pairs.py` — skip guard (M7).
- `deplodock/compiler/pipeline/passes/lowering/kernel/060_permute_lane_accesses.py` — skip guard (M7).
- `deplodock/compiler/pipeline/ARCHITECTURE.md`, `deplodock/compiler/ir/ARCHITECTURE.md` — docs (M10).
