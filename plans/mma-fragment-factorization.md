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
> - **The planner emits a lazy Fork tree via a generic builder.** `rewrite` returns `Graph | None | TileOp | Fork |
>   list[Fork]`; construction is delegated to `build_fork_tree(*, params, levels, materialize, score)` in
>   `pipeline/fork_tree.py` (~128 lines, generic over the row type `P`). The planner passes a flat list of
>   `Level((knob_names,), key)` entries — today, four of them: `BR → (BM,BN) → (FM,FN) → (BK,SPLITK)`. The builder
>   sorts siblings by max-propagated `-score` and **collapses single-value levels automatically** (no 1-child wrapper).
>   Each fork is an *op fork* (graph-preserving, separable — see `autotune_no_graph_forks` memory) consumed by the
>   two-level tuner. M3 (below) builds **two subtrees** with disjoint `levels` schemas — one per row type — and returns
>   them as sibling Forks, since the warp subtree's level keys (ATOM_KIND, WM, WN) and the scalar subtree's (BR, BM,
>   BN) don't share a schema. Earlier drafts of this plan referenced `_build_fork_tree_lazy` / `_group_level` /
>   `_bmbn_level` / `_fmfn_level` / `_br_level` — those named-level functions have been **deleted**; the docstring at
>   `010_partition_loops.py:417` referencing `_build_fork_tree_lazy` is stale.
> - **`TileParams` + the seven planner `Knob`s + the priority functions + `enumerate_cartesian` + the impl
>   cartesian live in `pipeline/passes/lowering/tile/_enumeration.py`** (a sibling `_`-prefixed module the rule loader
>   skips). `010_partition_loops.py` imports them. Earlier drafts of this plan pointed at `010_partition_loops.py` for
>   the enumerator — wrong location; M3 lands in `_enumeration.py`.
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
- `N_g` — the **new** warp tier, materialized by a `WarpTile` flavor (`plans/warptile-primitive.md`). Scalar **does not
  emit this layer at all** (its layer builder doesn't add a `Role.WARP` entry); MMA emits it with the per-N warp count.
- `N_c` — `RegisterTile`, replication count. Today's `FN` for scalar; the MMA cell count per warp.
  `010_split_register_axes` replicates the body per cell regardless of what the body contains.
- `N_a` — the hardware-atomic extent. Scalar **does not emit ATOM layers at all** (its layer builder doesn't add a
  `Role.ATOM` entry — see Design decision 2 and 6 for why this is cleaner than "emit extent-1 and rely on the size-1
  filter to drop it"). For MMA `ATOM_N = 16`. The materializer never iterates it; it dispatches on `ATOM_KIND` and
  emits one fragment instruction per cell.

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

2. **`AtomTile(ParallelTile)` flavor + planner-internal `Role.ATOM`.** *(Replaces the original `axis.py` `Role.ATOM`
   binding, which targeted a deleted enum.)* Add `Role.ATOM` to the planner enum in `010_partition_loops.py` (it stays
   planner-internal — never reaches downstream passes' discriminator logic, which keys on flavor type). Add an
   `AtomTile(ParallelTile)` flavor to `ir/tile/ir.py` beside `RegisterTile`. Extend `_layer_kind_for` with
   `Role.ATOM → "atom"` and add an `"atom"` case to `_wrap_tower`'s grouping that wraps in `AtomTile`. Pretty-label
   `"atom"`; render is `NotImplementedError("AtomTile must be consumed by the MMA materializer")` — matches how
   `RegisterTile.render` raises today (consumed before kernel render).

   **The scalar no-op holds because the scalar layer builder doesn't emit `Role.ATOM` entries at all.** Under the
   sum-type `TileParams` (Design decision 11 below), the matmul body-builder dispatches on the row type:
   `ScalarTileParams` → a layers list with `THREAD`/`REGISTER`/`BLOCK` only; `WarpTileParams` → a layers list with
   `ATOM`/`REGISTER`/`WARP`/`BLOCK`. The two builders import disjoint knob bundles. The size-1 filter in `_wrap_tower`
   is **not** the mechanism for tier exclusion — it is a normalization safety net for genuine knob=1 collapses
   (`FM=1`, `BR=1`, degenerate `BN=BM=1` corners). It still runs and still catches those. Following the same
   convention today's planner uses for REGISTER/THREAD (pointwise/reduce don't emit them, they aren't emitted
   extent-1 then filtered) keeps the model uniform: a role enters `layers` iff its tier is structurally present.

   The trade is that for MMA kernels, `AtomTile` is a new structural Stmt every tile / kernel pass must either
   recognise or pass through. Most passes already `with_bodies`-recurse opaquely through wrappers; the M1 audit (see
   Design decision 9) catalogues the ones that don't. The win: the materializer reads tower structure (presence of
   `AtomTile`) rather than only the `ATOM_KIND` knob, which is the stronger signal and matches the "Role tells you
   what to do with this Loop" invariant the rest of the planner uses; and a future async kind (wgmma/tcgen05) gets a
   place to hang issue/wait scheduling structurally.

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
   - **Default: 3a.** Committing to `AtomTile` (Decision 2) makes 3a the natural sibling — both encode their binding
     tier in the flavor type, both pretty-print as their own bracketed group via `_wrap_tower`'s grouping, and the
     `BLOCK > WARP > REGISTER > ATOM` chain reads directly off the IR. 3b would put `AtomTile` immediately under
     `ThreadTile` with the warp/lane split implicit, which loses the structural symmetry the rest of the planner now
     has. **Pick 3a in M1.**

4. **Eligibility predicate.** `is_atom_eligible(kind, loop_op, ctx) -> bool`, dispatched via the registry. The
   `wmma_m16n16k16_f16` predicate checks: (a) `is_matmul_reduce` fires on at least one reduce in the body; (b) every
   K-indexed Load and the Accum target dtype is `F16`; (c) `ctx.compute_capability >= (7, 0)` (BF16 kinds require
   `>= (8, 0)`); (d) output M, N extents divisible by 16; K extent divisible by 16 (defer the `% (16·BR)` check until
   BR is picked). Lives in `tile/_helpers.py` beside `is_matmul_reduce`, or in `tile/_atom.py`.
   `is_atom_eligible("scalar", …)` is always `True`.

5. **Atom-kind candidate set, v1.** `"scalar"` always; `"wmma_m16n16k16_f16"` when eligible (the WMMA "square" shape —
   broadest arch support, simplest lane mapping). BF16 + skewed WMMA shapes land in M9. NVFP4/MXFP4/wgmma wait for their
   own plans.

6. **Scalar path is a no-op refactor.** M1 lands the sum-type `TileParams` split with the existing scalar
   `Knob` bundle on `ScalarTileParams` and an empty `ATOM_REGISTRY` warp branch (only `"scalar"` registered yet —
   which is itself thread-tier, so the warp branch enumerates zero rows). Existing golden IR tests must pass
   byte-identical: every existing matmul produces a `ScalarTileParams` row, the scalar layer builder is structurally
   identical to today's, and there's no `Role.ATOM` / `Role.WARP` to appear anywhere. If anything in M1 changes a
   golden, that's a bug, not a re-bless.

7. **`pack_fp16_pairs` interaction.** For MMA kernels there are no scalar f16 `Init`/`Accum` to pair — the C-fragment IS
   the accumulator. Skip when `ATOM_KIND != "scalar"` (M7).

8. **`permute_lane_accesses` interaction.** *(File moved to `kernel/`.)* Permutes LDS.128 indices on staged Loads to
   break bank conflicts. WMMA uses its own swizzled `load_matrix_sync` access. Skip when `ATOM_KIND != "scalar"` (M7).

9. **Downstream passes pass `AtomTile` through opaquely.** The tile chain (`020_stage_inputs`,
   `030_hoist_invariant_compute`, `040_use_ring_buffers`, `050_use_tma`, `060_use_async_copy`, `070_pad_smem`,
   `080_pipeline_stages`, `090_mark_unroll`) operates on K_o / stage structure and doesn't touch the output-axis cell
   tower below the reduce, so an `AtomTile` deep inside the tower should be invisible to them. **M1 audit** (one-pass
   read of each rule's `PATTERN` + body walk): confirm each either (a) doesn't recurse below the reduce, or
   (b) recurses via `with_bodies` / `Body.map`, which descends through any wrapper Stmt. Same audit for the kernel
   chain except the MMA-aware passes (`010_split_register_axes`, `100_materialize_tile`) and the two M7-skipped passes
   (`060_permute_lane_accesses`, `070_pack_fp16_pairs`). Any pass that introspects flavor types explicitly gets a
   passthrough branch for `AtomTile`.

10. **MMA via `wmma::load_matrix_sync` / `mma_sync` / `store_matrix_sync` in v1, *if NVRTC ships `<mma.h>`*.** The
    intrinsic path is simplest to plumb and verify. **Gated on the M6/Failure-modes NVRTC probe** — if the header isn't
    reachable, fall back to a raw-PTX `mma.sync` + `ldmatrix` prelude (analogous to `_TMA_PRELUDE`). The IR Stmt
    abstraction (`MmaLoad` / `MmaSync` / `MmaStore`) is the same either way; only the `render()` body text differs.

11. **`TileParams` is a sum type: `ScalarTileParams | WarpTileParams`.** Thread-tier and warp-tier matmul rows carry
    disjoint knob bundles (no shared "binding-tier-discriminator" flag with half the fields ignored per tier). The
    sum type forces every reader — `_priority_*`, the layer builder, the perf-row key, the materializer dispatch —
    to discriminate at the type level, which catches a class of "I read `p.bn` on a warp row and got 0" bugs at the
    type checker. Concretely (in `_enumeration.py`):
    - `ScalarTileParams(bn, bm, fm, fn, bk, splitk, br, overhang)` — today's row, unchanged.
    - `WarpTileParams(wn, wm, fm, fn, bk, splitk, atom_kind, overhang)` — new. `BR` not carried (MMA gates BR=1).
    - `BN`/`BM`/`BR` `Knob`s are scalar-tier only; new `WN`/`WM`/`ATOM_KIND` `Knob`s are warp-tier only;
      `FM`/`FN`/`BK`/`SPLITK` are shared (same arithmetic role in both tiers).
    - `_priority_matmul_thread` and `_priority_matmul_warp` are separate functions, each pure on its row type.
    - `_enumerate_cartesian_impl` splits into `_enumerate_scalar_matmul_impl` and `_enumerate_warp_matmul_impl` —
      divisibility (`E_M % (bm·fm)` vs `E_M % (wm·fm·atom_m)`) and per-CTA thread budget (`bn·bm·br` vs
      `wn·wm·32·br`) differ enough that one parameterized impl would be more `if`s than shared algorithm.
    - The fork tree (M3) calls `build_fork_tree(..., levels=[...])` twice with disjoint per-tier `levels` schemas
      and returns sibling Forks. Scalar levels = today's four (`BR → (BM,BN) → (FM,FN) → (BK,SPLITK)`); warp levels
      = `ATOM_KIND → (WM,WN) → (FM,FN) → (BK,SPLITK)`. Only matmul forks the tier — `reduce` and `pointwise` stay
      thread-only.
    - The matmul layer builder in `_build_split_body` dispatches on `isinstance(params, ScalarTileParams |
      WarpTileParams)` and emits the tier-specific layers list. Neither tier emits placeholder extent-1 layers for
      the other tier's roles.

---

**Prerequisite landed:** `Axis.source_axis` is in place (`ir/axis.py` — every split sub-axis back-points to its parent
via `Axis.split`, `compare=False`/`hash=False` so Var-rename invariance holds). The MMA enumerator can use it for
BLOCK·GROUP·CELL·ATOM grouping without name-suffix matching.

## M1 — `AtomTile` / `WarpTile` flavors + `ATOM_KIND` registry + `TileParams` sum-type split, scalar no-op

**Why.** Establish the unified-factorization scaffolding (new flavors, new planner Roles, registry, sum-typed rows)
without changing any emitted CUDA. Scalar kernels still produce `ScalarTileParams` rows; the scalar layer builder
doesn't reference WARP/ATOM at all. The warp branch is wired up but has zero registered MMA kinds yet, so it
enumerates no rows. **Pick warp tier 3a here**: `WarpTile` lands as a real flavor (see `plans/warptile-primitive.md`
— this milestone subsumes M1/M2/M3 of that plan if it hasn't shipped first). If anything in M1 changes a golden,
the scaffolding is wrong; fix before proceeding.

**Prerequisite handling.** If `plans/warptile-primitive.md` has shipped, this milestone consumes its primitives
(`WarpTile` + `Role.WARP` + `_layer_kind_for("warp")` + `_wrap_tower` "warp" case + launch-bounds/tid wiring) and
shrinks proportionally. If it hasn't, M1 ships them inline.

**Change.**

- New file `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py`: define `AtomSpec` (frozen dataclass:
  `shape: tuple[int, int, int]`, `operand_dtypes: Mapping[str, DataType]`, `instruction: str`, `group_size: int`) and
  `ATOM_REGISTRY: dict[str, AtomSpec]` — **seeded empty** at M1 (M2 adds `"wmma_m16n16k16_f16"`). Helpers
  `atom_spec(kind)`, `atom_shape(kind)`, `atom_group_size(kind)`. Prefixed `_` so the rule loader skips it
  (`engine._load_rules` filters `startswith("_")`). The `"scalar"` "kind" is **not** in the registry — scalar is the
  absence of an atom, represented by the `ScalarTileParams` row type, not by a registered kind.
- `deplodock/compiler/ir/tile/ir.py`: add `AtomTile(ParallelTile)` beside `RegisterTile`. Same shape (`axes`, `body`),
  pretty-label `"atom"`, `render` raises `NotImplementedError("AtomTile must be consumed by the MMA materializer …")`
  matching `RegisterTile.render`'s pattern. Also add `WarpTile(ParallelTile)` (if not already landed by the WarpTile
  primitive plan) — see that plan for the render contract. Export both from `__all__`.
- `deplodock/compiler/pipeline/knob.py`: add `STR` to `KnobType` (parse = identity, pretty = `str()`).
- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` — **the load-bearing change of M1**:
  - Split `TileParams` into a sum type per Design decision 11:
    `ScalarTileParams(bn, bm, fm, fn, bk, splitk, br=1, overhang=())` (today's row, unchanged) and
    `WarpTileParams(wn, wm, fm, fn, bk, splitk, atom_kind, overhang=())`. Type alias
    `TileParams = ScalarTileParams | WarpTileParams` so importers downstream don't churn.
  - Move existing `BN`/`BM`/`BR` `Knob`s to a clearly-scalar-tier section; add `WN`/`WM`/`ATOM_KIND` `Knob`s
    (warp-tier, all `STR`/`INT` as appropriate); leave `FM`/`FN`/`BK`/`SPLITK` shared.
  - `format_tuning_knobs` (in `knob.py`): default-elide `ATOM_KIND` when not present (warp-only — never appears on
    scalar rows). No "drop scalar" special case needed because the scalar row type doesn't carry the field.
  - `_priority_matmul` → renamed `_priority_matmul_thread` (operates on `ScalarTileParams`). Add stub
    `_priority_matmul_warp` (operates on `WarpTileParams`) — fully wired in M3; at M1 it can be the same shape as the
    thread variant since no warp rows enumerate yet.
  - `_enumerate_cartesian_impl` → `_enumerate_scalar_matmul_impl` (today's impl, unchanged in M1 modulo the row
    type). Add stub `_enumerate_warp_matmul_impl` that returns `[]` at M1 (the warp branch is plumbed but enumerates
    nothing until M3 wires the atom kinds).
  - `enumerate_cartesian` dispatch: extend `priority_mode` to accept `("matmul", "thread")` / `("matmul", "warp")` /
    `"reduce"` / `"pointwise"`. The single-string forms keep working as aliases (`"matmul"` → `("matmul", "thread")`)
    so call sites in `010_partition_loops.py` don't churn at M1.
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py`:
  - Extend `class Role` (currently lines 106–119, members `BLOCK / THREAD / REGISTER / STAGE_INNER / SERIAL_OUTER /
    PIPELINE`): add `WARP` (if not already landed via the WarpTile primitive plan) and `ATOM`.
  - Extend `_layer_kind_for` (line 395): `Role.WARP → "warp"`, `Role.ATOM → "atom"`.
  - Extend `_wrap_tower` (line 311) grouping: `"warp" → WarpTile(...)` and `"atom" → AtomTile(...)` cases beside
    `"grid" / "thread" / "register"`. Both group like the other parallel kinds (consecutive same-kind axes coalesce).
    The non-BLOCK size-1 filter at line 345 is unchanged.
  - `_build_split_body` (line 638): dispatch the matmul branch on `isinstance(params, ScalarTileParams |
    WarpTileParams)` — scalar branch is today's code at lines ~835–880 (both the prologue-bearing and
    no-prologue tower constructors), unchanged; warp branch is stubbed (`raise _BuildSkipped("warp tier wired in
    M3")`). Scalar's `layers` list doesn't reference `Role.ATOM` or `Role.WARP`.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~30 lines, new — registry empty)
- `deplodock/compiler/ir/tile/ir.py` (~25 lines if `WarpTile` already landed via the primitive plan, ~75 if not)
- `deplodock/compiler/pipeline/knob.py` (~15 lines: `KnobType.STR` + warp-knob elision)
- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` (~120 lines: sum-type split, knob bundle split,
  per-tier priority/impl stubs, dispatch-grid `priority_mode`)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~25 lines: Role.WARP / Role.ATOM +
  `_layer_kind_for` + `_wrap_tower` grouping + `_build_split_body` dispatch stub)

**Verification.** `make test` byte-clean — no golden bless. Spot-check one matmul kernel's lowered CUDA dump under
`DEPLODOCK_DUMP_DIR` against a pre-M1 snapshot: identical. **Explicit tier-absence test**: build a `LoopOp`, run the
planner, assert (a) every emitted row is a `ScalarTileParams`, (b) the emitted `TileOp.body` contains zero `AtomTile`
/ `WarpTile` instances (recursive walk), (c) no `layers` entry has `Role.WARP` or `Role.ATOM`. Run the
Design-decision-9 audit (one pass per rule in the tile + kernel chains) and add `AtomTile` / `WarpTile` passthrough
branches to any that introspect flavor types.

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

## M3 — Wire the warp-matmul subtree end-to-end

**Why.** M1 plumbed the sum type with the warp branch stubbed to return `[]` and `_build_split_body`'s warp
dispatch raising `_BuildSkipped`. M3 fills both: the warp enumerator produces real `WarpTileParams` rows, the warp
layer builder emits the `WarpTile + AtomTile` tower, the fork tree gains its top-level binding-tier branch with the
warp subtree under it, and the materializer learns the `WarpTile` arm. After M3 the planner emits MMA variants
siblings-to-scalar in the fork tree; the materializer then routes them through M5's MMA dispatch.

**Change.**

- `_plan_kernel` (in `010_partition_loops.py`): compute the eligible kinds `eligible = [k for k in _ATOM_KINDS_V1 if
  is_atom_eligible(k, loop_op, ctx)]` once per matmul kernel and stash on the `_Plan` / `KernelShape`. Scalar is not
  in this list — scalar is the absence of an atom kind, modelled by `ScalarTileParams`. `eligible` only contains MMA
  kinds.
- `_enumeration.py`:
  - `_enumerate_warp_matmul_impl`: real cartesian, parallel to today's scalar impl but with the warp divisibility +
    budget formulae from Design decision 11 (`E_M % (wm·fm·atom_m) == 0`; `wn·wm·32 ≤ ctx.max_threads_per_cta`).
    Candidate tuples: `wn_choices = wm_choices = _TUNE_WARP_AXIS_CHOICES = (1, 2, 4, 8)`; FM/FN as divisors of the
    cell budget; BK/SPLITK as today; no BR (forced 1). For each eligible atom kind, emit one row stamped with
    `atom_kind=kind`.
  - `_priority_matmul_warp`: real implementation per Design decision 11. Threads = `wn·wm·32`. Cells-per-cell-owner
    cap = `min(fm·fn·atom_m·atom_n, 64)` (bigger cap than scalar's 32 — MMA amortizes K-loop overhead better).
  - `enumerate_cartesian` callers in `010_partition_loops.py` are at lines **510 (`priority_mode="matmul"`), 547
    (`"reduce"`), 558 (`"pointwise"`)**. The matmul call site now calls both `("matmul", "thread")` and
    `("matmul", "warp", atom_kinds=eligible)` and concatenates. When `eligible == ()` the warp call returns `[]`
    cleanly — no special-casing needed at the call site. `reduce` and `pointwise` are untouched (thread-only).
- `_build_split_body` warp branch (formerly stubbed in M1): real layer builder per the "What `_build_split_body`
  actually looks like" snippet you'll find in the planner discussion above. Emits a layers list with `Role.ATOM` (3
  innermost entries) + `Role.REGISTER` (N_r, M_r) + `Role.WARP` (N_w, M_w) + `Role.BLOCK` (N_b, M_b, K_s).
  No `Role.THREAD` entries. K-σ extended with `+ K_a`.
- Fork tree construction in `rewrite()`: today the planner calls `build_fork_tree(params, levels=[…], materialize,
  score)` once with a single `TileParams` row type and four levels (`BR → (BM,BN) → (FM,FN) → (BK,SPLITK)`). Under
  M3, call it **twice** — once per row type — with disjoint level schemas:
  ```python
  scalar_subtree = build_fork_tree(
      params=plan.scalar_params,                    # tuple[ScalarTileParams, ...]
      levels=[
          Level((BR.name,),               lambda p: (p.br,)),
          Level((BM.name, BN.name),       lambda p: (p.bm, p.bn)),
          Level((FM.name, FN.name),       lambda p: (p.fm, p.fn)),
          Level((BK.name, SPLITK.name),   lambda p: (p.bk, p.splitk)),
      ],
      materialize=lambda p: _materialize(plan, p),
      score=lambda p: _score_variant(plan, p, ctx),
  )
  warp_subtree = build_fork_tree(
      params=plan.warp_params,                      # tuple[WarpTileParams, ...]
      levels=[
          Level((ATOM_KIND.name,),        lambda p: (p.atom_kind,)),
          Level((WM.name, WN.name),       lambda p: (p.wm, p.wn)),
          Level((FM.name, FN.name),       lambda p: (p.fm, p.fn)),
          Level((BK.name, SPLITK.name),   lambda p: (p.bk, p.splitk)),
      ],
      materialize=lambda p: _materialize(plan, p),
      score=lambda p: _score_variant(plan, p, ctx),
  )
  ```
  `build_fork_tree` already collapses single-value levels, so when `eligible == ()` (no warp params) the call
  returns `[]` and the planner returns `scalar_subtree` unchanged (byte-identical to today). When both subtrees are
  non-empty, return `[*warp_subtree, *scalar_subtree]` — the engine treats sibling Forks as a top-level fork
  decision. `_priority_matmul_warp` is constructed to outscore `_priority_matmul_thread` so the
  `-score`-sorted siblings drill warp-first.
- `_Plan` (in `010_partition_loops.py`) gains a per-tier field split: `scalar_params: tuple[ScalarTileParams, ...]`
  and `warp_params: tuple[WarpTileParams, ...]`, replacing today's flat `params`. `_plan_kernel` populates both;
  callers that consumed `plan.params` get migrated (search: ~7 references today).
- `ir/kernel/render.py::_launch_bounds_for`: see WarpTile primitive plan M2 — if not already landed, add the
  `WarpTile` branch returning `prod(warp_extents) * 32`. Idempotent if shipped.
- `kernel/100_materialize_tile.py`: `_materialize_warp` arm (also from the WarpTile primitive plan M2) — idempotent.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` (~80 lines: real `_enumerate_warp_matmul_impl`,
  `_priority_matmul_warp`, `_TUNE_WARP_AXIS_CHOICES`, dispatch updates)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~70 lines: `_plan_kernel` eligible-kinds
  hook, real warp `_build_split_body` branch, `_kind_level` + `_wmwn_level` in the fork tree)
- `deplodock/compiler/ir/kernel/render.py` and `kernel/100_materialize_tile.py` — empty if the WarpTile primitive
  plan landed; otherwise see that plan's M2 for the line counts (~12 + ~80 respectively).

**Verification.** Unit test `tests/compiler/passes/test_partition_planner_mma.py` extensions: an eligible TinyLlama
matmul `LoopOp` → enumerated rows include ≥1 `WarpTileParams(atom_kind="wmma_m16n16k16_f16", …)`; a non-eligible
(f32) matmul → zero `WarpTileParams` rows, only `ScalarTileParams`; a pointwise kernel → zero `WarpTileParams` rows
(pointwise mode doesn't dispatch the warp tier). Fork-tree shape: when `eligible != ()` the tree's top is a 2-child
fork with warp first; when `eligible == ()` the tree top is the unchanged scalar-only chain (assert the *shape* is
identical to a pre-M3 run on the same kernel). Tower-shape test: materialize one warp variant; assert the body
contains exactly one `WarpTile` wrapping the cell tower and one `AtomTile` at the innermost position, both with the
expected axis extents. `_launch_bounds_for` reports `warps × 32`.

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
  through (`100_materialize_tile.py:553`, inside `_materialize`'s `_process_stmt`) because by then 010 has consumed
  it.
- **Recommended:** for `ATOM_KIND != "scalar"`, have `010_split_register_axes` route to a new
  `_replicate_mma_cells` helper that, per `(register_m, register_n)` cell, emits `MmaFragment` a/b/c decls + `MmaLoad`s
  + `MmaSync` (instead of scalar replicated `Init`/`Accum`), and the C-fragment `MmaStore` after the K loop. The
  materializer (`100`) then lowers the staged smem and surrounding scaffolding as today and treats the Mma Stmts as
  opaque leaves. This keeps the "where cells are replicated" logic in one pass and the "smem/sync scaffolding" logic in
  the other — matching the current split of responsibilities.
- Alternative: keep `RegisterTile` for MMA (010 skips when `ATOM_KIND != "scalar"`) and do all MMA emission in `100`.
  Rejected — duplicates the cell-walk that 010 already does well.

**Change** (in `kernel/010_split_register_axes.py`, plus a thin read in `100`):

1. Detect MMA **structurally**: if the body contains an `AtomTile` (M1/M3 only emits one for `atom_kind != "scalar"`),
   resolve `kind = root.op.knobs["ATOM_KIND"]` and `spec = atom_spec(kind)`. The structural signal is preferred over
   the knob alone because it can't drift out of sync. Scalar bodies have no `AtomTile` → existing
   `_replicate_register_tiles` path unchanged.
2. When `spec.instruction == "wmma"`: route through `_replicate_mma_cells`. Walk to the reduce-K
   `SerialTile(kind in {serial_outer, stage_inner})` and its containing `RegisterTile` cell tower wrapping the
   `AtomTile`. For each `(register_m, register_n)` cell:
   - emit a C `MmaFragment` in the prelude (before the K_o loop);
   - inside the K_o → stage_inner body emit a/b `MmaFragment` decls (operand dtypes from `spec.operand_dtypes`),
     `MmaLoad` from the staged smem slab at the warp-cooperative offset (offset uses `spec.group_size` for warp
     counting + the render-derived `lane`/`warp`), then `MmaSync(c, a, b)`;
   - after the K_o loop, `MmaStore` from each C fragment to the smem accumulator (or to gmem via `Write` if no
     combine).
   - The `AtomTile` is consumed in the rewrite — its axes contributed shape, not iteration; the materializer emits one
     fragment instruction per cell, no inner loop. The output body has no `AtomTile` instances.
3. Guard the MMA path behind `_MMA_ENABLED = config`-driven (`DEPLODOCK_MMA`, default on) so there's a safe off switch
   until M8 passes. Route this through `deplodock/config.py` (the sole owner of `DEPLODOCK_*` env reads, per
   CLAUDE.md), **not** a bare `os.environ.get`.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/010_split_register_axes.py` (~120 lines: `_replicate_mma_cells`
  + dispatch)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` (~12 lines: treat Mma Stmts as opaque
  leaves if any handling is needed; **and one-line invariant fix at `:93`**: change
  `KernelOp(body=new_body, name=root.op.name)` → `KernelOp(body=new_body, name=root.op.name, knobs=root.op.knobs)`
  so `ATOM_KIND` survives materialize for any downstream KernelOp-keyed introspection. `knobs` is declared on the
  base `Op` (`ir/base.py:62`) with a default `{}`, so the current materialize silently drops `TileOp.knobs` — not a
  bug for *this* plan (M5/M7 all run on TileOp pre-materialize), but a fragile invariant any future MMA-related work
  would trip on. Fix it here while we're in the file.)
- `deplodock/config.py` (~4 lines: `DEPLODOCK_MMA` accessor — use the existing `knob_var` / `knob_raw` pattern at
  `:49–62`, not a bare `os.environ.get`).

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

- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` **module docstring**: the docstring
  currently shows one loop-nest example for the scalar matmul tower (`for m_b … for n_b … for m_t … for n_t … for
  m_r … for n_r … for k_o … for k_i …`). Extend it with a **second loop-nest example for the warp/MMA mode** —
  same bracket-pretty-print style as `ir/tile/ir.py`'s convention, showing the `grid > warp > register > atom` tower
  with the `MmaFragment` / `MmaLoad` / `MmaSync` body. A reader landing on this module sees both modes side by side
  and can map any concrete kernel to the right factorization without cross-referencing the plans.
- `deplodock/compiler/pipeline/ARCHITECTURE.md`: extend the partition-planner factorization description (the
  `A → A_b·(T·R) + A_t·R + A_r` block) to `BLOCK · GROUP · CELL · ATOM`; document `ATOM_KIND` in the knob table;
  describe the `ATOM_REGISTRY` model and current entries. Update the `lowering/kernel/` order line (~line 349) to
  note the `split_register_axes` MMA-cell branch and the `pack_fp16_pairs` / `permute_lane_accesses` MMA skips.
- `deplodock/compiler/ir/ARCHITECTURE.md`: add the four new Kernel-IR Stmts to the kernel-dialect table (~line 296+);
  add `WarpTile` and `AtomTile` to the Tile-IR flavor list (~line 54) — both Stmts the planner emits but the MMA
  materializer consumes (analogous to today's `RegisterTile`).
- `CLAUDE.md`: nothing needed — README stays example-driven; `make tune-kernels` already exercises the path.
- Wrap all markdown at ~120 chars (CLAUDE.md doc convention). Python module docstrings wrap at the project's
  140-char Python limit.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~40 lines: warp-mode loop-nest example
  in the module docstring, side-by-side with today's scalar example)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` (~15 lines)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~12 lines)

**Verification.** `make lint` clean. Spot-read both docstring examples and confirm they match the actual emitted
`TileOp.body` for one scalar test case and one MMA test case (paste each from a fresh dump).

---

## Failure modes to watch

- **NVRTC `<mma.h>` unavailable (probe first).** The codebase forward-declares TMA intrinsics because NVRTC omits
  `<cuda.h>`/`<cuda/barrier>`. `<mma.h>` may be the same. **Mitigation: a 5-line NVRTC smoke compile of a
  `wmma::fragment` kernel before M4**, so the IR-Stmt render bodies are decided knowing whether `wmma::*` or raw-PTX
  `mma.sync`+`ldmatrix` is the target. This reorders risk to the front instead of discovering it at M6.
- **Silent miscompile on fragment lane mapping.** `wmma::load_matrix_sync(frag, ptr, ldm)` expects `ptr` at the first
  element of the warp's tile and `ldm` as the leading-dimension stride *in elements*. Easy off-by-one. M8's
  end-to-end test catches it; M5's golden IR test does not.
- **Warp-tier launch geometry (Design decision 3, committed to 3a).** `_launch_bounds_for` and `_build_linear_tid`
  compute per-CTA threads as the product of `ThreadTile` axis extents. M3's `WarpTile` arm must compute
  `prod(warp_extents) × 32` in both sites; missing either is a 32× wrong launch bound (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
  or a 32×-misaligned tid decode (silent wrong output).
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
- **Scalar no-op regression.** M1's whole purpose is byte-identical scalar output. Under the sum-type
  `TileParams` (Design decision 11) the primary guarantee is **structural**: the scalar layer builder is
  `ScalarTileParams`-typed and physically cannot reference `Role.ATOM` / `Role.WARP` — those names aren't even in
  scope of the function. The verification assertion ("every emitted row is a `ScalarTileParams`; zero `AtomTile` /
  `WarpTile` instances in any body") catches a violation immediately. Two residual ways this could still break:
  - `ATOM_KIND` leaks into a rendered knob string / `perf` context key on a scalar row. Shouldn't happen — scalar
    rows don't have the field — but worth a regression assert.
  - A downstream pass introspects flavor type with an `assert isinstance(s, ThreadTile)` (rather than `(ThreadTile,
    WarpTile)`) and a future warp-emitting consumer trips it. Caught by the M1 Design-decision-9 audit.
- **`KernelOp.knobs` silently empty post-materialize.** `100_materialize_tile.py:93` returns
  `KernelOp(body=new_body, name=root.op.name)` — `knobs` is inherited from `Op` (`ir/base.py:62`) with a default
  `{}`, so the TileOp's knobs are dropped. None of this plan's MMA-aware passes (M5 dispatch, M7 skip guards, M1
  audit) are affected — they all run on TileOp pre-materialize where `op.knobs["ATOM_KIND"]` is intact. But the
  invariant is fragile: any future pass that needs to introspect ATOM_KIND on a KernelOp (a debug analyzer, a
  KernelOp-level perf-key derivation) silently sees no atom kind. M5's checklist includes the one-line fix to
  forward `knobs` at the materialize boundary; verify with a unit test
  (`KernelOp(...).knobs == TileOp(...).knobs` after materialize).

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

- `deplodock/compiler/ir/tile/ir.py` — tile flavors; add `WarpTile` and `AtomTile` (new `ParallelTile` subclasses,
  consumed by the MMA materializer like `RegisterTile`).
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` — `AtomSpec`, `ATOM_REGISTRY`, `is_atom_eligible` (new).
- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` — sum-type `TileParams` split
  (`ScalarTileParams` | `WarpTileParams`), `BN`/`BM`/`BR` vs `WN`/`WM`/`ATOM_KIND` knob bundles, per-tier priority
  functions (`_priority_matmul_thread` / `_priority_matmul_warp`), per-tier `_enumerate_*_matmul_impl`,
  dispatch-grid `priority_mode`.
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — factorization, per-tier
  `build_fork_tree` invocations + sibling-Fork return (M3), planner `Role` (internal) gains `WARP` + `ATOM`,
  `_layer_kind_for` / `_wrap_tower` grouping, `_build_split_body` sum-type dispatch, `_Plan.scalar_params` /
  `_Plan.warp_params` field split. Docstring loop-nest examples for both scalar and warp modes (see M10).
- `deplodock/compiler/pipeline/fork_tree.py` — the generic `build_fork_tree[P](*, params, levels, materialize,
  score)` builder + `Level[P]` dataclass. Read-only for this plan; M3 calls it twice with disjoint level schemas.
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
