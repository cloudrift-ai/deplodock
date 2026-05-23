# MMA Fragment Factorization

## Context

Adding WMMA / `mma.sync` (sm_70+ FP16 and sm_80+ BF16) tensor-core support to deplodock is naturally framed as a new
*fragment shape* on the matmul reduce, **not** a new lowering branch alongside the existing scalar register-tile path.
Every matmul kernel — scalar or MMA — produces independent accumulator cells along output M and N, replicated by the
planner along `Role.REGISTER` axes. The only differences are (a) how many threads jointly own one cell (1 for scalar,
32 for WMMA, 128 for wgmma), and (b) what hardware instruction updates that cell per K step (`fma` for scalar,
`mma.sync` for WMMA, `wgmma` for sm_90+).

Unifying the two paths under a single **atom shape** knob — with the scalar path being just `(1, 1, 1)` — collapses
what would otherwise be two parallel planner enumerators, two parallel `register_tile_planned`-style passes, and two
parallel materializer branches into one factorization with a per-cell dispatch on `ATOM_SHAPE`.

### The unified factorization

For each output axis (one shown; M / N / K symmetric where applicable):

```
N → N_b · (GROUP_N · CELL_N · ATOM_N) + N_g · (CELL_N · ATOM_N) + N_c · ATOM_N + N_a

       BLOCK              GROUP                       CELL                ATOM
```

- `N_b` — `Role.BLOCK`, today's `N_b`.
- `N_g` — `Role.THREAD` for scalar (`GROUP_N=BN`, atom=1), or `Role.WARP` for MMA (`GROUP_N=warps along N`,
  atom=16). New `BIND_WARP` binding; derived implicitly from `GROUP_SIZE > 1`.
- `N_c` — `Role.REGISTER`, replication count. Today's `FN` for scalar, new `CELL_N` knob for MMA. The same role serves
  both — `006a_register_tile_planned` replicates the body per cell regardless of what's inside it.
- `N_a` — `Role.ATOM` (new), hardware-atomic extent. Materializer never iterates it; it dispatches on the carrying
  `TileOp.knobs["ATOM_SHAPE"]` and emits one scalar or fragment instruction per cell. Extent-1 atoms get inlined by
  `drop_size_one_free_axes` during normalization, so the scalar path's IR is structurally unchanged from today.

K side: today's `K_o · (BR · BK) + K_i · BR + K_c` factorization gains an inner ATOM_K extent. Scalar BK = today's BK
× 1; MMA BK = today's BK × 16 (where 16 is the WMMA K dim). The `FRAG_K` knob (MMA-only) controls how many MMA
instructions fire per K_o iteration — same role as today's BK for scalar paths.

### What this dissolves

- Two factorizations → one. `_split_kernel_fully` keeps its three-way detection (matmul / cooperative-reduce /
  pointwise) but the matmul branch enumerates a single cartesian over `(GROUP, CELL, ATOM_SHAPE, BK, SPLITK)`. The
  scalar case is `ATOM_SHAPE=(1,1,1) ∧ GROUP_SIZE=1`; MMA is `ATOM_SHAPE=(16,16,16) ∧ GROUP_SIZE=32`.
- Two `register_tile_planned` passes → one. The existing pass walks `Role.REGISTER` and replicates body stmts; the
  stmts being replicated (scalar `Init`+`Accum` vs `MmaFragment`+`MmaLoad`+`MmaSync`) come from the planner's
  σ-split output, not from a separate replication pass.
- Two materializer branches → one. `001_materialize_tile.py` reads `TileOp.knobs["ATOM_SHAPE"]` and dispatches on it.
  Scalar emission stays the default; MMA emission is a single new branch.
- The fp16 `__half2` packing pass (`006_pack_fp16_pairs.py`) becomes structurally equivalent to
  `ATOM_SHAPE=(1, 2, 1)` for f16. v1 keeps it as a separate post-pass; v2 could fold it into the framework.

### Scope guard

**Fragmentize only matmul reductions** (entry point `is_matmul_reduce` in `_helpers.py`). Pointwise, softmax,
cooperative-reduce, SDPA's non-matmul reduces — none of these get an atom shape. Forcing them into the framework
would add noise without enabling anything. For non-matmul kernels, `ATOM_SHAPE` is implicitly `(1,1,1)` and not
explored.

### Risk note up front

The riskiest single step is M5 (materializer dispatch on `ATOM_SHAPE`): an off-by-one in the fragment lane mapping
or smem→fragment address calculation will silently produce wrong matmul results. WMMA accumulators are warp-shared
across distributed registers; a single thread can't sanity-check its value. Verification has to compare full output
matrices against a PyTorch reference at end-to-end correctness, not at the per-thread level.

A secondary risk is the planner cartesian explosion: today's matmul enumerator already produces ~50-200 variants per
kernel post-prune. Adding an `ATOM_SHAPE` dimension with 4-5 options could 4-5× the autotune budget. Mitigation is
the `_priority_matmul` ordering, which already deprioritizes unpromising configs — the MMA variants land at the top
when eligible, and the scalar variants follow.

## Design decisions

1. **`ATOM_KIND` lives on `LoopOp.knobs` / `TileOp.knobs`, with a registry resolving it to a full spec.** A
   kernel-wide *string* knob naming the atom kind (`"scalar"`, `"wmma_m16n16k16_f16"`, …); a module-level
   `ATOM_REGISTRY: dict[str, AtomSpec]` maps the kind to a frozen record carrying shape `(M, N, K)`, per-operand
   dtype dict (`{"a": F16, "b": F16, "c": F32}`), the hardware instruction family, and the group size. A single
   string knob + registry lookup means future kinds (NVFP4 / MXFP4 scaled MMA, wgmma) extend the registry rather
   than the knob schema. Eligibility predicates, the cartesian enumerator, the materializer dispatch, and
   `launch_geometry`'s warp-group lifting all read from the registry instead of hardcoding atom-specific
   arithmetic. Read from `op.knobs["ATOM_KIND"]` (default `"scalar"`).

2. **`Role.ATOM` as the innermost role.** Marks the hardware-atomic layer the materializer collapses. Required
   structurally even for the scalar path so the σ-split shape stays uniform; `drop_size_one_free_axes` inlines the
   extent-1 case so today's emitted CUDA doesn't change. Without a `Role.ATOM` Loop, the materializer would need to
   reach into the parent REGISTER to discover atom_shape — adds coupling and breaks the "Role tells you what to do
   with this Loop" invariant.

3. **`BIND_WARP` as a new binding in `axis.py`.** `launch_geometry` synthesizes `warp_id = tid / 32` from `BIND_WARP`
   axes and `lane_id = tid % 32` is implicit. WMMA fragment loads / mma / store operate on the lane group.
   `BIND_BLOCK` and `BIND_THREAD` stay unchanged. For sm_90+ wgmma, a future `BIND_WGROUP` (128 threads) would slot in
   alongside — confirms the binding is per-tier and not a single warp/thread distinction.

4. **Eligibility predicate.** `is_mma_eligible(loop_op, ctx) -> bool`: (a) `is_matmul_reduce` fires on at least one
   reduce in body; (b) every K-indexed Load has dtype in `{F16, BF16}` (BF16 requires sm_80+); (c) `ctx.arch >= 70`;
   (d) M, N extents divisible by atom_m, atom_n; K extent divisible by `BR · BK · atom_k`. Predicate lives in
   `_helpers.py` alongside `is_matmul_reduce`.

5. **Atom-kind candidate set, v1.** `"scalar"` always; `"wmma_m16n16k16_f16"` when eligible (the WMMA "square" shape
   — only one with broad arch support and the simplest lane mapping). BF16 + skewed WMMA shapes land in M9 as
   additional registry entries. NVFP4 / MXFP4 / wgmma kinds wait for their own plans — they reuse this plan's
   factorization but add infrastructure (TMEM tier, async issue/wait, scaled-MMA primitive, multi-operand fragments)
   beyond extending the registry.

6. **Scalar path is a no-op refactor.** M1 plumbs `ATOM_KIND` and `Role.ATOM` end-to-end with `"scalar"` as the
   only registered kind. Existing golden IR tests must pass byte-identical post-normalization — the extent-1 ATOM
   Loops get inlined, the new `ATOM_KIND="scalar"` knob is dropped by `format_tuning_knobs` (default — needs an
   explicit drop rule), and the BLOCK·THREAD·REGISTER tower the planner emits today is structurally unchanged.
   If anything in M1 changes a golden, that's a bug, not a re-bless.

7. **`pack_fp16_pairs` interaction.** This kernel-level pass pairs scalar f16 accumulators into `__half2`.
   For MMA kernels (`ATOM_SHAPE != (1, 1, 1)`) there are no scalar f16 `Init` / `Accum` to pair — the C-fragment IS
   the accumulator. Add a structural guard: skip when body contains `MmaFragment` decls.

8. **`permute_lane_accesses` interaction.** Permutes LDS.128 indices on Stage Loads to break bank conflicts. WMMA
   uses `ld.matrix` (or the wmma::load_matrix_sync intrinsic), which has its own swizzled access pattern. Skip when
   `ATOM_SHAPE != (1, 1, 1)` to avoid double-permutation.

9. **Other downstream passes stay unchanged.** `007_stage_inputs`, `030_use_ring_buffers`, `040_use_tma`,
   `050_use_async_copy`, `060_pad_smem`, `015_pipeline_k_outer` all operate on K_o / STAGE_INNER and don't care about
   the cell shape. Confirmed by reading their PATTERN matches — none of them touch the output-axis tower below
   STAGE_INNER.

10. **MMA via `wmma::load_matrix_sync` / `mma_sync` / `store_matrix_sync` in v1, not raw `ld.matrix` + `mma.sync`
    PTX.** The intrinsic path is uglier on SASS (one extra register copy per fragment) but vastly simpler to plumb
    and verify. v2 can swap in raw PTX when we have ncu evidence the intrinsic path is the bottleneck — the IR
    abstraction (`MmaLoad` / `MmaSync` / `MmaStore` Stmts) doesn't change.

---

**Prerequisite landed:** `Axis.source_axis` (Phase A of the stage-wrap-body refactor) is now in place. Every
split sub-axis carries a back-pointer to its parent; the MMA enumerator can use this for BLOCK·GROUP·CELL·ATOM
grouping without name-suffix matching. See `plans/stage-wrap-body.md`.

## M1 — Plumb `Role.ATOM` + `ATOM_KIND` registry through the planner as a no-op

**Why.** Establish the unified factorization scaffolding without changing any emitted CUDA. The scalar path goes
through the new code path with `ATOM_KIND="scalar"` and `Role.ATOM` Loops of extent 1, which
`drop_size_one_free_axes` inlines during normalization. If this milestone shifts any golden IR or any test result,
the scaffolding is wrong; fix before proceeding.

**Change.**

- `deplodock/compiler/ir/axis.py`: add `Role.ATOM` to the enum with a docstring entry describing it as
  "hardware-atomic extent; materializer dispatches on `op.knobs['ATOM_KIND']`."
- New file `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py`: defines `AtomSpec` (frozen dataclass with
  `shape: tuple[int, int, int]`, `operand_dtypes: Mapping[str, DataType]`, `instruction: str`, `group_size: int`)
  and `ATOM_REGISTRY: dict[str, AtomSpec]` seeded with one entry: `"scalar" → AtomSpec((1,1,1), {"a":…,"b":…,"c":…}
  inferred-or-F32, "fma", 1)`. Public helpers `atom_spec(kind) -> AtomSpec`, `atom_shape(kind)`,
  `atom_group_size(kind)`.
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py`: in `_build_split_body` (line 443-530),
  read `kind = "scalar"`, `(atom_m, atom_n, atom_k) = atom_shape(kind)` (= `(1,1,1)`), extend the layers list with
  `Role.ATOM` Loops of the corresponding extents for N_a, M_a, K_a. Update the σ-substitution in
  `sigma_map[N_name]` / `sigma_map[M_name]` to include the `+ N_a` term (and equivalently for M), and the K-sigma
  in `_build_k_sigma` to include `+ K_a` (line 619-629). Every variant emitted gets
  `knobs["ATOM_KIND"] = "scalar"` stamped.
- `deplodock/compiler/pipeline/knobs.py`: register `ATOM_KIND` as a `KnobType.STR` (new type if absent — small
  addition to `KnobType` enum, with parsing/rendering that's literally identity). `format_tuning_knobs` drops
  `"scalar"` from rendered output (default-elision).

**Files.**

- `deplodock/compiler/ir/axis.py` (~6 lines)
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~40 lines new file)
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py` (~30 lines: layers extension +
  σ-rewrite + knob stamping)
- `deplodock/compiler/pipeline/knobs.py` (~15 lines: `KnobType.STR` + `ATOM_KIND` registration + default elision)

**Verification.** `make test` byte-clean — no golden bless. Spot-check one matmul kernel's `08_lowering_cuda.cu`
dump under `DEPLODOCK_DUMP_DIR` against a pre-M1 snapshot: must be identical.

## M2 — Per-kind eligibility predicate + planner enumerator split

**Why.** Before adding MMA atom kinds to the cartesian, the planner needs to know which kernels qualify for which
kind. A wrong predicate either silently disables MMA on eligible kernels (perf regression) or enables it on
ineligible ones (compile error or wrong output).

**Change.**

- `_atom.py`: add `"wmma_m16n16k16_f16"` to `ATOM_REGISTRY` with `AtomSpec((16,16,16), {"a":F16, "b":F16, "c":F32},
  "wmma", 32)`. Also add a per-entry `eligibility(loop_op, ctx) -> bool` callable on `AtomSpec` (or a parallel
  `ATOM_ELIGIBILITY` dict), checking the kind's specific requirements. Public helper
  `is_atom_eligible(kind: str, loop_op: LoopOp, ctx: Context) -> bool` dispatches via the registry.
- The WMMA-F16 predicate checks: (a) `is_matmul_reduce` on any body reduce; (b) every K-indexed Load and the Accum
  target dtype is `F16`; (c) `ctx.arch >= 70`; (d) E_M, E_N divisible by 16; E_K divisible by 16 (defer the
  `% (16·BR)` check until BR is picked).
- Add module-level constant `_ATOM_KINDS_V1: tuple[str, ...] = ("scalar", "wmma_m16n16k16_f16")` listing the
  enumerator's candidate set in priority order.
- `_split_kernel_fully` (line 213-241): when matmul branch fires, filter `_ATOM_KINDS_V1` by `is_atom_eligible`;
  pass the surviving kinds list to `_enumerate_cartesian` as a new parameter.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~30 lines: WMMA registry entry + eligibility helper)
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py` (~15 lines)

**Verification.** Unit test: build a `Context(arch=75)` + a TinyLlama-shape matmul `LoopOp` with f16 Loads →
`is_atom_eligible("wmma_m16n16k16_f16", …)` returns True. Build the same with f32 → False. Build with `arch=60`
(Pascal) → False. Build a softmax `LoopOp` (no matmul reduce) → False. `is_atom_eligible("scalar", …)` always
returns True for any LoopOp.

## M3 — Extend cartesian over `ATOM_KIND`

**Why.** Wire the eligible atom-kind candidates into the variant enumeration so the tuner sees scalar and MMA
configs as siblings.

**Change.** In `_enumerate_cartesian` (line 343-440), wrap the existing cartesian with an outer loop over the passed
`atom_kinds: tuple[str, ...]`. For each `kind`, resolve `spec = atom_spec(kind)` and read `(atom_m, atom_n, atom_k)
= spec.shape`, `group_size = spec.group_size`:

- Effective per-group divisibility check: `E_M % (bm_c * fm * atom_m) == 0` instead of `E_M % (bm_c * fm) == 0`.
  Similarly for N. K divisibility: `per_thread_K % (bk * atom_k) == 0`.
- When `group_size > 1`, force `bn_c, bm_c` to multiples consistent with `group_size · atom_m · atom_n` worth of
  output per group; add a separate `_TUNE_WARP_AXIS_CHOICES = (1, 2, 4, 8)` for the warps-per-axis enumeration.
- Stamp `knobs["ATOM_KIND"] = kind` per variant. No per-dim atom knobs — shape/dtypes/group_size live in the
  registry and are looked up at materialize time.

Priority: `_priority_matmul` ranks MMA variants strictly above scalar variants when both are present (lift
`min(p.fm * p.fn * atom_m * atom_n, 64)` instead of capping at 32; MMA fragments amortize K-loop overhead far better
than scalar). Final tiebreaker is the existing thread-count-near-256 heuristic.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py` (~40 lines: cartesian rewrite +
  priority adjust)

**Verification.** Unit test: an eligible TinyLlama matmul `LoopOp` → enumerated variants include at least one with
`ATOM_KIND="wmma_m16n16k16_f16"`. A non-eligible (f32) matmul → no MMA variants. A pointwise kernel → no `ATOM_KIND`
knob stamped (defaults to `"scalar"` via M1).

## M4 — Kernel-IR Stmts: `MmaFragment`, `MmaLoad`, `MmaSync`, `MmaStore`

**Why.** Hardware primitives for the materializer to emit when atom shape is MMA. Lives alongside `Smem`, `Sync`,
`TreeHalve` in `kernel/ir.py` — these are not Tile-IR concepts (Tile IR encodes scheduling, not hardware ISA).

**Change.**

- `deplodock/compiler/ir/kernel/ir.py`: four new `Stmt` subclasses.
  - `MmaFragment(name: str, role: str, shape: tuple[int, int, int], dtype: DataType)`. `role` is a free-form string
    (v1 values: `"a"`, `"b"`, `"c"`; future kinds may add `"a_scale"`, `"b_scale"`, etc.). Renders as
    `nvcuda::wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> name;` (role-dependent).
  - `MmaLoad(frag: str, src_buffer: str, src_offset: Expr, ldm: int)`. `src_buffer` names a smem allocation in v1;
    future kinds (NVFP4 reads accumulator from TMEM) may target other tiers via the same parameter. Renders as
    `wmma::load_matrix_sync(frag, &<buffer>[offset], ldm);`.
  - `MmaSync(c_frag: str, a_frag: str, b_frag: str)`. Synchronous 3-operand MMA. Renders as
    `wmma::mma_sync(c, a, b, c);`. **Note**: future scaled-MMA kinds (NVFP4) introduce a sibling Stmt with a
    5-operand signature — they are distinct enough that pretending they share one Stmt obscures more than it
    helps. Async kinds (Hopper / Blackwell) introduce `MmaIssue` + `MmaWait` Stmts; same principle.
  - `MmaStore(dst_buffer: str, dst_offset: Expr, frag: str, ldm: int, layout: Literal["row", "col"])`. Renders as
    `wmma::store_matrix_sync(&<buffer>[offset], frag, ldm, wmma::mem_row_major);`.
- Each implements `Stmt.render(ctx)` returning the CUDA source line(s).
- `pretty()` for IR dumps: `mma_frag c0 : c[16,16,16] f32` style.
- `structural_key()` excludes `frag` / `src_smem` / `dst_smem` *names* (rename-stable), includes shape + dtype +
  role + offset Expr's key.

**Files.**

- `deplodock/compiler/ir/kernel/ir.py` (~120 lines: four dataclasses + render + pretty + structural_key)
- `deplodock/compiler/ir/kernel/ARCHITECTURE.md` (~10 lines documenting the new primitives if such a doc exists;
  otherwise update `deplodock/compiler/ir/ARCHITECTURE.md` line 240-245)

**Verification.** Unit tests in `tests/compiler/ir/test_kernel_mma.py`: construct each Stmt, render with a stub
`RenderCtx`, assert the emitted CUDA matches a golden string. Structural-key dedup: two `MmaSync(c, a, b)` with
different name letters but the same Smem source structure → equal keys.

## M5 — Materializer dispatch on `ATOM_SHAPE`

**Why.** The load-bearing change. When `op.knobs["ATOM_M"] != 1`, the materializer emits MMA fragment chains
instead of scalar Init/Accum. This is where correctness can silently break.

**Change.** In `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py`:

- Read `kind = root.op.knobs.get("ATOM_KIND", "scalar")`. Resolve `spec = atom_spec(kind)` for shape / dtypes /
  group_size.
- When `kind != "scalar"` and `spec.instruction == "wmma"`, route through a new `_materialize_mma_body` helper that:
  1. Walks the body to find the reduce-K `Loop(STAGE_INNER)` and its containing tower of `Role.REGISTER` Loops.
  2. For each `(register_m, register_n)` cell, emits `MmaFragment(name=f"c_{m}_{n}", role="c", shape=spec.shape,
     dtype=spec.operand_dtypes["c"])` in the prelude (before the K_o loop).
  3. Inside the K_o → STAGE_INNER body: for each `(register_m, register_n)` cell, emits `MmaFragment` a-frag +
     b-frag decls with the operand dtypes from `spec.operand_dtypes`, `MmaLoad` for each from the staged smem slab
     at the correct offset (warp-cooperative; offset uses `spec.group_size` for warp counting), then
     `MmaSync(c_{m,n}, a_frag, b_frag)`.
  4. After the K_o loop, emits `MmaStore` from each `c_{m,n}` fragment to the appropriate smem accumulator location
     (or directly to gmem via `Write` if no combine is needed).
- Wrap the new path behind a feature flag `_MMA_ENABLED = os.environ.get("DEPLODOCK_MMA", "1") != "0"` until M8
  passes — gives a safe off switch if the path is wrong.
- Skip the scalar-Accum emission for cells covered by the MMA path; keep it unchanged for non-MMA kernels.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` (~120 lines: new helper + dispatch +
  env gate)

**Verification.** Two-pronged. (a) Golden IR test: a small synthetic f16 matmul (M=N=K=64) with
`ATOM_SHAPE=(16,16,16)` produces a `KernelOp` body with the expected `MmaFragment` decl count (1 c-frag + 1 a-frag +
1 b-frag per `(register_m, register_n, K_i)` triple) and the expected `MmaSync` chain length. Assert structurally
against an expected pretty-print. (b) **End-to-end numerical**: an existing TinyLlama f16 matmul test in
`tests/perf/` or `tests/compiler/test_matmul.py` runs with `DEPLODOCK_MMA=1` and produces output matching the
PyTorch reference within fp16 tolerance (max-abs-err ≤ 1e-2 against f32 reference, or ≤ existing test tolerance).
This is the gate on M5 being correct.

## M6 — CUDA render: `<mma.h>` include + namespace

**Why.** The new Stmts already render their own lines (M4); render-time scaffolding needs the include and
namespace declaration so the kernel compiles.

**Change.** In `deplodock/compiler/ir/kernel/render.py`:

- In the include-list builder, add `#include <mma.h>` and `using namespace nvcuda;` when any body Stmt is an
  `MmaFragment` / `MmaLoad` / `MmaSync` / `MmaStore`. Detection: walk the rendered body once before emission and
  set a `needs_mma_header` flag, then conditionally prepend.
- Verify NVRTC ships `<mma.h>` (it does — it's part of the CUDA runtime header set, not the libcu++ subset that's
  excluded). If not, fall back to forward-declarations in `_TMA_PRELUDE`-style raw asm wrappers (defer; M6 assumes
  the include works).

**Files.**

- `deplodock/compiler/ir/kernel/render.py` (~15 lines)

**Verification.** Render an `MmaSync`-bearing kernel, assert the output string starts with the include + namespace
preamble. Compile via NVRTC with `cupy.RawKernel` and assert no compile error (using the existing test fixture for
NVRTC compilation).

## M7 — Skip incompatible kernel passes

**Why.** `pack_fp16_pairs` and `permute_lane_accesses` don't apply to MMA kernels and would corrupt them if
they fired.

**Change.**

- `006_pack_fp16_pairs.py`: at top of `rewrite`, add
  `if root.op.knobs.get("ATOM_KIND", "scalar") != "scalar": raise RuleSkipped("non-scalar atom kind; pack-half2 not
  applicable")`. Idempotent — knob value doesn't change between runs.
- `005_permute_lane_accesses.py`: same guard via `root.op.knobs.get("ATOM_KIND", "scalar") != "scalar"`. The check
  generalizes to every future MMA kind (WMMA-BF16, NVFP4, wgmma) without re-editing the guard.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/006_pack_fp16_pairs.py` (~5 lines)
- `deplodock/compiler/pipeline/passes/lowering/tile/005_permute_lane_accesses.py` (~3 lines)

**Verification.** Run an MMA-eligible kernel through the full pipeline with `DEPLODOCK_MMA=1`; assert via the
`.rules.json` dump that both skipped passes log `RuleSkipped` with the expected reason on the MMA variant.

## M8 — End-to-end correctness + perf gate

**Why.** Beyond unit tests, the load-bearing question is whether the MMA path produces correct output across
realistic matmul shapes and whether it actually wins on perf vs the scalar register-tile path.

**Change.** No code beyond test additions.

- Add `tests/compiler/test_matmul_mma.py`: parametrize over (M, N, K) shapes — at minimum (64, 64, 64),
  (512, 512, 512), (4096, 4096, 4096) — and dtype (f16, bf16 on sm_80+ only); pin `DEPLODOCK_MMA=1` and
  `DEPLODOCK_KNOBS="ATOM_KIND=wmma_m16n16k16_f16"`; assert max-abs-err vs the f32 PyTorch reference within fp16
  tolerance.
- Add `tests/perf/test_matmul_mma_perf.py` (under the `perf` marker; runs only with `pytest -m perf`): bench
  TinyLlama-shape matmul with `DEPLODOCK_MMA=0` vs `DEPLODOCK_MMA=1`; assert MMA ≥ 2× scalar on sm_80+ hardware.
  Skip if `arch < 70`.
- Run `make bench-kernels-tuned` and confirm the autotune DB picks the MMA variant on every matmul kernel in a
  TinyLlama or Qwen 7B layer; record the table in the PR description.

**Files.**

- `tests/compiler/test_matmul_mma.py` (~100 lines)
- `tests/perf/test_matmul_mma_perf.py` (~50 lines)

**Verification.** All new tests pass. Bench shows MMA winning by the expected margin (2-8× on f16 matmul,
shape-dependent). If MMA *loses* on small shapes (M=N=K=64), that's expected — the launch overhead dominates;
the priority function will rank scalar variants higher there. Confirm via the autotune DB: which atom shape was
picked per shape.

## M9 — Additional atom kinds (skewed WMMA, BF16)

**Why.** WMMA on sm_80+ supports `(8, 32, 16)` and `(32, 8, 16)` for skinny matmul shapes (e.g. the projection
matmuls in attention have very different M and N extents). BF16 has the same shape menu but different operand
dtype. The square `(16, 16, 16)` F16 from M3 is the safe default but leaves perf on the table for skewed shapes
and bf16-native models.

**Change.** Add new `ATOM_REGISTRY` entries: `"wmma_m16n16k16_bf16"`, `"wmma_m8n32k16_f16"`,
`"wmma_m32n8k16_f16"`, and BF16 versions of the skewed shapes. Extend `_ATOM_KINDS_V1` →
`_ATOM_KINDS_V2 = (..., "wmma_m8n32k16_f16", "wmma_m32n8k16_f16", "wmma_m16n16k16_bf16", ...)`. The per-kind
eligibility predicate already gates BF16 on `ctx.arch >= 80` and the skewed shapes on divisibility — no new
checks at the planner level. The materializer is registry-driven (M5's `spec` lookup), so no codegen changes
beyond the new entries.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` (~20 lines: new registry entries)
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py` (~3 lines: candidate list)

**Verification.** Re-run M8's perf bench with the expanded candidate set; confirm the tuner picks
`"wmma_m8n32k16_f16"` on at least one skinny matmul in Qwen 7B (the gate/up projections in MLP are good
candidates) and that the picked kind outperforms `"wmma_m16n16k16_f16"` on it.

## M10 — Documentation + ARCHITECTURE.md updates

**Why.** Once the fragment factorization is the canonical model, the existing ARCHITECTURE.md descriptions of the
scalar register-tile path read incompletely. New contributors need to understand "every matmul has an atom shape;
scalar is (1, 1, 1)" as a first-class concept, not a footnote.

**Change.**

- `deplodock/compiler/pipeline/ARCHITECTURE.md`: extend the partition-planner factorization block (line 246-250) to
  describe `BLOCK · GROUP · CELL · ATOM` as the unified shape; document `ATOM_KIND` in the knob table; describe
  the `ATOM_REGISTRY` lookup model and list current entries.
- `deplodock/compiler/ir/ARCHITECTURE.md`: add the four new Kernel-IR Stmts to the kernel-dialect table.
- `deplodock/compiler/ir/axis.py`: `Role.ATOM` docstring already updated in M1; add `BIND_WARP` doc entry.
- `CLAUDE.md`: nothing needed — the README intentionally stays example-driven, and `make tune-kernels` already
  exercises the new path.

**Files.**

- `deplodock/compiler/pipeline/ARCHITECTURE.md` (~15 lines)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~10 lines)
- `deplodock/compiler/ir/axis.py` (~3 lines)

**Verification.** `make lint` clean (markdown wrapping at ~120). Spot-read the changed sections to confirm the
unified model reads naturally.

---

## Failure modes to watch

- **Silent miscompile on MMA fragment lane mapping.** The smem→fragment offset arithmetic in M5 is the most subtle
  part — `wmma::load_matrix_sync(frag, ptr, ldm)` expects `ptr` to point at the first element of the warp's tile
  and `ldm` to be the leading-dimension stride in elements (not bytes). Easy off-by-one. Mitigation: M8's
  end-to-end test against a PyTorch reference catches it; M5's golden IR test does not.
- **`MmaSync` is synchronous in v1.** Hopper / Blackwell tensor cores are async (issue + wait pattern); the
  registry's `spec.instruction == "wmma"` branch in M5 explicitly opts into the synchronous semantics. Future kinds
  (`"wgmma_*"`, `"tcgen05_*"`) will add their own materializer branches with `MmaIssue` / `MmaWait` Stmts.
  `_materialize_mma_body` is structured per-instruction-family so async kinds don't have to rewrite the WMMA path.
- **Autotune cartesian explosion.** Adding 2-4 atom shapes × existing variant count could 4× the per-kernel
  search space. Mitigation: `_priority_matmul`'s atom-shape boost (M3) puts MMA variants at the top of the queue;
  the patience-based stop catches the case where scalar is winning early.
- **MMA + cooperative-K conflict.** v1 cooperative-K constraint is `BR > 1 ⇒ BN = BM = 1`. With MMA, `BN`/`BM`
  are implicitly the warp count × 16, so `BN = BM = 1` is impossible. **Solution: gate MMA off when BR > 1** —
  the cartesian's MMA variants only enumerate `BR = 1`. Defer MMA + cooperative-K to v3.
- **fp16 accumulator drift.** WMMA's C-fragment is fp32 by default even when A/B are fp16. The scalar path's f16
  accumulator (when it exists — most matmuls accumulate in f32 already) has more drift than the MMA path; if a
  test was relying on scalar's accumulator dtype, it may need re-blessing. Audit the dtype-promotion logic during
  M8.
- **NVRTC `<mma.h>` availability.** If a target system's CUDA toolkit ships `<mma.h>` via a header-locked include
  path that NVRTC doesn't see, M6's include fails. Mitigation: detect at backend init; fall back to the raw
  asm wrappers approach (analogous to the TMA prelude). Defer until observed.
- **Extent-1 ATOM inlining doesn't fire.** M1 depends on `drop_size_one_free_axes` (in
  `ir/loop/normalize.py`) inlining the extent-1 ATOM Loops. If the normalizer skips them (e.g. because they
  carry a role tag), the scalar path's IR shape changes and every golden busts. Verify the normalizer handles
  role-tagged extent-1 Loops before M1's golden re-runs; if not, extend the normalizer's "ignore role" rule for
  the inline case.

## Future extensions (out of scope for this plan)

This plan deliberately ships WMMA on sm_70-sm_89 only. Future plans extend the framework without redesigning it:

- **NVFP4 / MXFP4 scaled MMA (Blackwell sm_100+).** Adds new `ATOM_REGISTRY` entries (`"tcgen05_nvfp4_*"`,
  `"tcgen05_mxfp4_*"`) with FP4 operand dtypes, an FP8 scale operand dtype, and a `scale_block_size` field on
  `AtomSpec`. Introduces a sibling `MmaScaledSync(c, a, a_scale, b, b_scale, kind)` Stmt next to `MmaSync`. Adds a
  `Tmem` allocation primitive next to `Smem` (Blackwell accumulators live in tensor memory). Adds async
  `MmaIssue` / `MmaWait` Stmts and threads them through `015_pipeline_k_outer`. Frontend work: quantized
  `MatmulOp` variant carrying scale-tensor operands, or a separate `ScaledMatmulOp`. Each of these is its own
  milestone-track plan; the *factorization* and the *Role.ATOM* / `Role.REGISTER` machinery stay unchanged.
- **wgmma (Hopper sm_90).** Same shape as NVFP4 minus the scale operands. New registry entries (`"wgmma_*"`),
  `BIND_WGROUP` binding in `axis.py` (128-thread warp groups), and the async issue/wait infrastructure that NVFP4
  also needs. Likely lands as a precursor to the NVFP4 plan since it shares the async + warp-group plumbing.
- **Sparse MMA (sm_80+ 2:4 sparsity).** New registry entries with a sparsity metadata operand. Same scaffolding;
  one more fragment role per matmul cell.

**Why these slot in cleanly.** Each future kind adds (registry entry, optional new Stmt, optional new memory
tier). The planner's σ-split, the per-cell replication, `launch_geometry`'s lifting, the autotune cartesian, and
the env-pinning workflow all read from the registry — none of them embed v1's "WMMA is special" assumptions. The
materializer's instruction-family dispatch (`spec.instruction`) is the single extension point for new codegen
paths.

## Test additions summary

- `tests/compiler/passes/test_partition_planner_mma.py` — `is_mma_eligible` predicate (M2), cartesian enumeration
  includes MMA variants when eligible (M3).
- `tests/compiler/ir/test_kernel_mma.py` — new Stmt construction + render + structural_key (M4).
- `tests/compiler/passes/test_materialize_tile_mma.py` — golden IR for a small MMA kernel (M5).
- `tests/compiler/test_matmul_mma.py` — end-to-end f16/bf16 correctness across shapes (M8).
- `tests/perf/test_matmul_mma_perf.py` — perf gate, MMA vs scalar (M8).
- `tests/compiler/test_register_tile_planned_mma_skip.py` — incompatible-pass skip guards (M7).

## Critical files

- `deplodock/compiler/ir/axis.py` — `Role.ATOM`, `BIND_WARP`
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` — `AtomSpec`, `ATOM_REGISTRY`, `is_atom_eligible`
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_loops.py` — factorization + cartesian
- `deplodock/compiler/ir/kernel/ir.py` — `MmaFragment` / `MmaLoad` / `MmaSync` / `MmaStore`
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` — dispatch on `ATOM_KIND`
- `deplodock/compiler/ir/kernel/render.py` — `<mma.h>` include
- `deplodock/compiler/pipeline/knobs.py` — `KnobType.STR` + `ATOM_KIND` registration
- `deplodock/compiler/pipeline/passes/lowering/kernel/006_pack_fp16_pairs.py` — skip guard
- `deplodock/compiler/pipeline/passes/lowering/tile/005_permute_lane_accesses.py` — skip guard
- `deplodock/compiler/pipeline/ARCHITECTURE.md` — documentation update
- `deplodock/compiler/ir/ARCHITECTURE.md` — documentation update
