# WMMA smem staging — follow-up to `plans/mma-fragment-factorization.md`

## Context

The MMA fragment factorization (PR #177) lowered f16 WMMA matmul end-to-end with operands read **directly from gmem**
via `wmma::load_matrix_sync`. v1 ships correctness (max-abs-err 0.0 on 16/64/128 squares) but leaves the gmem-direct
loads as a known perf gap — standard CUDA WMMA matmuls stage operands through smem first (cooperative gmem→smem
copy, then warp-fragment load), and on Hopper the canonical shape is TMA→smem→WMMA-fragment.

Three pieces of v1 plumbing carry the "no smem staging" assumption that this plan removes:

- `tile/020_stage_inputs.py:122-129` — `RuleSkipped("MMA path bypasses smem staging in v1")` guarded on `ATOM_KIND`.
- `kernel/005_lower_atom_tile.py:75-80` — defensive pass-through for `Stage`/`StageBundle` inside an AtomTile body.
- `kernel/005_lower_atom_tile.py:118-119` — `MmaLoad(... ldm=0)` reads `ctx.shapes[gmem_buf]`; for staged operands the
  buffer becomes an smem slab whose extents come from `Source.alloc_extents`.

Target: ≥ 1.5× perf vs gmem-direct WMMA on TinyLlama-shape matmuls (M·N·K ≥ 128³), where smem reuse pays off.

## The central design problem

The planner's MMA σ (in `_build_split_body_warp`, ~`010_partition_loops.py:1042`) emits Loads with non-unit strides
on every per-warp / per-cell axis:

    A[m_b·(WM·FM·atom_M) + m_w·(FM·atom_M) + m_r·atom_M, K_o·(bk·atom_K) + K_i·atom_K]

Each per-thread cache var (`M_w`, `M_r`, `K_i`) carries an σ stride that includes the **atom factor**
(`atom_M = atom_K = 16` for `wmma_m16n16k16_f16`). `020_stage_inputs._classify` (~`020_stage_inputs.py:567`) currently
does a coef-1 affine check on the per-source-dim σ output. With atom-strided σ no var has coef 1 → today this would
fall back to template addressing. But `Source.alloc_extents` (`ir/tile/ir.py:372-377`) returns naked cache-axis
extents *without the atom multiplier*: for cache_dims `(M_w, M_r, K_i)` with extents `(WM, FM, bk)`, the slab is
allocated as `(WM, FM, bk)` instead of the correct `(WM, FM·atom_M, bk·atom_K)`. The slab is missing a 256× area
factor for fp16 m16n16k16, and the producer cooperative-load under-fills the slab — WMMA reads garbage from the
255/256 that nobody wrote.

The fix is to recognize the σ literal coefficient as a structural per-cache-dim multiplier and carry it on the
addressing object, so the slab geometry, producer iteration range, and consumer index reconstruction all see the
same number while staying affine.

## Recommended approach

**Lift the addressing-mode payload off `Source` and add a `block` multiplier to `AffineAddressing`.** Today
`Source.template_index: tuple[Expr, ...] | None` is an addressing-mode payload masquerading as a Source field —
`Source.addressing` is a derived property that branches on it. Pulling it into `TemplateAddressing.exprs`, making
`Source.addressing: AffineAddressing | TemplateAddressing` a stored field, and giving `AffineAddressing` a per-
cache-dim `block: tuple[int, ...] = ()` lets the staging machinery describe both scalar and atom cases through a
single typed channel:

- **Scalar matmul / RMSNorm / softmax** — `Source.addressing = AffineAddressing(dims=(...,), block=())`. Empty
  `block` means "trivial multiplier, all 1s" — byte-identical to today's behavior.
- **Atom-strided MMA** — `Source.addressing = AffineAddressing(dims=(...,), block=(1, atom_M, 1, atom_K))`. The
  slab grows to `extent[i] * block[i]` per cache dim, the producer cooperative-load iterates the grown range, the
  consumer cache vars stay at their original extent, and `affine_decode_per_dim` folds `block`-scaled strides into
  the per-source-dim composite.
- **Collapsed-reshape views** — `Source.addressing = TemplateAddressing(exprs=...)`. Unchanged semantics; the
  field that used to live on `Source` as `template_index` now lives where it always belonged.

This collapses two prospective new fields on `Source` (`atom_factor`, `mma_view`) into a single addressing-mode
extension. Source's surface stays focused on slab identity (`name`, `buf`, `cache_dims`, `origin`, `pad`, `dtype`)
and the addressing-mode-discriminator branch in `Source.addressing` becomes a no-op accessor of the stored field.

The MMA cell materializer `kernel/005_lower_atom_tile.py` walks into the `StageBundle` body, finds the per-warp
scalar Loads already rewritten to read from smem, and emits `MmaLoad` reading from the smem slab with a folded
rank-2 warp-relative tile-base offset. The fold is computed via a new
`AffineAddressing.source_index(coord_for, origin)` method that uses `block`-scaled strides — same shape as the
existing free-function `affine_decode_per_dim`, just lifted onto the type so `005` doesn't have to know about the
`block` field directly.

A pre-existing latent issue is already mitigated in `affine_decode_per_dim` (`ir/tile/ir.py:395-432`): the previous
`dict(zip(dims, coord_for))` shape silently dropped one of two cache axes sharing a `source_dim`. That fix already
landed; M4 below only needs to thread `block` into the same composite-stride formula.

## Milestones

### M1 — Probe gate + plumbing smoke

Remove the `ATOM_KIND` skip in `020_stage_inputs.py:122` behind a `DEPLODOCK_MMA_STAGE_PROBE` env gate (default OFF
so M6 below can flip the default ON only once all the staged-WMMA pieces are green). Add the config accessor in
`config.py` mirroring `mma_enabled()`. **Verify**: a new test asserts the post-020 body for an MMA matmul contains a
`StageBundle` with A + B `Source`s when the probe is on; existing `test_matmul_mma.py` is byte-clean (probe off).

**Files**: `tile/020_stage_inputs.py` (~10 lines), `config.py` (~5 lines),
`tests/compiler/passes/test_stage_inputs_mma_probe.py` (new).

### M2 — Addressing-mode refactor (pure cleanup, no behavior change)

Pure refactor — no compiler behavior changes; this milestone exists so M3+ can work against a clean addressing
model.

1. **Extend `AffineAddressing` with `block: tuple[int, ...] = ()`.** Default `()` means "all-1s, byte-identical
   to today." Validate in `__post_init__` that non-empty `block` has `len(block) == len(dims)` and every entry is
   `≥ 1`.
2. **Move `template_index` into `TemplateAddressing.exprs`.** Drop `Source.template_index`. The contents flow into
   the stored `TemplateAddressing(exprs=...)` when staging classifies a load as template-addressed.
3. **Make `Source.addressing` a stored field, not a derived property.** Replace the property at
   `ir/tile/ir.py:364-368` with a dataclass field of type `AffineAddressing | TemplateAddressing`. Construction
   sites (`020_stage_inputs.py:742`, hand-written test fixtures) build the addressing object explicitly.
4. **Update `Source.alloc_extents`.** When addressing is affine and `block != ()`, return
   `extent[i] * block[i] + pad[i]` per cache dim. When affine and `block == ()`, behavior is identical to today.
   When template, behavior is unchanged.
5. **Update read sites.** Three call sites need to switch from `src.template_index is not None` to
   `isinstance(src.addressing, TemplateAddressing)`:
   `025_unify_sibling_stages.py:179-180`, `ir/tile/ir.py:475-477` (`_source_pretty`),
   `ir/tile/ir.py:492-512` (`_source_decl_line`), `ir/tile/ir.py:1141-1142` (`free_vars` walk).

**Files**: `ir/tile/ir.py` (~80 lines net), `tile/020_stage_inputs.py` (~15 lines — construction site flip),
`tile/025_unify_sibling_stages.py` (~5 lines), `ir/ARCHITECTURE.md` (`Source` field list update),
`tests/compiler/ir/tile/test_addressing_refactor.py` (new — covers `block != ()` slab sizing,
`isinstance(addr, TemplateAddressing)` read sites, AffineAddressing `__post_init__` validation).

**Verify**: full test suite is byte-clean — no IR-dump diffs in `tests/compiler/snapshots/`, every scalar
matmul/rmsnorm/softmax case passes unchanged. This is the gate that the refactor is purely structural.

### M3 — `_classify` recognition stamps `AffineAddressing.block`

In `020_stage_inputs._classify`: when `unit_sigma.reduce(load.index[d], ctx)` simplifies to
`Literal(c) * Var(ax.name)` with `c > 1`, record `c` in the per-cache-dim block tuple and keep affine addressing.
The resulting `Source.addressing` is `AffineAddressing(dims=(..., d, ...), block=(..., c, ...))` instead of a
template fallback. **Mitigation for off-by-one risk**: assert the σ output has *exactly* `Var(ax) * Literal(c)` (no
other terms on this cache axis) before stamping; reject ambiguous cases and fall back to template.

**Files**: `tile/020_stage_inputs.py` (~50 lines),
`tests/compiler/passes/test_stage_inputs_block_recognition.py` (new).

### M4 — `affine_decode_per_dim` honors `block`

Extend `affine_decode_per_dim` (`ir/tile/ir.py:395`) to accept a `block: tuple[int, ...] = ()` argument and fold
`block[j]` into the composite stride: when summing the per-source-dim contributions, the i-th cache axis's stride
is `prod(cache_axes[j].extent * (block[j] if block else 1) for j > i where dims[j] == d)`. Plumb the addressing
object's `block` through every call site (`_stage_expand`, `025_unify_sibling_stages._reconstruct_global_index`,
`_source_decl_line`). Add `AffineAddressing.source_index(coord_for, origin) -> tuple[Expr, ...]` that calls into
this — single source of truth for both scalar `Load.index` reconstruction and the rank-N MMA source-dim fold M5
needs.

This is the **load-bearing correctness step** — an off-by-stride silently produces a slab striped wrong by
`atom_M` (16×); only end-to-end output-vs-reference comparison catches it.

**Files**: `ir/tile/ir.py` (~30 lines on `affine_decode_per_dim` + `AffineAddressing.source_index`),
`kernel/_stage_expand.py` (~20 lines — pass `addressing.block` through),
`tile/025_unify_sibling_stages.py` (~10 lines),
`tests/compiler/passes/test_stage_expand_blocked_strides.py` (new — golden expected `cooperative-load` index for a
shared-source-dim block case).

### M5 — `005_lower_atom_tile` walks into StageBundle

Replace the defensive Stage pass-through in `kernel/005_lower_atom_tile.py:75-80` with a recursive walk: descend
into `StageBundle.body`, find the K_o/K_i tower + AtomTile + reduce body as today, but now `a_load.input` is the
smem name and `a_load.index` is the rank-3 cache-coord index. Construct
`MmaLoad(src_buffer=a_load.input, src_index=source.addressing.source_index(cache_coord_for, origin=...), ldm=0)`.
`ldm` resolves at render via the existing `ctx.shapes[smem_name][-1]` path (`Smem.render` registers the slab
extents, which `M2.4` already grew by `block[i]`).

**Files**: `kernel/005_lower_atom_tile.py` (~80 lines),
`tests/compiler/passes/test_lower_atom_tile_smem.py` (new),
`tests/compiler/test_matmul_mma.py` (extend parametrize with `staged=[False, True]`).

### M6 — Flip the probe default ON

Delete the `DEPLODOCK_MMA_STAGE_PROBE` gate; just delete the `ATOM_KIND` skip block outright. The autotuner picks
staged vs direct via the existing `STAGE` knob enumeration (mask=0 keeps the gmem-direct path as a peer). Extend
`test_matmul_mma.py` to grep for `__shared__ a_smem` + `wmma::load_matrix_sync(a_frag, &a_smem[...]` in the
rendered kernel for the canonical pin.

**Files**: `tile/020_stage_inputs.py` (~10 lines), `config.py` (~5 lines).

### M7 — Skip `tile/070_pad_smem.py` for blocked Sources

`070_pad_smem`'s `+1` padding on the inner dim breaks WMMA's `ldmatrix` 16-byte alignment requirement. Skip the
bundle when *any* `Source.addressing` is `AffineAddressing` with non-empty `block`. The predicate is a one-liner
isinstance + tuple check; no atom-specific knowledge leaks into the pad pass.

**Files**: `tile/070_pad_smem.py` (~5 lines),
`tests/compiler/passes/test_pad_smem_mma_skip.py` (new).

### M8 — BUFFERED / ASYNC / TMA upgrade audit

Audit `040_use_ring_buffers.py`, `050_use_tma.py`, `060_use_async_copy.py`, `080_pipeline_stages.py` for any field
introspection on `Source` that the addressing-mode refactor could break. Each is structured around `StageBundle`
opaque to Source internals, but `050_use_tma`'s TMA descriptor box-shape derives from `alloc_extents` — verify the
`block`-grown extents produce valid TMA box dims (each ≥ 16 bytes, multiple of element size; for fp16 the 32 / 64-
element rows we emit fit).

When BUFFERED prepends the phase coord, `MmaLoad.src_index` must absorb it too. Since `AffineAddressing.source_index`
takes the `coord_for` dict, `040_use_ring_buffers` can either (a) inject the phase term into the dict before
calling `source_index`, or (b) leave `addressing` in pure cache-coord space and have `005_lower_atom_tile`
synthesize the phase prefix when the enclosing `StageBundle.buffer_count > 1`. **Pick (b)** — keeps addressing
canonical and isolates the phase math in one place.

**Files**: audit-only across the four passes (~0-15 lines of passthrough branches as needed);
`tests/compiler/test_matmul_mma_buffered.py` (new), `tests/compiler/test_matmul_mma_tma.py` (new, sm_90+ gated).

### M9 — Perf gate

`tests/perf/test_matmul_mma_staged_perf.py` (new, marker `perf`) — bench TinyLlama-shape matmul, staged vs direct,
both at `-O3`. Assert staged ≥ 1.5× direct on sm_80+. Skip on `compute_capability < (8, 0)`.

### M10 — Docs

Update module docstrings on `tile/020_stage_inputs.py`, `ir/tile/ir.py:Source` / `AffineAddressing`,
`kernel/005_lower_atom_tile.py`, and the kernel-pipeline section of `pipeline/ARCHITECTURE.md`. Add an MMA-staged
loop-nest example mirroring the existing scalar + warp-tier examples in `010_partition_loops.py`'s docstring. Note
the M2 refactor in `ir/ARCHITECTURE.md`'s `Source` field table — `template_index` row removed, `addressing` row
gains the `AffineAddressing.block` / `TemplateAddressing.exprs` breakdown.

## Reused machinery (no new code, just call sites)

- `Source.cache_dims` / `Source.addressing` (`ir/tile/ir.py:323-394`) — existing slab-descriptor structure; M2
  promotes `addressing` from derived property to stored field but the call-site shape (`isinstance(addr, ...)`)
  stays the same.
- `_stage_expand.emit_stage` (~`_stage_expand.py:34`) — cooperative producer scaffolding; M4 threads
  `addressing.block` through its stride computation. The composite-stride dict-overwrite that the original plan
  flagged is **already fixed** in `affine_decode_per_dim` (`ir/tile/ir.py:395-432`, docstring lines 416-422); M4
  only adds `block` to the same formula.
- `Smem.render` (`ir/kernel/ir.py:95-122`) — already registers `ctx.shapes[name] = extents`; `MmaLoad._resolve_ldm`
  (`ir/kernel/ir.py:738`) consumes this path unchanged.
- `_ATOM_REGISTRY` / `atom_spec` (`tile/_atom.py`) — atom shape and operand dtypes per kind; M3/M5 read
  `spec.shape` to validate the block tuple matches the atom geometry per cache dim.
- `Mma*` Stmt `rewrite.register` handlers (`ir/kernel/ir.py:1010-1060`) — already thread `rename` / `sigma`
  through `src_index`; M5's per-cell replication via `kernel/010_split_register_axes` keeps working unchanged.

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — M2 (addressing refactor), M4 (block-aware decode).
- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` — M1, M2 (construction), M3 (block stamp),
  M6.
- `deplodock/compiler/pipeline/passes/lowering/tile/025_unify_sibling_stages.py` — M2, M4.
- `deplodock/compiler/pipeline/passes/lowering/kernel/_stage_expand.py` — M4.
- `deplodock/compiler/pipeline/passes/lowering/kernel/005_lower_atom_tile.py` — M5.
- `deplodock/compiler/pipeline/passes/lowering/tile/070_pad_smem.py` — M7.
- `deplodock/config.py` — M1 + M6 (probe gate then removal).
- `tests/compiler/test_matmul_mma.py` (extend), `tests/perf/test_matmul_mma_staged_perf.py` (new).

## Verification

End-to-end correctness gate per milestone using `tests/compiler/test_matmul_mma.py` extended to parametrize
`staged ∈ {False, True}` across `(M, N, K) ∈ {(16,16,16), (64,64,64), (128,128,128), (256,256,64), (4096,4096,4096)}`.
The (256, 256, 64) skewed shape catches asymmetric N_b × WN × FN arithmetic; the 4096³ shape is the perf-gate
proxy. Each case compiles via `nvcc --cubin` (NVRTC lacks `crt/mma.h`), runs on cupy, compares max-abs-err vs the
f32 `numpy @ numpy` reference (tolerance ≤ 1e-2 for f16 operands · f32 accumulator at these K values).

M2 is its own gate: full test suite must be byte-clean. No snapshot diffs, no perf regressions on the scalar
matmul row of `make bench-kernels`. This proves the addressing-refactor is purely structural before M3 starts
making compiler-visible changes.

For M4's load-bearing fix, run the post-M4 test suite *without* M3's block stamping (revert M3 locally) — the
scalar matmul tests should still pass byte-identically (proves the `block` plumbing degenerates correctly when
every entry is the implicit 1).

Perf gate (M9): `make bench-kernels-tuned` should rank staged WMMA above gmem-direct WMMA on TinyLlama matmuls.
The autotune DB row picks should be staged-favoring; record the latency table in the follow-up PR.

## Out of scope

- **MMA + cooperative-K** (`BR > 1` + MMA) — `WarpTileParams` enforces `BR=1`; preserved here.
- **Pipelined-K MMA** beyond the natural cascade through `080_pipeline_stages` (no new scheduling math).
- **Skewed WMMA staging** (m8n32k16 / m32n8k16) — the per-cache-dim `block` tuple composes uniformly; no separate
  code path needed when those kinds enable in `_ATOM_KINDS_V1` (they're already registered post-M9 of the MMA
  plan).
- **NVFP4 / wgmma / tcgen05** — each future hardware kind brings its own tmem / async issue/wait infra.
- **MMA-aware bank-conflict padding** — M7 skips `070_pad_smem`. A future plan could add an MMA-friendly swizzle
  (e.g. XOR-style row shuffle matching `ldmatrix.x4`'s access pattern) once the baseline staged path lands.

## Failure modes to watch (load-bearing risks)

1. **M2 refactor leaks a behavior change.** The whole point of M2 is "byte-clean refactor." Mitigation: M2's
   verification step requires the snapshot tests to be unchanged before merging. Any IR-dump diff is a bug in the
   refactor, not in the plan.

2. **`block` stamped on the wrong cache dim (M3).** σ-coefficient extraction must match the *cache* var's σ
   output, not a neighboring var's. Mitigation: explicit structural assertion in `_classify` that the σ output is
   exactly `Var(ax) * Literal(c)` with no other terms before stamping; reject otherwise.

3. **`affine_decode_per_dim` block-stride composition order (M4).** Two cache axes sharing source_dim 0 — the
   stride for the outer one (`M_w`) is `inner_extent * inner_block`, not just `inner_extent`. Mitigation:
   unit-test the function with hand-computed expected strides for shared-source-dim cases on both the `block=()`
   and `block=(1, 16)` shapes.

4. **`AffineAddressing.source_index` rank-N fold direction (M5).** A-side: outer-M × inner-K (M is the leading
   dim, ldm = K-extent). B-side: outer-K × inner-N. Folding the wrong order produces a transposed fragment.
   Mitigation: golden-IR test asserts the rendered `wmma::load_matrix_sync` argument order matches a hand-written
   reference per side.

5. **`MmaFragment.layout` vs smem layout mismatch.** Today both A and B fragments are declared `row_major`.
   Staged A is row-major M×K (matches `wmma::matrix_a, row_major`); staged B is row-major K×N (matches
   `wmma::matrix_b, row_major`). If a future plan flips B to `col_major` for an attention-style kernel, the
   staged slab layout must follow. Add an assertion in M5 that `MmaFragment.layout == "row_major"` when staging
   is in play.

6. **Probe-gate removal in M6 racing test changes.** The probe-gated M1-M5 must keep `test_matmul_mma.py` passing
   with the probe off; M6 only flips the default once every staged-shape correctness test is green.
