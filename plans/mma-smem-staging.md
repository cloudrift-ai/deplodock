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
fall back to `template_index`. But `Source.alloc_extents` (`ir/tile/ir.py:372-377`) returns naked cache-axis extents
*without the atom multiplier*: for cache_dims `(M_w, M_r, K_i)` with extents `(WM, FM, bk)`, the slab is allocated as
`(WM, FM, bk)` instead of the correct `(WM, FM·atom_M, bk·atom_K)`. The slab is missing a 256× area factor for
fp16 m16n16k16, and the producer cooperative-load under-fills the slab — WMMA reads garbage from the 255/256 that
nobody wrote.

The fix is to recognize the σ literal coefficient as a structural per-source-dim multiplier (`atom_factor`) that the
slab absorbs, while keeping affine addressing.

## Recommended approach

**Extend `Source` with `atom_factor: tuple[int, ...] = ()`** — a per-cache-dim multiplier that grows the slab and
the producer-side iteration range without changing the cache-var identities. `020_stage_inputs._classify` recognizes
the σ literal coefficient as the atom factor (keeping affine addressing, not falling back to template). The MMA cell
materializer `kernel/005_lower_atom_tile.py` walks into the `StageBundle` body, finds the per-warp scalar Loads
already rewritten to read from smem, and emits `MmaLoad` reading from the smem slab with a folded rank-2
warp-relative tile-base offset (computed structurally from cache vars × atom_factor strides).

A pre-existing latent bug surfaces under MMA: `_stage_expand.emit_stage` (~`_stage_expand.py:88`) uses a
`{source_dim: decoded_var}` dict that **silently drops** one of two cache axes sharing a source_dim (M_w + M_r both
on dim 0). Scalar matmul doesn't hit it because its cache axes always map to distinct source dims. MMA's per-tier
splits put M_w *and* M_r on source dim 0 of A — fixing the dict overwrite into a proper per-source-dim summation is
part of M3.

## Milestones

### M1 — Probe gate + plumbing smoke

Remove the `ATOM_KIND` skip in `020_stage_inputs.py:122` behind a `DEPLODOCK_MMA_STAGE_PROBE` env gate (default OFF
so M5 below can flip the default ON only once all the staged-WMMA pieces are green). Add the config accessor in
`config.py` mirroring `mma_enabled()`. **Verify**: a new test asserts the post-020 body for an MMA matmul contains a
`StageBundle` with A + B `Source`s when the probe is on; existing `test_matmul_mma.py` is byte-clean (probe off).

**Files**: `tile/020_stage_inputs.py` (~10 lines), `config.py` (~5 lines),
`tests/compiler/passes/test_stage_inputs_mma_probe.py` (new).

### M2 — `Source.atom_factor` field + `_classify` recognition

Add `atom_factor: tuple[int, ...] = ()` to `Source` (default `()` = no multiplier; scalar path unchanged). Update
`Source.alloc_extents` (~`ir/tile/ir.py:372`) to multiply by `atom_factor[i]` when set. In
`020_stage_inputs._classify`: when `unit_sigma.reduce(load.index[d], ctx)` simplifies to
`Literal(c) * Var(ax.name)` with `c > 1`, record `c` as the cache-dim's atom_factor and keep affine addressing.
**Mitigation for off-by-one risk**: assert the σ output has *exactly* `Var(ax) * Literal(c)` (no other terms on this
cache axis) before stamping atom_factor; reject ambiguous cases and fall back to template.

**Files**: `ir/tile/ir.py` (~35 lines), `tile/020_stage_inputs.py` (~50 lines),
`tests/compiler/ir/tile/test_source_atom_factor.py` (new).

### M3 — `_stage_expand.emit_stage` producer-side reconstruction

Fix the `decoded_per_dim` dict-overwrite (`_stage_expand.py:88`): switch to per-source-dim summation
`per_dim_offsets[dim] += decoded[ax.name] * stride_for_axis(ax)` where `stride_for_axis` walks the cache_dims order
and applies `atom_factor` to the inner stride. This is the **load-bearing correctness step** — an off-by-stride
silently produces a slab striped wrong by atom_M (16×); only end-to-end output-vs-reference comparison catches it.

**Files**: `kernel/_stage_expand.py` (~50 lines),
`tests/compiler/passes/test_stage_expand_atom_factor.py` (new).

### M4 — `005_lower_atom_tile` walks into StageBundle

Add `mma_view: tuple[Expr, ...] | None = None` to `Source` (parallels `template_index`). `020_stage_inputs` computes
this for atom-factor-bearing Sources as the rank-2 affine fold of cache vars × atom_factor strides (e.g. A-side:
`(m_w * (FM*atom_M) + m_r, k_i)` in cache coords). `kernel/005_lower_atom_tile.py` replaces its defensive Stage
pass-through with a recursive walk: descend into `StageBundle.body`, find the K_o/K_i tower + AtomTile + reduce body
as today, but now `a_load.input` is the smem name and `a_load.index` is the rank-3 cache-coord index. Construct
`MmaLoad(src_buffer=a_load.input, src_index=sigma(source.mma_view, with_cache_vars→a_load.index), ldm=0)`. `ldm`
resolves at render via the existing `ctx.shapes[smem_name][-1]` path (`Smem.render` registers the slab extents).

**Files**: `ir/tile/ir.py` (~10 lines), `tile/020_stage_inputs.py` (~30 lines),
`kernel/005_lower_atom_tile.py` (~80 lines), `tests/compiler/passes/test_lower_atom_tile_smem.py` (new),
`tests/compiler/test_matmul_mma.py` (extend parametrize with `staged=[False, True]`).

### M5 — Flip the probe default ON

Delete the `DEPLODOCK_MMA_STAGE_PROBE` gate; just delete the `ATOM_KIND` skip block outright. The autotuner picks
staged vs direct via the existing `STAGE` knob enumeration (mask=0 keeps the gmem-direct path as a peer). Extend
`test_matmul_mma.py` to grep for `__shared__ a_smem` + `wmma::load_matrix_sync(a_frag, &a_smem[...]` in the rendered
kernel for the canonical pin.

**Files**: `tile/020_stage_inputs.py` (~10 lines), `config.py` (~5 lines).

### M6 — Skip `tile/070_pad_smem.py` for MMA-staged Sources

`070_pad_smem`'s `+1` padding on the inner dim breaks WMMA's `ldmatrix` 16-byte alignment requirement. Skip the
bundle when *any* Source has nonempty `atom_factor` or `mma_view` (catches mixed-bundle corner cases too).

**Files**: `tile/070_pad_smem.py` (~5 lines),
`tests/compiler/passes/test_pad_smem_mma_skip.py` (new).

### M7 — BUFFERED / ASYNC / TMA upgrade audit

Audit `040_use_ring_buffers.py`, `050_use_tma.py`, `060_use_async_copy.py`, `080_pipeline_stages.py` for any field
introspection on `Source` that the new `atom_factor` / `mma_view` fields could break. Each is structured around
`StageBundle` opaque to Source internals, but `050_use_tma`'s TMA descriptor box-shape derives from `alloc_extents`
— verify the atom_factor-grown extents produce valid TMA box dims (each ≥ 16 bytes, multiple of element size; for
fp16 the 32 / 64-element rows we emit fit).

When BUFFERED prepends the phase coord, `MmaLoad.src_index` must absorb it too: `040_use_ring_buffers` should
update `Source.mma_view` to prepend the same phase term it prepends to scalar Load indices, OR `005` must read
phase from the `StageBundle.phase` field directly. **Pick the latter** — keep `mma_view` in pure cache-coord space,
have `005` synthesize the phase prefix when the enclosing `StageBundle.buffer_count > 1`.

**Files**: audit-only across the four passes (~0-15 lines of passthrough branches as needed);
`tests/compiler/test_matmul_mma_buffered.py` (new), `tests/compiler/test_matmul_mma_tma.py` (new, sm_90+ gated).

### M8 — Perf gate

`tests/perf/test_matmul_mma_staged_perf.py` (new, marker `perf`) — bench TinyLlama-shape matmul, staged vs direct,
both at `-O3`. Assert staged ≥ 1.5× direct on sm_80+. Skip on `compute_capability < (8, 0)`.

### M9 — Docs

Update module docstrings on `tile/020_stage_inputs.py`, `ir/tile/ir.py:Source`,
`kernel/005_lower_atom_tile.py`, and the kernel-pipeline section of `pipeline/ARCHITECTURE.md`. Add an MMA-staged
loop-nest example mirroring the existing scalar + warp-tier examples in `010_partition_loops.py`'s docstring.

## Reused machinery (no new code, just call sites)

- `Source.cache_dims` / `Source.addressing` (`ir/tile/ir.py:323-394`) — existing slab-descriptor structure.
- `_stage_expand.emit_stage` (~`_stage_expand.py:34`) — cooperative producer scaffolding; M3 fixes the dict-overwrite
  inside it.
- `Smem.render` (`ir/kernel/ir.py:95-122`) — already registers `ctx.shapes[name] = extents`; `MmaLoad._resolve_ldm`
  (`ir/kernel/ir.py:738`) consumes this path unchanged.
- `_ATOM_REGISTRY` / `atom_spec` (`tile/_atom.py`) — atom shape and operand dtypes per kind; M2/M4 read
  `spec.shape` to validate atom_factor matches.
- `Mma*` Stmt `rewrite.register` handlers (`ir/kernel/ir.py:1010-1060`) — already thread `rename` / `sigma` through
  `src_index`; M4's per-cell replication via `kernel/010_split_register_axes` keeps working unchanged.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` — M1, M2, M4.
- `deplodock/compiler/ir/tile/ir.py` (`Source` class) — M2, M4.
- `deplodock/compiler/pipeline/passes/lowering/kernel/_stage_expand.py` — M3.
- `deplodock/compiler/pipeline/passes/lowering/kernel/005_lower_atom_tile.py` — M4.
- `deplodock/compiler/pipeline/passes/lowering/tile/070_pad_smem.py` — M6.
- `deplodock/config.py` — M1 + M5 (probe gate then removal).
- `tests/compiler/test_matmul_mma.py` (extend), `tests/perf/test_matmul_mma_staged_perf.py` (new).

## Verification

End-to-end correctness gate per milestone using `tests/compiler/test_matmul_mma.py` extended to parametrize
`staged ∈ {False, True}` across `(M, N, K) ∈ {(16,16,16), (64,64,64), (128,128,128), (256,256,64), (4096,4096,4096)}`.
The (256, 256, 64) skewed shape catches asymmetric N_b × WN × FN arithmetic; the 4096³ shape is the perf-gate
proxy. Each case compiles via `nvcc --cubin` (NVRTC lacks `crt/mma.h`), runs on cupy, compares max-abs-err vs the
f32 `numpy @ numpy` reference (tolerance ≤ 1e-2 for f16 operands · f32 accumulator at these K values).

For M3's load-bearing fix, run the post-M3 test suite *without* M2's atom_factor stamping (revert M2 locally) — the
scalar matmul tests should still pass byte-identically (proves the dict-overwrite fix doesn't regress the
single-axis-per-source-dim case).

Perf gate (M8): `make bench-kernels-tuned` should rank staged WMMA above gmem-direct WMMA on TinyLlama matmuls. The
autotune DB row picks should be staged-favoring; record the latency table in the follow-up PR.

## Out of scope

- **MMA + cooperative-K** (`BR > 1` + MMA) — `WarpTileParams` enforces `BR=1`; preserved here.
- **Pipelined-K MMA** beyond the natural cascade through `080_pipeline_stages` (no new scheduling math).
- **Skewed WMMA staging** (m8n32k16 / m32n8k16) — the `atom_factor`-per-cache-dim design composes uniformly; no
  separate code path needed when those kinds enable in `_ATOM_KINDS_V1` (they're already registered post-M9 of the
  MMA plan).
- **NVFP4 / wgmma / tcgen05** — each future hardware kind brings its own tmem / async issue/wait infra.
- **MMA-aware bank-conflict padding** — M6 skips `070_pad_smem`. A future plan could add an MMA-friendly swizzle
  (e.g. XOR-style row shuffle matching `ldmatrix.x4`'s access pattern) once the baseline staged path lands.

## Failure modes to watch (load-bearing risks)

1. **`atom_factor` stamped on the wrong cache dim (M2).** σ-coefficient extraction must match the *cache* var's
   σ output, not a neighboring var's. Mitigation: explicit structural assertion in `_classify` that the σ output
   is exactly `Var(ax) * Literal(c)` with no other terms before stamping; reject otherwise.

2. **`_stage_expand.emit_stage` per-source-dim summation order (M3).** Two cache axes sharing source_dim 0 — the
   stride for the outer one (`M_w`) is `inner_extent * inner_atom_factor`, not just `inner_extent`. Mitigation:
   helper `_source_dim_stride(src, cache_axis_idx)` with a unit test enumerating shared-source-dim cases.

3. **`mma_view` rank-2 fold direction (M4).** A-side: outer-M × inner-K (M is the leading dim, ldm = K-extent).
   B-side: outer-K × inner-N. Folding the wrong order produces a transposed fragment. Mitigation: golden-IR test
   asserts the rendered `wmma::load_matrix_sync` argument order matches a hand-written reference per side.

4. **`MmaFragment.layout` vs smem layout mismatch.** Today both A and B fragments are declared `row_major`. Staged
   A is row-major M×K (matches `wmma::matrix_a, row_major`); staged B is row-major K×N (matches
   `wmma::matrix_b, row_major`). If a future plan flips B to `col_major` for an attention-style kernel, the staged
   slab layout must follow. Add an assertion in M4 that `MmaFragment.layout == "row_major"` when staging is in play.

5. **Probe-gate removal in M5 racing test changes.** The probe-gated M1-M4 must keep `test_matmul_mma.py` passing
   with the probe off; M5 only flips the default once every staged-shape correctness test is green.
