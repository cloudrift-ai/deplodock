# Fix: TMA-swizzle shared-memory base alignment

## Context

deplodock's TMA path computes the `ldmatrix` swizzle XOR from a **buffer-relative** element offset
(`&a_smem[(o) ^ (((o >> 6) & mask) << 3)]`, see `_LDMATRIX_SWIZZLE_XOR` in `ir/kernel/ir.py`). The TMA hardware,
however, deposits the tile by swizzling the **absolute** smem address. The two agree only when the buffer base zeroes
the swizzle's source address bits — i.e. when the base is aligned to the full swizzle atom (8 rows × width): **1024 B
for B128, 512 B for B64, 256 B for B32**. Today swizzled buffers are emitted with only `__align__(128)`
(`_TMA_ALIGN_BYTES = 128`) and the dynamic pool with `__align__(16)`. Correctness currently rides on luck (the driver
1024-aligns dynamic smem; ptxas happens to 1024-align the static segment; and swizzled buffers, being whole 1024 B
atoms emitted first, never misalign each other).

We **reproduced** the bug as a clean numeric failure: taking the generated warp-specialized 2048³ fp16 matmul and
shifting `a_smem`/`b_smem` off the 1024 B atom corrupts the result (`max_diff≈3`) for any non-1024-multiple pad, while
1024-multiple pads stay bit-exact. The risk is latent today but becomes live the moment a fused/multi-buffer kernel
orders a non-1024-sized buffer before a swizzled operand, or the pool base isn't 1024-aligned. This change makes
correctness explicit instead of incidental, matching CUTLASS (swizzled smem aligned to the atom).

## Approach (full hardening)

Three coordinated changes plus a regression test.

### 1. Per-buffer atom alignment — `compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py`

`src.swizzle` (a `SwizzleMode`) is in scope at both `Smem(...)` construction sites. `SwizzleMode` and
`pick_swizzle_atom` are already imported; add `_SWIZZLE_BY_BYTES` to that import block
(`from deplodock.compiler.ir.tile.ir import (...)`).

- Add a helper next to `_TMA_ALIGN_BYTES = 128`:
  ```python
  def _swizzle_align_bytes(mode: SwizzleMode) -> int:
      """Swizzled TMA smem aligns to its full swizzle atom (8 rows × width):
      B128→1024, B64→512, B32→256 — the coordinate-only ldmatrix XOR only
      reproduces the hardware deposit when the base zeroes the swizzle's
      source address bits. NONE keeps NVIDIA's 128 B box recommendation."""
      for wb, m in _SWIZZLE_BY_BYTES:
          if m == mode:
              return 8 * wb
      return _TMA_ALIGN_BYTES
  ```
- Site 1 (~line 278): `smem_align = _swizzle_align_bytes(src.swizzle) if is_tma else (16 if smem_dtype == "__half" else 0)`
- Site 2, `emit_tma_stage` (~line 490): `align = _swizzle_align_bytes(src.swizzle)` and rewrite the now-stale comment
  (lines ~486-489) that claims every TMA stage runs at the 128 B base recommendation.

Non-swizzled TMA (`NONE`) still yields 128, and non-TMA still yields 16/0 — no behavior change off the swizzle path.

### 2. Single padding-aware smem total — `compiler/ir/kernel/ir.py`

Today `KernelOp.smem_bytes()` returns the **unpadded sum** of buffer sizes, while the renderer's pool packer pads each
buffer to its align. With larger aligns these can diverge → the launch under-allocates the dynamic pool (the hang seen
during investigation). Make one source of truth:

- Add module-level `pack_smem(smems) -> tuple[dict[str, int], int]` that walks buffers in order, aligns each cursor to
  `max(nbytes_of(dtype), s.align)`, and returns `(name→offset, total_bytes)` (mirrors the existing loop in
  `_compute_dynamic_smem_offsets`; reuse `nbytes_of` via the same local import already used by `smem_bytes`).
- `KernelOp.smem_bytes()` returns `pack_smem(self.smem_buffers.values())[1]`.

### 3. Pool base alignment + consistent gate — `compiler/ir/kernel/render.py`

- `_compute_dynamic_smem_offsets`: compute offsets/total via `pack_smem` (same packer as `smem_bytes`), so the
  static-vs-dynamic gate (`total <= STATIC_SMEM_CAP`) and the launch size agree.
- Pool decl (~line 289): replace `__align__(16)` with `__align__({pool_align})` where
  `pool_align = max(16, *(max(nbytes_of(s.dtype), s.align) for s in smems))` — i.e. ≥1024 whenever a swizzled buffer is
  present. Update the adjacent comment (the "`__align__(16)` satisfies TMA's 16-byte requirement" note).

### 4. Regression test — `tests/compiler/test_matmul_mma_tma.py`

Reuse the file's `_compile_and_render(M, N, K, out_dtype)` helper and `requires_cuda` / `_supports_tma()` markers.

- **Codegen assertion (fast):** compile the TMA swizzle matmul (e.g. `M=256, N=256, K=128`, the shape the existing TMA
  test uses). Assert the swizzled operand `Smem`s carry atom alignment — A→B64 (`align == 512`), B→B128
  (`align == 1024`) — by iterating `kop.smem_buffers` / `kop.body`, and that the rendered source contains
  `__align__(1024)` and `__align__(512)` (static path) or an `__align__(>=1024)` pool decl (dynamic path).
- **Numeric:** the existing `test_tma_mma_matches_f32_reference` already launches this path vs an f32 reference and must
  stay green (proves the new alignment doesn't regress correctness). No new GPU test needed.

### 5. Cleanup + docs

- Remove the scratch reproducer `scripts/swizzle_repro.py`. (`kernel.cu` in the repo root is git-untracked scratch —
  leave it alone, do not commit it.)
- Per repo contribution rules, update `ARCHITECTURE.md` in each modified dir if it documents smem alignment: check
  `compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md` and `compiler/ir/kernel/ARCHITECTURE.md` (and
  `compiler/ARCHITECTURE.md`) — add a line that swizzled TMA smem aligns to `8 × swizzle_width`.

## Critical files

- `compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — atom-align helper + two Smem sites
- `compiler/ir/kernel/ir.py` — `pack_smem` + padding-aware `KernelOp.smem_bytes()`
- `compiler/ir/kernel/render.py` — pool packer via `pack_smem` + `__align__(pool_align)` decl
- `tests/compiler/test_matmul_mma_tma.py` — new alignment assertion test

## Reused, not rewritten

- `_SWIZZLE_BY_BYTES` / `SwizzleMode` (`ir/tile/ir.py`) — mode→width source of truth
- `nbytes_of` (`backend/cuda/dtype.py`) — natural dtype size, already used by `smem_bytes`
- `_compile_and_render`, `requires_cuda`, `_supports_tma` (`tests/compiler/test_matmul_mma_tma.py`)

## Verification

1. `./venv/bin/pytest tests/compiler/test_matmul_mma_tma.py -v` — new alignment test passes; the existing
   `test_tma_mma_matches_f32_reference` (numeric) stays green.
2. Inspect codegen: `M="torch.randn(2048,2048,dtype=torch.float16,device='cuda')`;
   `DEPLODOCK_KNOBS="TMA=1,ATOM_KIND=mma_m16n8k16_f16,WM=1,WN=4,FM=4,FN=2,BK=2,BUFFER_COUNT=2,WARP_SPECIALIZE=1,SPLITK=1"
   deplodock compile --code "a=$M;b=$M;torch.matmul(a,b)" --ir cuda` → `a_smem`/`b_smem` now show `__align__(512)` /
   `__align__(1024)`.
3. End-to-end accuracy unchanged: run both blog configs (the WS one above and the no-WS
   `WM=2,WN=4,FM=4,FN=4,BUFFER_COUNT=3`) via `deplodock run --code ... -v` → `max_diff=0.000000 PASS`, and a larger
   dynamic-pool matmul to exercise the bumped pool decl.
4. `make lint` (run `make format` if it flags) and finally `make test`.
