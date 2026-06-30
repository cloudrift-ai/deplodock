# TODO: Matmul Tuning & Strategy Integration

## Completed this session

- [x] Contraction + epilogue fusion (bias add, activation, residual)
- [x] M-aware k_splits and thread_m clamping
- [x] >2D fusion for non-contraction patterns (softmax on 4D)
- [x] Noop removal → buffer aliases
- [x] Codegen/IR/Program layering (moved to backend/)
- [x] Pretty printer for program (40_program.txt in dumps)
- [x] `--dump-dir` flag on `emmy bench`
- [x] `smem` strategy: SIMT shared-memory kernel with splitK support
- [x] Bench experiment with rectangular shapes and batch sizes

## Next: Wire smem into compiler pipeline

The `smem` strategy works and is correct (488 tests pass) but is not yet
auto-selected by the compiler. Currently only usable via explicit hints
or bench_matmul.py `--strategy smem`.

### 1. Auto-select smem for small M in `_compile_matmul`

In `backend.py:_compile_matmul()`, after tuning profile selection:

```python
# When M is small and TMA overhead dominates, use smem strategy.
if m <= 64 and strategy == "tma_db":
    strategy = "smem"
    matmul_hints["strategy"] = "smem"
```

The smem grid uses `(ntx, nty, k_splits)` — already handled by
`strategy == "smem"` branch in the grid computation.

### 2. Tune smem k_splits per shape

From standalone benchmarks (M=32, N=2048, K=2048):
- k_splits=16 → 19us (best, 58% cuBLAS)
- k_splits=8 → 22us
- k_splits=1 → 155us

Optimal k_splits depends on grid_blocks vs SM count:
```
desired_ks = min(SM_count / grid_blocks, K / BK, 16)
```

### 3. Smem epilogue support

The smem kernel currently doesn't support contraction epilogue
(bias add, activation). The write section needs epilogue codegen
similar to the TMA/naive paths in `_contraction_epilogue_code()`.
For now, smem + epilogue falls back to TMA.

### 4. Fix bench_matmul naive strategy

The naive strategy fails in `generate_benchmark_program` because M/N/K
are `#define`d but the kernel has `int M, int N, int K` params (same
issue fixed for smem via `_dim_decls()`). The fix already applies to
smem/naive via `kernel.tma_params` check.

### 5. cuBLAS kernel analysis (reference)

For M=32, cuBLAS dispatches `cutlass_80_simt_sgemm_128x32_8x5` +
`splitKreduce_kernel`. Key design choices:
- 128×32 CTA tile (tall N, narrow M)
- 8×5 = 40 thread block (not 128 or 256)
- Separate splitK reduce kernel (not atomicAdd)
- SIMT FMA only (no TMA on sm_120 for small M)

A dedicated splitK reduce kernel would avoid atomicAdd contention
and potentially match cuBLAS. Low priority since we're already at
58% with atomicAdd.

### 6. Performance summary (RTX 5090, batch=1)

| Shape | Our TMA | Our smem(ks=16) | cuBLAS | Eff |
|-------|---------|-----------------|--------|-----|
| 32×2048 | 79us | 19us | 11us | 58% |
| 32×3584 | 137us | 40us(ks=8) | 24us | 60% |
| 1024sq | 44us | 70us | 44us | 100% (TMA) |
| 4096sq | 2.3ms | 0.6ms | 2.0ms | 313% (smem!) |

Strategy selection: smem for M≤64, TMA for M≥128 or batch>1.
