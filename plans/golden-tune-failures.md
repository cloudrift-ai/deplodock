# Golden-tune compile/bench failures — repro log

Failures hit while tuning the golden configs (`deplodock tune --golden <name> --clean`). Each entry: symptom, exact
error, hypothesis, and a repro. These are *kernel-variant* failures the inner search pins as `bench_fail @ 2e6 us` and
walks past — so the tune still completes, but the failing variants are silently excluded from the search.

## 1. Unused `cp_async_bulk_tensor_5d` TMA helper → NVRTC warning promoted to error

**Symptom (from `deplodock tune --golden square.512 --clean`):**

```
compile failed for kernel 'k_matmul_207791':
k/cubin/tmpc8n3guqs/k.cu(79): warning #177-D: function "cp_async_bulk_tensor_5d" was declared but never referenced
  static __attribute__((device)) __inline__ __attribute__((always_inline)) void cp_async_bulk_tensor_5d(
                                                                                ^
1 error detected in the compilation of "/home/.../cubin/tmpc8n3guqs/k.cu".
)) — pinning bench_fail @ 2000000.0 us for 1 kernel(s)
```

**Hypothesis:** the CUDA codegen emits the `cp_async_bulk_tensor_5d` TMA bulk-tensor helper *declaration* into a kernel
that never calls it (a thread-tier fp32 matmul variant on `square.512` — it has no business using a 5-D TMA bulk copy),
and the NVRTC/nvcc invocation treats the resulting `#177-D` "declared but never referenced" **warning as an error**
("1 error detected" with only a warning shown ⇒ a `-Werror`-class flag is promoting it). Two independent bugs:

1. **Codegen**: the TMA helper preamble is emitted unconditionally (or over-broadly) rather than only when a TMA bulk
   op is actually present in the kernel body. The fix is to emit `cp_async_bulk_tensor_*` helpers only when referenced.
2. **Flags**: a warning is fatal. Find the `-Werror`/`--Werror` (or NVRTC `default` promotion) in the cubin-compile
   flags (`DEPLODOCK_NVCC_FLAGS` / `compiler/backend/cuda/nvcc.py`) and either drop it or suppress `#177-D` for the
   generated preamble.

**Repro:**

```bash
DEPLODOCK_DUMP_DIR=/tmp/golden_fail deplodock tune --golden square.512 --clean
# inspect the emitted source of the failing variant:
grep -rl cp_async_bulk_tensor_5d /tmp/golden_fail
```

To pin the exact failing variant for a minimal repro, capture its knobs from the `[tune]` line and re-run
`DEPLODOCK_KNOBS="…" deplodock compile -c "torch.matmul(torch.randn(512,512), torch.randn(512,512))" --ir cuda`.

**Impact:** ranking only — the variant is excluded (pinned `bench_fail`), so the tune still finds a (possibly
sub-optimal) best among the variants that *do* compile. Worth fixing so the search isn't silently blind to a slice of
the space, and so the unused-helper emission doesn't mask a real TMA codegen path.
