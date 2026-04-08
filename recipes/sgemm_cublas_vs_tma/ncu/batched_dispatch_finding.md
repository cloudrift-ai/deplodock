# Batched-mode dispatch: 5090 vs Pro 6000 ncu comparison

The article notes that our TMA kernel beats cuBLAS by ~60% in batched mode on
the 5090 but only matches cuBLAS on Pro 6000. Both GPUs are sm_120 Blackwell
with the same TMA hardware. This file documents the actual cause: a
**cuBLAS dispatcher heuristic difference**, not a hardware difference.

## ncu of `cublasSgemmStridedBatched(N=4096, batch=8)` on both GPUs

Both runs use CUDA 13.2.51, cuBLAS 13.3.0, driver 595.58.03, our bench harness.

| Metric                | 5090 (170 SMs) — cuBLAS | Pro 6000 Max-Q (188 SMs) — cuBLAS |
|-----------------------|-------------------------|-----------------------------------|
| Dispatched kernel     | `cutlass_80_simt_sgemm_**128x32_8x5**_nn_align1` (small) | `cutlass_80_simt_sgemm_**256x128_8x4**_nn_align1` (large) |
| Threadblock tile      | 128 × 32                | 256 × 128                         |
| Block dim             | (128, 1)                | (256, 1)                          |
| Cycles active (M)     | 60.8                    | 31.3                              |
| IPC                   | 2.30                    | 3.26                              |
| FMA pipe utilization  | **41.6%**               | **73.5%**                         |
| Registers / thread    | ~210                    | 210                               |

Same toolkit, same compute capability, same source CUTLASS template — but the
runtime dispatcher inside `libcublas.so` picks a **completely different size**
for batched workloads on these two GPUs.

## ncu of our `fused_matmul` (TMA) at the same workload

| Metric                | 5090 (TM=28)            | Pro 6000 (TM=24/26)               |
|-----------------------|-------------------------|-----------------------------------|
| Cycles active (M)     | 38.0                    | 34.3                              |
| IPC                   | 2.95                    | 2.99                              |
| FMA pipe utilization  | 67.5%                   | 68.7%                             |
| Registers / thread    | 241                     | 213                               |

Our kernel is essentially identical between the two GPUs (same template, same
TMA double-buffer, just a slightly different `thread_m`). FMA pipe utilization
is the same ~68% on both.

## What this means

The 5090 batched-mode "win" is **not** a TMA win in isolation — it's:

> (TMA being good) × (cuBLAS picking a sub-optimal small kernel for 5090 batched)

On the 5090, cuBLAS's batched-dispatch heuristic switches to the small
`128x32_8x5` kernel which only achieves 41% FMA pipe utilization. Our TMA
kernel hits 67%, so the ratio is `67% / 41% ≈ 1.6×` — matching the observed
~158% efficiency.

On the Pro 6000, cuBLAS's heuristic keeps the large `256x128_8x4` kernel for
batched, which achieves 73% FMA pipe utilization. Our TMA kernel still hits
~68%, so the ratio is `68% / 73% ≈ 0.93` — matching the observed ~93%
efficiency.

The ~5pp gap between our 68% and cuBLAS's 73% is the same per-instruction
SASS scheduling gap we documented for 5090 single mode in the main article —
ptxas can't extract the same instruction density from generated C source as
NVIDIA's hand-tuned PTX templates do. This gap is not specific to Pro 6000
batched and isn't closable from the C generator side.

## TM sweep on Pro 6000 (4096×4096 batch=8)

Reality check: maybe a different `thread_m` value happens to schedule better
on the Pro 6000's 188-SM topology. Sweep at 4096×4096 batch=8:

| TM | Eff vs cuBLAS |
|----|---------------|
| 16 | 91.0% |
| 18 | 91.0% |
| 20 | 92.2% |
| 22 | 90.4% |
| 24 | 92.8% (was the default) |
| 26 | **93.5%** (new default) |
| 28 | 92.1% |

And at 8192×8192 batch=8:

| TM | Eff vs cuBLAS |
|----|---------------|
| 20 | 92.6% |
| 22 | 91.9% |
| 24 | 94.9% |
| 26 | **95.1%** |
| 28 | 94.9% |

TM=26 wins by 0.2–0.7pp over the previous TM=24 default. The surface is
extremely flat from TM=24 to TM=28 — there is no architectural lever that
would close the gap to cuBLAS in any meaningful way. The Pro 6000 ceiling
in batched mode is ~95% and structural.

## Could we make cuBLAS dispatch the small kernel on Pro 6000?

No public knob. cuBLAS's algorithm-selection heuristics are not exposed
through `cublasSetMathMode` or any other documented API. `cuBLASLt` has
algorithm querying (`cublasLtMatmulAlgoGetIds`) which lets you list and try
specific kernels, and some FAST_TF32 / FAST_16BF compute modes pick
different paths — but for plain `cublasSgemmStridedBatched` with FP32 inputs
and FP32 accumulator, the kernel is whatever `libcublas.so` decides at the
moment of the first call, with no override.

This finding is also useful in the opposite direction: if you wanted to
improve **cuBLAS itself**, you'd note that on the 5090 it's leaving ~60%
performance on the floor by picking a 41%-FMA-pipe kernel for batched
workloads where the 73% kernel would clearly be better. That's a real bug
in NVIDIA's dispatcher heuristic, surfaced by this comparison.
