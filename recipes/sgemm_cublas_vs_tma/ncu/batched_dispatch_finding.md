# cuBLAS batched FP32 SGEMM dispatcher: full per-arch / per-size sweep

The article notes that our TMA kernel beats cuBLAS by ~60% in batched mode on
the 5090 but only matches cuBLAS on Pro 6000. Both GPUs are sm_120 Blackwell
with the same TMA hardware. After investigating, the cause turned out to be
**three separate bugs in cuBLAS's batched FP32 SGEMM dispatcher**, all
visible from a single full-sweep `ncu` matrix.

All measurements: CUDA 13.2.51, cuBLAS 13.3.0, driver 595.58.03, batch=8,
square matmul, captured by `scripts/diagnostics/ncu_compare.sh`.

## The full dispatch table

| Size | RTX 5090 cuBLAS kernel              | FMA% | RTX PRO 6000 Max-Q cuBLAS kernel       | FMA% |
|------|-------------------------------------|-----:|----------------------------------------|-----:|
| 256  | `cutlass_80_simt_sgemm_128x32_8x5`  |  33% | `magma_sgemmEx_kernel<float, ...>`     |  32% |
| 512  | `cutlass_80_simt_sgemm_128x32_8x5`  |  39% | `magma_sgemmEx_kernel<float, ...>`     |  32% |
| 1024 | `cutlass_80_simt_sgemm_128x32_8x5`  |  41% | `cutlass_80_simt_sgemm_128x64_8x5`     |  64% |
| 2048 | `cutlass_80_simt_sgemm_128x32_8x5`  |  41% | `cutlass_80_simt_sgemm_128x128_8x4`    |  69% |
| 4096 | `cutlass_80_simt_sgemm_128x32_8x5`  |  42% | `cutlass_80_simt_sgemm_256x128_8x4`    |  73% |
| 8192 | `cutlass_80_simt_sgemm_128x32_8x5`  |  42% | `cutlass_80_simt_sgemm_256x128_8x4`    |  73% |

Both GPUs are the same `compute_cap = 12.0`, same toolkit, same CUTLASS source
templates, same library binary. The only thing that differs is the runtime
heuristic inside `libcublas.so` that maps `(M, N, K, batch_count, GPU)` to a
kernel ID. **That heuristic is wrong in three distinct ways:**

### Bug 1: The 5090 dispatcher never escalates the batched kernel

From 256×256 (sub-microsecond per call) all the way to 8192×8192 (~50 ms per
call), the 5090 dispatcher picks the same tiny `128x32` cutlass tile and
stays there. Even at 8192×8192×8 batches — a workload that easily justifies
a 256×128 tile and that the *same library shipping the same templates*
escalates correctly on the Pro 6000 — the 5090 path stays on the small
kernel forever. FMA pipe utilization is stuck at ~40% across the entire
batched-mode operating range.

Concretely: at 4096×4096 b=8, the 5090 dispatched kernel finishes in
60.8 M cycles. The same cuBLAS library on Pro 6000, against a workload
of identical shape, finishes in 31.3 M cycles. **The library has a kernel
that runs at 1.94× the throughput. The 5090 dispatcher just doesn't pick it.**

### Bug 2: The Pro 6000 dispatcher falls into a MAGMA code path at small sizes

At 256 and 512, Pro 6000 dispatches `magma_sgemmEx_kernel<...>` at 32% FMA
pipe utilization — even though the same library has `simt_sgemm_128x64_8x5`
that hits 64% on the very next size up (1024). This isn't "the threshold is
wrong"; it's a *different kernel family* being selected. MAGMA is the
open-source linear algebra library NVIDIA forked into cuBLAS years ago,
and parts of it still exist in the dispatcher. Apparently those parts get
preferred over cutlass at very small sizes on sm_120 — for no good reason.

### Bug 3: Both dispatchers cap at ~73% FMA pipe util even when picking the right kernel

When the dispatcher *does* pick the large `256x128_8x4` kernel (Pro 6000
at 4K+), the FMA pipe utilization tops out at 73%. The same cap shows up
on the H200 (`sm80_xmma_gemm` at 79%) and on the 5090 single-mode kernel
(`simt_sgemm_256x128_8x4` at 73%). Our generated TMA kernel hits 68% on
all three architectures. The remaining 5pp gap is the irreducible
"generated C source vs hand-tuned PTX" cost — ptxas can't extract the
same instruction-level density from C as it can from hand-rolled PTX
templates.

This gap is real but it's a *constant* across architectures and workloads.
It's not specific to the buggy dispatch paths.

## Where our advantage actually comes from

Putting all of this together, our kernel's "wins" against cuBLAS are
quantitatively explained by a combination of (a) the broken dispatcher
bugs above and (b) the constant 5pp generator-vs-hand-tuned gap. Nothing
else.

| Workload                      | cuBLAS FMA% | Ours FMA% | Ratio | Observed eff |
|-------------------------------|------------:|----------:|------:|-------------:|
| 5090 batched 4K b=8           | 42%         | 67%       | 1.60× | 158% ✓       |
| 5090 batched 8K b=8           | 42%         | 68%       | 1.62× | 159% ✓       |
| Pro 6000 batched 256 b=8      | 32%         | 54%       | 1.69× | (small-size variance dominates) |
| Pro 6000 batched 4K b=8       | 73%         | 68%       | 0.93× | 93% ✓        |
| Pro 6000 batched 8K b=8       | 73%         | 68%       | 0.93× | 95% ✓        |
| 5090 single 8K                | 73%         | 68%       | 0.93× | 95% ✓        |
| H200 single 8K                | 79%         | 71%       | 0.90× | 92% ✓        |
| Pro 6000 single 4K            | 73%         | 69%       | 0.94% | 93% ✓        |

**Our TMA kernel is consistently a 5-pp generator gap below the best cuBLAS
kernel for the workload.** When cuBLAS dispatches the right kernel, we lose
by 5-10%. When cuBLAS dispatches the wrong kernel — which it does in batched
mode on the 5090 across the entire size range, and on Pro 6000 at small
sizes — we win by 30-60%. The win is structural to the dispatcher bugs,
not to TMA giving us novel compute capacity.

## What TMA actually buys us

This re-framing isn't a knock against TMA. The TMA double-buffer template:

1. Achieves **~68% FMA pipe utilization** with ~80 lines of inline PTX inside
   a ~300-line generated kernel template. Cutlass's hand-tuned cooperative-
   loading kernels achieve **~73%** with thousands of lines of templated C++
   plus embedded PTX scheduling hints. **TMA buys you ~93% of cuBLAS's peak
   performance for ~3% of cuBLAS's source-code complexity.** That's a real
   engineering win even if it's not a peak-throughput win.

2. Lets us write a fully-pipelined kernel with **zero shared-memory store
   instructions** (TMA writes smem via hardware DMA). The instruction count
   for the same workload is roughly 4x lower than cuBLAS's cooperative
   loading path. The kernel is much easier to read and debug.

3. Gives us **uniform behavior across architectures**. The same template
   runs at 68% FMA pipe util on every Blackwell SKU we tested (5090, Pro 6000)
   *and* on Hopper (H200, 71%). cuBLAS, by contrast, dispatches completely
   different kernel families per arch and per workload size, with all the
   heuristic bugs that entails.

## What this means for the article

The headline "beats cuBLAS by 58% in batched mode" is technically accurate
but misleading. The honest version:

> *I wrote a TMA-based FP32 SGEMM kernel template to learn TMA. While
> benchmarking it I noticed that the same `cublasSgemmStridedBatched` call
> dispatches a 5x throughput-different kernel on the 5090 vs the Pro 6000
> at the same shape — both Blackwell, same library, same compute capability.
> Tracing the dispatch revealed three separate cuBLAS heuristic bugs. My
> kernel "beats" cuBLAS by 60% on the 5090 specifically because it
> happens to schedule itself sanely while cuBLAS picks a kernel running at
> 41% FMA pipe utilization. On any path where cuBLAS picks the right kernel,
> my generated template loses by 5pp — the cost of being a code generator
> vs hand-tuned PTX. TMA's role in the story isn't a novel performance unlock;
> it's a clean way to write a kernel that's competitive with hand-tuned
> cuBLAS using ~300 lines of C instead of thousands of lines of templates.*

That's the version backed by the data.
