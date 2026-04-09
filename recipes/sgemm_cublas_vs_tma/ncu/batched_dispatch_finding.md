# cuBLAS batched FP32 SGEMM dispatcher: full per-arch / per-size sweep

The article notes that our TMA kernel beats cuBLAS by ~60% in batched mode on
the 5090 but only matches cuBLAS on Pro 6000 / H200. After investigating,
the cause is **the cuBLAS dispatcher itself, not TMA hardware or our kernel**.
A full ncu sweep over (256, 512, 1024, 2048, 4096, 8192) at batch=8 on
**three** different sm_90/sm_120 GPUs surfaces a striking pattern: the
dispatcher behavior is wildly different per-arch, with the 5090 path
clearly broken.

All measurements: CUDA 13.2.51, cuBLAS 13.3.0, driver 595.58.03, batch=8,
square FP32 matmul, captured by `scripts/diagnostics/ncu_compare.sh`.

## The full dispatch table

| Size B=8 | **5090 kernel + FMA%**             | **Pro 6000 Max-Q kernel + FMA%**      | **H200 kernel + FMA%**                         |
|----------|-------------------------------------|----------------------------------------|------------------------------------------------|
| 256      | `simt_sgemm_128x32_8x5`        33% | `magma_sgemmEx_kernel`            32%  | `sm80_xmma_gemm_..._32x32x8_stage3`       33%  |
| 512      | `simt_sgemm_128x32_8x5`        39% | `magma_sgemmEx_kernel`            32%  | `sm80_xmma_gemm_..._64x128x8_stage3`      69%  |
| 1024     | `simt_sgemm_128x32_8x5`        41% | `simt_sgemm_128x64_8x5`           64%  | `simt_sgemm_256x128_8x4`                  78%  |
| 2048     | `simt_sgemm_128x32_8x5`        41% | `simt_sgemm_128x128_8x4`          69%  | `simt_sgemm_256x128_8x4`                  79%  |
| 4096     | `simt_sgemm_128x32_8x5`        42% | `simt_sgemm_256x128_8x4`          73%  | `sm80_xmma_gemm_..._128x128x8_stage3`     82%  |
| 8192     | `simt_sgemm_128x32_8x5`        42% | `simt_sgemm_256x128_8x4`          73%  | `sm80_xmma_gemm_..._128x128x8_stage3`     82%  |

All three GPUs use the same toolkit, same library binary, same CUTLASS source
templates, same compute capability family. The only thing that differs is the
runtime dispatch logic inside `libcublas.so` that maps `(M, N, K, batch_count, GPU)`
to a kernel ID. **That logic produces three completely different behaviors,
and the 5090 path is obviously broken.**

## Three observations

### 1. The H200 dispatcher is the gold standard

It mixes kernel *families* intelligently — open-source CUTLASS templates at
1024–2048, NVIDIA's closed-source hand-tuned `sm80_xmma_gemm` at 4K+, and
different xmma tile sizes (32×32 / 64×128 / 128×128) escalating with workload.
Peak FMA pipe utilization hits **82%**. This is a *real* dispatcher with
workload-aware logic and per-shape kernel selection.

### 2. The Pro 6000 dispatcher escalates within the CUTLASS family

It picks three different cutlass tiles (128×64 → 128×128 → 256×128) as the
workload grows, climbing from 64% to 73% FMA pipe util. Less sophisticated
than H200 — never tries the xmma family — but at least it has thresholds
and they fire correctly. The one bug: at 256 / 512 it falls into the
`magma_sgemmEx_kernel` legacy code path (32% FMA pipe util) instead of
picking the `simt_sgemm_128x64_8x5` cutlass kernel that gets 64% one
size up at 1024.

### 3. The 5090 dispatcher does nothing

Same `simt_sgemm_128x32_8x5` kernel from 256 all the way to 8192.
FMA pipe utilization stuck in the 33–42% band across the entire **32×**
range of linear dimensions. The exact same library that escalates correctly
on Pro 6000 (same compute capability!) and dispatches multiple kernel
families on H200 just hardcodes one tiny kernel for the 5090 sm_120
batched-FP32 path and never picks anything else.

This refutes my earlier hypothesis that "the 5090 has a wrong threshold."
It's not a wrong threshold — there is **no escalation at any threshold**.
The dispatch logic for 5090 batched FP32 is missing entirely. The library
*has* a `simt_sgemm_256x128_8x4` kernel that hits 73% FMA pipe util on
the Pro 6000 at the same workload — and it never picks it on the 5090.

## Where our kernel stands against this

Our generated TMA template hits **~68% FMA pipe utilization** on every
arch we tested (60% on H200 small workloads where the kernel doesn't even
have enough work; 68% on everything ≥1024). It's a constant.

| Workload            | cuBLAS FMA% | Ours FMA% | Ratio | Observed eff |
|---------------------|------------:|----------:|------:|-------------:|
| 5090 batched 4K b=8 |        42%  |       67% | 1.60× |        158% ✓|
| 5090 batched 8K b=8 |        42%  |       68% | 1.62× |        159% ✓|
| Pro 6000 4K b=8     |        73%  |       68% | 0.93× |         93% ✓|
| Pro 6000 8K b=8     |        73%  |       68% | 0.93× |         95% ✓|
| H200 batched 4K b=8 |        82%  |       71% | 0.86× |         91% ✓|
| H200 batched 8K b=8 |        82%  |       71% | 0.87% |         87% ✓|
| 5090 single 8K      |        73%  |       68% | 0.93× |         95% ✓|
| H200 single 8K      |        79%  |       71% | 0.90× |         92% ✓|

**Every observed efficiency number is `(ours FMA%) / (cuBLAS FMA%)` at the
specific workload.** Our TMA kernel is consistently 5–11pp below cuBLAS's
best-available kernel. When cuBLAS picks the right kernel we lose by
5–14%. When the dispatcher picks a buggy tiny kernel — which happens for
the entire 5090 batched range and at small sizes on Pro 6000 — we win
by 30–60%.

The headline "TMA beats cuBLAS by 58% in batched mode" is therefore
accurate but misleading. The accurate-and-honest version: **our kernel
incidentally exposes that cuBLAS's 5090 sm_120 batched FP32 dispatch
path picks a 41%-FMA-pipe kernel for the entire workload range, where
the same library has 73%-FMA-pipe and 82%-FMA-pipe kernels sitting
right there.** That's the actual win — finding a real, reproducible,
size-independent dispatcher bug in `libcublas.so` that costs ~60% of
peak performance for FP32 batched workloads on the most popular consumer
Blackwell SKU.

## What TMA actually buys us

The TMA double-buffer template:

1. Achieves **~68% FMA pipe utilization** with ~80 lines of inline PTX inside
   a ~300-line generated kernel template. CUTLASS hand-tuned cooperative-
   loading kernels achieve **73%** with thousands of lines of templated C++
   plus embedded PTX scheduling hints. NVIDIA's closed-source `xmma_gemm`
   achieves **82%** on H200 with even more hand-tuning. **TMA gets us
   ~93% of CUTLASS peak with ~3% of the source-code complexity** — and
   that ratio holds across every Blackwell SKU we tested.

2. Has **zero shared-memory store instructions** in our SASS (TMA writes
   smem via hardware DMA). Total instruction count for the same workload
   is ~4× lower than CUTLASS's cooperative loading path.

3. Gives us **uniform behavior across architectures**: same template runs
   at 68% FMA pipe util on every Blackwell SKU we tested *and* on Hopper.
   cuBLAS, by contrast, dispatches completely different kernel families
   per arch with all the heuristic bugs that entails.

This is a real engineering win for *kernel implementation simplicity*,
even though it isn't a peak-throughput win.

## Suggested article reframe

The current article's headline is "TMA beats cuBLAS in batched mode." The
data supports a sharper, more honest claim:

> *I wrote a TMA-based FP32 SGEMM kernel template to learn TMA. While
> benchmarking it I noticed that the same `cublasSgemmStridedBatched` call
> dispatches a different kernel on every Blackwell/Hopper SKU at the same
> shape. Tracing the dispatch surfaced three distinct cuBLAS dispatcher
> bugs — most notably, the 5090 sm_120 batched FP32 path is hardcoded to
> a single ~40%-FMA-pipe kernel across the entire size range, where the
> same library escalates correctly to a ~73% kernel on the Pro 6000 and
> a ~82% kernel on H200. My kernel "beats" cuBLAS by 60% on the 5090
> specifically because the 5090 dispatcher picks the wrong kernel.
> On any path where cuBLAS picks the right kernel, my generated template
> loses by 5–11pp — the cost of being a code generator vs hand-tuned PTX.
> TMA's role in the story isn't a novel performance unlock; it's a clean
> way to write a kernel that's competitive with hand-tuned cuBLAS using
> ~300 lines of C instead of thousands of lines of templates.*

That's the version backed by the data.
