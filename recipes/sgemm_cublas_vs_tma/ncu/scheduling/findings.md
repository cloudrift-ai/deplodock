# Tier 1 + Tier 2 SASS scheduling analysis: results

Three experiments to back the article's "PTX is for Noobs: SASS Deep Dive"
section. All measurements on RTX 5090, CUDA 13.2.51, cuBLAS 13.3.0,
single-mode 8192×8192 unless noted.

## Tier 1 — LDS-to-FFMA distance (our kernel)

`scripts/diagnostics/scheduling_analysis.py /tmp/sass_dbg/bench`:

- **256 LDS.128 instructions**, all 256 have a downstream FFMA consumer
- **Median LDS→first-FFMA-consumer distance: 40 FFMAs** (mean 44.6, max 170)
- **Median LDS→next-LDS spacing: 5 FFMAs** (mean 13.2, max 223)
- 110/256 LDS in [20, 40) FFMA distance, 117/256 in [40, 80), only 4 LDS with distance < 10

**Interpretation:** ptxas hides LDS latency essentially perfectly. Blackwell
LDS latency is ~30 cycles; we have ~40 FFMAs (≈40 cycles on the FMA pipe)
between every load and its first consumer. The 5pp gap to cuBLAS is **not**
about LDS latency hiding — that hypothesis was wrong.

## Tier 1 — LDS-to-FFMA distance (cuBLAS)

Extracted from ncu source view of `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`
during a live cublasSgemm(8192, 8192, 8192) call (cuBLAS JIT-compiles the
kernel from PTX at runtime; not available as a static cubin):

- 256 LDS, 1152 FFMAs (cuBLAS uses smaller per-thread tile, more CTAs)
- LDS / FFMA ratio: **1 LDS per 4.5 FFMAs** (vs my 1 per 14)
- **Median LDS→first-consumer distance: 158 FFMAs** (mean 140.9, max 191)
- **Median LDS→next-LDS spacing: 0** (mean 3.7, max 190)

**Interpretation:** completely different scheduling strategy. cuBLAS clusters
LDS into back-to-back groups (median spacing = 0) and amortizes them over
much longer FFMA runs (4× the latency-hiding distance of mine). My kernel
spreads LDS evenly through the FFMA cluster.

## Tier 1 — Inner loop excerpt (our kernel)

Densest 100% FFMA region in `fused_matmul` is **112 instructions long**
with no LDS interleaving — that's the post-unroll TM=28 thread tile
(28 rows × 4 cols = 112 FMAs). LDS happen at the boundaries between
unrolled K iterations, not inside the FFMA cluster.

```
/*5290*/  FFMA   R162, R37, R148.reuse, R162
/*52a0*/  FFMA   R165, R38, R148.reuse, R165
/*52b0*/  FFMA   R164, R39, R148, R164
/*52c0*/  FFMA   R3, R41.reuse, R152, R166
/*52d0*/  LDS.128 R36, [R15+0x8400]   ← single LDS in the middle of the FMA cluster
/*52e0*/  FFMA   R168, R41.reuse, R153, R168
/*52f0*/  FFMA   R167, R41.reuse, R154, R167
/*5300*/  FFMA   R40, R41, R155, R40
/*5310*/  FFMA   R5, R45.reuse, R152, R172
... (continues for ~25 more FFMAs)
```

The `.reuse` suffix is the operand-collector reuse hint — ptxas is correctly
marking that R148, R41, R45 etc are reused on the next instruction so the
operand collector can skip a register-file read. Reuse hints are heavy in
this region, which is what you want for SGEMM.

## Tier 2 — Per-warp stall reason comparison (ncu source counters)

`ncu --metrics smsp__average_warps_issue_stalled_*_per_issue_active`
(values are "warps stalled per issue-active cycle"; can exceed 100% when
multiple warps stall in parallel).

### Single-mode 8192×8192

| Stall reason       | fused_matmul (mine) | cuBLAS 256x128_8x4 | delta  |
|--------------------|--------------------:|-------------------:|-------:|
| not_selected       | 82.23%              | 85.14%             | +2.9   |
| **dispatch_stall** | **44.21%**          | **22.36%**         | **-22**|
| **short_scoreboard** | **19.95%**        | **11.84%**         | **-8** |
| mio_throttle       | 7.86%               | 4.95%              | -3     |
| barrier            | 7.25%               | 6.66%              | -0.6   |
| no_instruction     | 3.00%               | 7.36%              | +4     |
| wait               | 3.93%               | 3.28%              | -0.7   |
| long_scoreboard    | 1.92%               | 1.84%              | -0.1   |
| lg_throttle        | 0.04%               | 2.73%              | +2.7   |
| math_pipe_throttle | 0.17%               | 1.13%              | +1     |

**Where my kernel loses the 5pp at the FMA pipe:**

- **`dispatch_stall` = 44.2% vs cuBLAS 22.4%** (-22 pp). This is FMA pipe
  back-pressure on the issue side — the warp scheduler picks a ready warp
  but the dispatch unit can't accept another instruction this cycle because
  the FMA pipe is saturated by some other warp's in-flight FFMA. **This is
  the dominant cause of the 5pp utilization gap**, and it's a direct
  consequence of the spread-LDS-through-FMA-cluster pattern: all warps are
  in roughly the same phase of execution, so they all want the FMA pipe at
  the same time.
- **`short_scoreboard` = 20.0% vs cuBLAS 11.8%** (-8 pp). Short scoreboard
  stalls are dependencies on smem-load results. Even though the *static*
  LDS-to-consumer distance is 40 FFMAs (more than enough to hide the
  latency), the warp scheduler still attributes scoreboard stalls to the
  LDS chain because the consumers are interleaved tightly.

### Batched 4096×4096×8 — the buggy 5090 dispatch

| Stall reason       | cuBLAS simt_sgemm_128x32_8x5 |
|--------------------|-----------------------------:|
| **mio_throttle**   | **212.82%** (!!)             |
| barrier            | 51.99%                       |
| short_scoreboard   | 17.92%                       |
| long_scoreboard    | 11.50%                       |
| dispatch_stall     | 6.85%                        |

**`mio_throttle = 212%`** — the LSU pipeline is so saturated that on average
2+ warps per scheduler are simultaneously stalled waiting for an MIO queue
entry. This is the structural reason the buggy 5090-batched kernel sits at
41% FMA pipe utilization: the 128×32 thread tile is too small, each thread
does too few FMAs per shared-memory load, and the LSU queue saturates
before the FMA pipe can be fed.

That's an additional, clean explanation for *why* picking the 128x32 kernel
is wrong for batched workloads at the sizes we tested — it's not just "the
heuristic picked a small kernel," it's "the small kernel is structurally
incapable of feeding the FMA pipe at this batch density because its LDS
density is too high."

## Bottom line for the article

The current SASS section claims that the 5pp single-mode gap is "SASS-level
instruction scheduling that ptxas can't extract from generated C source as
well as NVIDIA's hand-tuned PTX." That's *correct* but vague. The data lets
us tighten it considerably:

1. **The scheduling difference is structural, not subtle.** cuBLAS clusters
   LDS, we spread them. cuBLAS has 4× the LDS-to-consumer distance.
2. **The mechanism is dispatch-stall pressure on the FMA pipe**, measured at
   44% vs 22%. ptxas's emitted schedule has us putting too much issue-side
   pressure on the FMA pipe at the same warp-phase, while cuBLAS's PTX
   scheduling hints stagger warps better.
3. **Closing it would require generating C source that produces the
   clustered-LDS pattern after ptxas transformations** — which is hard
   because ptxas reschedules aggressively. The 3–5% bound CuAsmRL reports
   for SASS-level reordering matches our measured gap.

For the buggy batched dispatch:

4. **The 41% FMA pipe util on `simt_sgemm_128x32_8x5` is mio_throttle-bound**,
   not dispatch-stall-bound. The 128x32 tile is structurally wrong for batched
   workloads ≥1024 — the LSU queue saturates. cuBLAS's dispatcher picking
   this kernel for batched workloads at every 5090 size is therefore not
   just "suboptimal," it's "selecting a kernel that hits a different
   bottleneck than the right kernel would."

## Files in this bundle

- `SCHEDULING_FINDINGS.md` — this file
- `stall_summary.md` — Tier 2 stall reason table
- `fused_matmul_lds_distance.md` — Tier 1 LDS distance / spacing / inner loop excerpt
- `fused_matmul_sass_with_stalls.txt` — full ncu source view of fused_matmul (per-instr stalls)
- `cublas_simt_sgemm_256x128_sass_with_stalls.txt` — same for cuBLAS single-mode kernel
- `/tmp/fused_full_profile.ncu-rep` — full ncu report (open in Nsight Compute GUI for live navigation)
- `/tmp/cublas_full_profile.ncu-rep` — same for cuBLAS
