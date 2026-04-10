# NCU Comparison: TMA vs cuBLAS at 8192x8192x8192, single batch

Captured by scripts/diagnostics/ncu_compare.sh, median across 3 ncu runs.

## RTX 5090 (sm_120)

### fused_matmul

| Metric | Value |
|---|---:|
| sm__cycles_active.avg | 37,602,324.57 |
| sm__inst_executed.avg.per_cycle_active | 2.95 |
| sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active | 68.00 |
| sm__warps_active.avg.pct_of_peak_sustained_active | 16.67 |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | 9.40 |
| launch__registers_per_thread | 241 |
| launch__shared_mem_per_block | 91,392 |

### cutlass_80_simt_sgemm

| Metric | Value |
|---|---:|
| sm__cycles_active.avg | 34,831,449.99 |
| sm__inst_executed.avg.per_cycle_active | 3.23 |
| sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active | 72.92 |
| sm__warps_active.avg.pct_of_peak_sustained_active | 16.66 |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | 9.80 |
| launch__registers_per_thread | 210 |
| launch__shared_mem_per_block | 50,176 |

## H200 (sm_90)

### fused_matmul

| Metric | Value |
|---|---:|
| sm__cycles_active.avg | 45667110.18 |
| sm__inst_executed.avg.per_cycle_active | 3.26 |
| sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active | 71.25 |
| sm__warps_active.avg.pct_of_peak_sustained_active | 37.10 |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | 5.31 |
| launch__registers_per_thread | 80 |
| launch__shared_mem_per_block | 50432 |

### sm80_xmma_gemm

| Metric | Value |
|---|---:|
| sm__cycles_active.avg | 41106730.76 |
| sm__inst_executed.avg.per_cycle_active | 3.57 |
| sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active | 79.19 |
| sm__warps_active.avg.pct_of_peak_sustained_active | 12.50 |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | 4.74 |
| launch__registers_per_thread | 254 |
| launch__shared_mem_per_block | 68608 |

