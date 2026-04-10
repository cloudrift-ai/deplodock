# Per-warp stall reason comparison: 5090, batch=1 8192 single + batch=8 4096 batched

Source: `ncu --metrics smsp__average_warps_issue_stalled_*_per_issue_active`
on /tmp/sass_dbg/bench (fused_matmul), /tmp/cublas_single_8192, /tmp/cublas_batched.

Each cell is "warps stalled on this reason per issue-active cycle". Sum can
exceed 100% because multiple warps can stall in parallel.

## Single-mode 8192

| Stall reason       | fused_matmul (mine, TM=28) | cuBLAS simt_sgemm_256x128_8x4 |
|--------------------|---------------------------:|------------------------------:|
| not_selected       | 82.23%                     | 85.14%                        |
| **dispatch_stall** | **44.21%**                 | **22.36%**                    |
| **short_scoreboard** | **19.95%**               | **11.84%**                    |
| mio_throttle       | 7.86%                      | 4.95%                         |
| barrier            | 7.25%                      | 6.66%                         |
| no_instruction     | 3.00%                      | 7.36%                         |
| wait               | 3.93%                      | 3.28%                         |
| long_scoreboard    | 1.92%                      | 1.84%                         |
| lg_throttle        | 0.04%                      | 2.73%                         |
| math_pipe_throttle | 0.17%                      | 1.13%                         |

## Batched 4096 b=8 (the buggy 5090 dispatch)

| Stall reason       | cuBLAS simt_sgemm_128x32_8x5 (broken) |
|--------------------|--------------------------------------:|
| **mio_throttle**   | **212.82%**                           |
| not_selected       | 98.91%                                |
| **barrier**        | **51.99%**                            |
| short_scoreboard   | 17.92%                                |
| long_scoreboard    | 11.50%                                |
| dispatch_stall     | 6.85%                                 |
| math_pipe_throttle | 6.40%                                 |
| no_instruction     | 6.34%                                 |
| wait               | 6.77%                                 |
