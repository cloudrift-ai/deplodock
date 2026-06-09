# Qwen3-Embedding-0.6B layer tune: bench-cache reuse, per-kernel-vs-torch, and a hung-kernel codegen bug

Context: a clean, isolated two-layer tune of `Qwen/Qwen3-Embedding-0.6B` on an RTX 5090 (sm_120) to (a) confirm the
two-level autotuner's per-op bench cache transfers between structurally-identical layers, and (b) see which kernels are
well-tuned vs lagging torch. The run surfaced a **codegen bug** (a generated kernel that never terminates) and a
**bench-harness robustness gap** (the deployable `--bench` hung on it ~109 min). This doc records the findings; the fix
landed alongside it (branch `fix/bench-hung-kernel-watchdog`).

All runs used isolated cache paths (`DEPLODOCK_TUNE_DB` / `DEPLODOCK_PRIOR_FILE` / `DEPLODOCK_CUBIN_CACHE` → a temp dir)
so the developer's accumulated 725 MB tune DB and golden prior were never touched.

## 1. Model shape

All 28 decoder layers are structurally identical (dense Qwen3: `sliding_window: null`, no `mlp_only_layers` /
`layer_types`, shared RoPE). hidden 1024, 16 q-heads / 8 kv-heads (GQA), head_dim 128, intermediate 3072. Layers differ
only in weight *values*, which don't affect kernel selection — so a kernel tuned for layer 0 transfers to all 28 via the
structural `op_cache_key`.

## 2. Bench-cache reuse (layer 0 cold → layer 1 warm)

Each layer fused to one terminal: 10 unique kernels over 11 positions; `Σ best-per-op` ≈ 640 µs (an **-O1 ranking**
sum, not deployable). The inner per-op search explored ~the same number of terminals both times, but layer 1 replayed a
third of them from layer 0's DB:

| | Layer 0 (cold) | Layer 1 (warm) |
|---|---|---|
| Terminals explored | 767 | 746 |
| **New GPU benches** | 753 | **509** |
| **Cache-hit replays** (`ok, cached`) | 13 | **236** |
| Tune wall-clock | 751.6 s | 705.8 s |

So **~32% (236/746)** of layer 1's configs were served from cache with no GPU bench (log: `cache hit for 1 kernel(s) —
skipping bench`). Reuse is *partial by design*: the kernels are structurally identical (same `op_cache_key`) so configs
transfer, but the global learned prior absorbed ~850 benches during layer 0, so the warm search steers down a new
trajectory and benches the genuinely-new configs it surfaces while replaying the ones it has already seen. Wall-clock
dropped only ~6% because the 509 new configs still pay nvcc compile cost, which dominates the per-variant time.

## 3. Per-kernel vs torch (layer 0, deployable -O3)

`tune --bench` re-benches each kernel's `.torch.json` reproducer at -O3 vs eager / `torch.compile`. Sorted by deplodock
latency:

| Kernel | eager µs | tcompile µs | deplodock µs | vs eager |
|---|---|---|---|---|
| k_linear_mean_reduce | 215 | **47** | **135** | 1.59x |
| k_linear_reshape_transpose_sdpa_reduce | 79 | 93 | 46 | 1.70x |
| k_reshape_linear_mean_reduce | 218 | 310 | 39 | 5.60x |
| k_sdpa_transpose_unsqueeze_cat_slice_reduce | 426 | 583 | 37 | 11.62x |
| k_reshape_linear_mean_reduce | 219 | 303 | 27 | 8.20x |
| k_linear_reduce | 20 | 20 | 20 | 1.00x |
| k_sdpa_transpose_reshape_linear_reduce | 71 | 71 | 16 | 4.44x |
| k_linear_reduce | 14 | 14 | 10 | 1.40x |
| k_linear_reduce | 10 | 12 | 6 | 1.74x |

End-to-end full model (layer 0): **eager 240 µs · torch.compile 89 µs · Deplodock 271 µs (0.89× eager)**.

### Reading these honestly

- **The end-to-end number is the truth: Deplodock (271 µs) is ~3× behind torch.compile (89 µs) and slightly behind
  eager** for the whole layer. The per-kernel table *flatters* Deplodock because each reproducer is benched in
  isolation, which strips torch.compile of its cross-kernel fusion — so its per-kernel µs are inflated and the per-kernel
  "wins" don't translate to the fused end-to-end result. The gap is dominated by 11 separate kernel launches (dispatch
  overhead torch.compile fuses away) plus the slow `k_linear_mean_reduce`.
- **The one kernel lagging torch even in isolation is `k_linear_mean_reduce`** — 135 µs vs torch.compile's 47 µs
  (~2.9× slower), and it's also Deplodock's single slowest kernel. Every other kernel matches or beats torch.compile
  in isolation. This is the kernel to optimize.

## 4. Codegen bug: `k_linear_mean_reduce` can generate a non-terminating -O3 kernel

On the **layer 1** deployable bench, the greedy -O3 pick for `k_linear_mean_reduce_23ab9c` produced a kernel that
**never returns**: GPU pinned at 100% utilization, host spin-waiting. The full-model bench's per-launch watchdog caught
it (`kernel 'k_linear_mean_reduce_23ab9c' did not complete within 1000 ms — variant marked bench_fail`), but the kernel
stayed resident on the device.

Key facts:

- **It's a pick divergence, not a fixed defect.** Layer 0's pick for the same structural kernel ran fine (135 µs). The
  prior evolved after layer 0 and led layer 1's search to a *different* best -O1 config that, recompiled at -O3, hangs.
  A later re-run picked yet another (healthy) config and completed — so the bad config is reachable but not always
  chosen. This matches the standing `fp16-prior-needs-o3-coverage` note: with sparse -O3 reservoir data the prior can
  select an -O3-pathological config. Here the failure mode is worse than "deploys slow" — it's a hang.
- **-O1 vs -O3.** Tuning compiles at `-Xcicc -O1` (a ranking signal). The hang only manifests at -O3 (the deployable
  recompile), so the -O1 sweep never saw it. The kernel is almost certainly miscompiled or has a loop bound that -O3
  optimization turns non-terminating (a register-tiled linear+mean reduce — consistent with the BK1/STAGE1 collapse
  family).
- **Status: unfixed.** This doc and the accompanying PR make the bench *robust* to the hang; the kernel itself still
  needs a codegen fix. Root-cause next step: pin the exact knobs of the hanging variant (re-tune layer 1 to reproduce,
  read its `cuda_op` row), dump its `.cu` at -O3, and inspect the reduce loop bound.

## 5. Second codegen issue: RMSNorm reproducer fails to lower in isolation

`k_mean_b6c7b1` is the input RMSNorm (`pow → mean → +eps → rsqrt → mul → weight-mul`, output `mul_1`). It tunes and runs
fine *inside the full-model graph*, but its isolated `.torch.json` reproducer fails the -O3 re-bench:

```
k_mean: skipped (CudaBackend: node 'mul_1' has non-CudaOp 'LoopOp'; lowering must produce Graph[CudaOp].)
```

The trailing weight-multiply (`mul_1`) doesn't lower to a `CudaOp` when the kernel is re-compiled standalone, so there's
no deployable per-kernel number for RMSNorm. Path-specific (reproducer re-lowering only); lower priority than the hang.

Minor, non-fatal: three reproducers (`mul_12` SwiGLU 3072, `mul_3`/`mul_5` Q/K-proj+RoPE) produce NaN on random inputs
(out-of-domain for RoPE/activation), so their per-kernel accuracy is unverified — full-model accuracy passed.

## 6. Bench-harness robustness gap (fixed in this PR)

Why the hang wedged the whole run for ~109 min:

- **Tuning sweep** runs each variant in a SIGKILL-able subprocess (`bench_wall_timeout_s` → `benchmark_program_isolated`)
  — a hung kernel dies with its worker and the device resets. Tuning never wedged.
- **Deployable `--bench`** runs **in-process** (`benchmark_program`) — required because its interleaved `on_iter`
  peer-bench can't cross a subprocess boundary — so there is **no SIGKILL backstop**. Its per-launch polling watchdog
  (`_KERNEL_TIMEOUT_MS`, 1 s) *did* fire and abort the *measurement*, but in-process it can't *evict* the kernel, so the
  device stayed poisoned (the watchdog's own caveat: "a hung kernel is still queued on the device after we give up").
- The per-kernel sweep then launched into the poisoned device, and the torch peer-bench's **blocking
  `torch.cuda.synchronize()`** (no polling watchdog) queued behind the still-running kernel and blocked indefinitely.

**Fix (`fix/bench-hung-kernel-watchdog`):** the watchdog now raises a distinct `HungKernelError(RuntimeError)`, and
`_run_bench` treats a `HungKernelError` from the full-model bench as device-poisoning and **skips the per-kernel sweep**
(process exit resets the device) instead of marching into it. A benign `RuntimeError` (slow compile, healthy device)
still falls through to per-kernel. Verified by a real-GPU test (`test_hung_kernel_watchdog.py`: watchdog raises in 0.9 s
on a genuine non-terminating kernel) and control-flow tests (`test_tune_bench_hung_kernel.py`); the wedging invocation
now completes in 637 s.

Also removed in this PR: the `tune --bench-timeout` CLI flag, which was parsed and documented but **consumed nowhere**
(dead — the value in effect was the hardcoded 1 s watchdog constant). The separate, functional `--bench-timeout` args in
`scripts/bench_full_model.py` / `scripts/tune_golden_set.py` are unrelated and untouched.

## 7. Follow-ups

1. **Fix the `k_linear_mean_reduce` -O3 hang** (the real codegen bug) — pin the variant, dump -O3 `.cu`, inspect the
   reduce loop bound. Highest priority.
2. **Give the prior -O3 coverage for this shape** so it stops selecting -O3-pathological configs (`tune --patience 40
   DEPLODOCK_O3_TOL=0.30`, per the fp16 note).
3. **Close the RMSNorm reproducer lowering gap** (`mul_1` LoopOp → CudaOp in the standalone re-lower path).
4. **Reduce per-layer launch overhead** — the end-to-end gap vs torch.compile is largely 11 unfused launches.
