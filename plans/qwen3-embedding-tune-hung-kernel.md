# Qwen3-Embedding-0.6B: `k_linear_mean_reduce` WS=1 deadlock (FIXED)

Status: **root-caused and fixed**. The hang was not an -O3 miscompile — the WS=1 (warp-specialized) variant of this
kernel was emitted with an **empty producer branch**, a structural deadlock at every nvcc opt level. The
`085_warp_specialize` eligibility check now runs the same producer/consumer split the transform uses and rejects the
shape, so WS=1 is no longer in this op's enumeration; the deployed greedy pick compiles and completes (123 µs at -O3).

## 1. The kernel

The slice is the decoder MLP path: post-attention RMSNorm (`pow → mean → +eps → rsqrt → mul → weight-mul`) feeding the
gate/up projections (1024 → 3072, fp16) and the SwiGLU combine (`silu(gate) * up`), output `mul_12` of shape
`(1, 32, 3072)`. Healthy picks lower it to a register-tiled fused linear+mean kernel; the hanging variant was the same
tile family (BM=8 BN=32 BK=64 RING=2 STAGE=100) **plus WARPSPEC=1**.

## 2. Root cause (the real one)

`085_warp_specialize` splits the consumer tile's body into producer (TMA `StageBundle`s) and consumer (waits + reduce +
writes) halves via `_split_by_role`, which recurses only through `SerialTile(serial_outer)` / `RegisterTile` /
`AtomTile`. This fused linear+mean kernel wraps its whole body in a `SerialTile(kind='plain')` per-thread M-fragment
loop, which the splitter classifies consumer wholesale. But `_eligible` used a **deep** walk (`op.body.iter()`) to find
the TMA depth-2 bundle, so it declared WS=1 eligible anyway. Result: a `WarpSpecialize` with `producer_body=()` — the
producer warp ran only `setmaxnreg.dec` and exited, while the TMA issues landed in the **consumer** branch behind an
`if (threadIdx.x == 0)` guard that no consumer-branch thread can satisfy (thread 0 is a producer-warp thread). Nobody
ever issued the TMA loads; every consumer `mbarrier.wait` spun forever. GPU pinned at 100 %, the launched kernel
resident until the owning process exits, CUDA context teardown blocked behind it.

Two earlier claims in this doc were wrong, in instructive ways:

- **"The -O1 tuning sweep benches the variant fine"** — it was never benched. The tune DB has **zero** WS=1 `perf`
  rows for this op's shape (`S_ext_free_prod=98304`); the sweep's patience ran out before the WS=1 fork was explored.
  Re-run under `--nvcc-flags "-Xcicc -O1"`, the variant hangs identically (verified): the deadlock is structural, not
  an optimizer artifact.
- **"-O3 turns a loop non-terminating"** — the opt level was a red herring. The deploy path compiles at -O3, the
  tuning sweep at -O1, so the hang *correlated* with -O3 only because the deploy path was the only one that ever
  launched the variant.

How the prior came to pick it: the learned `CatBoostPrior` generalizes across ops. WS=1 is a measured win on warp-tier
MMA matmuls (12–14 µs bests in the same DB), so the prior predicted WS=1 fast for this scalar-path op too — a config
no bench had ever executed. The DB also holds 21 `bench_fail` rows (2 s watchdog) for WS=1 on non-MMA square shapes:
the same stranded-TMA deadlock, already caught by the tuner on other shapes without anyone connecting them.

## 3. The fix

`085_warp_specialize._eligible` now runs `_split_by_role` (the exact split `_ws_transform` performs) and rejects when
no TMA depth-2 bundle lands producer-side ("TMA StageBundle not reachable by the producer split"); `_ws_transform`
asserts the same invariant as defense in depth. Ineligible shapes stamp `WARPSPEC=False` as before; a pinned
`DEPLODOCK_WARP_SPECIALIZE=1` fails loudly with the reachability reason.

Regression tests:

- `tests/compiler/passes/test_warp_specialize.py` — unit: the plain-SerialTile-wrapped synthetic shape stamps
  WS=False / raises on a WS=1 pin.
- `tests/compiler/passes/test_warp_specialize_deadlock.py` — the real fused RMSNorm + gate/up + SwiGLU slice with the
  hanging knob family pinned: compile-only (GPU-less, `set_target(sm_120)`) WS=1-pin raise + never-offers-WS=1
  stamping, plus a CUDA test that the family compiles, completes under the watchdog, and matches numpy.

## 4. Verification

```bash
DEPLODOCK_DUMP_DIR=/tmp/dd deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0
deplodock run --ir /tmp/dd/07_lowering_cuda.kernels/k_linear_mean_reduce_*.torch.json --bench
# pre-fix: HungKernelError within ~1 s (at -O3 AND -O1); post-fix: completes, 123 µs vs eager 70 µs
```

The op remains deplodock's worst on this model (torch.compile 14 µs / eager 63–70 µs on the healthy picks) — that's a
performance gap, not a correctness bug, and is out of scope here.

## 5. Hardening that already landed (kept from the original report)

- Per-launch watchdog + SIGKILL-able bench worker (#215): a hung kernel fails one variant/row, not the run.
- PR #216: the parent no longer wedges writing to a worker stuck in CUDA teardown; dynamo's recompile limit is reset
  before every bench compile; bench timings are CUDA-graph-captured; the torch reference honors declared dtypes;
  reproducer slices keep constant-derived boundaries.

## 6. Remaining

- Re-tune layer 0 into the default caches so the DB/prior rows for this op reflect the post-fix enumeration (the
  greedy pick already deploys fine — WS=1 simply no longer exists for the shape — so this is about prior hygiene, not
  correctness).
- The 21 WS=1 `bench_fail` square-shape rows in the tune DB are the same bug; they'll fall out of the enumeration the
  same way on their next tune.
