# Warp-Specialized Fused-Prologue Kernels (`setmaxnreg`)

## Why

Fused-prologue matmuls — `silu_mul_matmul` (down_proj on a SiLU-gated MLP),
`gelu_mul_matmul`, `bias_relu_matmul`, etc. — currently lose 2-3× to the
equivalent two-kernel sequence (`silu_mul → temp; matmul(temp, w)`) on
RTX 5090 fp32. The fused kernel is structurally forced into a worse
schedule than the pure matmul:

| Aspect | Pure matmul (`F=8×4 BN=128`) | Fused `silu_mul_matmul` (`F=4×4 BN=64`) |
|---|---|---|
| Outputs / thread | 32 | 16 |
| Arithmetic intensity / CTA | 1.0× | 0.5× |
| Regs / thread | 80 | 80 (after fused-class fix; was 145) |
| Occupancy | 50 % | 50 % |
| FMA% | ~65 % | ~46 % |
| SiLU pipeline ↔ FMA dependency chain inside K-loop | n/a | 5 stages × FM live values |

The wall isn't bank conflicts (~5 % overhead) and it isn't pure
register pressure once we drop to `F=4×4`. The wall is that **every
K-iteration of the inner loop carries a serial `LDS_g → neg → exp →
add → rcp → mul_g → mul_u → A → mul_w → accum` chain**, the SFU
(`exp`, `rcp`) competes with the FMA pipe for issue bandwidth on the
same warp scheduler, and the per-CTA tile shape is forced down to
make the live SiLU values fit. Defusion solves all three at once but
costs an extra global-memory round-trip on the `A` operand.

The Hopper / Blackwell answer is **warp specialization with dynamic
register reallocation** (`setmaxnreg.dec/inc.sync.aligned.u32`).
Producer warps run the fused prologue (silu·u) into an in-CTA
ping-pong smem buffer; consumer warps run the canonical matmul over
that buffer at the standard `F=8×4` tile. The producer/consumer
warp-groups have *different register budgets*: producers release
their reg-file slice with `setmaxnreg.dec`, consumers claim it with
`setmaxnreg.inc`. Same kernel, no global-memory round-trip on `A`,
but the consumer body looks identical to a pure-matmul kernel.

## Goal

Get `silu_mul_matmul` (and the rest of the fused-prologue family) to
**≥ 0.85× of cuBLAS** on the qwen.s512 / s128 cases (currently
0.28-0.39× with the fused-class tile fix, 0.80-0.87× with the
banned defusion-via-fusion-guard hack).

## Sketch of the target Tile IR

```
matmul = TileOp(silu_one, g, u, w)
  kernel k_silu_mul_matmul_warp_specialized

      Tile(axes=(a0:N_block=BLOCK, a1:M_block=BLOCK)):

          # Warp-group declarations (new IR)
          WarpGroup(name="producer", warps=4, max_regs=40)
          WarpGroup(name="consumer", warps=8, max_regs=240)

          # Multi-stage producer→consumer A buffer (new IR)
          PipelineBuffer(name="A_smem",
                         extents=(BM=64, BK=32),
                         buffer_count=4,
                         producer_signal="A_full",
                         consumer_signal="A_empty")

          WarpGroupBody("producer"):
              setmaxnreg.dec(40)
              for a4 in 0..K_outer:
                  TmaBufferedStage(g_smem, ...)        # standard TMA load
                  TmaBufferedStage(u_smem, ...)
                  AsyncWait(g_smem, u_smem)
                  PipelineWait("A_empty")              # consumer drained slot?
                  for r,k: A_smem[slot][r,k] = silu(g[r,k]) * u[r,k]
                  PipelineSignal("A_full")             # consumer can read

          WarpGroupBody("consumer"):
              setmaxnreg.inc(240)
              register_acc[8][4]                       # F=8×4
              for a4 in 0..K_outer:
                  TmaBufferedStage(w_smem, slab=(BK, BN=128))
                  PipelineWait("A_full")
                  for a5 in 0..BK:
                      A_r = load A_smem[slot][m_t*8 + r, a5]   for r=0..7
                      w_c = load w_smem[a5, n_t*4 + c]         for c=0..3
                      acc[r][c] += A_r * w_c                    # pure FMA, no SFU
                  PipelineSignal("A_empty")
              # epilogue
              for r,c: matmul[...] = acc[r][c]
```

The consumer body is **byte-for-byte the canonical matmul body** —
no SiLU pipeline in its instruction stream at all. The producer body
is a tight elementwise loop that uses the SFU at peak. The two warp
groups talk through `A_smem` + mbarriers and never serialize on data
dependencies.

## Why this beats both fused and defused

| | Current fused | Defused (banned) | Warp-specialized |
|---|---|---|---|
| Matmul tile | F=4×4 BN=64 | F=8×4 BN=128 (cuBLAS-class) | F=8×4 BN=128 |
| Outputs / thread (consumer) | 16 | 32 | 32 |
| SiLU pipeline location | inline in K-loop body | separate kernel | producer warps (concurrent) |
| Cross-pipe data dep in inner loop | yes (silu→FMA) | no | no |
| `A` tensor traffic | smem-only | global write+read (~M·K·8 B) | smem-only (ping-pong) |
| Producer/consumer parallelism | none | sequential kernels | concurrent warp-groups |
| Expected ratio vs cuBLAS | 0.4× | 0.9× | **0.85-0.95×** |

The warp-specialized kernel inherits the matmul-quality ratio of
defusion *without* the global-memory round-trip, because the SFU
producer work runs concurrently with consumer FMA work rather than
sequentially.

## What's missing

We need five compiler pieces. Listed in dependency order:

### 1. Warp-group IR primitives

New `Stmt` types in `ir/tile/ir.py`:

- `WarpGroup(name, warps, max_regs)` — declares a contiguous range
  of warps within the CTA assigned to a body. Lowers to a
  warp-id range check + `setmaxnreg` PTX intrinsic.
- `WarpGroupBody(group_name, body)` — body executed only by the
  named group; siblings can run different `WarpGroupBody`s
  concurrently.
- `PipelineBuffer(name, extents, buffer_count, producer_signal,
  consumer_signal)` — multi-slot smem buffer for producer→consumer
  hand-off. Like `BufferedStage` but with paired mbarriers per
  slot.
- `PipelineWait(signal)` / `PipelineSignal(signal)` — per-slot
  mbarrier ops, parameterized by the current slot index (rotating
  modulo `buffer_count`).

### 2. Tile-IR pass `00X_warp_specialize_fused_prologue.py`

Detect the fused-prologue pattern (the same `_has_fused_prologue`
predicate already used by the fused-class tile pick), then rewrite:

- Split the body into a "producer subtree" (loads + prologue
  compute, ending in a `Stage`-like store of A) and a "consumer
  subtree" (the matmul loop).
- Wrap each in a `WarpGroupBody`.
- Replace the producer's terminal Stage with a `PipelineBuffer`
  store; replace the consumer's load of the prologue output with
  a `PipelineBuffer` load.
- Pick warp counts (typical: 4 producer + 8 consumer) and register
  budgets (40 / 240) based on the matmul tile (currently
  `F=8×4` → 240 regs/consumer, leaves headroom).

Skip when the prologue itself is a non-trivial reduction (the
producer would need its own internal sync), when the matmul has
small K (pipeline drain dominates), or when smem won't fit a
`buffer_count=4` ping-pong.

### 3. Materializer / Kernel-IR lowering for warp-groups

`passes/lowering/kernel/100_materialize_tile.py` extension:

- A `WarpGroup` declaration emits a per-warp-id branch dispatching
  to each `WarpGroupBody`.
- `setmaxnreg.dec/inc.sync.aligned.u32 N` lowered as inline-PTX
  asm wrappers, similar to how `MbarrierArriveExpectTx` is
  emitted today (see `_TMA_PRELUDE` in `kernel/render.py`).
- `PipelineBuffer` lowered to (a) a `Smem` decl with
  `buffer_count` slots, (b) `MbarrierInit` for each slot's pair
  of full/empty mbars, (c) per-iter slot-index update.
- `PipelineWait("full" / "empty")` lowered to
  `mbarrier.try_wait.parity` on the matched mbar — the same
  primitive the existing TMA path uses; we already have
  `mbarrier_wait_parity` in `_TMA_PRELUDE`.

### 4. CUDA renderer plumbing

- `__launch_bounds__` becomes per-CTA, but each warp-group's
  register count is set via `setmaxnreg`. The static
  `__launch_bounds__` should reflect the *consumer* group's
  budget (the larger one), so PTXAS schedules accordingly.
- A small alignment guard at kernel entry: `setmaxnreg` requires
  warp-aligned dispatch, so we need an `if (warp_id < N_PROD)
  setmaxnreg.dec(40); else setmaxnreg.inc(240);` prelude before
  any of the warp-group bodies run.
- The `_NCU_METRICS` set already covers `launch__registers_per_thread`;
  the warp-specialized kernel will report the *consumer* count
  (the producer's smaller count is invisible to ncu's per-launch
  reporting).

### 5. Tuning fixtures + perf coverage

- Add a `warp_specialize_fused` tuning knob in `tuning.py`
  (default-on for sm_90+, off below — Ampere doesn't have
  `setmaxnreg`).
- The existing `naive_attn` perf cases regression-net the SDPA
  fusion path; for the warp-specialized path, the existing
  `silu_mul_matmul.*` cases cover it directly. Add at least one
  `gelu_mul_matmul`-shaped case (different prologue) so the pass
  is exercised on more than one elementwise.

## Validation plan

Phase 1 — IR + lowering (no perf gate):
- Hand-write a `silu_mul_matmul.qwen.s128`-shaped Tile IR with the
  warp-group form, run through compile + lowering, confirm the
  emitted PTX has `setmaxnreg.dec.sync.aligned.u32` and
  `setmaxnreg.inc.sync.aligned.u32` in the right places.
- Smoke-test correctness against eager (the existing
  `_check_accuracy` path).

Phase 2 — pass that auto-creates the warp-group form:
- Sweep on `silu_mul_matmul.qwen.{s32, s128, s512}` and
  `silu_mul_matmul.tinyllama.{s32, s128, s512}`. Target
  ≥ 0.80× of cuBLAS on each (≥ 0.85× on the larger ones).
- Run `make bench-kernels` to confirm no regression on
  `naive_attn`, the matmul cluster, sdpa.

Phase 3 — extend to other fused-prologue shapes:
- `gelu_mul_matmul`, `bias_add_matmul`, `tanh_matmul` — same pass
  applies. The producer body just runs different elementwise ops.
- A `matmul_silu_mul_add` (epilogue + prologue together) would
  also fit, but is one shape removed from this work.

## Risk / open questions

- **smem budget for `buffer_count=4`** on the qwen.s512 case:
  `BM=64 × BK=32 × 4 buffers × 4 B = 32 KB` for `A_smem`, plus
  `g_smem + u_smem + w_smem`. Static-smem cap is 48 KB; we'd need
  dynamic smem (extern `__shared__` + `cudaFuncAttributeMaxDynamicSharedMemorySize`).
  Already supported by the existing materializer — see
  `STATIC_SMEM_CAP` in `kernel/render.py`.
- **producer warp count** (4 vs 2 vs 1): too few producers can't
  keep up with the consumer's SFU-free FMA pipe; too many waste
  registers. Likely needs a small tuning sweep, parameterized
  the same way `register_tile_shape` is.
- **pipeline drain on small K**: with K_outer < `buffer_count`, the
  pipeline never reaches steady state. The pass should skip
  warp-specialization on the small-K cases (s32 shapes).
- **fp16 / bf16 future**: warp specialization combined with `wgmma`
  (Hopper warp-group MMA) is the canonical CUTLASS-style kernel,
  but we'd have to cross the SIMT-FMA → tensor-core line first.
  For now: SIMT-FMA consumer body works on every sm_90+ GPU.

## Estimated scope

- New IR primitives: ~150 LOC across `ir/tile/ir.py` and
  `ir/kernel/ir.py`.
- New tile pass `00X_warp_specialize_fused_prologue.py`: ~250 LOC.
- Materializer extension: ~150 LOC in
  `passes/lowering/kernel/100_materialize_tile.py`.
- Renderer + PTX preludes: ~80 LOC in `kernel/render.py`.
- Tests + tuning fixture: ~100 LOC.

Total: ~700 LOC of new code plus ~50 LOC of edits to existing
materializer paths. Comparable in size to the existing TMA / mbar
path that's already in the tree.

## When to do this

Now (this branch) is too soon — the LDS.128 vectorization, chunk
pass, fused-prologue tile class, and `naive_attn` regression net
already extend the perf-tuning surface meaningfully. This warp-
specialization work is its own multi-week effort and should land
on a separate branch with the perf bench-kernels infra committed
first as the regression baseline.
