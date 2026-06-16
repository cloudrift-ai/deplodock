# Fused symbolic-K SDPA P@V via smem-staged `xn` (capacity-capped at the hint)

Status: **design / not started.** Goal: make the dynamic (symbolic-`seq_len`) SDPA P@V lower to a single
tensor-core kernel that matches the static path's fusion — softmax producer + `mma` P@V in one launch, with the
softmaxed `P` (`xn`) staged in **smem** (capped at `DEFAULT_SEQ_HINT = 512`) instead of round-tripped through HBM.
A runtime `seq_len > hint` guard traps. This is the user-requested "(A)" path from the Qwen3-Embedding layer-0
static-vs-dynamic investigation (`plans/qwen3-embedding-layer0-static-vs-dynamic-tune-findings.md`).

## Why (evidence)

At seq 512, static layer-0 = 184 µs / 1.21x eager; dynamic = 336 µs / 0.65x. The gap is the SDPA P@V:

- **Static fused P@V** (`k_sdpa_linear_reduce_3d2635`): `__launch_bounds__(1024)`, TMA-staged, `ldmatrix` +
  `mma.sync` over the static K=512 reduce, softmax folded into an in-kernel smem-staged `xn`.
- **Dynamic, kept fused** (`SPLIT_CONE=0`): `__launch_bounds__(128)`, **0 `mma`**, a scalar 3-pass online-softmax
  + weighted-V sum looping `for (a6 < seq_len)`, 3× HBM re-reads of the seq×seq matrix → **1610 µs**.
- **Dynamic, split** (deployed): softmax producer materializes `xn` to HBM; masked-K `mma` consumer
  (`for a5 < (seq_len+511)/512`, 15 `mma.sync`) re-reads it → **336 µs**. Correct and tensor-core, but pays the
  HBM round-trip and the consumer runs at 26% occupancy / 3.67M smem bank-conflicts.

The symbolic K can't be a fused-prologue warp tile because the masked-K slab **zero-fills** the tail past
`seq_len` (correct for a clean P@V — 0 contributes nothing) but wrong for the softmax max/sum. Static escapes
this only because static K lets it stage the whole softmaxed row; the deployable symbolic equivalent is to stage
`xn` in smem capped at the hint.

## Key insight that simplifies the build

The softmax phase can stay **scalar** (it already masks correctly via `if (a6 < seq_len)` and writes `xn = 0`
past `seq_len`). Then phase 2 is the **existing** masked-K `mma` consumer, reading `xn` from **smem** instead of
HBM — and its zero-fill is already correct because `xn` is already 0 past `seq_len`. So we do **not** need an
`−inf` fill-mode on the masked-K slab; we need a kernel that holds `xn` in smem between a scalar producer phase
and the `mma` consumer phase. The win is purely removing the `xn` HBM round-trip + replacing the scalar P@V with
`mma`.

## Realistic ceiling

Removes the `xn` HBM round-trip and the degenerate scalar P@V → `mma`. Does **not** fix the masked-K `mma`
consumer's 26% occupancy / 3.67M bank-conflicts (that gap, 38 µs vs static's 15.8 µs, is a separate slab-layout
issue). Expected landing ≈ **270 µs**, not the full 184 µs. Closing to 184 needs the masked-K slab-layout fix too
(separate track, also helps the shipping 336 µs path).

## Where it lives (code map)

- `005_split_demoted.py` — today the cut emits *separate* Graph nodes (→ separate kernels, HBM hand-off). New:
  a **fused-smem** cut variant that keeps one node whose body is `producer-phase ; Smem(xn) ; consumer-phase`,
  gated on `xn`'s capped extent ≤ hint fitting smem. Offered as a third structural option beside keep-fused /
  split (knob `SPLIT_CONE` → extend to a tri-state, or a new `FUSE_SMEM` knob).
- `010_partition_loops.py:710` — the `not prologue` warp gate stays for the *raw* prologue; the fused-smem body's
  consumer phase is a clean matmul (P=`xn` is now a stored smem operand), so it reaches the warp tier normally.
- `kernel/_stage_expand.py` — the consumer reads `xn` from smem; the producer writes `xn` to smem. Reuse the
  existing masked-K zero-fill on the consumer (no `−inf` mode needed, per the insight above).
- `kernel/100_materialize_tile.py` — size the `xn` smem buffer at `BM × hint` (capped); emit both phases in one
  kernel body with a `__syncthreads()` between.
- Guard: host-side in `backend/cuda/program.py` (where symbolic dims resolve from input shapes at launch) — raise
  if `seq_len > hint` for a capped kernel. Cheaper and clearer than an in-kernel trap; fully unit-testable.

## Staging (each step independently landable + tested)

1. **Host-side `seq_len > hint` guard** + a kernel flag marking "capacity-capped". Inert until a capped kernel
   exists, but unit-testable (resolve a too-large shape → raises). Lands the guard infrastructure.
2. **Masked-K slab-layout fix** (occupancy / bank-conflicts) on the *existing* split consumer — separable, helps
   the 336 µs path now, and is a prerequisite for the fused kernel not inheriting the conflicts. A/B `PAD_SMEM` /
   `PERMUTE_LANES` first (the prior dynamic report showed these were no-ops on the masked-tile layout — so this is
   likely a new swizzle, not a knob).
3. **Fused-smem cut** in `005_split_demoted` + the two-phase materialization. The big step. Verify accuracy on the
   `k_sdpa_linear_reduce` reproducer (static-and-dynamic parity test), then e2e bench vs the 336 µs split.

## Tests

- Accuracy: `tests/compiler/passes/test_knob_pinning.py` style — fp16 SDPA P@V, dynamic mode, pinned to the
  fused-smem cut, asserted against the numpy/torch reference at the hint (and a sub-hint seq to exercise the mask).
- Guard: unit test that resolving `seq_len > hint` on a capped kernel raises a clear error.
- Bench: `run --ir <reproducer> --bench --ab` fused-smem vs split, expect ≤ split and document the gap to static.
