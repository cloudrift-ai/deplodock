# Close the two pipelining gaps: scalar operand staging + TMA ring

## Context

The tile-IR rebuild (`plans/tile-ir-rebuild.md`, branch `refactoring/tile-ir-rebuild`) has the warp/tensor-core
operand pipeline landed: the `STAGE` codec (`d<depth>/sync|cp|tma[/ring][/p<n>]`), the cp.async gmem→smem ring at
`depth>1`, the smem→register double-buffer (`/p<n>`), and TMA single-buffer staging. Two pipelining gaps remain
(tile-ir-rebuild.md, "Wiring cost & open gaps", final paragraph):

1. **Scalar operand staging** — the `Stage` struct is wired onto `SemiringKernel.schedule.stage`, but the scalar
   register-tile materializer **ignores it**: it always emits gmem-direct loads. The only real scalar smem staging is
   the hand-crafted fused-prologue *shared row* (one buffer, the norm→linear pattern, in `_reduce`). Scalar matmul has
   "no STAGE slab to pipeline."
2. **TMA ring** — TMA staging is **single-buffer only**. `_warp` clamps `tma_ok` to `depth=1` and the comment at
   `010_materialize.py:961-963` says "TMA's depth is ignored here (single-buffer)." A `d<depth>/tma` ring (depth>1)
   never materializes.

**Outcome:** scalar register-tile matmul (`Semiring·Scalar`, the `TILE` codec) stages A/B operands through smem
honoring `schedule.stage` (single-buffer + cp.async ring), and TMA staging gains a depth>1 ring. Both are pure perf
transforms — output stays bit-identical to gmem-direct.

**Decisions (from user):** scalar staging is scoped to **`Semiring·Scalar` only** (leave `Monoid·Scalar` on the
existing fused-prologue shared row). Deliver as **two PRs, scalar first, then the TMA ring.**

### What I verified during exploration (resolves a contradiction in the notes)

- `_reg_tile` (`010_materialize.py:249`) **does not read `schedule.stage`** — it replicates per-cell gmem loads with
  operand dedup, no smem slab.
- The "passing scalar staging tests" are misleading: `test_staged_scalar_matmul_matches_reference`
  (`test_matmul_coverage.py:243`) pins `DEPLODOCK_STAGE` to **legacy binmask** values (`"11"/"10"/"01"/"00"`), which
  `_stage_spec` (`020_schedule.py`) silently degrades to `""` (gmem-direct). It proves the register-tile matmul is
  correct, **not** that scalar smem staging materializes. `test_scalar_matmul_stages_through_pipeline` only asserts the
  *schedule struct* carries the `Stage`. So scalar staging genuinely does not exist yet.
- The IR already supports per-slot mbarrier arrays: `MbarrierInit`/`MbarrierArriveExpectTx`/`MbarrierWait`/`TmaLoad`
  all carry a `slot` / `mbar_slot` `Expr` and render `&mbar[slot]` (`ir/kernel/ir.py:456-561`). The TMA ring needs **no
  new IR** — only the materializer + `_stage.py` helpers, which hardcode `slot=0` today.
- `_stage.py`'s `CtaTile` seam + `cp_async_fill`/`slab_smem` are explicitly written "so the same helper drives both the
  warp (`_warp`) and the future scalar (`_reg_tile`) tiers" (`_stage.py:11-13`). Gap 2 is the intended reuse.

---

## PR 1 — Scalar operand staging (`Semiring·Scalar`)

Make `_reg_tile` honor `schedule.stage`: stage the CTA-wide A/B operand slabs into smem and drain with scalar smem
loads, reusing the `_stage.py` helpers. The register-tile already computes a CTA tile of `tile_m × tile_n` output cells
(`tile_m = par_m·reg_m`, `tile_n = par_n·reg_n`) over a reduce axis K — exactly the warp tier's slab geometry, drained
by scalar register loads instead of `ldmatrix`.

### Files

- **`deplodock/compiler/pipeline/passes/lowering/kernel/010_materialize.py`** — `_reg_tile` (the materializer change).
- **`deplodock/compiler/pipeline/passes/lowering/kernel/_stage.py`** — reuse `cp_async_fill`, `cp_async_barrier`,
  `cp_async_commit`/`cp_async_wait`, `slab_smem`, `CtaTile` as-is (already tile-agnostic). No change expected; add a
  scalar-tile restage helper here if it grows beyond `_reg_tile`.
- **`tests/compiler/e2e/test_matmul_coverage.py`** — new accuracy + source tests.
- **`deplodock/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md`** — extend the STAGE section to say staging now
  drives the scalar register tile, not only the warp tier and the fused-prologue.

### Steps

1. **Eligibility gate** (`_can_stage_scalar`, mirror `_can_stage_warp` at `010_materialize.py:536`): `Semiring`,
   transport `sync` or `cp.async`, **static tile-divisible K**, canonical `A[m,k]` / `B[k,n]` (no B-transpose),
   reduce/free axes recognized. Otherwise stay gmem-direct (current path).
   - **K-chunk for the slab.** The scalar register tile's reduce loop is monolithic today; staging needs a K-slab chunk
     `BK` (analog of `WarpTile.bk`). Reuse the legacy `BK` ingest (the test already pins `DEPLODOCK_BK=16`) as the slab
     K-chunk: outer loop steps K by `BK`, fills the `tile_m × BK` (A) and `BK × tile_n` (B) slabs, inner reduce over
     `BK` reads smem. Default `BK = K` (whole-K single slab) when unpinned and it fits the smem cap.
2. **Build the `CtaTile` seam** from the scalar block axes + a linear thread id (`par_m·par_n` threads). Mirror the warp
   tier's `linear_tid` construction (`010_materialize.py:699`) but off the scalar block/thread vars.
3. **Stage the slabs.** In `_reg_tile`, when `_can_stage_scalar`:
   - allocate `_a_smem` (`tile_m × BK`) and `_b_smem` (`BK × tile_n`) via `slab_smem`;
   - single-buffer (`d1/sync`): cooperative `cp_async_fill` (sync degrades to a plain copy+`Sync`; confirm the `sync`
     transport path in `cp_async_fill`/`cp_async_barrier`) → `__syncthreads` → drain;
   - cp.async ring (`d<depth>/cp`): mirror `_warp_staged_kloop`'s `depth≥2` ring (prologue primes `ring-1` chunks,
     prefetch `depth-1` ahead, `cp_async_wait(group=ring-1)`, clamped tail). Reuse the same row-offset slot scheme.
4. **Restage the per-cell loads.** Rewrite each replicated scalar `A[m,k]` load to `_a_smem[local_m][k_in_chunk]` and
   `B[k,n]` to `_b_smem[k_in_chunk][local_n]`, where `local_m`/`local_n` are the thread's cell offsets within the CTA
   tile. This is the 2-D analog of the existing `_restage_loads` (`010_materialize.py:347`, currently 1-D shared row) —
   generalize it or write a sibling for the matmul slab.
5. **Smem-cap clamp** (mirror `010_materialize.py:962-965`): clamp ring depth to the 48 KB static cap from the slot
   byte size; fall back to a shallower ring.
6. **Non-goals (state in the PR):** no `/p<n>` register double-buffer for scalar (scalar reuses via per-cell dedup, not
   `ldmatrix` fragments); no TMA for scalar; `Monoid·Scalar` stays on the fused-prologue.

### Verify (PR 1)

- **Bit-for-bit:** a new `test_scalar_staged_matches_gmem_direct_bit_for_bit` (mirror
  `test_staged_matches_gmem_direct_bit_for_bit:` the warp version) — pin `STAGE=d1/sync` then `d2/cp`, assert
  bit-identical to the unpinned gmem-direct baseline, and assert `_a_smem`/`_b_smem` + `cp.async` appear in source for
  the `cp` case. Replace the misleading binmask `test_staged_scalar_matmul_matches_reference` with real `d1/sync` /
  `d<depth>/cp` codec values (keep its masked-axis shape coverage).
- **Accuracy on GPU** (`@requires_cuda`, sm_80+): the staged scalar matmul matches `a @ b` across the masked / divisor
  shapes already parametrized.
- `make test` green; `make lint` clean. Watch the xfail registry stays empty (no regressions).

---

## PR 2 — TMA ring (depth>1)

Turn `_warp_tma_staged_kloop` from single-buffer into a depth-D ring, structurally mirroring the landed cp.async ring
(`_warp_staged_kloop` depth≥2 branch), swapping the cp.async commit/wait handshake for per-slot mbarriers.

### Files

- **`deplodock/compiler/pipeline/passes/lowering/kernel/010_materialize.py`** — `_warp_tma_staged_kloop` (the ring) +
  the `_warp` dispatch (`:956-965`, stop ignoring TMA depth).
- **`deplodock/compiler/pipeline/passes/lowering/kernel/_stage.py`** — split `tma_fill` into `tma_issue` (producer:
  `arrive.expect_tx` + box `TmaLoad`s onto a chosen mbarrier slot) and `tma_wait` (consumer: `MbarrierWait`), and
  extend `tma_mbar_prologue` to init D slots. Keep a single-buffer convenience wrapper.
- **`tests/compiler/e2e/test_matmul_coverage.py`** — TMA ring tests.
- **`deplodock/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md`** — update the STAGE section: TMA is no longer
  single-buffer-only.

### Steps

1. **`_stage.py` helpers:**
   - `tma_mbar_prologue(mbar, tid0, slots=D)`: `Smem(mbar, extents=(D,))` + issuer-only `MbarrierInit(slot=s)` for each
     `s in 0..D-1` + `Sync`.
   - `tma_issue(loads, mbar, tid0, slot, total_bytes)`: `Cond(tid0, [MbarrierArriveExpectTx(slot), *TmaLoad(mbar_slot=slot)])`.
   - `tma_wait(mbar, slot, phase)`: `[MbarrierWait(mbar, phase, slot)]`.
   - `tma_fill` stays as `tma_issue` + `tma_wait` at `slot=0` for the depth-1 path.
2. **`_warp_tma_staged_kloop(depth=…)`:** `depth==1` keeps today's single-buffer body. `depth≥2`:
   - D-slot slabs (`D·tile_m × bk_elems` A, `D·bk_elems × tile_n` B), 128 B-aligned (`TMA_SLAB_ALIGN`); D-slot mbarrier.
   - **Prologue:** issuer `tma_issue`s the first `D-1` chunks into slots `0..D-2` (each onto `mbar[s]`).
   - **Main loop step `i`** (mirror `_warp_staged_kloop:792-811`): `tma_issue` the prefetch chunk `i+D-1` (clamped tail,
     reuse the `k0_pref` ternary) into slot `(i+D-1)%D`; `tma_wait(slot=i%D, phase=(i//D)%2)`; drain slot `i%D` via
     `_staged_inner_atom_loop` with `a_slab_off`/`b_slab_off` = slot row offsets; trailing **`Sync()`** (the
     empty-slot proxy — guarantees all consumers finished reading slot `i%D` before a later step's `tma_issue`
     overwrites it, exactly as the cp.async ring's trailing `Sync`).
   - **Phase rule:** mbarrier slot `s` toggles every D chunks, so chunk `i` waits on slot `i%D` at phase `(i//D)%2`
     (single-buffer's `(k0/bk)%2` is the D=1 special case). The clamped tail still issues one `arrive.expect_tx` per
     step, keeping the phase count uniform across all CTA threads (barrier-under-mask invariant).
3. **`_warp` dispatch (`:956-965`):** compute `tma_depth = min(stage.depth, 48KB-cap)` for the TMA slabs (same
   `_slot_bytes` clamp, but the slot includes 128 B alignment rounding); pass `depth=tma_depth` to
   `_warp_tma_staged_kloop`. Remove the "TMA's depth is ignored" restriction. Composes with `reg_depth` (`/p<n>`,
   already threaded through `_staged_inner_atom_loop`).
4. **Note (not a blocker):** deep TMA rings may exceed the 48 KB static smem cap; mirror cp.async's clamp (fall back to
   a shallower ring) rather than reaching for dynamic smem now.

### Verify (PR 2)

- **Source (CPU render, force sm_90):** extend `test_pinned_transport_and_shape_fire:` with `STAGE=d3/tma` — assert
  `cp.async.bulk.tensor` + `CUtensorMap` still fire AND a D-slot mbarrier array (`mbar[`) + D-slot slab appear.
- **Bit-for-bit:** `test_tma_deep_ring_matches_gmem_direct_bit_for_bit` (mirror
  `test_cp_async_deep_ring_matches_gmem_direct_bit_for_bit:`) — `d3/tma` output bit-identical to gmem-direct.
- **Accuracy on GPU** (sm_90+, the `test_static_dynamic_mma_parity` TMA path): `d2/tma` and `d3/tma` match torch.
- `make test` + `make lint` green; xfail registry stays empty.

---

## Cross-cutting

- **Docs to update before each commit** (per CLAUDE.md contribution steps): the kernel `ARCHITECTURE.md` STAGE section,
  and the **status paragraph in `plans/tile-ir-rebuild.md`** — move "TMA ring" and "scalar-tier operand staging" out of
  the remaining-gaps list as each PR lands. (Do not reference `plans/*.md` from durable docs/code.)
- **No schedule/codec changes needed:** `Stage`, the `STAGE` codec, `_stage_spec`, and the featurizer already carry
  `depth`/`transport`/`ring`. Both PRs are materializer + helper + test changes only.
- **Auto-fork is out of scope:** both stay pin-/prior-driven (a cold greedy compile won't auto-propose staging),
  consistent with the rest of the warp tier.
