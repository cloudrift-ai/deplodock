# Composer-only green suite — phased plan to fix the remaining 69 failures

**Branch:** `feature/move-composer` **Status:** in progress — **10 / 69 green**. The legacy partition planner and the
`DEPLODOCK_MOVE_COMPOSER` feature flag are **deleted** — the move composer (`passes/lowering/tile/partition/`) is the
sole, unconditional, strict partitioner (a kernel it can't lower raises). Every remaining failure is therefore a
**composer-completion gap** (each passed on the now-deleted legacy planner). This plan itemizes all 69 and groups them
into 6 phases by root cause + shared machinery, ordered so foundational codegen lands before the search/ranking phases
that depend on it.

Discipline (per the sibling plans): each phase lands behind a green per-kernel `deplodock compile` compare under a fixed
`PYTHONHASHSEED` + the named tests; no phase may regress a passing test. Run the masked-warp phases in isolation first
(a misaligned-store / OOB fault wedges xdist workers) before the full `-n auto` sweep.

Counts below sum to 69. Tests are written `file::test` (parametrize ids collapsed as `[…]`). Each test line is
tagged ✅ (green) / ❌ (red) as of the last sweep (`PYTHONHASHSEED=0`, sm_120 host).

## Current state (10 / 69 green)

| Phase | Theme | Green / Total |
| ----- | ----- | ------------- |
| 1     | Symbolic free-axis masked `mma.sync` (store guard + operand clamp) | **7 / 7 ✅ DONE** |
| 2     | Masked-K `mma.sync` tile (symbolic reduce)                         | 0 / 11 |
| 3     | TMA descriptor for masked / symbolic tiles                         | 1 / 17 |
| 4     | Composer regime coverage (declined shapes)                         | 2 / 5  |
| 5     | Scalar / transport codegen details + masked cooperative clamp      | 0 / 6  |
| 6a    | Structural-fork search integration                                 | 0 / 13 |
| 6b    | TMA / warp-specialize gate + cold-prior pick                       | 0 / 10 |

**Landed:** Phase 1 in full (`2178b911` — masked warp tier for symbolic M/N: the warp builder emits the boundary
`Cond` + the per-role masked structural feature, `021`/`005` clamp + guard, the tree gate lets symbolic-N reach the warp
tier). Phase 4's **global-reduce** half (`673d2e5b` — `iter_dag` synthesizes a degenerate size-1 row for a no-free-axis
reduction, routing it through the cooperative-reduce regime). As a Phase-1 side effect,
`test_demoted_symbolic_n_b_operand_reaches_tma_and_warpspec` (Phase 3) also went green. Separately, the `OVERHANG`
tuning knob was replaced by per-role `S_masked_{m,n,k}` structural features (`7dc5adcd`) — byte-identical codegen, zero
test movement, but it renames what the masked tests assert (reflected below).

**Attempted + reverted:** Phase 4's **fused-prologue** half (`test_fused_rmsnorm_linear_blocked_prologue` + the two
`test_norm_linear_fp16_scalar_reduce_tma_alignment`). A working fused-prologue matmul regime (prologue detection in
`iter_dag`, a dedicated `M_r`-serial builder, M/N pin-honoring in the matmul offers) made all three pass, but it is
**entangled with Phase 6a**: making `classify` recognize the fused-prologue shape changes `005_split_demoted`'s
keep-vs-split structural offer, which regressed `test_outer_descends_prior_preferred_branch_first` (a then-passing 6a
test) and the broad pin-honoring change wedged the full `-n auto` sweep. Per the plan's own ordering (6 last, presupposes
every shape lowers), these three move to **be done with Phase 6a** — 005 must offer the keep-vs-split as a real fork so
the fused regime composes instead of suppressing the split.

---

## Phase 1 — Symbolic free-axis masked `mma.sync` (store guard + operand clamp) — 7 tests ✅ DONE (`2178b911`)

The composer's warp tier covers a **symbolic M** axis (accuracy passes) but only because its unguarded epilogue stores
land in benign OOB memory; a **symbolic N** (inner, contiguous) axis corrupts adjacent rows and misaligns the vectorized
store. The fix is one mechanism: generate the per-cell **store guard** + **operand runtime-extent clamp** for any masked
warp output axis, and make the `__half2` store alignment-safe.

- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype0-128]`
- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype0-130]`
- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype0-200]`
- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype1-128]`
- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype1-130]`
- ✅ `test_matmul_mma_transposed_b.py::test_transposed_b_mma_symbolic_mn[out_dtype1-200]`
- ✅ `test_matmul_mma_masked.py::test_symbolic_m_masked_mma_kernel_structure`

**As-built:** the actual landed fix was smaller than the three coupled fixes anticipated below — emitting the boundary
`Cond` in `build_warp_matmul_tile` (mirroring the scalar `_assemble`) re-used the existing 021 clamp + 005 guard
machinery, the `n_guard` already forces per-element scalar stores (dodging the `__half2` alignment hazard), and the gate
flip (`scalar_only` drops the `inner_n.symbolic` term) let symbolic-N reach the warp tier. The masked axes now surface as
the `S_masked_{m,n}` structural feature (the `OVERHANG` knob was retired in `7dc5adcd`), which is what the structure test
asserts. Original three-fix analysis kept for reference:

1. **`__half2`-over-odd-`ldm` store** (`ir/kernel/ir.py::RegStore.render`): the C-fragment row-pair `*reinterpret_cast<
   __half2*>(&o[base + _g*ldm + _t*2])` is 4-byte aligned only when `ldm` is a static **even** int; a symbolic `ldm`
   (the `(seq, seq)` QK^T output) is odd at runtime → `MISALIGNED_ADDRESS`. Gate both vectorized sites (unguarded +
   `m_guard`-only) on `_vec2_store_safe(ldm) = isinstance(ldm, int) and ldm % 2 == 0`, else use the existing per-element
   scalar fallback. Byte-identical for static even `ldm` (verified).
2. **Missing `n_guard` / `m_guard`** (`materialize._warp_axis` + `materialize.build_warp_matmul_tile` +
   `kernel/005_lower_atom_tile`): `_warp_axis` returns a `bound` (the symbolic extent `Expr`) for a masked axis but
   `build_warp_matmul_tile` **discards** it (`_n_bound` / `_m_bound`), and `real_extent` is only set for non-divisible
   *static* axes (`None` for symbolic). Propagate the symbolic bound onto the block axis (a `real_extent` analogue) so
   `005_lower_atom_tile` stamps the per-cell `RegStore` `m_guard` / `n_guard` (`base + _g (+8) < bound` per row,
   `base + _t*2 (+1) < bound` per col) — the same guard the masked-M structure test expects.
3. **Unclamped staged B/A operand** (`020_stage_inputs`): the staged smem-copy load
   `kT[k*seq_len + (a1*16 + n)]` reads N columns past `seq_len` at the boundary block (no `cols_left` clamp; the
   gmem-direct loaders have `dpl_mma_load_b_gmem_nclamp` but the staged path doesn't). Clamp the symbolic-N column (and
   the symbolic-M / masked-K row) in the staged slab fill via `real_extent`, mirroring the gmem-direct clamp.

Then flip the gate: `tree.build_matmul_tree`'s `scalar_only = dag.inner_n.symbolic or n_reduce > 1` → drop the
`inner_n.symbolic` term so symbolic-N reaches the warp tier.

---

## Phase 2 — Masked-K `mma.sync` tile (symbolic reduce) — 11 tests ❌ (0 / 11)

A matmul whose **reduce (K) axis** is symbolic (SDPA P@V over `seq_len`) must tile K at the hint and **zero-fill** the
final partial K slab (clamp the gmem read, force the loaded value to 0 past the runtime extent — a clamped duplicate
would corrupt the sum). The composer's matmul `classify` currently requires a **static** K (`if not k_dim.is_static:
return None`), so these decline. The masked-K helpers (`dpl_mma_load_*_kzero`) already exist; the gaps are the composer
admitting a symbolic-K matmul + the **batched** B operand (`xnb[head, k, n]`, K after a leading batch axis).

- ❌ `test_matmul_mma_masked.py::test_symbolic_k_masked_mma_accuracy[16|31|130|512|700]` (5)
- ❌ `test_matmul_mma_masked.py::test_batched_symbolic_mk_masked_mma_accuracy[16|31|130|512|700]` (5)
- ❌ `test_matmul_mma_masked.py::test_batched_symbolic_mk_reaches_warp`

Approach: relax `classify`'s matmul static-K requirement to admit a symbolic K (tile at the hint; `k_bound` = runtime
extent); thread the masked-K zero-fill through `_replace_k_warp` (the K-tower analogue of the scalar `_replace_k_scalar`
masked-K path); teach `classify_matmul_operands` the batched B layout (one var dim — the N output — follows K after a
leading batch axis). Depends on Phase 1's operand-clamp machinery.

---

## Phase 3 — TMA descriptor for masked / symbolic tiles — 17 tests ❌ (1 / 17)

The composer's masked warp staging is **not TMA-eligible** (a pinned `TMA=1` raises "no BUFFERED/ASYNC StageBundle …
eligible for TMA"), and the static-N TMA path mis-aligns the swizzle box. The TMA box `globalDim` must be the
**runtime** extent (not the hint) so `cp.async.bulk.tensor` zero-fills the masked overhang instead of reading past the
buffer. This
is the largest single body of new codegen (`050_use_tma` + the box descriptor) and depends on Phases 1–2.

- ❌ `test_matmul_mma_masked.py::test_symbolic_m_masked_mma_tma_accuracy[1|31|512|700]` (4)
- ❌ `test_matmul_mma_masked.py::test_symbolic_m_masked_mma_tma_structure`
- ❌ `test_matmul_mma_masked.py::test_demoted_masked_k_pv_reaches_tma`
- ❌ `test_matmul_mma_masked.py::test_demoted_masked_k_pv_tma_accuracy[16|31|130|512|700]` (5)
- ✅ `test_matmul_mma_masked.py::test_demoted_symbolic_n_b_operand_reaches_tma_and_warpspec` (went green with Phase 1)
- ❌ `test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[dynamic-tma]`
- ❌ `test_matmul_mma_parity.py::test_static_dynamic_mma_parity[dynamic-tma-256|512]` (2)
- ❌ `test_matmul_mma_tma.py::test_mma_sync_matches_reference[static-128-256-128-out_dtype1]`
- ❌ `test_matmul_mma_tma.py::test_tma_swizzle_smem_aligns_to_atom`

Approach: make the composer's warp staging emit a TMA-eligible `StageBundle`; build the descriptor's `globalDim` per
launch from the runtime shape; align the swizzle smem box to the atom. Hold the masked-warp staging to the same box math
`050_use_tma` re-derives (disagreement → deadlock).

---

## Phase 4 — Composer regime coverage (declined shapes) — 5 tests ⚠️ (2 / 5 — global-reduce done; fused-prologue → 6a)

Shapes whose `LoopOp` the composer's `classify` returns `None` for → hard error (empty kernel). Add the regime.

- ✅ `test_dtype_cuda.py::test_fp16_reduction_uses_fp32_accumulator_on_cuda` — **global reduction** `x[1024] → s[1]`: a
  single reduce loop with **no free axis**, so `dag.parallel` is empty and `classify` declines. **Done** (`673d2e5b`):
  `iter_dag` synthesizes a degenerate size-1 PARALLEL row → the cooperative-reduce regime tiles it as a single-CTA
  tree reduce.
- ✅ `test_dtype_cuda.py::test_fp16_max_reduction_stays_in_fp16` — same global-reduce shape, `max` carrier. **Done**.
- ❌ `test_lowering_blocked_gemm.py::test_fused_rmsnorm_linear_blocked_prologue` — **fused norm+linear prologue**: a matmul
  whose operand is produced by a prologue REDUCE (RMSNorm) in the same nest — the composer has no fused-prologue matmul
  regime (its matmul `classify` rejects a map/reduce-loop prologue). Either add the regime or have `005_split_demoted`
  cut it. **A working regime was built + reverted** — it's entangled with 6a's `005_split_demoted` structural offer
  (recognizing the fused shape suppresses the split fork the 6a tests expect, and regressed `test_outer_descends…`).
  **Moved to Phase 6a**: 005 must offer keep-vs-split as a real fork so the fused regime composes alongside the split.
- ❌ `test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment[static|dynamic]` (2) — same fused norm+linear
  class (also needs the regime + M/N pin-honoring in the matmul offers). **Moved to Phase 6a** with the above.

The global-reduce half is independent of Phases 1–3 and landed early; the fused-prologue half is not independent of 6a.

---

## Phase 5 — Scalar / transport codegen details + masked cooperative clamp — 6 tests ❌ (0 / 6)

Smaller, localized codegen gaps in the composer's scalar matmul + cooperative-reduce towers and the transport-gate
passes reading them.

- ❌ `test_run.py::test_compile_fp16_matmul_window_emits_half2` — the scalar fp16 matmul **`FK` half2 window** (`MMA=0`
  vectorized-K) codegen isn't emitted by the composer scalar tier.
- ❌ `test_matmul_mma_residual.py::test_epilogue_warp_rows_stay_splitk_one` — a fused-residual (`matmul_add`) warp row must
  force `SPLITK=1`; verify the composer's warp epilogue path stamps it (the scalar path already does via
  `matmul_reduce_offers`' `has_epilogue` gate).
- ❌ `test_masked_tile.py::test_masked_n_clamps_cooperative_load_index` — masked-N **cooperative** reduce load must clamp
  the N index to the runtime extent (the coop-reduce analogue of Phase 1's operand clamp).
- ❌ `test_masked_tile.py::test_symbolic_m_cooperative_load_clamps_to_runtime_extent` — symbolic-M cooperative load clamp.
- ❌ `test_ring_buffer_fp16_smem.py::test_fp16_ring_buffer_rejects_when_real_bytes_overflow` — `040_use_ring_buffers` must
  decline a ring slot whose real (fp16) bytes overflow, reading the composer's tower.
- ❌ `test_tma_smem_alignment.py::test_fp16_subaligned_ring_slot_declines_tma_fp32_keeps_it` — TMA/ring smem-alignment gate
  on the composer's staged tower.

---

## Phase 6 — Structural-fork search integration + cold-prior ranking — 23 tests ❌ (0 / 23)

The two-level tune + greedy structural pick + the cold `AnalyticPrior` were tuned against the deleted legacy planner.
With the composer as the sole partitioner, the structural offers (`005_split_demoted`), the kernel-set Σ costs, and the
cold pick must re-integrate. Two sub-groups; depends on Phases 1–5 (a structural pick can't deploy a kernel the composer
can't yet lower — `falls_back_on_lowering_failure`). **The reverted Phase-4 fused-prologue regime + its three tests
(`test_fused_rmsnorm_linear_blocked_prologue`, `test_norm_linear_fp16_scalar_reduce_tma_alignment[static|dynamic]`) land
here**: `005_split_demoted` must offer keep-vs-split as a real structural fork (today it returns a single option), so the
fused regime can recognize the shape *without* suppressing the split the 6a tests assert.

**6a — structural-fork search (`005_split_demoted` + two-level + replay):**
- ❌ `test_split_demoted.py::test_rule_offers_fused_first_then_split`
- ❌ `test_split_demoted.py::test_rule_knob_guard_skips_reconsider`
- ❌ `test_split_demoted.py::test_greedy_compile_keeps_fused_kernel`
- ❌ `test_split_demoted.py::test_greedy_trained_prior_deploys_split`
- ❌ `test_split_demoted.py::test_greedy_cold_stub_prior_keeps_fused`
- ❌ `test_split_demoted.py::test_greedy_structural_pick_falls_back_on_lowering_failure`
- ❌ `test_split_demoted.py::test_tune_explores_fused_and_split_terminals`
- ❌ `test_two_level.py::test_decomposition_rows_sum_kernel_set_costs`
- ❌ `test_two_level.py::test_identical_offer_sites_take_the_same_side`
- ❌ `test_two_level.py::test_outer_branches_on_structural_fork`
- ❌ `test_two_level.py::test_split_kernels_attribute_to_pre_decision_op`
- ❌ `test_structural_push.py::test_split_demoted_fork_pushes_structural`
- ❌ `test_resolve.py::test_structural_replay_consulted`

(Watch for regressions to `test_two_level.py::test_outer_descends_prior_preferred_branch_first`, currently green — the
fused-prologue regime tripped it; the keep-vs-split fork must keep it passing.)

**6b — TMA / warp-specialize gate + cold-prior pick over composer towers:**
- ❌ `test_use_tma_gates.py::test_oversized_box_declines_tma`
- ❌ `test_use_tma_gates.py::test_oversized_box_pinned_tma_raises`
- ❌ `test_use_tma_gates.py::test_reentered_pipeline_declines_tma`
- ❌ `test_use_tma_gates.py::test_reentered_pipeline_pinned_tma_raises`
- ❌ `test_use_tma_gates.py::test_single_trip_cell_loop_keeps_tma`
- ❌ `test_use_tma_gates.py::test_hang_knob_family_completes_and_matches`
- ❌ `test_warp_specialize_deadlock.py::test_mlp_slice_completes_and_matches`
- ❌ `test_warp_specialize_deadlock.py::test_mlp_slice_never_offers_ws1`
- ❌ `test_warp_specialize_deadlock.py::test_pinned_ws1_on_mlp_slice_raises`
- ❌ `test_masked_tile.py::test_planner_masks_symbolic_m_axis_at_hint` — the cold `AnalyticPrior` ranks a degenerate `BM=1`
  whole-axis bind above the real hint-sized masked tile; needs offer-order / prior tuning so the cold pick tiles the
  symbolic axis at the hint.

---

## Suggested order & gating

1. ~~**Phase 4** (regime coverage)~~ — global-reduce **done**; fused-prologue split out to Phase 6a (see below).
2. ~~**Phase 1** (symbolic free-axis store guard + clamp)~~ — **done** (`2178b911`).
3. **Phase 2** (masked-K) → **Phase 3** (TMA) — build on Phase 1's clamp/guard. **Next up.**
4. **Phase 5** (scalar/transport details + coop clamp) — small, parallelizable with 2–3.
5. **Phase 6** (search + cold-prior) — last; the structural picks and cold ranking presuppose every shape lowers. Now
   **also carries the 3 fused-prologue tests** from Phase 4: do the `005_split_demoted` keep-vs-split fork rework first,
   then re-land the fused-prologue regime (its diff is preserved in the `7dc5adcd`-era history / this session's reverts)
   on top of the fork so it composes without suppressing the split.

Remaining: **59 / 69 red**. Next-up is Phase 2 → 3 (the masked-K + TMA codegen body), then Phase 5, then Phase 6.

Each phase is its own PR-sized unit with the listed tests as the acceptance gate; the full `-n auto --dist=loadgroup`
sweep stays green-or-better after each.
