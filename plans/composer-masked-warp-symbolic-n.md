# Composer masked warp tier — symbolic-N (and the remaining masked-warp gaps)

**Branch:** `feature/move-composer` **Status:** investigated, not landed. Scopes the masked warp-tier MMA work the cutover
deferred. Symbolic-**M** (outer / row axis) is now landed and verified (`tree.build_matmul_tree`:
`scalar_only = skel.inner_n.symbolic or n_reduce > 1` — symbolic-M reaches the warp tier, symbolic-N stays scalar). This
note records the concrete root causes found for the symbolic-**N** (inner / contiguous axis) crash so the follow-up is a
scoped fix rather than a re-investigation.

## Why symbolic-M works but symbolic-N faults

The composer's masked warp tile (`_warp_axis` ceil-div + `real_extent`; `020_stage_inputs` clamp + `005_lower_atom_tile`
per-cell store guard) is verified-correct for a symbolic **M** axis: accuracy passes at runtime sizes 1 / 31 / 512 / 700,
and the staging is not TMA-eligible so the TMA box-overrun fault can't fire (a pinned `TMA=1` raises cleanly). The
difference is the **output row stride**: for symbolic-M the output is `[seq_len, N]` with `N` static (even, a multiple of
`atom_n`), so the leading dim is aligned; for symbolic-N (e.g. the rotary `Q @ K^T` scores `[seq, seq]`) the output is
`[seq_len, seq_len]` and the row stride **is** `seq_len` — odd at many runtime sizes.

## The three coupled root causes (a symbolic-N `Q@K^T` repro: `q[seq,128] @ kT[128,seq] → o[seq,seq]`)

1. **`__half2` C-fragment store over an odd row stride (`MISALIGNED_ADDRESS`).** `RegStore.render`
   (`ir/kernel/ir.py`) stores each mma row-pair as one vectorized `__half2` (`*reinterpret_cast<__half2*>(&o[base +
   _g*ldm + _t*2])`). The comment asserts "base is tile-aligned and `2t` is even, so the pair is 4-byte aligned" — true
   only when `ldm` is a static **even** int. With `ldm = seq_len` (a symbolic C expression), an **odd** `seq_len` and an
   odd `_g` make `_g*ldm` odd → the `__half2` store is 2-byte aligned → fault. **Fix (verified to remove this fault):** a
   `_vec2_store_safe(ldm)` guard (`isinstance(ldm, int) and ldm % 2 == 0`) gating both the unguarded and the
   `m_guard`-only vectorized paths onto the existing per-element scalar fallback. Byte-identical for static/even `ldm`.
   This was found, implemented, and confirmed to clear the store fault, then reverted — it has no live beneficiary until
   the other two causes below are also fixed (symbolic-N stays gated to scalar), so it is not committed on its own.

2. **The symbolic-N output store is not boundary-guarded.** After fix (1) the store is scalar but still unguarded
   (`o[(row)*seq_len + col + _g*seq_len + _t*2 + i] = …` with no enclosing per-element `if`), so a tile straddling the
   runtime extent writes past `o`. The symbolic-M path emits the `RegStore` `m_guard`; the symbolic-N path does not emit
   the `n_guard` (the per-column `base + _t*2 (+1) < bound` predicate). Root cause is in how `_warp_axis` / the guard
   plumbing derives the N boundary for a symbolic inner axis — it produces the masked grid (ceil-div) but loses the
   per-cell store guard.

3. **B-operand N-clamp is missing (out-of-bounds gmem read).** The `kT_smem` staging load
   (`kT[k*seq_len + (a1*16 + n)]`) reads N columns past `seq_len` at the boundary block — there is no `cols_left` clamp on
   the staged B slab (the gmem-direct fragment loaders DO have `dpl_mma_load_b_gmem_nclamp`, but the **staged** smem-copy
   path used here does not clamp the N column). `020_stage_inputs`'s `real_extent` clamp covers the M / K slab fill but
   not the symbolic-N column of the B operand.

## Scope of the follow-up

Landing symbolic-N is causes (1) + (2) + (3) together (each alone leaves a fault), then flipping the gate to
`scalar_only = n_reduce > 1`. The remaining masked-warp test groups beyond symbolic-N are separate, larger efforts and
stay deferred: the **TMA** path (box `globalDim` must be the runtime extent, not the hint — the composer's masked warp
staging isn't TMA-eligible at all today), the **masked-K** mma tile (symbolic reduce), the **batched** P@V split
consumer, and the **demoted** (`005_split_demoted`) symbolic-N / masked-K-PV consumers under TMA. All of these pass on
the legacy planner (composer off), so the gap is composer coverage, not algorithm.
