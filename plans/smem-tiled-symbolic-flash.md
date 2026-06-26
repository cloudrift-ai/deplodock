# Plan — smem-tiled / tensor-core flash for a **symbolic** `seq_len`

**Status:** Phases 1–3 landed + Phase 4 causal (symbolic S; equal-head + GQA; causal+symbolic mask; **warp chain is the
deployed symbolic default — the ~100× perf win, measured**). Remaining: Phase 4 additive-mask, Phase 5 search
integration (both deferred — neither is the layer-0 win). **Goal:** make symbolic-`seq_len` SDPA fast — the last wall on
the
Qwen3-Embedding-0.6B layer-0 e2e (and any dynamic-shape attention model). Companion to
[`plans/qwen3-embedding-0.6b-layer0-tune-findings.md`](qwen3-embedding-0.6b-layer0-tune-findings.md) (the autonomous
e2e-perf session that diagnosed the wall) and the static fused-TC flash this extends,
[`plans/tensor-core-streaming-flash-mma.md`](tensor-core-streaming-flash-mma.md).

## 1. Why this is the wall (measured)

At the deployment length (seq=512), nsys ground truth on the layer-0 forward:

```
k_sdpa_linear_reduce   8.16 ms/call   <-- ~90% of the 8.8 ms layer (eager whole layer = 224 µs)
k_linear_0837e7        0.21 ms/call
… everything else      2–40 µs
```

The symbolic SDPA deploys today as the **scalar serial-KV streaming flash** (`enumeration/070_coop_reduce` +
`_build.chain_build`, the FA-2 shared-score nest): each query *thread* streams all `seq` keys, **re-reading K/V from
gmem per query** with no smem tile sharing, at low occupancy (block 8 / 168 regs). That is O(seq²) and memory-bound —
~8 ms at seq=512, captured **and** uncaptured (there is no CUDA-graph capture penalty; that was a seq=32-vs-512
confound). The streaming flash is the **only correct** symbolic SDPA today — the materialized QK^T→softmax→P@V
decomposition is wrong for `seq < hint` (the masked-N QK^T zero-fills padding keys, but softmax needs −inf — `exp(0)=1`
corrupts the denominator; this is why `010_recognize_flash` FORCES flash for symbolic). So we must make the *flash
itself* fast, not route around it.

## 2. The vehicle already exists — for STATIC shapes

`assembly/_warp_chain.py::build_warp_chain_tileop` (driven by `split/005_warp_chain.py` under `DEPLODOCK_CHAIN=1`) is
the **fused tensor-core flash**: `GridTile > WarpTile > [init] SerialTile[kv] [epilogue]`, QK^T `mma` → fragment
online-softmax (`_frag_softmax.realize_fragment_softmax`) → C→A smem handoff → P@V `mma`. It tiles the queries 16-wide
(`qb` grid) and streams the KV in 16-wide tiles (`kv` SerialTile over `S//16`); within a query tile the `ldmatrix`'d KV
tile is reused across all 16 query rows by the `mma` — **exactly the K/V sharing the scalar path lacks**. It already
handles causal (`_frag_softmax.realize_score_mask` masks score columns `kv_col > q_row` to −1e30) and fp16/bf16. It is
validated by `tests/compiler/e2e/test_flash_tensorcore_generated.py` and the `*_matches_torch` flash tests under
`DEPLODOCK_CHAIN=1`.

**What blocks our case** (`warp_chain_eligible(... symbolic, mask, group ...)`):

```python
return not symbolic and not mask and group == 1 and D % 16 == 0 and S % 16 == 0 and 16 <= D <= 256 and S >= 16
```

1. `not symbolic` — the KV-stream count and query-tile grid are static `S // 16`.
2. `group == 1` — Qwen3-Embedding is GQA (16 q-heads / 8 kv-heads, group 2).
3. CHAIN-pinned only (`split/005_warp_chain._chain_pinned`) — never the default.

`S % 16 == 0` (the partial-tile gap) is also implied by the symbolic case — a runtime `seq` is rarely a multiple of 16.

## 3. Correctness requirements for a symbolic KV/query axis

These are the load-bearing parts — get them wrong and embeddings are silently incorrect for `seq` not a multiple of 16
(the `*_dynamic_matches_torch` tests at seq=8/16/37 are the oracle; they already caught the materialized-path bug).

- **KV tiling → runtime ceil-div.** The `kv` SerialTile bound becomes `ceil(seq_len / 16)`; `seq_len` enters the kernel
  signature (like every other symbolic kernel — `int seq_len`). The query-tile `qb` grid becomes `ceil(seq_len / 16)`.
- **Score boundary mask (the crux).** The final partial KV tile has columns `kv_col >= seq_len`. Their score must be
  masked to **−1e30 BEFORE the rowmax / exp** so the online-softmax denominator excludes them — identical mechanism to
  the **causal** mask, just a different column predicate (`kv_col >= seq_len` instead of `kv_col > q_row`). Reuse
  `_frag_softmax.realize_score_mask` with a boundary predicate (compose with causal when both apply — AND the
  predicates). **Zero-fill is wrong here** (the masked-K matmul's `(k<seq_len)?v:0` is correct for a *sum* reduce but
  not for softmax); the score is the QK^T mma **output**, masked at the fragment, exactly as causal does.
- **K/V gmem loads for the partial tile.** The `ldmatrix` / `dpl_mma_load_b_gmem_*` of the final KV tile must not read
  past `seq_len`. Reuse the masked-K clamped-load helpers (`kernel/005`'s `dpl_mma_load_*_kzero` / the `_stage_expand`
  clamp) — clamp the row index; the loaded garbage is harmless because the score boundary mask sets those columns to
  −inf regardless. (Q loads for the partial query tile clamp the same way; their output store is guarded — next.)
- **Query-row store guard.** The final partial query tile (`q_row >= seq_len`) must not write its `O[d]` epilogue.
  Reuse the warp-tier **masked-M** store guard (`_warp_axis` `real_extent` → boundary `Cond(σ(M) < seq_len)`), the same
  one the masked matmul uses.
- **`l_i == 0` guard.** A fully-masked row (only an out-of-range query tile, already store-guarded) must not divide by
  zero in `O_i / l_i`; the store guard covers it, but assert no NaN in the test.

All four reuse existing masked-tile machinery (`_warp_axis` masked-M, `realize_score_mask` causal→boundary,
`dpl_mma_load_*` clamp) — the work is **composing** them in the warp-chain assembler, not inventing new masking.

## 4. GQA (`group > 1`)

K/V are indexed at `head // group` (Qwen3: kv-head = q-head // 2), reading the kv_heads-many K/V without materializing
the q-head expansion. `_flash.build_flash_frag` (the scalar path) already does exactly this; port the `head // group`
index into the warp-chain's K/V `Mma` operand indices (the `bh` grid axis decodes `(batch, head)`; the K/V load uses
`head // group`). No new masking — purely an index change on the K/V operands.

## 5. Phased implementation (each phase independently testable + commit-able)

**Phase 1 — symbolic S, non-causal, equal-head. ✅ LANDED.** `warp_chain_eligible` accepts `symbolic=True` (keeps
`group==1`, drops `S%16==0`); `build_warp_chain_tileop` ceil-divs the `kv`/`qb` extents over `Var(seq_var)`, uses
`seq_var·D` for the (b,h) row stride, masks the partial final KV tile's score columns (`realize_boundary_mask` →
`FragmentBoundaryMask`, `kv_col >= seq_len` → `-1e30` before the rowmax), stamps the operand load guards on the QK^T /
P@V `Mma`s (`m_guard`/`n_guard`/`k_zero` → `kernel/005` routes them to clamp Q rows / clamp K cols / zero-fill V rows),
and guards the partial query store (`RegStore.m_guard`). Oracle: `test_warp_chain_dynamic_matches_torch` (CHAIN=1, seq ∈
{8, 16, 37, 64}) — all green, NaN-free. CHAIN-pinned. Original plan text below.

**Phase 1 (original) — symbolic S, non-causal, equal-head.** Relax `warp_chain_eligible` to accept `symbolic=True` (keep
`group==1`, drop the `S%16==0` requirement when symbolic). In `build_warp_chain_tileop`: ceil-div the `kv`/`qb` extents
to `Var("seq_len")`-derived runtime bounds; add the score boundary mask (`kv_col >= seq_len`) via `realize_score_mask`;
clamp the partial-tile K/V loads; add the masked-M query store guard. Oracle: a new
`test_warp_chain_dynamic_matches_torch` mirroring `test_flash_sdpa_dynamic_matches_torch` but `CHAIN=1`, asserting
parity at seq ∈ {8, 16, 37, 64} (37 exercises the partial tile in BOTH the kv and query axes). Keep CHAIN-pinned for
now.

**Phase 2 — GQA. ✅ LANDED.** `build_warp_chain_tileop` reads K/V at the kv-head `bh_kv = batch·H_kv + head // group`
(Q/O keep the q-head `bh`; `group==1` collapses byte-identically), and `warp_chain_eligible` accepts `group > 1`
(`H % group == 0`). Oracle: `test_warp_chain_gqa_dynamic_matches_torch` + `…_gqa_static_…` (Hq/Hkv ∈ {4/2, 16/8}, seq ∈
{8, 16, 37, 64}). `_GqaSdpa` traces as GQA **and** causal (the public-API form), so the test also exercises the
**causal mask composed with the symbolic boundary mask** (both write `-1e30` before the rowmax — so the causal+symbolic
cross-product of Phase 4 is already covered for the masked-at-fragment case; only an *additive* mask remains Phase 4).
Note: relaxing `group` also routes a **static** GQA flash under `CHAIN=1` to the warp chain (previously scalar
`chain_build`) — verified by the static GQA oracle + the existing `test_flash_chain_causal_and_gqa_match_torch`.

**Phase 2 (original) — GQA.** Add `head // group` to the warp-chain K/V operand indices; accept `group > 1` in
`warp_chain_eligible`. Oracle: `test_warp_chain_gqa_dynamic` (16/8 heads, seq 8/16/37). This is the Qwen3 shape.

**Phase 3 — make it the symbolic default (the perf win lands). ✅ LANDED (the "Simplest" option).**
`split/005_warp_chain.rewrite` now fires the warp chain for an **eligible symbolic** flash regardless of the `CHAIN`
pin (CHAIN stays the *static* opt-in); a non-eligible symbolic flash (fp32, odd `D`, additive mask) falls through to the
scalar `chain_build`. **Measured** (RTX 5090, fp16, the Qwen3-Embedding attention shape `1×16×512×128`, seq=512, -O3,
CUDA-graph captured): deplodock warp chain **68 µs vs eager 39 µs (0.57×)**, max_diff 2.4e-4 vs eager — versus the old
scalar streaming nest's **~8 ms** at this shape: a **~100× kernel-level win**, landing the goal (Section 8's "~50–150 µs,
~0.2–0.4× eager"). NOTE: the `deplodock run --layer 0` path traces in **fp32**, so the Qwen3-Embedding layer-0 e2e keeps
the scalar `chain_build` for the SDPA (the warp chain is fp16/bf16) — the win lands at the fp16 **deployment** dtype, not
the fp32 accuracy-trace. The two-level OptionFork (the "Cleaner" option, also covering the static fork) is deferred to
Phase 5. Original plan text below.

**Phase 3 (original) — make it the symbolic default (the perf win lands).** Today symbolic SDPA → scalar streaming
flash by default (`070_coop_reduce`); the warp chain fires only under `CHAIN=1`. Route an **eligible** symbolic flash to
the warp chain by default, falling back to the scalar streaming nest when out of scope (odd D, additive mask until Phase
4, …):

- Simplest: in `split/005_warp_chain.rewrite`, fire when the nest is an eligible symbolic flash **regardless of the
  CHAIN pin** (CHAIN stays the *static* opt-in). The scalar streaming flash remains the fallback for non-eligible.
- Cleaner (preferred): make it a **two-level structural fork** — `split/005` offers `{warp_chain, scalar_stream}` as an
  `OptionFork` so the tuner picks per shape and greedy defaults to the warp chain when eligible (the `FLASH` knob's
  docstring already anticipates this "two-level OptionFork"). Greedy/cold default = warp chain (eligible) else scalar.

Re-run the layer-0 e2e: expect the SDPA ~8 ms → ~50–150 µs (tensor-core, K/V shared), layer ~8.8 ms → **~0.5–1 ms**
(~3–5× eager — "reasonable"; the residual is the small projections / norms, already competitive).

**Phase 4 — causal + additive mask, symbolic. ⬖ PARTIAL (causal ✅, additive mask pending).** The **causal** half
landed for free with Phases 1–2: `build_warp_chain_tileop` emits the causal `realize_score_mask` (`kv_col > q_row`) and
the symbolic `realize_boundary_mask` (`kv_col >= seq_len`) in sequence — both write `-1e30` before the rowmax, so the
composition IS the AND of the keep predicates. Oracles: `test_warp_chain_causal_dynamic_matches_torch` (equal-head) +
`test_warp_chain_gqa_dynamic_matches_torch` (GQA, traced causal), seq ∈ {8, 16, 37, 64}. The **additive-mask** half is
still pending — it needs net-new fragment codegen (a `FragmentBiasAdd` op loading the `mask[b,h,q,k]` tile into the score
C-fragment before the rowmax) + accepting the 4th rank-4 input in `_flash_params`. Currently an additive-mask symbolic
flash falls back to the scalar `chain_build` (correct, slow). Qwen3-Embedding is non-causal/no-bias, so this is general
coverage, not the layer-0 win — deferred.

**Phase 5 — prior / search integration. PENDING.** Surface the warp-chain-vs-scalar fork + the warp tile knobs (WM/WN/BK)

to the two-level tuner and the `AnalyticPrior` cold-start, so the search can tune the warp chain (KV tile, warp count)
and the prior cold-picks it. Fix the **per-op tune at the hint** issue (symbolic per-op slices currently bench at the
trace seq, mis-ranking O(seq²) kernels — see the findings doc) so the prior ranks the warp chain on its real seq=512
cost. (The warp chain currently emits a single fixed-geometry kernel — `knobs={}` — so it is not yet tunable; this phase
opens it to the search. Deferred — the greedy default already deploys the win.)

## 6. Testing strategy

- **Correctness oracle (non-negotiable):** the `*_dynamic_matches_torch` family at seq ∈ {8, 16, 37, 64} — 37 is the
  key partial-tile case. These caught the materialized-path −inf bug; they are the gate for every phase. Run with the
  warp chain active (CHAIN-pinned in phases 1–2, default in phase 3+).
- **Perf:** nsys (`cuda_gpu_kern_sum`) on the uncaptured layer-0 forward at seq=512 — trust nsys, not the deplodock
  `run --bench` per-kernel table (its solo-window attribution is broken; the whole-program number is fine). Target:
  SDPA ≤ ~200 µs, layer ≤ ~1 ms.
- **No-regression:** the full `tests/compiler/e2e` + `tests/compiler/passes` suites (static flash, materialized static
  SDPA, the scalar streaming fallback) must stay green — the warp chain is gated by eligibility, so out-of-scope shapes
  keep the correct scalar/decomposition paths (no correctness risk for anything it doesn't claim).

## 7. Risks & mitigations

- **Boundary-mask correctness** (the −inf-vs-0 trap). Mitigated by reusing the *proven* causal `realize_score_mask`
  mechanism and the seq=37 oracle. Highest-risk item — write that test FIRST (red), then implement.
- **`D % 16 != 0`** head dims fall outside the mma tile. Keep the `D%16==0` gate; non-conforming D falls back to the
  scalar streaming flash (correct, slow) — acceptable, and Qwen3 D=128 conforms.
- **Partial-tile perf.** The masked final KV/query tile runs the full 16-wide mma then masks — a small constant
  overhead per row-tile, negligible vs the O(seq²) win.
- **Scope creep into the static path.** Phases 1–2 only *add* a symbolic branch; the static warp chain and its tests are
  untouched. The fork (Phase 3) is the only change to the deployed default, and only for eligible symbolic flash.
- **Fallback always available.** The scalar streaming flash stays the correct path for any shape the warp chain
  declines, so a bug in the warp chain degrades to slow-but-correct, never wrong (provided eligibility is honest).

## 8. Expected outcome

Symbolic SDPA at seq=512: **~8 ms → ~50–150 µs** (tensor-core, smem-shared K/V). Layer-0 e2e: **~8.8 ms → ~0.5–1 ms**,
i.e. from 0.02× to **~0.2–0.4× of eager** — "reasonable" for a from-scratch compiler vs PyTorch's cuBLAS+flash, with
the remaining gap in the small projections (Finding 5) rather than attention. This closes the qwen3-embedding-0.6b
layer-0 perf goal and unlocks fast dynamic-shape attention generally.
