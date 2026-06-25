# Tensor-core streaming flash (mma.sync) — flash as an edge placement, not a kernel

**Branch:** `feature/tile-ir-block-dag`
**Status:** **The compiler GENERATES a working fused tensor-core flash** (`DEPLODOCK_CHAIN=1`, fp16, matching torch
across `(B,H,S,D)`). Phases 0–3 landed green: the view layer (1a–1c, incl. a GPU-validated scalar shared-score flash),
the atom layer (the two cells via `atomize`), and the codegen (a `KernelOp` built from the shared mma ops + the new
`Fragment*` ops, rendered by the standard renderer — no source template). The kernel **structure** is a dedicated
`assembly/_warp_chain` assembler (the flash's fused-streaming shape doesn't fit the generic single-cell assembly — see
**Architecture** below); the default path is byte-unchanged (pin-gated). Remaining is **functional** (Phases 4–7:
bf16 / causal / GQA / symbolic-seq / the geometry forks + cost-based deploy / the perf bench) — see **Remaining work**.
Full `tests/compiler/` green throughout. Detailed history + the architectural boundary are in the live status below.

The streaming-flash `MONOID` tier is built and correct but **scalar** — `QK^T` and `P@V` lower to FMA loops, the
online-softmax carrier `(m, d, O)` updates one KV element at a time. This plan brings the **`mma.sync` tensor-core tier**
to flash **without adding a single flash-specific build move**. The whole point is the architecture's load-bearing
invariant — *purely algebraic moveset, no shape specializations* (the composer dispatches on carrier algebra, never on a
named shape). A bespoke FA-2 tiler would be a second attention-shaped code path the architecture forbids; the ambitious,
clean solution makes flash *fall out* of the moves that already exist, applied to a slightly richer **view** of the same
algebra. If a phase below adds a `flash_tile_build`, it has failed — even if it works.

Target hardware: **RTX 5090 (sm_120)**. Tensor-core family: **`mma_m16n8k16_{f16,bf16}` only** (the sole entries in
`ATOM_REGISTRY`: 16-bit operands, f32 accumulate, sm_80+). **No `wgmma` / Hopper / TMA-bulk / FP8 / warp-specialized
ping-pong** — that is FA-3 and is explicitly out of scope (see [Out of scope](#out-of-scope)).

Motivation: the blog post `learning-flashattention-the-hard-way-part-2` shows deplodock emitting the *scalar* streaming
flash kernel from the carrier, then hits a wall — the tensor-core, FA-2, and Hopper sections are placeholders because the
streaming tier never reaches the warp/MMA tier. This plan closes the mma-reachable subset. The
[placeholder map](#mapping-to-the-blog-placeholders) ties each phase to the section it unblocks.

## The architecture — three unifications, zero flash code

Everything below is one idea seen from three angles: **flash is the SDPA split with its score edge placed in
registers.** Each unification already has a seam in the codebase pointing at it.

### Unification 1 — the score is an `INLINE` edge (flash = the demoted-SDPA split at a different placement)

`split/010_split_demoted` already un-fuses the score-materializing SDPA into an `xn` producer + a clean gemm consumer
(P@V), wiring the score across a **GMEM** edge — the full `[S_q, S_k]` matrix in HBM. And the block-DAG IR already has
the unifying view: `TileGraph.placement(edge) ∈ {INLINE, SMEM, GMEM}`, read off the `Schedule`, where `stage` and `split`
are *two values of one query* (`plans/dag-edge-placement-split-as-enumeration.md`). So:

> **Flash = the demoted-SDPA split with the score edge placed `INLINE` (a `[BM,BN]` register fragment per KV tile),
> bounded by the `BN` tile, with the online-softmax carrier supplying the cross-tile recombine.**

- materialized SDPA → score edge at **GMEM** (full `[S_q, S_k]`, the `010_split_demoted` cut today);
- flash → score edge at **INLINE** (register, `BN`-tiled, the carrier rescales `O` across tiles).

Same producer/consumer structure, one extra value in the placement lattice. The **C→A handoff** the original Phase 3
fretted about (mma-#1's C fragment becoming mma-#2's A operand) stops being a bespoke mechanism — it is simply *what an
`INLINE` edge between two contractions is*. The v1 smem bounce is the **SMEM** placement of that same edge; the v2
register-reuse is **INLINE**. `stage`, `split`, and flash collapse into one question: *where does this edge live.*

### Unification 2 — the carrier is `MONOID` over `SEMIRING` (P@V is an ordinary reduce axis)

The deferred "crux" (surface the contraction embedded in the carrier's combine) is cleanest as a **classification**
change, not a build change. `flash_combine`'s `O = O·α + p·v` is a `SEMIRING` accumulation (`O += p·v`) twisted by a
`MONOID` rescale (`α`). So `classify`/`iter_dag` should read the carrier's algebra as **compositional** —
`MONOID(SEMIRING)` — making the `kv` axis appear in the DAG as a `SEMIRING` reduce (for P@V) *simultaneously* with being
the `MONOID` carrier's reduce (for the stream). Once `kv` is a first-class reduce axis with two algebra roles,
`_moves.legal_decomps` already knows what to do: it gates on **carrier traits**, and both traits license the same `BN`
split. The embedded contraction is then just *in the DAG* — no recognizer, no `op.streaming` branch — and
`_atom.atomize_cell` (factored out in Phase 0, provenance-agnostic) fires on it in Phase 2 with **zero new code**. "A
twisted monoid is a monoid" generalizes to "a monoid over a semiring is lowered by composing the monoid moves with the
semiring moves."

### Unification 3 — flash is a **carried contraction chain** on a dual-role axis

What blocks naive tiling: head-dim `d` is modeled as a free axis *outside* everything (`_flash.build_flash_frag` emits
`Loop d > Loop m > … score …`), so the score `s[m,kv]` — independent of `d` — is recomputed per `d` (the "correct but
naive" scalar form). The clean fix is to stop treating `d` as plain-free and represent flash as what it is — a **chained
contraction sharing an intermediate axis**:

```
S[m,k] = Σ_e Q[m,e]·K[k,e]        # e = dd (reduce);  k = kv (free output of matmul-1)
P[m,k] = softmax_k(S)              # the MONOID carrier, over k
O[m,n] = Σ_k P[m,k]·V[k,n]         # k = kv (reduce of matmul-2);  n = d (free output)
```

`kv` is the **dual-role hinge**: free-output of matmul-1, reduce of matmul-2, reduce of the carrier. `d`/`n` is free for
matmul-1 but an *output* of matmul-2. A `reduce_decomp` generalized to **a chain on a shared axis** then does all of
Phase 1 in one move: tiling `k` by `BN` simultaneously tiles matmul-1's free output (→ the `score[BM,BN]` INLINE edge),
matmul-2's reduce (→ the P@V cell, `O[BM,D]` its accumulator fragment), and the carrier (→ the softmax-between via
`Monoid.merge`, the cross-tile `O` rescale via `Monoid.combine_states` — **both already authored** on `flash_combine`).
This generalizes beyond attention to any `A@B → f → @C` fusion; it is not an attention feature.

## Why flash is scalar today (precise diagnosis, in the new frame)

1. **The score edge has only two placements today.** `010_split_demoted` materializes it at GMEM; the fused streaming
   nest keeps it implicit (recomputed per `d`). There is no `INLINE` placement for a producer→consumer intermediate, so
   the register-resident score the warp tier needs is unreachable.
2. **The carrier's embedded contraction is invisible to the DAG.** `classify` returns a flat `MONOID`; the P@V inside
   `flash_combine` is not a reduce axis, so `atomize`/`legal_decomps` never see a second matmul to tile or fuse.
3. **The hinge axis `kv` is modeled single-role.** `iter_dag` tags it `REDUCE` (the stream) but not also the free output
   of QK^T / the reduce of P@V, so the chain structure — and the score-sharing across `d` — is not expressed.

All three are **view** gaps (`iter_dag` / `classify` / the edge lattice), not missing build machinery. That is why the
fix is clean: enrich the derived view, and the algebraic moves consume it unchanged.

## Target kernel shape (the *output* of the moves, not a hand-coded target)

Per `(batch, head)`, tile of `BM` query rows, accumulator `O[BM, D]` in registers (f32):

```
init m[BM] = -inf, l[BM] = 0, O[BM,D] = 0           # per-row carrier + accumulator fragment
for kv0 in 0..S_k step BN:                          # KV tile loop (serial, registers carried)
    S[BM,BN] = Q[BM,D] @ K[kv0:kv0+BN, D]^T         # mma #1 (reduce D; B = K^T, b_trans)  — INLINE score edge
    m_new[BM] = max(m, rowmax(S));  P = exp(S - m_new)   # carrier.merge, over the BN register cells
    l = l·exp(m - m_new) + rowsum(P);  α = exp(m - m_new)
    O[BM,D] *= α                                     # carrier.combine_states rescale (the twist)
    O[BM,D] += P[BM,BN] @ V[kv0:kv0+BN, D]           # mma #2 (reduce BN; A = P fragment)  — same INLINE edge as A
    m = m_new
O /= l                                               # epilogue normalize
```

`S` and `P` are the **INLINE score edge** (register fragments, never HBM). `D = 64`, `BN ∈ {16, 32, 64}` map onto the
`m16n8k16` atom. Crucially this listing is *derived* — it is `reduce_decomp` tiling the shared `kv` axis of the chain at
an INLINE score placement, then `atomize` on the two cells. No code emits this shape directly.

## Phases

### Phase 0 — Consolidate MONOID lowering, factor the atom layer — **DONE (green)**

Landed in four commits. One `MONOID` build move (`_build.monoid_build`) + one pass (`070_coop_reduce`) lower both the
flat cooperative reduce and the streaming flash; `080_streaming` / `streaming_build` deleted; the `streaming` flag is
derived on demand (`IterDag.streaming`); the validator's `COOP`/`STREAMING` tiers collapsed into one `MONOID` tier
(`BK`/`FK` legal on it). The `atomize` body edit is factored into the **provenance-agnostic atom layer**
(`_atom.atomize_cell`, unit-tested cell→`Mma`) — the reuse boundary Phase 2 depends on. Full `tests/compiler/` suite
(1635) green. This is the foundation: one MONOID move to enrich, one reusable atom layer.

### Phase 1 — the view layer: the carried contraction chain (the crux), as classification + an edge value — **DONE (green)**

The whole crux is here, and it is a **view** change plus one new value in the placement lattice — no bespoke build path.
Three coordinated, individually structural-testable sub-steps:

- **1a — `iter_dag` represents the carried contraction chain.** Tag the hinge axis `kv` with its dual role
  (free-output of matmul-1, reduce of matmul-2 + carrier), and link the QK^T → softmax → P@V chain through the shared
  axis. *Test:* the DAG exposes two SEMIRING contractions sharing `kv` + the `Monoid` carrier on it.
- **1b — `classify` recognizes `MONOID(SEMIRING)`.** Read the carrier's combine as a SEMIRING accumulation twisted by a
  MONOID rescale, surfacing the embedded P@V as a SEMIRING reduce axis. *Test:* `classify` returns the compositional
  algebra and `legal_decomps` offers the `BN` split on `kv` for both the carrier (associative) and P@V (semiring) traits.
- **1c — `INLINE` edge placement + the shared-axis `reduce_decomp`.** Add `INLINE` as a placeable value for a
  producer→consumer intermediate (extending `stage`=SMEM / `split`=GMEM). `reduce_decomp` on the shared `kv` axis tiles
  the chain by `BN`: emits the `score[BM,BN]` INLINE edge, makes QK^T and P@V two cells, and wires the carrier's `merge`
  (softmax-between) + `combine_states` (cross-tile `O` rescale). `BN=1` is degenerate-identical to today. *Test:* the
  tiled `TileGraph` carries an `INLINE` score edge + two SEMIRING cells + the carrier rescale; **end-to-end**
  register-`BN` flash SDPA matches torch (scalar FMA P@V — the first accuracy check of the crux).

**Already landed (step 1, `d6feb244`):** the serial KV re-bracket (`S_k → S_k/BK·BK`) via the existing
`_replace_k_monoid` — `BK` honored as a streaming-axis knob (`BK ∈ {2,4}` verified). It tiles the stream but keeps the
score implicit; 1a–1c are what surface it as the `INLINE` edge and split the carrier into the two cells.

The deep equivalence to keep in view: **flash = the `010_split_demoted` cut with the score edge at `INLINE` instead of
GMEM.** If 1c is built right, the materialized-SDPA split and flash share one fission + one edge-placement fork.

### Phase 2 — `atomize` composes over the two cells (no new build move)

After Phase 1, the two inner contractions are ordinary SEMIRING cells, so `_atom.atomize_cell` (provenance-agnostic,
factored in Phase 0) fires on both — the same `OptionFork` shape as `020_tensorize` (greedy default = atom when eligible,
scalar fallback), offered on the MONOID pass over the coupled geometry. Reused verbatim: `_atom.atomize_cell`, the `Mma`
op + `kernel/005_lower_atom_tile` codegen, `eligible_atoms`'s SEMIRING-cell recognizer + `cc ≥ (8,0)` + 16-bit gate, and
the free-axis warp tower for `BM` / `D`.

The genuinely new constraints (the reuse boundary):

1. **Two cells per body.** Walk both (`Mma(c=S, a=Q, b=K, b_trans)` reducing `D`; `Mma(c=O, a=P, b=V)` reducing `BN`).
2. **Register-fragment operand provenance.** mma-#1's `S` feeds the softmax and becomes mma-#2's `A` operand `P` *in
   registers/smem, never gmem* — i.e. the `INLINE` (v2) / `SMEM` (v1) score edge of Unification 1. `atomize` is already
   provenance-agnostic; only the edge's *placement* is parameterized (Phase 3), and it is shared work, not duplicated.
3. **Joint, not independent, geometry.** mma-#1's N-fragment layout must match the softmax row-reduction *and* mma-#2's
   A-operand layout, and `BM`/`BN`/`D` are shared. The MONOID pass **propagates** one coupled atom geometry across
   the two cells rather than enumerating each independently — this is "compose over the chain," not "stitch two boxes."

### Phase 3 — realize the `INLINE` edge + fragment-layout softmax (kernel tier)

The hard codegen, all keyed on the **score edge's placement** + the atom's C-layout — no flash special case:

- **The C→A handoff = the edge placement.** v1: place the score edge at **SMEM** (`ldmatrix` it back into the A layout) —
  correct, one slab, still no HBM. v2 (perf follow-up): **INLINE** (keep `P` in registers, shuffle into the A layout —
  the FA-2 register-reuse trick), only if the bounce shows in the profile. Same `_slab` / placement machinery as stage.
- **Fragment row reduction.** `rowmax(S)` / `rowsum(P)` are `__shfl_xor_sync` butterflies over the `m16n8` C-fragment's
  N-direction lane set — a fragment-aware emitter keyed on the atom's C layout, distinct from the existing whole-state
  `MonoidWarpShuffle`. Write it against the documented C layout; unit-test the reduction in isolation.
- **Accumulator rescale (the twist) reuses the carrier.** `α = exp(m_old − m_new)` per row + in-place `O_frag *= α` is
  the register-fragment form of `combine_states`' `O = O·α + …`; the `m`/`l` update reuses `combine_partials`.

### Phase 4 — dtype (f16/bf16 in, f32 accumulate)

The atom is 16-bit only. Thread the operand dtype through the MONOID atom gate; the carrier + accumulator stay f32
(matches the numerics section: fp32 stats under low-precision matmuls, `α ≤ 1` never amplifies). Keep the scalar fp32
path as the cc<8 / fp32 fallback. This is a gate, not a phase of work.

### Phase 5 — masking at the fragment tier (causal + symbolic `seq_len`)

- **Symbolic `seq_len` KV** composes the existing masked-K mma machinery (`_replace_k_warp` ceil-div `K_o` +
  `dpl_mma_load_*_kzero` zero-fill) onto the KV-tile loop: ceil-div the tile loop, zero-fill the overhang K/V tile,
  `-inf`-mask the score fragment past `seq_len` (reuse `_mask_streaming_carrier`'s predicate at the fragment).
- **Causal** skips whole KV tiles above the diagonal (`kv0 > m_tile_max`) and per-element `-inf`-masks the diagonal
  tile's score fragment.

### Phase 6 — fork integration + cold pick

- The atom-vs-scalar choice, the `BN` tile factor, **and the score edge placement (`INLINE`/`SMEM`)** are `OptionFork`s
  on the unified MONOID pass, keyed structurally (`op_cache_key`), so the two-level tuner explores them and greedy picks
  the atom when the prior prices it cheaper.
- Promote `FLASH` from a hard env pin to a cost-based offer at the recognizer (`_composer_wants_flash` has the hook).
- Add an `AnalyticPrior` cold ranking for the MONOID atom knobs (`BN`, `WM/WN`, `FM/FN`, edge placement).

### Phase 7 — validation (RTX 5090)

Bench on sm_120, CUDA-graph-captured, fp16: tensor-core flash **vs** scalar flash **vs** the materialized 2-kernel
baseline **vs** eager SDPA / `torch.compile` / FlashAttention, at seq ∈ {512, 2048, 8192}, causal and non-causal, MHA and
GQA; accuracy vs fp64 across fp32-scalar / fp16-mma / bf16-mma; per-kernel `tune --bench` + `eval` drill-down; record an
mma-flash golden for the layer-0 attention shape. Fills the blog's Validation + Numerics tables.

## Out of scope

- **`wgmma` / Hopper warpgroup MMA**, **TMA bulk-tensor** copies, **FP8** attention, **warp-specialized ping-pong**
  (producer/consumer named-barrier overlap of softmax with the next MMA). That is FlashAttention-3 and a separate plan.
  RTX 5090 is sm_120 *consumer* Blackwell: Ampere-style `mma.sync.m16n8k16` 16-bit tensor cores (the target here),
  not the FA-3 async/warpgroup model.
- **cp.async double-buffering of KV tiles** (sm_80+) is an **optional perf follow-up** layered on the score edge's SMEM
  placement, not required for the tier to exist. `130_transport` already knows cp.async; composing it with the streaming
  slab is a later step.

## Mapping to the blog placeholders

| Blog section (placeholder)                                     | Closed by                                                                                                                                                                                                                             |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| §Tensor cores in the softmax seam (mma + fragment softmax)     | Phases 1, 2, 3, 4                                                                                                                                                                                                                     |
| §Work partitioning (FA-2: q-block parallel, warp split-Q)      | Phase 2 (warp tower) + the free-axis grid split; cp.async KV double-buffer = follow-up                                                                                                                                                |
| §Async / Hopper (FA-3)                                         | **out of scope**                                                                                                                                                                                                                      |
| §Long context = split-KV (Flash-Decoding)                      | the `BR` cooperative-KV lane already folds carrier partials; the score-edge `INLINE`/`GMEM` placement *is* the streaming-vs-materialized choice; compose with the atom tier + a standalone decode combine (`monoid_reduce_tilegraph`) |
| §Numerics at the metal (fp32 stats, α ≤ 1, fp8 caveat)         | Phases 4, 7 (fp8 caveat stays narrative)                                                                                                                                                                                              |
| §The payoff (one MONOID pass + atom, beside hand-written FA-2) | Phases 1–3 — the payoff *is* the unification: flash = SDPA split @ INLINE, lowered by the same moves as the matmul                                                                                                                    |
| §Validation (bench vs eager / torch.compile / FlashAttention)  | Phase 7                                                                                                                                                                                                                               |

## Sequencing & risk

- **Phase 0 is the foundation and is landed green** — one MONOID move, one reusable atom layer, retired streaming flag.
- **Minimum working tensor-core flash = Phase 1 + 2 + 3** (non-causal, static, fp16; Phase 4 dtype folded in). The whole
  crux concentrates in **Phase 1's view layer** (`iter_dag` chain + `classify` MONOID(SEMIRING) + the `INLINE` edge).
- **Top risk: the view layer, not the atom.** If `iter_dag`/`classify` cannot cleanly express the carried contraction
  chain + the compositional carrier, the `INLINE`-edge `reduce_decomp` has nothing to tile and Phase 2 has nothing to
  atomize. De-risk it first, where the oracle is exact: `BN=1` must reproduce today's scalar flash byte-for-byte through
  the enriched view before any atom lands, and each of 1a/1b/1c carries a structural unit test.
- **No-end-to-end-green-intermediate caveat (measured, not assumed):** the IR dump confirms the score is recomputed per
  head-dim `d`, so surfacing it couples the `score` INLINE edge + the `O[BM,D]` fragment + the restructured
  online-softmax + the P@V cell into one change whose *accuracy* oracle (1c) only lights up once 1a–1c land together.
  Hence 1a/1b are
  **structural** unit tests (DAG / classify contracts), accuracy at 1c — the bottom-up order that keeps "view is derived,
  moves are algebraic" intact.
- **Second risk: the joint atom geometry** (Phase 2.3) — one coupled `WM/WN/FM/FN` across both cells. Mitigated by
  propagating a single choice rather than two forks.
- **Third risk: the fragment row-reduction layout** (Phase 3) — a fixed shuffle for the one atom family; unit-test in
  isolation against the documented C layout before wiring it into the stream.
- **Validation foundation already exists:** the scalar streaming flash is correct end-to-end, so every Phase 1/2/3 output
  can be checked against it at each KV tile, not just at the end.

## Implementation status (live)

- **Phase 0 — done, green** (commits `66483c9d` derive-streaming, `79f5db1b` factor atom layer, `a4bbd02a` unify MONOID,
  `134086db` collapse validator). Full `tests/compiler/` = 1635 passed.
- **Phase 1 step 1 — done, green** (`d6feb244`): serial KV re-bracket, `BK` honored on the streaming axis, `BK ∈ {2,4}`
  verified vs torch.
- **Phase 1a — done, green.** `IterDag.chain` derives the carried contraction chain (`ContractionChain` in
  `enumeration/_iterdag.py`): the dual-role hinge `kv`, the nested SEMIRING QK^T contraction, and the `Monoid` carrier
  (its first partial = the INLINE score edge). A derived view (`None` off a non-streaming nest), structurally unit-tested
  (`tests/compiler/passes/test_contraction_chain.py`). No build-path change yet — 1b/1c consume it.
- **Phase 1b — done, green.** `classify` returns the compositional algebra `MONOID(SEMIRING)` for a streaming nest
  (`_Regime.inner_algebra=SEMIRING`, derived from `dag.chain`); `legal_decomps` licenses the hinge `kv` split under both
  the carrier's associative trait (serial re-bracket) and its commutative trait (the embedded P@V's THREAD partition),
  which is what makes the shared-axis tiling sound. Structural tests in `test_contraction_chain.py`.
- **Phase 1c — done, green.** `_build.chain_build` restructures a static `MONOID(SEMIRING)` streaming flash into the
  **FA-2 shared-score** form: the P@V output `d` becomes a REGISTER domain axis (`O[BM,D]` register accumulator), so the
  register-replication pass (`kernel/010_split_register_axes`) shares the score across `d` instead of recomputing it per
  `d` block (the INLINE score edge), and `_split_carrier` splits the twisted carrier into a **scalar stats** `Monoid` +
  a **register-tiled accumulation** `Monoid` (the two SEMIRING cells Phase 2 atomizes) — the accumulation reads the
  stats carrier's rescale `α` / probability `p` temps, which render inline (visible to the sibling carrier). The key
  realization that unlocked it: a single register tile over the whole block + the split carrier shares the score
  *automatically*, because the replication keys on the `d` var and the split made the stats/score `d`-independent. The
  `DEPLODOCK_CHAIN=1` pin opts in (`070_coop_reduce`); greedy default stays the scalar streaming nest (search-fork =
  Phase 6). Matches torch end-to-end (`max_diff ≈ 5e-7`) across static non-causal / causal / GQA / additive-mask SDPA
  (`tests/compiler/e2e/test_flash_attention.py::test_flash_chain_*`); structural tests in `test_contraction_chain.py`.
  Symbolic-`seq_len` + cooperative-KV (`BR>1`) under the chain form, and the search-fork, are follow-ups.
### Phase 2 — `atomize` composes over the two cells — **done (green)**

The chain restructuring (1c) emits the two SEMIRING cells — the inner QK^T producing the `INLINE` score fragment + the
register-tiled P@V accumulation `O[BM,D]`. Phase 2 composes `_atom.atomize_cell` on each, reusing the `Mma` op +
`kernel/005` codegen. The atom-layer reuse boundary (2.1/2.2 below) is unit-tested; the deployed warp-chain kernel
derives BOTH cells' layout via `atomize_cell` (`split/005_warp_chain._classify_cell`). NOTE: the *deployment* took the
dedicated-assembler route (see **Architecture**), not the `020_tensorize`-fork-on-`chain_build` route originally
sketched — the fused-streaming structure doesn't fit the generic warp tower; the atom-layer composition is the part that
generalized.

- **2.1 — QK^T fragment-output atomization — done, green.** `atomize_cell` gained an `out_index` param so a cell whose
  result is an INLINE register fragment with **no `Write`** (the flash QK^T score) can supply its `(M=query, N=kv)` coords
  explicitly; the transposed-B Q@K^T then fuses to `Mma(c=score, a=Q, b=K, b_trans=True)` reducing `dd`. Unit-tested
  (`tests/compiler/passes/test_atomize_cell.py::test_fragment_output_cell_uses_explicit_out_index`). The first reuse
  boundary the test file's note anticipated; the warp-chain-build (below) calls it.

- **2.2 — P@V fragment-`A` atomization (atom-layer part) — done, green.** `atomize_cell` gained a `frag_a` flag for the
  fragment-`A` cell shape: one gmem `B` `Load` + a register `A` fragment → `Mma(c=O, a=P, b=V)` reducing the KV tile (the
  score-derived probability `P` is the `A` operand in registers — the C→A handoff). OFF by default and the
  caller opts in (the one-`Load` `mul` cell is ambiguous with a scalar-scaled reduce `acc += x·s`), so the generic
  `warp_build` matmul path is unchanged. Unit-tested
  (`test_atomize_cell.py::test_fragment_a_pv_cell_fuses_with_register_a` + the off-by-default guard). The atom-layer
  reuse boundary for BOTH chain cells is now complete (2.1 QK^T fragment-output + 2.2 P@V fragment-`A`).

> **What actually shipped** (vs the originally-planned 2.2-build-side / 2.3-joint-geometry below): the warp-tiled fused
> kernel is emitted by the dedicated `assembly/_warp_chain` (one warp / 16 query rows, the score `[16,BN=16]` tile, the
> two cells via `atomize_cell`), NOT by tiling `chain_build`'s scalar nest through the generic warp tower. The
> joint-geometry / `BN`/`WM`/`WN` forks are deferred to Phase 6 (v1 fixes one warp / one BM tile). The notes below are
> the original sketch, kept for the design rationale.

- **2.2 build-side (sketch).** Split the carrier's `O = O·α + p·v` into a separate rescale (`O *= α`) + a clean `Accum`
  (`O += p·v`) so the P@V cell is canonical for the `frag_a` atomize.
- **2.3 — joint geometry (deferred to Phase 6).** ONE coupled `WM/WN/FM/FN` across both cells (QK^T's `N`-fragment
  layout = the softmax row-reduction = P@V's `A`-operand layout), `BN ≥ atom_k`, `D % atom_k == 0`.

**Phase 3 design — validated end-to-end on hardware (the codegen target).** Before writing the codegen, the whole fused
tensor-core flash was proven by a hand-written FA-2 kernel that matches torch SDPA across the KV stream (S ∈ {16…128},
`max_diff ≈ 1e-4`, fp16) — `tests/compiler/e2e/test_flash_tensorcore_reference.py`. It de-risks every Phase-3 unknown at
once and is the **executable spec** the warp-chain codegen must generate:

- **Fragment row-reduction** (the plan's "third risk", unit-tested in isolation first): `rowmax`/`rowsum` over the score
  C-fragment's N (kv) lanes is `max(in-lane col pair, across the 2 N-tiles)` then a `__shfl_xor` butterfly (`xor 2`,
  `xor 1`) over the 4-lane col group — exact vs numpy. The C-fragment layout is rows `g`/`g+8` (`g=lane/4`), cols
  `(lane%4)*2+{0,1}` (the `ir/kernel` `RegStore` layout).
- **C→A handoff (v1 SMEM):** the `P` C-fragment writes row-major to smem, `ldmatrix.x4`-loads back as the P@V `A` — no
  register shuffle (v2 is the perf follow-up).
- **Operand layouts confirmed:** `Q`/`P` → `ldmatrix.x4` A; `K` (transposed-B Q@K^T) → native col-major manual pack
  (`n=lane/4`, `k=(lane%4)*2{+8}`); `V` (canonical B) → `ldmatrix.x2.trans`. The `α` rescale + `m`/`l` update are the
  carrier's `merge`/`combine_states` in fragment-distributed (per-row, 2 rows/lane) form.

Remaining: emit this from the compiler (the warp-chain build wiring the two `Mma`s + the fragment-softmax codegen + the
smem C→A), gated under `CHAIN=1` + an atom pin, then validate the generated kernel against this reference.

**Warp-chain codegen — the compiler now GENERATES a working fused tensor-core flash (v1).** A fp16 non-causal SDPA
compiled with `DEPLODOCK_CHAIN=1` lowers — via `split/005_warp_chain` → `assembly/_warp_chain.assemble_warp_chain` — to a
single `mma.sync` kernel that matches torch end-to-end (`max_diff ≈ 5e-4`, fp16) across `(B,H,S,D)`
(`tests/compiler/e2e/test_flash_tensorcore_generated.py`). The generated kernel is the validated reference generalized
over the shape, reusing the `FragmentRowReduce` op for the fragment softmax and the atom-layer's two cells (QK^T
fragment-output + P@V fragment-`A`). The default path (no `CHAIN` pin) is byte-unchanged — this only fires under the
explicit opt-in. **v1 scope:** fp16, non-causal, equal-head, `D%16==0`, `S%16==0`; out of scope falls back to the scalar
chain / materialized path.

**The kernel is built through the IR (the source template is gone).** `build_warp_chain_kernelop` now produces a
**`KernelOp`** (kernel-IR — `GridTile > WarpTile >` the leaves), rendered by the standard `render_kernelop` and lowered
by the standard `cuda/010_lower_kernelop`. The QK^T / P@V mma + the A/V ldmatrix loads are the **same** kernel-IR ops
the warp-tier matmul uses (`MmaSyncPtx` / `LdmatrixLoad` / `RegStore`, `dpl_mma_m16n8k16_f16` + `dpl_mma_load_*`); the
fragment
online-softmax is `FragmentRowReduce` + the new `FragmentExp` / `FragmentScale` ops + the carried `m`/`l` recurrence
(`Init` + the new `Reassign`). The C→A handoff is `RegStore` → the `flash_pv_smem` slab → `ldmatrix.x4` A (`ldm=16`, the
BN stride — the bug that bit D>16). Validated vs torch across `(B,H,S,D)` D∈{16,32,64}, S≤256 (`max_diff ≤ 5e-4`); full
`tests/compiler/` = 1668 passed.

### Architecture — what's on the shared/algebraic path, and the genuine boundary

The warp-chain flash's **dispatch + codegen + cell-classification** are on the shared / algebraic path; only the kernel
**structure** is hand-assembled (in `assembly/_warp_chain`), because the flash's fused-streaming shape genuinely does not
fit the generic single-cell assembly. What landed (all committed, green):

| Layer                             | State                                                                                                                                                                                                                      |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| kernel form                       | a **`KernelOp`** (kernel-IR), rendered by the standard `render_kernelop` + lowered by `cuda/010_lower_kernelop` — the source-string template is **deleted**                                                                |
| TC primitives                     | the **shared** `dpl_mma_m16n8k16_f16` / `dpl_ldmatrix_x4` / `dpl_ldmatrix_x2_trans` (the exact helpers the warp-tier matmul emits)                                                                                         |
| mma / softmax codegen             | the standard kernel-IR ops: `MmaSyncPtx` / `LdmatrixLoad` / `RegStore` + the new `FragmentRowReduce` / `FragmentExp` / `FragmentScale` / `Reassign`                                                                        |
| cell **layout** (`b_trans`, atom) | **derived** by the `atomize` move (`split/005_warp_chain._classify_cell` → `atomize_cell` → `classify_matmul_operands`) — QK^T transposed-B, P@V canonical-B fall out, not hard-coded                                      |
| cell **lowering**                 | `kernel/005._lower_cell` now lowers a **fragment-output cell** (operand Loads + `Mma`, no `Write` → the fragment chain, no `RegStore`, the C-fragment stays live) — the flash QK^T can be lowered by the matmul's own pass |
| dispatch                          | a `split/`-phase structural fork (`005_warp_chain`), keyed on the algebra (`dag.chain` + atom-eligibility) — analogous to `010_split_demoted`, NOT a named-shape recognizer                                                |
| layering                          | clean (`assembly/` ↛ `enumeration/`; the `atomize` derivation lives in the `split` phase)                                                                                                                                  |

**The genuine boundary (why the structure stays hand-assembled).** Routing the *structure* through the generic
assembly + kernel passes is not a reroute — it needs a **new model** in the shared assembly, because the flash differs
from a matmul in three ways the single-cell machinery can't express: (1) a **streaming accumulator** — `O` persists
across the **kv-stream loop**, which is *outside* the cell, whereas the matmul accumulator is cell-local (persists
across the cell's K-reduce); `005` would reset `O` per kv tile. (2) **two mmas + a shared reduce loop** — `_scan_cell`
assumes one `Mma` per `AtomTile`, but the kv loop holds QK^T *and* P@V *and* the softmax, sharing the loop (the score
must flow QK^T→softmax→P@V within one iteration). (3) it is a **matmul→softmax→matmul** fusion — the one fusion the
generic assembly does (`_fused.py`, the SMEM edge) is a MAP/rmsnorm producer + a matmul, not two tensor-core mmas. So a
focused dedicated assembler for this shape (like `_fused.py` is for the SMEM-edge shape) is the architecturally
reasonable form; forcing it into the generic single-cell assembly would be a large extension to shared matmul code for
one consumer. **Conclusion: the dedicated assembler is a correct *waypoint*, not the endpoint** — the invariant ("no
shape-specific *build move* in the composer") holds, so it is a legitimate ship-it-first form; but the three "boundary"
items above are facets of *one* limitation in the assembly model, and the principled endpoint dissolves all three. See
**Generalized `_tower`** next.

### Generalized `_tower` — the principled endpoint (the re-bracketable reduction)

The "genuine boundary" above is genuine *relative to today's `_tower` model*, not fundamental. All three items —
streaming accumulator outside the cell / two mmas sharing a loop / matmul→softmax→matmul — are facets of a **single**
limitation: `assembly/_tower._wrap_tower` hard-codes *one bracketing* of the reduction (one reduce level, accumulator
cell-local, one `Mma` per `AtomTile`, the body a single compute site). Make the bracketing a parameter and all three
dissolve at once — and `_fused.py` + `_warp_chain.py` collapse into the one `_assemble` path.

**The key observation: the IR already carries the model.** `TileGraph` is a DAG of `Block`s with derived `Edge`s and a
three-value placement lattice `Placement ∈ {INLINE, SMEM, GMEM}` (`ir/tile/ir.py`). `assemble_block` realizes `GMEM`
(→ launch groups, `_assemble_multi`) and the one SMEM-fused shape (→ the dedicated `_fused.py`). **`INLINE` between two
blocks *within one kernel* is the unimplemented cell of the lattice** — and that cell is exactly the flash gap. The
generalization is not a new abstraction; it is filling in the missing placement realization plus making the accumulator
level a parameter. Two orthogonal moves:

**G1 — placeable carry scope (the re-bracketing).** Today the reduce nest (`SerialTile(K_o)` + `RegisterTile(reduce)`)
is always innermost, accumulator reset per cell. Lift "which serial axis carries accumulator state, and how it combines"
into an explicit input:

```python
@dataclass(frozen=True)
class CarryScope:
    """A reduce axis that carries state ACROSS its serial loop — state init'd ABOVE the
    loop, combined per-iteration, not reset inside. Matmul's K-reduce and flash's kv-stream
    are both CarryScopes; they differ ONLY in (init, combine) = the carrier algebra."""
    axis: Axis
    init: tuple[Stmt, ...]      # Init / RegFragment hoisted ABOVE the SerialTile
    combine: tuple[Stmt, ...]   # the per-iteration merge at the loop tail (the carrier's algebra)
```

- **SEMIRING K-reduce** (matmul): `init=[Accum=0]`, `combine=[]` (the `Mma` writes the accumulator in place). Innermost
  — byte-identical to today.
- **MONOID kv-stream** (flash): `init=[m, l, O-fragment]`, `combine=[FragmentRowReduce, FragmentExp, FragmentScale,
  Reassign…]` — the online-softmax merge `_warp_chain` hand-writes. There can be **more than one** carry nested (flash
  has both QK^T's cell-local `dd`-reduce and the kv-stream over the whole QK^T→softmax→P@V body). "Placeable accumulator
  level" = the carry scope is any `SerialTile`, not forced innermost. The MONOID-over-SEMIRING carrier *interleaves* its
  merge/rescale/update around the inner SEMIRING accumulate (compute S → merge stats → rescale O → accumulate P@V →
  update m/l), which is precisely Unification 2's `MONOID(SEMIRING)`.

**G2 — the reduce body is a cell sub-DAG, not one cell.** Let the carry-scoped body be a sub-`TileGraph` and make
`assemble` **recurse** on it; the cell-DAG + edge-placement machinery is then the *same* machinery as the top level,
nested one level down. The internal edge (flash's score) is realized at its `Placement`: `INLINE` → registers (v2: the
mma C→A register shuffle), `SMEM` → a `_slab` (v1: `RegStore`→smem→`ldmatrix`-back — exactly what `_fused` already does
for its producer→matmul edge), `GMEM` → impossible inside a kernel. The spine:

```python
def _assemble_group(blocks, schedule, carries):     # blocks sharing one launch group, edges INLINE/SMEM
    order  = topo(blocks)
    cells  = [cell_body(b) for b in order]           # G2: each block -> its AtomTile(s)
    realize_internal_edges(order, schedule)          # INLINE=registers / SMEM=_slab
    return wrap_tower(layers, sequence(cells), carries=carries)   # G1: hoist init, emit combine
```

**How the three assemblers collapse into one:**

| Assembler | `CellDAG` | internal edge | `carries` |
|-----------|-----------|---------------|-----------|
| matmul (`_assemble`)      | 1 cell                                     | —                     | `[K-reduce]` (SEMIRING, innermost)        |
| fused prod+matmul (`_fused`) | 2 cells (MAP/MONOID producer + matmul) | 1 × `SMEM`            | `[K-reduce]` (+ producer reduce if MONOID)|
| flash (`_warp_chain`)     | 2 cells (QK^T + P@V)                        | 1 × `SMEM`(v1)/`INLINE`(v2) | `[QK^T K-reduce, kv-stream MONOID]` |

`_fused`'s "patch the `StageBundle`" trick *is* "realize this edge at SMEM"; `_warp_chain`'s hand-written `m`/`l`/`O`
recurrence *is* a MONOID `CarryScope.combine`. Both become inputs to one tower builder, not separate files.

**Migration oracle (byte-identical, the discipline `_assemble.py` already follows).** Mirrors the plan's own "BN=1
reproduces scalar flash" de-risking:

1. Realize a 2-block `INLINE`/`SMEM` group in one tower — replace `_assemble_multi`'s `NotImplementedError` for
   multi-block launch groups with `_assemble_group`. Target: reproduce `_fused`'s `TileOp` **byte-for-byte** → delete
   `_fused.py`. (Single SEMIRING carry; no carry-scope work yet.)
2. Add the MONOID `CarryScope` → reproduce `_warp_chain`'s output byte-for-byte → delete `_warp_chain.py`'s hand-rolled
   body.

Each step has an exact oracle (an existing assembler's output) — the "migration oracle = byte-identical CUDA" invariant.

**What does *not* fall out for free** (so this is not oversold): (a) the MONOID `combine` is **layout-aware** — it runs
in the mma C-fragment layout (`FragmentRowReduce`/`FragmentScale`), so `CarryScope.combine` must be produced by the same
`atomize` move that produced the cells, not a generic scalar combine (Phase 3 already built those fragment ops — they
just need to be emitted by the carry mechanism instead of hand-written); (b) `INLINE`-between-two-atoms is genuinely new
codegen (the register C→A shuffle, v2) — `SMEM`-between-atoms (v1) is reuse; (c) keeping `Block.compute` as one site and
nesting sub-`TileGraph`s under a carry scope is cleaner than relaxing `Block`, but needs `synthesize_staging`/`_slab` to
compose under nesting — the main unknown to spike.

**Implementation status.**
- **G1 abstraction — landed.** `CarryScope` + `wrap_carry_tower` (`assembly/_tower`); the warp-chain KernelOp body is
  rebuilt on it (the kv-stream MONOID carry + the two-cell body through the helper, the hand-listed
  `Init`/`Reassign`/softmax wiring deleted).
- **Migration step 1 — done.** `assemble_block` is the **single assembly entry that realizes the placement lattice**
  (single / SMEM-fused-group / GMEM-launch-split); `_fused.py` is **deleted**, its logic absorbed as `_assemble_group`,
  and the assembly pass lost its special-case dispatch (now just `assembly_ready` + `assemble_block`).
- **Migration step 2 — in progress** (route flash through the generic tower→kernel lowering, so `_warp_chain` collapses
  into `_assemble`). Landed:
  - **Cell lowering in the shared `kernel/005`.** The QK^T **fragment-output** cell (no `Write` → the C-fragment stays
    live) and the P@V **fragment-A** cell (A a live register fragment, C a carried accumulator → only `ldmatrix B +
    mma`, no A/C decl, no `RegStore`) both lower through the matmul's own pass.
  - **The flash is a `TileOp`, routed through the generic kernel passes** (`kernel/005`…`100` → `cuda`), not a KernelOp
    that bypassed them. This surfaced (and fixed) that the flash-only fragment-softmax ops
    (`FragmentScale`/`FragmentRowReduce`/`FragmentExp`/`Reassign`) had no `rewrite` registration — they were terminal
    (KernelOp→render only); now registered, so they survive the SSA-rewriting passes. The mma cells are **still
    hand-emitted** (`MmaSyncPtx`) in the tower body — the next step routes them through `005`'s AtomTile lowering.
  - **Blocker found for the AtomTile cell routing (the next slice).** Converting the P@V `consume` to a fragment-A
    `AtomTile` mostly works but mis-renders the **grid axis** `bh` in the `005`-derived `LdmatrixLoad`'s `src_index`:
    it renders as a **value** name (`v4`, colliding with the `alpha` float) instead of the grid coord (`a0`) — the
    epilogue store in the *same* kernel renders `bh` correctly, so the AtomTile lowering specifically drops `bh` from
    the σ axis-substitution within the cell scope (`_mma_src_index` returns `load.index` verbatim; the materializer
    doesn't σ-map the outermost grid axis there). Resolve the axis-var σ-mapping before converting the cells, then
    delete `_warp_chain`'s hand-emitted mma codegen.
  - **Remaining (the large part):** the AtomTile cell routing (above) + the carried `O` / fragment-softmax glue as a
    first-class tower, validated against the dedicated builder / torch, then delete `_warp_chain`'s hand-built
    structure.

### Remaining work (functional — Phases 4–7 + the scalar-chain follow-ups)

- **Phase 4 dtype:** bf16 (a trivial atom swap; v1 is f16-only). The fp32/`cc<8` fallback already routes to the scalar
  chain / materialized path.
- **Phase 5 masking + coverage:** causal (skip above-diagonal KV tiles + diagonal `-inf` mask), GQA (K/V at
  `head//group`), symbolic-`seq_len` (the masked-K machinery onto the KV loop), partial tiles (`S`/`D` not a multiple of
  16). v1 is non-causal / equal-head / static / `D%16==0` / `S%16==0`.
- **Phase 6 geometry forks + cold pick:** the warp chain is `CHAIN=1` pin-only — make `BN`/`WM`/`WN`/multi-warp-`BM` /
  the v2 register C→A `OptionFork`s the tuner explores + the prior prices, and promote it from a pin to a cost-based
  greedy pick (so it deploys by default when cheaper than the materialized path).
- **Phase 7 validation:** the perf bench (TC flash vs scalar vs materialized vs eager / `torch.compile` / FlashAttention
  at seq 512/2048/8192, causal/non-causal, MHA/GQA), accuracy vs fp64, and an mma-flash golden. Correctness vs torch is
  validated; the speedup is not yet measured.
- **Scalar-chain (1c) follow-ups** (off the critical path): symbolic-`seq_len` masked streaming + cooperative-KV
  (`BR>1`) under the scalar chain form, and a generalized `_chain_axes` for layouts where the P@V output is the inner
  free axis.
