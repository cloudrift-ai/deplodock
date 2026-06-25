# Tensor-core streaming flash (mma.sync) — flash as an edge placement, not a kernel

**Branch:** `feature/tile-ir-block-dag`
**Status:** Phase 0 landed (green); Phase 1 step 1 landed (green); the rest is the ambitious-clean design below.

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

### Phase 1 — the view layer: the carried contraction chain (the crux), as classification + an edge value

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

| Blog section (placeholder)                                   | Closed by              |
| ------------------------------------------------------------ | ---------------------- |
| §Tensor cores in the softmax seam (mma + fragment softmax)   | Phases 1, 2, 3, 4      |
| §Work partitioning (FA-2: q-block parallel, warp split-Q)    | Phase 2 (warp tower) + the free-axis grid split; cp.async KV double-buffer = follow-up |
| §Async / Hopper (FA-3)                                       | **out of scope**       |
| §Long context = split-KV (Flash-Decoding)                    | the `BR` cooperative-KV lane already folds carrier partials; the score-edge `INLINE`/`GMEM` placement *is* the streaming-vs-materialized choice; compose with the atom tier + a standalone decode combine (`monoid_reduce_tilegraph`) |
| §Numerics at the metal (fp32 stats, α ≤ 1, fp8 caveat)        | Phases 4, 7 (fp8 caveat stays narrative) |
| §The payoff (one MONOID pass + atom, beside hand-written FA-2)| Phases 1–3 — the payoff *is* the unification: flash = SDPA split @ INLINE, lowered by the same moves as the matmul |
| §Validation (bench vs eager / torch.compile / FlashAttention)| Phase 7                |

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
- **Next: Phase 1 the view layer (1a → 1b → 1c).** Start at `iter_dag`: represent the carried contraction chain + the
  dual-role hinge axis; then `classify` → `MONOID(SEMIRING)`; then the `INLINE` score edge + the shared-axis
  `reduce_decomp`. Structural tests at 1a/1b, end-to-end accuracy at 1c. Phase 2 then composes `_atom.atomize_cell` with
  no new build move — the boundary Phase 0 set up.
