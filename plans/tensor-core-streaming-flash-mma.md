# Tensor-core streaming flash (mma.sync) — closing the scalar-flash gap

**Branch:** TBD (off `feature/tile-ir-block-dag`)
**Status:** planned. The streaming-flash `MONOID` tier (`080_streaming`) is built and correct but **scalar** — `QK^T`
and `P@V` lower to FMA loops, the online-softmax carrier `(m, d, O)` updates one KV element at a time. This plan first
**consolidates all monoid lowering onto one composable path** (retiring the `streaming` flag and the bespoke
`streaming_build` / `080_streaming` pass — Phase 0), then brings the **`mma.sync` tensor-core tier** to that unified path.
The order matters: adding tensor cores to the bespoke streaming pass would deepen a specialization the architecture says
it does not have; unifying first means the atom arrives as the *same* `atomize` move the matmul path already uses.
Target hardware: **RTX 5090 (sm_120)**. Tensor-core family:
**`mma_m16n8k16_{f16,bf16}` only** (the sole entries in `ATOM_REGISTRY`: 16-bit operands, f32 accumulate, sm_80+).
**No `wgmma` / Hopper / TMA-bulk / FP8 / warp-specialized ping-pong** — that is FA-3 and is explicitly out of scope (see
[Out of scope](#out-of-scope)).

Motivation: the blog post `learning-flashattention-the-hard-way-part-2` shows deplodock emitting the *scalar* streaming
flash kernel from the carrier, then hits a wall — the tensor-core, FA-2, and Hopper sections are placeholders because the
streaming tier never reaches the warp/MMA tier. This plan closes the mma-reachable subset of those gaps. The
[placeholder map](#mapping-to-the-blog-placeholders) at the end ties each phase to the section it unblocks.

## Why flash is scalar today (precise diagnosis)

Three independent facts, each load-bearing:

1. **`classify` routes flash to the streaming tier, which forbids the atom.** `_classify.classify` flags the nest
   `streaming=True` (a `MONOID` carrier + a **nested** contraction + a tuple `Monoid`). That selects `080_streaming` →
   `streaming_build`, which serial-transforms **both** contraction axes (`bk=fk=splitk=1`), forces `FM=FN=1` ("the
   coupled `Monoid` carrier can't span register cells"), and never calls `atomize`. The warp passes
   (`020_tensorize`…`050_warp_build`) gate off streaming (they need a `SEMIRING` seed). So the atom is structurally
   unreachable for flash.

2. **The Loop-IR flash nest streams one KV element at a time.** `loop/recognize/_flash.build_flash_frag` emits (see the
   `--ir loop` dump in the blog):

   ```python
   for kv in 0..S_k:                         # streaming reduce, ONE key per step
       Init sacc = 0
       for dd in 0..head_dim: sacc += Q[m,dd]·K[kv,dd]   # QK^T — a clean [Load,Load,mul,Accum] cell ✓
       s = sacc · scale
       Monoid((m_i,l_i,O_i), (s, V[kv,d]))   # P@V folded into the carrier as a scalar rank-1 update ✗
   ```

   `QK^T` is an atomizable matmul cell, but `P@V` is **not a cell** — it is hidden inside `flash_combine` as
   `O = O·α + p·v` (a rank-1 update per key). There is no `[Load,Load,mul,Accum]` for `atomize` to fuse, and the score
   `s` is a single scalar, not a `[BM,BN]` tile. Tensor cores need a KV **tile** (BN keys) per step so that `P@V` is a
   real `P[BM,BN] @ V[BN,D]` matmul.

3. **`atomize` fuses ONE standalone matmul; flash is two chained matmuls with a reduction between them.** `_atomize_cell`
   / `_try_atomize_here` recognize a single canonical cell. Flash is `S = QK^T` → row-softmax over `S` (in the fragment
   layout) → `O += P@V`, where matmul-1's output (after softmax) is matmul-2's **A operand**. That C→A fragment handoff,
   plus the row reduction over the score fragment, is the whole difficulty and has no equivalent in the single-matmul
   `atomize`.

The fix is therefore *not* a knob flip and *not* a fourth bespoke build move. Fact 1 is a self-inflicted wound — the
`streaming` flag is a structurally-honest classifier (nested MONOID reduce, not a named shape), but the **pass** it gates
(`080_streaming` / `streaming_build`) is a parallel code path that reimplements free-axis tiling while hardcoding the
rest (serial both contractions, `FM=FN=1`, no stage, no atom) and is unreachable from the warp moves. **Phase 0 removes
fact 1** by dissolving that pass into the general composer; Phases 1–3 then add the atom as composition over the unified
path (Loop-IR KV tiling becomes the `reduce_decomp` move on the outer axis; `atomize` fires on the inner cells).

## Target kernel shape (the standard mma flash, for reference)

Per `(batch, head)`, tile of `BM` query rows, accumulator `O[BM, D]` in registers (f32):

```
init m[BM] = -inf, l[BM] = 0, O[BM,D] = 0           # per-row carrier + accumulator fragment
for kv0 in 0..S_k step BN:                          # KV tile loop (serial, registers carried across)
    S[BM,BN] = Q[BM,D] @ K[kv0:kv0+BN, D]^T         # mma #1  (reduce D; B = K^T, b_trans)   — f16 in, f32 acc
    (+ causal / seq_len mask on the S fragment)
    m_new[BM] = max(m, rowmax(S))                    # row reduction over BN, IN the mma C-fragment layout
    P[BM,BN]  = exp(S - m_new)                        # elementwise on the fragment
    l = l·exp(m - m_new) + rowsum(P)                 # row reduction over BN
    α[BM]     = exp(m - m_new)
    O[BM,D]  *= α                                     # in-place rescale of the accumulator fragment (the twist)
    O[BM,D]  += P[BM,BN] @ V[kv0:kv0+BN, D]          # mma #2  (reduce BN; A = P fragment)    — f16 in, f32 acc
    m = m_new
O /= l                                               # epilogue normalize
```

`S` and `P` are **register fragments**, never HBM (the FlashAttention invariant, now enforced at the fragment level).
`D = 64` and `BN ∈ {16, 32, 64}` map onto the `m16n8k16` atom (D and BN multiples of the 16/8 fragment dims).

## Phases

### Phase 0 — Consolidate monoid lowering, retire the `streaming` flag (the foundation)

Make the general composer lower **both** flat and nested `MONOID` nests by *composition of the existing moves*, and
delete the bespoke streaming path. This is a refactor with a hard correctness oracle (today's scalar kernels) and no new
capability — it exists so that Phases 1–3 add the atom *once*, on a shared path, instead of forking a fourth build move.

Concretely:

- **Remove the stored `streaming` flag.** Drop `_Regime.streaming` from `_classify`. Nested-ness is a *derived* property
  of the iteration DAG (a reduce whose parent is a reduce) — if a move needs it, it queries the DAG on demand, matching
  the "computed on demand, never stored" discipline already used for `Block.reads` / `carrier` (and so it never enters
  `op_cache_key`). `classify` returns `MONOID` for flash and for softmax/RMSNorm alike; what differs is the *number and
  nesting of contraction axes the DAG exposes*, not a tier label.
- **Unify `070_coop_reduce` + `080_streaming` into one MONOID pass** driven by `reduce_decomp` (re-bracket K) +
  `free_tile`, with the recombine taken from the carrier's `combine_partials()` (already carrier-general — this is what
  gives split-KV/`BR` for free). `coop_build` (flat: cooperative `K_c` lane) and `streaming_build` (nested: serial both
  axes) collapse into the single statement "apply the reduce-decomposition move to **each** contraction axis the DAG
  exposes, MONOID carrier recombine; tile the free axes." A flat monoid has one contraction; a nested monoid has the
  outer (KV) plus the inner (QK^T) — same move, applied per axis.
- **The crux — a carrier whose combine embeds a contraction.** The one thing the flat moves do not express today is the
  twist: `flash_combine` does `O = O·α + p·v`, i.e. the `P@V` reduce lives *inside* the monoid's combine, not as a
  sibling loop. Phase 0's real work is teaching `reduce_decomp` (or the carrier) to surface that embedded contraction as
  a decomposable axis so the same machinery tiles it. Getting this right is what makes the unification real rather than
  cosmetic; it is also exactly the structure mma-#2 needs in Phase 2.
- **Collapse the validator tier split.** `_validate.py` today intersects against five tiers, two of which (cooperative
  MONOID, streaming flash) are the split this phase removes; they become one MONOID tier with a uniform legal knob slice
  (the K-chunk knobs become legal on the nested monoid, since split-KV *is* legal — Part 1's associativity — undoing the
  current "streaming forbids `BK`/`FK`" pin).
- **Factor the atom layer out of the SEMIRING staging** (sets up the Phase 2 reuse; a worthwhile SEMIRING cleanup on its
  own). The matmul lowering today entangles two concerns: the **atom layer** (the `atomize` body edit `_atomize_cell` →
  `Mma`, the `Mma` op, and `kernel/005_lower_atom_tile`'s `ldmatrix` / `mma.sync` / fragment-store codegen — all
  provenance-agnostic, naming operands by SSA value) and the **matmul-staging layer** (gmem `Load` → `Mma` → gmem/smem
  `Write`, plus per-matmul free-axis geometry). Separate them so the atom layer is a clean, independently-testable unit
  callable with operands of *any* provenance (gmem `Load` **or** a register/smem fragment). No behavior change for a
  plain matmul — it still composes the two layers — but afterward a MONOID nest in Phase 2 can invoke the atom layer
  directly on its inner contractions without dragging in the gmem-I/O assumption. Unit-test the atom layer in isolation
  (cell → `Mma` → expected PTX) so the reuse has a fixed contract before flash depends on it.

**Oracle / exit criterion:** the consolidated path emits **accuracy-identical** (ideally byte-identical) scalar kernels
to today's `070`/`080` outputs across the existing softmax / RMSNorm / flash tests; the matmul path stays green with the
atom layer now factored (byte-identical SEMIRING kernels); and `080_streaming` / `streaming_build` / the `streaming`
field are gone. No tensor cores reach flash yet. This proves the moves compose over the nest **and** that the atom layer
is reusable, before any atom is wired into a monoid.

### Phase 1 — KV tiling as the `reduce_decomp` move on the outer axis (was: a bespoke re-bracket)

On the unified path, "process a BN-wide KV tile per step" is just the reduce-decomposition move offering a tile factor
`BN` on the **outer** MONOID contraction (`S_k → S_k/BN · BN`), with the `BM×BN` score surfaced as a register
intermediate (`TileGraph` buffer, `INLINE` placement — never `SMEM`/`GMEM`). The inner score reduce (`QK^T`, reduce `D`)
and the carrier's embedded `P@V` reduce (reduce `BN`, exposed in Phase 0) are the two SEMIRING cells; the online-softmax
over `BN` sits between them. Scalar tier = `BN=1` (the degenerate case, identical to today). `BN` is an ordinary
tile-tier search knob, ranked by the prior — no recognizer change, no Loop-IR rewrite. The Loop-IR `build_flash_frag`
stays the scalar canonical form.

### Phase 2 — `atomize` composes over the nested reduce (reuse the SEMIRING lowering, no new build move)

Flash is a **monoid over a semiring**: the outer carrier is a `MONOID` (online-softmax LSE), the two inner contractions
(`QK^T`, `P@V`) are `SEMIRING` matmuls. The goal of this phase is to lower those inner contractions with the **same atom
lowering the matmul path already uses** — not to reimplement `mma.sync` emission for flash. After Phase 0 made both inner
contractions ordinary SEMIRING cells and Phase 1 gave them tile shapes, the existing `atomize` move (R4, `_atomize_cell`
/ `warp_build`'s σ-split) fires on them directly.

**Reused verbatim (the atom layer Phase 0 factored out — the bulk of the tensor-core complexity):**

- the `atomize` body edit `_atomize_cell` (cell `[Load,Load,mul,Accum]` → `Mma`), which already names operands by SSA
  value and is *agnostic to why the matmul exists or where its operands come from*;
- the `Mma` op + the atom codegen `kernel/005_lower_atom_tile` (`ldmatrix` / `mma.sync.m16n8k16` / fragment store);
- the free-axis warp tower `_warp_axis` (GRID/WARP/REGISTER/ATOM), applied to the nest's free axes `BM` / `D` exactly as
  for a matmul — the per-row carrier `m[BM]` / `l[BM]` rides the M register tier, `O[BM,D]` is the mma-#2 accumulator
  fragment;
- `eligible_atoms` / `_atom_eligible`'s SEMIRING-cell recognizer and the `cc ≥ (8,0)` + 16-bit gate.

**The three things that are genuinely new (the reuse boundary):**

1. **Two cells per body.** `_atomize_cell` stops at the first cell today; the monoid body exposes two
   (`Mma(c=S, a=Q, b=K, b_trans=True)` reducing `D`; `Mma(c=O, a=P, b=V)` reducing `BN`). Walk both. `eligible_atoms`
   enumerates the two (today `020_tensorize` skips MONOID entirely; after Phase 0 there is no separate skip).
2. **Register-fragment operand provenance.** The matmul staging assumes gmem `Load` → `Mma` → gmem/smem. Here mma-#1's
   output `S` feeds the softmax and becomes mma-#2's A operand `P` *in registers/smem, never gmem*. The atom move is
   already provenance-agnostic; what must be parameterized is the *staging/source* layer around it — exactly the C→A
   handoff handled in Phase 3. This is the one place the SEMIRING path's assumptions (operands live in gmem) need
   loosening, and it is shared work with Phase 3, not duplicated.
3. **Joint geometry, not two independent fork trees.** You cannot call the SEMIRING fork tree twice and take independent
   `WM/WN/FM/FN`: mma-#1's N-fragment layout must match the softmax row-reduction *and* mma-#2's A-operand layout, and
   `BM`/`BN`/`D` are shared across both matmuls and the carrier. So the monoid pass **constrains/propagates** the atom
   geometry across the two cells (one coupled choice) rather than enumerating each cell's atom independently. This is the
   real reason it is "compose over the nest," not "lower each sub-block as a black box and stitch."

The atom-vs-scalar choice is the same `OptionFork` shape as `020_tensorize` (greedy default = atom when eligible, scalar
fallback), now offered on the unified MONOID pass over the coupled geometry.

### Phase 3 — fragment-layout softmax + accumulator rescale (kernel tier)

The hard codegen, in `kernel/005_lower_atom_tile` + `kernel/_combine` + `kernel/100_materialize_tile`:

- **Row reduction over the score fragment.** The `m16n8k16` C-fragment scatters each row across a known set of lanes;
  `rowmax(S)` / `rowsum(P)` are `__shfl_xor_sync` butterflies over **that** lane set (the N-direction of the fragment),
  distinct from the existing `MonoidWarpShuffle` (which folds a whole-state carrier across all warp threads). Add a
  fragment-aware row-reduce emitter keyed on the atom's C layout.
- **Accumulator rescale (the twist).** Emit `α = exp(m_old - m_new)` (per row) and the in-place `O_frag *= α` before
  mma-#2 accumulates the tile. This is the register-fragment form of `flash_combine`'s `O = O·α + p·v`, split across the
  two matmuls. Reuse the carrier's algebra (`combine_partials`) for the `m`/`l` update; the `O` rescale is a fragment
  broadcast-multiply.
- **C→A handoff for P.** mma-#2's A operand `P[BM,BN]` is mma-#1's D/C fragment after `exp`. v1: **bounce P through smem**
  (`ldmatrix` it back into the A layout) — correct, one smem slab, still no HBM traffic, so the flash invariant holds.
  v2 (perf follow-up): keep P in registers and shuffle into the A layout directly (the FA-2 register-reuse trick), only
  if the bounce shows up in the profile.

### Phase 4 — dtype (f16/bf16 in, f32 accumulate)

The atom is 16-bit only — there is no fp32/tf32 tensor-core entry. So the tensor-core flash path requires **f16/bf16**
`Q/K/V`; the carrier and accumulator stay **f32** (matches the blog's numerics section: fp32 stats under low-precision
matmuls, the rescale `α ≤ 1` never amplifies). Work: thread the operand dtype through the MONOID atom gate; switch the
blog's running example + the new tests to `dtype=torch.float16`; keep the scalar fp32 path as the cc<8 / fp32 fallback.

### Phase 5 — masking at the fragment tier (causal + symbolic seq_len)

- **Symbolic `seq_len` KV.** The masked-K mma machinery already exists for standalone matmul (`_replace_k_warp` ceil-div
  `K_o` + `dpl_mma_load_*_kzero` zero-fill of the partial tile). Compose it for the KV stream: ceil-div the KV-tile loop,
  zero-fill the overhang K/V tile, and `-inf`-mask the score fragment past `seq_len` (reuse `_mask_streaming_carrier`'s
  predicate, applied to the fragment instead of the scalar score).
- **Causal.** Skip whole KV tiles strictly above the diagonal (`kv0 > m_tile_max`), and per-element `-inf`-mask the
  diagonal tile's score fragment (`kv > m`). This is the "tile-skip is a tensor-core-tier follow-up" the recognizer
  docstring names.

### Phase 6 — fork integration + cold pick

- Make the Phase-2 atom-vs-scalar choice a real `OptionFork` on the unified MONOID pass, keyed structurally
  (`op_cache_key`), so the two-level tuner explores both and greedy deploys the atom when the prior prices it cheaper.
- Promote the `FLASH` knob from a hard env pin to a cost-based offer at the recognizer (`_composer_wants_flash` already
  has the hook) — the recognizer's named follow-up — so greedy `compile`/`run` can pick fused flash without the env var.
- Add an `AnalyticPrior` cold-start ranking for the MONOID atom knobs (`BN`, `WM/WN`, `FM/FN`) so the cold pick is
  sane before any tune.

### Phase 7 — validation (RTX 5090)

Bench on sm_120, CUDA-graph-captured, fp16:

- tensor-core streaming flash **vs** scalar streaming flash **vs** the materialized 2-kernel baseline **vs** PyTorch
  eager SDPA / `torch.compile` / FlashAttention, at seq ∈ {512, 2048, 8192}, causal and non-causal, MHA and GQA.
- accuracy vs an fp64 reference across fp32-scalar / fp16-mma / bf16-mma (the blog's numerics table).
- per-kernel `tune --bench` + `eval` drill-down; record an mma-flash golden for the layer-0 attention shape.

This fills the blog's **Validation** and **Numerics** tables and produces the `080_streaming`+atom kernel listing the
blog's payoff section wants to print beside the hand-written FA-2 kernel.

## Out of scope

- **`wgmma` / Hopper warpgroup MMA**, **TMA bulk-tensor** copies, **FP8** attention, **warp-specialized ping-pong**
  (producer/consumer named-barrier overlap of softmax with the next MMA). That is FlashAttention-3 and a separate plan.
  RTX 5090 is sm_120 *consumer* Blackwell: it has Ampere-style `mma.sync.m16n8k16` 16-bit tensor cores (the target here)
  but the FA-3 async/warpgroup model is not what this plan builds.
- **cp.async double-buffering of KV tiles** (sm_80+, available on sm_120) is an **optional perf follow-up** layered on
  Phase 3's smem bounce, not required for the tier to exist or be correct. The transport move (`130_transport`) already
  knows cp.async for matmul; composing it with the streaming slab is a later step.

## Mapping to the blog placeholders

| Blog section (placeholder)                                   | Closed by              |
| ------------------------------------------------------------ | ---------------------- |
| §Tensor cores in the softmax seam (mma + fragment softmax)   | Phases 0, 1, 2, 3, 4   |
| §Work partitioning (FA-2: q-block parallel, warp split-Q)    | Phase 2 (warp tower) + the existing free-axis grid split; cp.async KV double-buffer = follow-up |
| §Async / Hopper (FA-3)                                       | **out of scope**       |
| §Long context = split-KV (Flash-Decoding)                    | mostly present — the scalar `BR>1` cooperative-KV lane already folds carrier partials, and Phase 0 makes the K-chunk knobs legal on the monoid; compose with the atom tier + add a standalone decode combine kernel (carrier-general `monoid_reduce_tilegraph`) |
| §Numerics at the metal (fp32 stats, α ≤ 1, fp8 caveat)        | Phase 4, 7 (fp8 caveat stays narrative) |
| §The payoff (one MONOID pass + atom, beside hand-written FA-2)| Phases 0–3 (makes the comparison real, not hand-written-only — and the unified pass *is* the payoff: same moves as the matmul) |
| §Validation (bench vs eager / torch.compile / FlashAttention)| Phase 7                |

## Sequencing & risk

- **Phase 0 is the foundation and gates everything** — but it is a *refactor with a parity oracle*, not a leap: it must
  reproduce today's scalar `070`/`080` kernels through the unified moves before any atom lands. Land it on its own,
  green, with `streaming_build` / the `streaming` flag deleted, before starting Phase 1.
- **Minimum working tensor-core flash = Phase 0 + 1 + 2 + 3** (non-causal, static, fp16). Phase 4 is folded in (dtype is
  a gate, not a phase of work). Phases 5/6/7 add masking, fork integration, and proof.
- **Top risk lives in Phase 0, not the atom: the carrier whose combine embeds a contraction** (`P@V` inside the twist).
  If `reduce_decomp` cannot surface that embedded reduce as a decomposable axis, the unification stalls and Phase 2 has
  nothing to atomize. De-risk it first, in the scalar parity refactor, where the oracle is exact.
- **Second risk: the C→A handoff for P** (Phase 3, the mma C-fragment → mma A-operand layout). Mitigated by the v1 smem
  bounce — correct and simple, defers the register-reuse layout puzzle to a profile-driven v2.
- **Third risk: the fragment row-reduction layout** (rowmax/rowsum over the `m16n8` C-fragment N-direction). It is a
  fixed, known shuffle pattern for the one atom family; write it against the atom's documented C layout and unit-test the
  reduction in isolation before wiring it into the stream.
- **Validation foundation already exists:** the scalar streaming flash is correct end-to-end, so every Phase 0/1/2/3
  output can be checked against it for accuracy at each KV tile, not just at the end.

## Implementation status & notes (live)

**Phase 0 — DONE, green** (4 commits): `streaming` flag derived on demand (`IterDag.streaming`); the `atomize` body
edit factored into the provenance-agnostic atom layer (`_atom.atomize_cell`, unit-tested); `coop_build` + `streaming_build`
unified into one `monoid_build` move + one pass (`070_coop_reduce`), `080_streaming`/`streaming_build` deleted; the
validator's COOP+STREAMING tiers collapsed into one `MONOID` tier (`BK`/`FK` legal on it). The full `tests/compiler/`
suite (1635) stays green.

**Phase 1 — step 1 DONE, green** (1 commit `d6feb244`): the reduce-decomposition move offers a tile factor `BK` on the
outer streaming contraction (`S_k → S_k/BK · BK`). `monoid_build`'s `_replace_k_monoid` already re-brackets the streaming
axis; `_streaming_leaves` honors a `DEPLODOCK_BK` pin (divisor-gated). This is the **serial** re-bracket — each `K_i` step
still folds one key through the `Monoid`. Verified `BK ∈ {2,4}` flash SDPA matches torch.

**Phase 1 — the crux (remaining): surface `score[BM,BN]` as a register tile + expose `P@V` as a SEMIRING cell.**
Grounded in the actual tile IR of a flash nest (free axes `d=a0` outer, `m=a1` inner; reduces `kv=a2` MONOID streaming,
`dd=a3` QK^T nested):

```
Loop d (a0, free):                      # ← the head/output dim, OUTERMOST free
  Loop m (a1, free):
    Init m_i, l_i, O_i
    Loop kv (a2, MONOID reduce):
      Init acc; Loop dd (a3): acc += Q[m,dd]·K[kv,dd]   # QK^T
      s = acc·scale                                      # ← score s[m,kv], INDEPENDENT of d
      Load v_e = V[kv, d]
      Monoid((m_i,l_i,O_i), (s, v_e))                    # P@V folded per key
    Write out[m,d] = O_i / l_i
```

The score `s` depends on `(m, kv)` only, but sits **inside** the free `d` loop → recomputed per `d` (the "correct but
naive" scalar streaming flash). Surfacing `score[BM,BN]` shared across `d` inherently requires moving `d` from an *outer
free axis* into the *output dimension of the `P@V` cell* (`O[BM,D]` becomes a register fragment, `d` its accumulator dim).
That one move couples four things into a single end-to-end change:

1. the `score[BM,BN]` intermediate (`TileGraph` INLINE buffer, never SMEM/GMEM);
2. the `O[BM,D]` register fragment (the d-axis register/warp tile);
3. the restructured online-softmax (tile-max + `exp` + rescale over the `BN` register cells, between the two cells);
4. the `P@V` SEMIRING cell (reduce `BN` → `D`), the one Phase 2's `atomize` then fires on unchanged.

**Consequence for sequencing:** unlike Phase 0 (clean parity-oracle decomposition), the crux has **no smaller
end-to-end-green intermediate** between step 1 (KV re-bracket, landed) and the full FA-2 register-tiled structure — the
accuracy oracle only lights up once all four land. The tractable path is **bottom-up with per-layer structural unit
tests** (each committable green), end-to-end accuracy last:

- **1a** IR/build: a `monoid_build` path that, for a register-`BN` factor on the streaming axis (d-register-tiled),
  emits the `score[BM,BN]` INLINE buffer + the two cells + the tile-fold online-softmax. Structural unit test (assert the
  buffer + two SEMIRING cells + the softmax-between).
- **1b** kernel: lower the new structure in `kernel/100_materialize_tile` + `_combine` (the tile-fold rowmax/rowsum +
  the `O[BM,D]` rescale). Structural unit test on the emitted kernel IR.
- **1c** end-to-end: register-`BN` flash SDPA vs torch (scalar FMA P@V); this is the first accuracy check of the crux.
- Phase 2 then composes `_atom.atomize_cell` onto cell #4 — the reuse boundary the atom-layer factoring set up.

This is FlashAttention-2's core expressed in the algebra and is the plan's named top risk; it is a focused multi-cycle
effort, not a knob.
