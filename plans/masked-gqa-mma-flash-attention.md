# Masked / GQA / MMA-tier flash attention — design plan

## Implementation status (2026-06-21)

**Parts A + B (scalar tier) landed** — GQA, causal, and explicit additive-mask recognition, static and dynamic seq,
GPU-verified vs torch (~3–8e-7) on RTX 5090. What changed and what was learned:

- **Masking/GQA recovered structurally** from the fused Loop body (`025_recognize_flash.py` + `_flash.py`), no frontend
  change. The score feeding the rowmax `Accum` is classified: bare Load (no mask), `add(score, Select(kv ≤ m))`
  (causal — drives the pre-existing `causal=True` per-element nest), or `add(score, Load(mask))` (additive bias — a
  per-`(m,kv)` mask Load in the nest). GQA group = `q_heads // kv_heads`, deployed as a `head // group` K/V index (A2,
  no materialized broadcast). `flash_shape_eligible` relaxed for GQA + broadcast mask. Tests in
  `tests/compiler/e2e/test_flash_attention.py` (GQA+causal, causal, additive-mask, + dynamic twins).
- **Correctness trap fixed.** The plan assumed causal "fails the recognizer's classifier" and falls through. It did
  **not** — a causal equal-head SDPA under `FLASH=1` was recognized and built as an *unmasked* nest → silently wrong
  output (measured `max_diff ≈ 1.94`). Detecting the mask (Part B) is therefore a correctness requirement, now enforced.
- **Tracer quirk (pre-existing, frontend, out of scope).** `enable_gqa=True` is a bool kwarg; dynamo drops the default
  `is_causal=False`, and `trace/torch.py`'s "first bool arg = is_causal" scan grabs `enable_gqa` → every GQA SDPA traces
  as **GQA + causal** (`is_causal=True`). This is the only GQA form reachable via the public torch API here, and it
  matches the real Qwen3-Embedding layer-0 trace (16 q / 8 kv heads, `is_causal=True`, no explicit mask).

**Blocker found — Parts A+B do NOT yet close Finding 1 on Qwen3.** The plan's `_extract_qk` recovers Q/K by matching a
score producer with exactly **two plain Loads** (a clean scaled-QK). On a real decoder/embedding layer the generic
fuser **fuses RoPE into the QK score reduce**: Q/K are computed inline from `mul_*` + `position_embeddings` (rotate-half
+ cos/sin), and the causal mask + scale also live in the producer. So `_extract_qk` returns `None` and the SDPA falls
through to `010_sdpa` (4 kernels) — confirmed on Qwen3-Embedding-0.6B layer 0. **Next step:** recover the score
*computation* wholesale (inline the producer's per-`(head,m,kv)` body — which already carries RoPE + the GQA `//group`
index + the mask — into the flash nest's score reduce, via the `Sigma`/`splice_graph` axis-remap machinery), replacing
the synthetic `score_reduce`. That generalizes Parts A+B to real layers and is the prerequisite for the integration
validation below.

**Remaining (unchanged from below):** the RoPE-fused score-body recovery (above), Part C (MMA tensor-core tier), and
Part D (FLASH structural-fork offer + analytic cold-start). Until D, flash deploys only under `DEPLODOCK_FLASH=1`.

---

Extend the landed **scalar-tier** flash path (PR #263, `plans/online-softmax-flash-attention.md`) to the three cases it
falls through on today, so flash deploys on real decoder / embedding layers instead of only toy non-causal,
equal-head, unmasked SDPA:

1. **GQA** (grouped-query attention — e.g. 16 query / 8 KV heads).
2. **Explicit additive mask** (the `(1,1,S,S)` float bias HF threads through whole-model / layer traces).
3. **MMA tensor-core tier** (the score computed once and `O` carried as a register fragment — the actual perf win).

## Hard constraint — no frontend-pass changes

**All work lands in `loop/` (incl. `loop/fusion`), `lowering/` (`tile` / `kernel` / `cuda`), `ir/`, and `search/`.
The frontend decomposition passes (`frontend/decomposition/010_sdpa.py`, `100_softmax.py`, the tracer) are NOT
modified.** This is the design #263 already chose — flash is recognized **after** the generic fuser at Loop IR (the
"Loop-fusion rewrite" entry point), *not* by a direct lowering from `SdpaOp` (the original plan's strategy A, which
would have been a frontend change and is rejected here). Every reference to `010_sdpa.py` / `100_softmax.py` /
`trace/*` below is **read-only** — the graph structure the Loop-IR recognizer consumes — never an edit target. The
consequence: GQA broadcast, the additive mask, and the softmax stats must all be recovered **structurally from the
fused Loop IR**; there is no frontend provenance stamp to lean on.

## Motivation — this is the gap on a real model

`plans/qwen3-embedding-0.6b-layer0-attn-retune-2026-06-20.md` (Finding 1) measured Qwen3-Embedding-0.6B layer 0 and
found flash #263 **cannot reach it**: it is GQA (`num_attention_heads=16`, `num_key_value_heads=8`), its SDPA carries a
mask, and `025_recognize_flash.py:127` hard-skips unless `DEPLODOCK_FLASH=1` (default off). So the layer keeps **4
separate SDPA kernels (~40 µs)** where torch.compile fuses **one flash kernel (~25 µs)** — and that ~15 µs, plus the
`ldmatrix` bank conflicts in the spread-out MMA matmuls, is the bulk of the ~45 µs gap to torch.compile (deplodock 195
µs vs tcompile 150 µs). Closing it needs masked + GQA + MMA-tier flash. None of it is reachable by tuning — it is
compiler feature work. This plan is the actionable form of that finding.

## What already landed (the foundation — do not rebuild)

- **`FlashCombine` `ReduceCarrier`** (`ir/stmt/leaves.py`) — the `(m, l, O)` tuple LSE monoid; `render` lowers the
  rescale directly; `LoopOp` validation knows `Init` + `FlashCombine`. This **already encapsulates** the three IR gates
  the naive nest would trip:
  - `_helpers.py::accums_independent` (lines 43–50) rejects cross-`Accum` reads (`l_i`/`O_i` read `alpha` from `m_i`).
  - `_atom.py::classify_fragment_epilogue` (lines 393–394) bails on `slice_in_reduce` (accumulator consumed inside the
    reduce loop). Quoted: `return None, "the accumulator is consumed inside a reduce loop (mid-reduction use, not a
    store-time fold)"`.
  - The loose rescale never appears as body statements for those gates to see — it lives in `FlashCombine.render`.
- **Loop-IR recognizer** `loop/fusion/025_recognize_flash.py` + `_flash.py` — runs after the generic fuser, anchors on
  the softmax-P@V kernel, recovers Q/K/V, rewrites to the fused flash `LoopOp` (`build_flash_frag`).
- **Scalar nest, non-causal, static + dynamic seq** — GPU-verified vs torch (~1–3e-7). Dynamic seq reuses the masked-K
  zero-fill (`_stage_expand.py:331-344`, the `(k < seq_len) ? v : 0` ternary).

## What blocks the three cases today (precise gates)

> The `frontend/*` citations here describe the graph the recognizer **reads** — they are the structure to match, not
> code to change (see "Hard constraint" above).

### GQA — a graph-level `IndexMapOp`, not yet threaded into the nest

`frontend/decomposition/010_sdpa.py:76,137` (read-only) broadcasts K and V to the query-head count via `_maybe_gqa`
(`_helpers.py:147-158`), which inserts an `IndexMapOp` whose head-axis coordinate is `h // group_size`:

```python
coord_map.append(BinaryExpr("/", p, Literal(group_size, "int")) if d == head_axis else p)
```

So by the time the fuser produces the two `LoopOp`s, K/V are already (logically) expanded to 16 heads. The recognizer's
eligibility rejects it anyway — `_flash.py::flash_shape_eligible` (the `has_mask` / batch-dim path):

```python
if [_static(d) for d in k_shape[:-2]] != batch or [_static(d) for d in v_shape[:-2]] != batch:
    return False  # GQA / mismatched batch dims
```

### Explicit mask — recognizer hardcodes `has_mask=False` (a correctness trap)

`010_sdpa.py:90-99` adds the mask before softmax: `scores += broadcast_to(attn_mask, scores_shape)`. The mask is the 4th
SDPA input (`010_sdpa.py:52` `inp_mask`), a `(1,1,S,S)` float bias (`0` keep / `-inf` mask)
(`trace/huggingface.py:86-88`, `trace/torch.py:573`). Softmax is mask-agnostic — `exp(-inf)=0` zeroes masked positions.
But the flash recognizer calls
`flash_shape_eligible(..., has_mask=False)` **unconditionally** (`025_recognize_flash.py:139`) and never reads the mask
add — so a masked SDPA that still matched the softmax-P@V anchor would build a nest **without the mask** → wrong output.
`has_mask` must be *detected*, not assumed.

### MMA tier — the partition planner can't model a reduced-N matmul

`lowering/tile/010_partition_loops.py:617-619`:

```python
if matmul_reduces and not combine_reduces:
    if outer_m is None:
        return None
```

The efficient (vector-`O`) flash nest computes the score `[m, kv]` once per `(m, kv)` and reuses it across all `d`. But
the inner QK^T score-reduce is a **matmul reduce whose output lands on the reduce axis `kv`** (consumed by the streaming
softmax, never written as a free output). The planner needs two free *output* axes (M and N); flash has only `m` (plus
batch/head). For `B·H=1` the gate fires outright; for `B·H>1` (Qwen3: 16 heads) `outer_m` exists but the score's N axis
(`kv`) is still the reduce axis, so the generic matmul-reduce model does not fit. The scalar nest sidesteps this by
keeping `d` as a grid axis and recomputing the score per `d` (correct, redundant). The `Mma` `c` fragment is also
**init-once, written at kernel end** today (`ir/kernel/ir.py:705-707` zero-inits `c` at declaration); flash needs it
**loop-carried across `K_o` and scaled by `alpha` each step**.

---

## Plan of attack — three parts, scalar-correct first, then fast

Ordering is by dependency and by "deploys on Qwen3 soonest." **GQA + mask are recognizer/nest changes that work at the
existing scalar tier** — landing them first makes flash *correct and deployable* on Qwen3-Embedding (B·H=16>1, so the
redundant scalar nest already plans), giving a measured baseline. The **MMA tier** is then a pure perf rewrite of the
same recognized nest. The **FLASH fork wiring** (Part D) is what makes any of it actually deploy under greedy
`compile`/`run`.

### Part A — GQA flash (scalar tier)

**Recognizer (`loop/fusion`, no frontend change).** In `025_recognize_flash.py`, when recovering K/V operands, detect
the GQA `IndexMapOp` (head-axis coord `h // group`) sitting between the input and the matmul/P@V Load **in the fused
Loop IR**, and recover `group_size` structurally (from the IndexMapOp's coord divisor, or the q-head / kv-head shape
ratio). Pass the kv-head count + group to `build_flash_frag`. All of this reads graph nodes the frontend already
emitted.

**Eligibility.** Relax `flash_shape_eligible`: accept `q_heads == group · kv_heads` (exact division) instead of
requiring equal batch dims. Keep the symbolic-non-seq and head_dim checks.

**Nest.** Two options for the K/V head index in `_flash_loop_body`:
- **(A1) Read the materialized broadcast buffer** (simplest): the IndexMapOp already produces a 16-head K/V; flash
  Loads it with the query head index. Cost: 4× redundant K/V smem traffic for the duplicated heads. Fine for a
  correctness-first landing.
- **(A2) Thread the head remap into the Load index** (recommended for the MMA tier): index K/V at
  `kv_head = q_head // group` directly (`q_idx` uses `b/h`, `k_idx`/`v_idx` use `b/(h//group)`), so the 8-head K/V are
  read without materializing the expansion. This is the same `//group` the `IndexMapOp` encodes, moved into the nest's
  index expression. Build it now (it is cheap) so the MMA tier inherits it.

**Verify.** A GQA SDPA parity test under the `shape_mode` fixture (`tests/compiler/conftest.py`), static green; the
kernel table shows one `k_sdpa…` kernel; accuracy vs torch `enable_gqa=True` (`backend/torch_ref.py:164-173` already
does GQA in the reference).

### Part B — explicit-mask flash (scalar tier)

**Recognizer (`loop/fusion`, no frontend change).** In `025_recognize_flash.py`, detect the additive-mask `Assign`/
`add` between the QK^T scale and the softmax rowmax **in the fused Loop body** (the op `010_sdpa.py:96` emitted),
capture the mask buffer id + its broadcast shape, and pass it through. Set `has_mask` from *detection*, not the
hardcoded `False` at `025_recognize_flash.py:139`.

**Eligibility.** `flash_shape_eligible` should *accept* an additive mask (it is just a per-`(m,kv)` bias), not reject
it. Keep rejecting anything that isn't a plain additive bias (e.g. non-broadcastable shapes).

**Nest.** In `_flash_loop_body`'s `kv_body`, after `s = sacc·scale`, load `mask_e = mask[..., m, kv]` (honoring the
`(1,1,S,S)` broadcast — leading dims index to 0) and `s = s + mask_e` **before** the `flash_combine`. The `-inf`
entries make `exp(s − m_new)=0`, so masked keys contribute nothing — identical to the score-matrix path's softmax. This
**subsumes the existing causal special-case** (`_flash.py:227-239` builds a per-element `kv ≤ m` Select); causal is just
an additive mask, so once the additive path lands, causal can either keep its tile-skip optimization (Part C / the
existing flash plan's Step 5) or fold into the additive-mask load. Keep causal tile-skip as a separate optimization;
do not regress it.

**Dynamic interaction.** A symbolic-seq mask is `(1,1,seq_len,seq_len)`; the masked final KV tile already zero-fills
past `seq_len` (`_stage_expand`), and the additive mask's own `-inf` past the real extent is consistent with that — no
new dynamic machinery, but add a dynamic-mode parity test (xfail until the dynamic lowering covers the mask load).

**Verify.** A padding-mask SDPA parity test (a real `(1,1,S,S)` bias with some `-inf` rows), static green; accuracy vs
torch SDPA with `attn_mask`.

> After Parts A + B, flash is **correct and deployable on Qwen3-Embedding-0.6B layer 0** at the scalar tier (B·H=16,
> redundant-score nest plans). Measure it: `DEPLODOCK_FLASH=1 deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0
> --dynamic seq_len@x:1 --bench`. Expect one fused `k_sdpa…` kernel; it may not yet beat the 4-kernel split (scalar
> score recompute per `d`), but it establishes the baseline the MMA tier improves and proves correctness end-to-end.

### Part C — MMA tensor-core tier (the perf win)

This is the real Step 3 of the existing flash plan: the score computed **once** per `(m, kv)` tile, the P@V accumulator
carried as a register fragment across the `K_o` streaming loop and scaled by `alpha` each step. Two chained MMAs:

- **QK^T**: `M = m` (query rows, tiled `BR`), `N = kv` (key tile, tiled `BK`), `K = dd` (head_dim). Output `S[BR, BK]`
  — a register/smem fragment consumed by the rescale, never written to gmem.
- **P@V**: `M = m`, `N = d_v` (a real free output axis), `K = kv` (the streamed key tile). Accumulator `O[BR, d_v]`
  **is the carried `FlashCombine` `O` state**, scaled by `alpha` between tiles.

**The planner decision (key architectural choice).** Generalizing `010_partition_loops` to model a matmul-reduce whose
N axis is reduced away (the QK^T score) is invasive and risks every other matmul. Recommended instead: a **dedicated
flash MMA lowering rule keyed on the `FlashCombine` carrier** — flash is structurally specific enough (two chained MMAs
+ an online rescale + a carried fragment) that a bespoke lowering is more tractable and lower-blast-radius than teaching
the generic planner. The rule consumes the recognized flash `LoopOp` and emits the `K_o` streaming kernel directly,
reusing the existing `Mma`/`RegFragment`/`ldmatrix` staging primitives for the two cells. Document the rejected
alternative (generalize the planner) and why.

**Sub-steps:**
- **C1 — loop-carried, per-step-scaled `Mma` fragment.** Teach kernel lowering to keep an `Mma` `c` fragment alive
  across a serial-outer (`K_o`) loop and emit a per-step `O_i *= alpha` (a `RegScale` on the fragment regs). Today `c`
  is zero-init-at-declaration + store-at-end (`ir/kernel/ir.py:705-707`). Verify in isolation with a
  matmul-with-per-step-rescale micro-kernel (the existing flash plan's Step 3): accuracy + emitted CUDA shows the carry
  living across `K_o`.
- **C2 — QK^T as an `Mma` cell** producing the score fragment `S[BR, BK]` (reduce over `dd`), staged like any other MMA
  B operand. The per-row `alpha` maps to the `m16n8k16` `c[4]` register→row layout (the per-query-row scale the scalar
  nest can't express because it keeps `d` as a grid axis).
- **C3 — P@V as an `Mma` cell** whose accumulator is the carried `O` fragment (reduce over the `kv` tile), with the
  online rescale (`FlashCombine.combine_states` / `render`) scheduled between C2's softmax stats and C3's accumulate.
- **C4 — register pressure.** `O[BR, d_v]` fragment + Q/K/V staging coexisting across `K_o` will force a smaller `BR`
  than a plain gemm; expect the tuner to pick conservative `BR`; watch occupancy in `--bench --profile`. This is the
  same headroom Finding 3 of the retune report flags (deplodock MMA matmuls run at 4–10% occupancy with heavy
  `ldmatrix` bank conflicts vs cutlass's 0) — a swizzled smem layout for the flash score/`O` staging is in scope here.

**GQA + mask in the MMA nest.** Inherit A2 (head-remap index on the K/V MMA B operands) and B (mask load added to the
QK^T `S` fragment before the rowmax). The mask add is a fragment-level epilogue on C2's output.

**Verify.** MMA-tier SDPA parity (non-causal, GQA, masked), static green; kernel table shows **one** `k_sdpa…` kernel
with `MMA=mma_m16n8k16_f16`; `--bench` beats the split path and closes toward torch.compile; `--bench --profile` shows
the score never round-trips through gmem and occupancy/conflicts improved over the 4-kernel split.

### Part D — FLASH structural-fork offer + cold-start (make it deploy)

From the existing flash plan ("not yet wired"). Until this lands, flash only fires under the `DEPLODOCK_FLASH=1` env
pin — greedy `compile`/`run`/serve never deploy it.

- **Two-level offer.** Make `FLASH` an `OptionFork` in the outer SP-MCTS, exactly like `005_split_demoted`'s
  `SPLIT_CONE` — stamp the decision on `op.knobs`, idempotence guard, no `off=` (preserve the absent-vs-declined
  distinction the prior trains on). The tuner then A/Bs flash-vs-split per shape.
- **AnalyticPrior cold-start.** Add the single gated term to `AnalyticPrior.score` (`search/prior/analytic.py`) —
  reward `FLASH=True` once `S_ext_reduce_max ≥ flash_seq_threshold`, penalize below — plus the `_extents` symbolic-hint
  companion change (`992_stamp_structural_features.py` stamps the symbolic axis's hint into `S_ext_reduce_*` and adds an
  `S_ext_reduce_symbolic` flag) so the gate sees the dynamic seq. **Note the blast radius**: this edits the shared
  structural-feature contract (`ShapeKey.from_matmul`, `search/data/shape.py`, the `test_data` mirror assert, the
  `_W_A_DYN` refit, and the CLAUDE.md / `shape.py` / `sample.py` "symbolic axes excluded" wording) — see the existing
  flash plan's "FLASH knob + analytic cold-start" section for the exact list.
- **Default-on for eligible shapes.** Once the MMA tier is fast, the analytic term + learned prior should pick
  `FLASH=True` for long/eligible SDPA without the env pin. Confirm via `eval analytic` / `eval prior`.

---

## Integration validation — close Finding 1 on Qwen3-Embedding

After Parts A–D, re-run the retune report's exact commands and diff:

```bash
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench \
    --dump-dir _tune/flash-qwen3-emb-l0/dump
deplodock compare _tune/tune-attn-qwen3-emb-l0/dump _tune/flash-qwen3-emb-l0/dump
deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench --profile
```

Success criteria:
- The attention deploys as **one** fused `k_sdpa…` kernel (no `_xn` score producer; no separate QK^T / softmax / P@V).
- Attention cost drops from ~40 µs (4 kernels) toward torch.compile's ~25 µs fused flash.
- Layer e2e closes meaningfully on torch.compile (baseline deplodock 195 vs tcompile 150 µs).
- `grep -c flash tune.log > 0` (flash actually fired) and `eval prior` shows `FLASH=True` deployed greedily.
- Then re-attempt the serving A/B (`deplodock serve … --bench` vs `--stock`) the report deferred.

## Test strategy — reuse the parity harness

Per the existing flash plan: every test runs under the `shape_mode` fixture (`tests/compiler/conftest.py`,
static/dynamic params from one body), threading `dyn_M` into the seq axis, with the `dynamic` param `xfail(strict=True)`
until the dynamic lowering covers that case. Extend `tests/compiler/e2e/test_flash_attention.py` with: GQA (Part A),
masked / padding (Part B), MMA-tier (Part C). Accuracy vs torch SDPA (`enable_gqa`, `attn_mask`) under the shared
`_run_module_with_eager` harness. Golden: a flash SDPA golden with a static entry + `.dynM` twin once Part D's analytic
term is in (so `eval analytic`/`eval prior` exercise the FLASH cold-start clause).

## Risks / open questions

- **Bespoke flash MMA lowering vs generalizing the planner (Part C).** Plan assumes a dedicated `FlashCombine`-keyed
  rule. If a small generalization of `010_partition_loops` (modeling a reduced-N matmul-reduce) turns out clean, prefer
  it — but do not gate the perf win on a risky planner refactor.
- **Register pressure / occupancy.** The carried `O[BR, d_v]` fragment is the central cost; `BR` and a swizzled smem
  layout for the score staging decide whether the MMA tier actually beats the split at seq=512 (it may only win clearly
  at longer seq — confirm with `--bench` across seq lengths, the open threshold question the cold-start term needs).
- **GQA head-remap correctness.** A2's `kv_head = q_head // group` index must match `_maybe_gqa`'s `IndexMapOp` exactly;
  cross-check against `backend/torch_ref.py`'s reference GQA.
- **Mask broadcast generality.** Start with the HF `(1,1,S,S)` additive bias; reject (fall through to `010_sdpa`) any
  mask shape the per-`(m,kv)` load can't address, rather than silently mis-broadcasting.
- **Numerics.** Rescale + `exp` in fp32 accumulators; carried `O` is the fp32 MMA fragment, cast to write dtype only at
  finalize — matches eager fp16 attention (the cosine-0.23 class of bug in the #260 masked-K P@V is exactly a
  fragment-dtype mistake; mirror its resolution).
- **Convergence with the masked-K split P@V.** The existing fused-prologue P@V stays degenerate at `FM=FN=1` for
  symbolic K; flash is its proper replacement. Part C and that path should converge, not both be maintained.

## Scope

**In:** GQA flash, additive-mask flash, MMA tensor-core tier, and the FLASH fork + cold-start wiring that makes them
deploy — static and dynamic seq at test parity (`shape_mode`), landing scalar-correct (A+B) then fast (C), with D
making it deployable under greedy compile/run.

**Out (future):** flash-decoding / split-KV across CTAs (the monoid makes it reachable, but its own work), paged-KV,
sliding-window, arbitrary (non-additive) mask functions. Causal tile-skip is an optimization tracked by the existing
flash plan's Step 5 — keep it, don't regress it, but it is not the focus here.
</content>
