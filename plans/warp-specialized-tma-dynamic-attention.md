# Enable warp-specialized TMA on the dynamic (masked-tile) attention kernels

## Motivation

After the transposed-B + causal-mask-epilogue fix (PR #251), the Qwen3-Embedding QK^T scores matmul reaches the
tensor-core (`mma_m16n8k16_f16`) tier on **both** the static and dynamic (symbolic-`seq_len`) paths. But the deployed
latencies diverge sharply — and the gap is **transport + schedule**, not the math:

| QK^T reproducer (seq 512)            |  eager |  tc |  deplodock | transport / schedule (emitted CUDA)                                         |
|--------------------------------------|-------:|----:|-----------:|-----------------------------------------------------------------------------|
| **static** (`k_sdpa_reduce_82f310`)  |    148 |  25 |     **31** | **TMA** (78 mbarrier/TensorMap) + **warp-specialized**, no boundary guards  |
| **dynamic** (`k_sdpa_reduce_77b0f0`) |    148 |  25 |     **73** | cp.async double-buffer, **no** warp-spec, 36 `< seq_len` masked-tile guards |

Static nearly matches torch.compile (31 vs 25 µs); dynamic — the *deployable* artifact — is 2.3× slower. The static
kernel wins because the shape-specialised path is allowed onto the TMA + warp-specialized pipeline; the dynamic
(masked-tile) path is gated off it. This plan closes that gap.

Evidence dumps (loop / tile / cuda, both modes): `_tune/tune-model-qwen3-l0-v3/dumps{,-static}/`. The **loop** IR is
identical across modes (same RoPE+QK^T+causal `Select`); all divergence appears at **tile/cuda**, where the static
consumer renders `warp_specialize(ring=2) { producer: bundle tma…; consumer: …mma… }` over fixed `0..8` bounds and the
dynamic consumer renders `bundle async depth=2` over `0..((seq_len+127)//128)` with per-element `if (coord < seq_len)`
store guards.

## Root cause — the gating chain (cited)

Three passes interlock to keep the dynamic path off TMA + warp-spec:

1. **`tile/085_warp_specialize.py`** only fires when the body already contains a **TMA `StageBundle` with
   `pipeline_depth == 2`** (`_has_tma_depth2`, lines 202-206; eligibility 65-66). **No TMA ⇒ no warp-specialization.**
   So TMA is the prerequisite — warp-spec follows for free once TMA lands.

2. **`tile/050_use_tma.py`** declines TMA for:
   - a **symbolic *innermost* dim** (lines 186-198): a symbolic inner extent gives a global stride that isn't
     16 B-aligned → `cuTensorMapEncodeTiled` fails (`CUresult=1`). A symbolic **outer/middle** dim is already fine
     (lines 161-184) — the descriptor's `globalDim` is encoded per launch and TMA's hardware OOB zero-fill covers the
     masked overhang (this is the existing **M-masked** TMA path).
   - **masked-K (symbolic-reduce) sources** (lines 201-209): kept on cp.async because the partial-K-slab zero-fill is a
     hand-rolled `(k < seq_len) ? v : 0` ternary in `_stage_expand`. The comment explicitly notes TMA's hardware OOB
     zero-fill **could replace it** but "needs the 4-D leading-batch box derivation fixed first."

3. **`tile/040_use_ring_buffers.py`** SYNC-pins masked-K bundles (lines 62-68) for the same zero-fill-ternary reason, so
   050/060/080 also see SYNC.

**Mapped onto the attention kernels:**

- **QK^T scores.** Reduce axis is `head_dim` (static, 128) — *not* masked-K. Operands: A = rotated-Q `xna[m, k]`
  (inner `k`=128 static → TMA-OK); B = rotated-K `xnb[k, n]` canonical, **inner `n`=`seq_len` symbolic → TMA declined**
  (rule 2a). Because no operand bundle is TMA, warp-spec (rule 1) can't fire. → the dynamic QK^T's only blocker is the
  **B operand's symbolic innermost dim**.
- **P@V.** Reduce axis is `seq_len` (the key dim) → **masked-K** → SYNC-pinned (rules 2b + 3). Blocked by the harder
  4-D-box TMA-OOB-zero-fill work.

## Workstreams

### WS1 — TMA the QK^T B operand (unblocks warp-spec automatically)

The QK^T's sole blocker is `xnb`'s symbolic innermost dim. Two ways to make the inner dim static (either re-enables the
existing M-masked-style TMA path; warp-spec then fires via WS2):

**Option A (recommended) — static-pad the materialized `xnb` inner (N) extent.** `005_split_demoted` /
`_split_demoted.py` materializes the rotated-K cone as `xnb[a0, k, n]` (canonical, K second-to-last). Allocate its inner
`n` dim at a **static TMA-aligned width** `W` (e.g. `ceil(DEFAULT_SEQ_HINT / boxN) * boxN`, ≥ the runtime seq) instead of
the symbolic `seq_len`. Then:
  - the producer writes the real `[k, seq]` columns into the `[k, W]` buffer;
  - `k`-stride = `W` (static, 16 B-alignable) → `050_use_tma` accepts it;
  - the TMA descriptor uses runtime `globalDim[n] = seq_len` and hardware OOB zero-fill for the `W − seq` overhang —
    same mechanism as the existing M-masked outer-dim path;
  - the consumer's canonical `x2.trans` staged ldmatrix path is **unchanged** (no transposed-B emit needed).
  - Touch points: `_split_demoted.py` (buffer extent), `ir/tile` `Source`/`Smem` alloc, `050_use_tma.py` eligibility
    (accept a symbolic inner dim whose *allocated* extent is static + aligned), `100_materialize_tile` /
    `_stage_expand` descriptor `globalDim`/`boxDim` derivation.

**Option B — transpose-materialize `xnb` as `[n, k]`** (inner `k`=128 static). This reuses the existing M-masked TMA
(symbolic `n` is now the outer dim) with no padding, but the consumer's B is then transposed-B and must load via
**staged `ldmatrix` without `.trans`** — the path PR #251 left as future work (currently transposed-B is gmem-direct
only; see `ir/kernel/ir.py` `LdmatrixLoad.b_trans` and the `NotImplementedError` guard in the staged branch). More
general (no padding, no wasted smem) but requires implementing that staged lane map first.

Recommendation: **start with Option A** (smaller blast radius, reuses canonical staging + existing OOB semantics);
keep Option B as the follow-up that also generalises to other transposed-B operands.

### WS2 — Warp-specialization (free once WS1 lands)

`085_warp_specialize` fires automatically on the TMA depth-2 bundle WS1 produces. Verify:
  - the producer/consumer split (`_split_by_role`) reaches the QK^T's TMA bundle (it recurses through
    `SerialTile(serial_outer)`/`RegisterTile`/`WarpTile` — confirm the masked-tile boundary `Cond` wrapping the cell
    doesn't sit between the split root and the bundle, blocking the recursion; if it does, hoist it as 021 does);
  - the per-element **store** guards stay in the consumer body (TMA only removes the *load*-side masking; stores still
    need `if (coord < seq_len)` — that's correct and stays);
  - the **1024-thread CTA limit** (`085:141`): the role split adds `producer_warps` on top of the WM·WN consumer tile;
    every fp16 TMA+WARPSPEC bench_fail in the 2026-06-12 sweep was 33 warps × 32 = 1056 threads. Ensure the QK^T tile
    leaves a producer-warp's headroom (gate enumeration so `(WM·WN + producer_warps)·32 ≤ 1024`).

### WS3 — Masked-K TMA for the P@V (harder; codebase-acknowledged)

`050_use_tma:204` already names this: replace `_stage_expand`'s manual `(k < seq_len) ? v : 0` partial-K-slab zero-fill
with **TMA's hardware OOB zero-fill**, which re-enables the async ring + warp-spec for the masked-K P@V. The blocker is
the **4-D leading-batch box derivation** (the P@V `xnb[head, k, n]` "box can't collapse arr" bench-fail). Steps:
  - fix the descriptor derivation for a 4-D source with a leading batch (head) axis so the box collapses correctly;
  - confirm TMA OOB zero-fill on the K (reduce) axis matches the mma-accumulation requirement (overhang must read **0**,
    not a clamped duplicate — TMA zero-fill is exactly right, unlike the M/N edge-clamp);
  - then `040_use_ring_buffers` + `050` + `085` can drop the masked-K SYNC pin (lines 62-68, 201-209).

## Risks

- **fp16 TMA+WARPSPEC CTA overflow** (`085:141`) — mitigate by gating tile size (WS2).
- **TMA descriptor constraints** — `boxDim ≤ 256` per dim (`050:94`), inner stride 16 B-aligned. Option A's padding must
  respect both.
- **Wasted smem / DRAM** from Option A's inner padding (`W − seq` columns) — bounded (one operand, padded to a box
  multiple); measure vs the gain.
- **Store guards remain** — TMA OOB only helps loads; the per-element causal-mask + scale + boundary store epilogue is
  unchanged (already correct from PR #251).
- **Re-tune cost** — the warp/TMA variants are compile-heavy (the tuner CPU-thrash noted in
  `qwen3-embedding-0.6b-layer0-low-performer-analysis.md`); a dynamic re-tune to make greedy deploy them is part of
  validation, not free.

## Validation

1. **Unit** — a test forcing TMA+WS on a symbolic-N matmul (extend `tests/compiler/test_matmul_mma_transposed_b.py` or a
   new `test_matmul_mma_tma_masked.py`): assert `cp.async.bulk`/TMA + the warp-spec role split appear in the emitted
   CUDA for a `Dim('seq_len')`-N matmul, and accuracy vs reference at a straddling seq.
2. **A/B** — `deplodock run --ir <dynamic QK^T repro> --bench --ab "TMA=1,WARPSPEC=1,…"`; target ≈ the static 31 µs.
3. **Re-tune + greedy** — re-tune the dynamic prior; confirm greedy deploys TMA+WS; layer-0 e2e accuracy PASS + latency
   (target: dynamic e2e 210 → toward tc 147), `deplodock compare` dynamic-before vs after.
4. **No-regression** — full compiler suite + the static path unchanged.

## Expected payoff

The dynamic QK^T should approach the static 31 µs (from 73 µs) — most of the static-vs-dynamic gap is exactly this
transport+schedule. WS3 extends the same win to the P@V. Phasing: **WS1 (biggest, self-contained) → WS2 (free) → WS3
(P@V, harder)**.

## Status (implementation outcome)

**WS1 + WS2 — DONE.** The realization differs from the literal Option A in one correctness-critical way: a fixed
static inner width `W ≈ hint` would overflow the buffer at runtime `seq_len > hint` (the masked-tile kernel's whole
point is seq-agnostic correctness; intermediate scratch is sized from the runtime `seq_len` via
`program.resolve_shape`, and the liveness-reused slab is not re-zeroed per launch). Instead the inner N extent is
padded to `round_up(seq_len, 64)` — a **runtime-sized, symbolic-but-provably-aligned** extent
(`_split_demoted._pad_inner_for_tma`), so the K dim's gmem stride is 16 B-aligned at any `seq_len` and
`050_use_tma` accepts it (`_inner_stride_aligned`). No descriptor `globalDim` change is needed: the padded
`[seq_len, round_up)` overhang columns feed the mma only into store-masked output positions, so they're
value-neutral. Warp-spec then fires automatically on the resulting TMA depth-2 bundle (`085`'s existing 1024-thread
gate already covers the CTA-overflow risk). Verified end-to-end on the synthetic rotary-QK^T shape (computed-B cone,
`M = N = seq_len`): greedy deploys `TMA=True, WARPSPEC=True, RING=2, mma_m16n8k16_f16`, emits
`cp.async.bulk.tensor` + `mbarrier.arrive.expect_tx` (no legacy `cp.async.commit_group`), and accuracy holds at
`seq ∈ {31, 130, 512, 700}` (below / at / above the hint). Tests:
`tests/compiler/test_matmul_mma_masked.py::test_demoted_symbolic_n_b_operand_reaches_tma_and_warpspec` +
`::test_demoted_symbolic_n_tma_accuracy`. Touch points landed: `_split_demoted.py` (inner-extent padding +
`_pad_inner_for_tma`), `050_use_tma.py` (`_inner_stride_aligned`, relaxed `inner_symbolic_bufs`). The descriptor /
materializer needed **no** change — the existing per-launch `globalDim`-from-`arr.shape` path handles the padded
buffer directly.

**WS3 — NOT LANDED (blocked; deeper than the plan's framing).** Investigation found compounding blockers, several
with real device-hang risk, so it was not landed rather than shipped half-working/unverified:
  1. The masked-K **K_o serial loop has a symbolic extent** `ceil(seq_len / (BK·atom_k))`, and
     `040_use_ring_buffers._maybe_promote_kouter` refuses to ring-buffer a symbolic-extent K loop. There is **no
     existing precedent** for a symbolic-count ring pipeline — M-masked TMA rings a *static* reduce; the masked M is
     a parallel grid axis, never the serial K_o pipeline.
  2. Force-allowing the symbolic ring exposes a second decline in `050` upstream of the eligibility gate (the
     masked-K bundle never reaches `_bundle_eligible`).
  3. The double-buffered K pipeline's **drain / parity schedule** (hardcoded parities, assumes ≥ `depth`
     iterations) is unverified over a runtime `K_o ∈ {1, 2, …}` — the exact stale-parity re-entry hang the `050`
     docstring already documents for `k_linear_mean_reduce`.
  4. The **foundational hypothesis** — that TMA's *per-dimension* OOB zeroes a **middle** K coordinate (rather than
     reading the next head's data via the linear address) — could **not be verified**, because TMA never fired far
     enough to test it. This must be confirmed before any masked-K TMA work is trusted.
All WS3 experimental edits were reverted; the masked-K SYNC pins (`040:62-68`, `050`'s masked-K decline,
`_stage_expand`'s `has_kmask → SYNC`) remain in place. Recommend a dedicated effort that first answers (4) in
isolation, then designs the symbolic-count ring pipeline (1)/(3) before touching the pins.

## References

- Gates: `deplodock/compiler/pipeline/passes/lowering/tile/050_use_tma.py`, `085_warp_specialize.py`,
  `040_use_ring_buffers.py`; staging `_stage_expand.py`, `020_stage_inputs.py`.
- Prior work: PR #251 (transposed-B + causal-mask epilogue);
  `plans/qwen3-embedding-0.6b-layer0-low-performer-analysis.md` (Finding 1 + re-tune addendum + the per-kernel table).
- Dumps: `_tune/tune-model-qwen3-l0-v3/dumps{,-static}/k_sdpa_reduce_*.{loop,tile,cuda}.txt`.
