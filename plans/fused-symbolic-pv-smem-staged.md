# Dynamic SDPA P@V — closing the static↔dynamic gap (premise corrected mid-flight)

Status: **steps 1–2 landed; producer step in progress.** This plan began as "fuse the symbolic-`seq_len` SDPA P@V into
one smem-staged tensor-core kernel to **match the static path's fusion**." That premise turned out to be **factually
wrong** — see below — so the plan was redirected to the two changes that actually separate static from dynamic. The
host-side capacity-cap guard (step 1) shipped first as reusable infrastructure; the masked-K consumer's bank-conflict
fix (step 2) shipped next as the real, evidence-backed win.

## Premise correction (the original plan was built on a misread)

The original "Why" claimed the **static** SDPA P@V is a single fused kernel — *"softmax folded into an in-kernel
smem-staged `xn`"* — and that the dynamic path should be made to match it. The dumped kernels
(`_tune/tune-model-qwen3-l0-staticdyn/{static512,dynamic}-dump/07_lowering_cuda.kernels/`) show otherwise:

**Both static and dynamic split P@V into a separate softmax producer + an mma consumer with an HBM round-trip.**

|                       | Static (`k_sdpa_linear_reduce_3d2635`)     | Dynamic (`…_a76a28`)                          |
|-----------------------|--------------------------------------------|-----------------------------------------------|
| Softmax producer      | separate kernel `_xn`, **writes `xn` to HBM** | separate kernel `_xna`, writes `xn` to HBM    |
| Consumer reads `xn`   | from **HBM** (TMA → smem)                   | from HBM (cooperative copy → smem)            |
| Consumer `xn` smem    | **12 swizzle XORs** → no bank conflicts     | **0 swizzle** (plain stride-512) → 3.67M conflicts |
| Producer threads      | `__launch_bounds__(64)` cooperative warp-reduce | `__launch_bounds__(16)` (BR=16) low-occupancy |

So static is **not** fused — it does the same HBM round-trip. It's faster for two concrete, measured reasons, and those
are the only two real opportunities:

1. **Consumer slab layout.** Static swizzles its `xn` smem slab; dynamic's masked-K slab is a plain row-major
   stride-512 `[M][K]` → the 3.67M ld-bank-conflict storm. → **step 2.**
2. **Producer occupancy.** Static's softmax is a 64-thread cooperative warp-reduce (83 % occ, 11 µs); dynamic's is a
   16-thread (BR=16) reduce (38 % occ, 22 µs). → **producer step.**

The originally-proposed "fused single kernel" (old step 3) would go *beyond* what static does — neither config fuses
today — and its justification ("match static") rests on the misread. It is **not** pursued; flash-style fused symbolic
attention remains the real future-work item (CLAUDE.md / the static-vs-dynamic findings name it).

## Step 1 — host-side capacity-cap guard (LANDED, `568630cc`)

A `symbolic_caps` map on `_Compiled` + a hard-error in `_resolve_symbolic` (`backend/cuda/program.py`) when a runtime
input extent exceeds a capacity-capped kernel's baked hint. Inert until a capped kernel exists (no lowering populates it
yet), but fully unit-tested (`tests/compiler/backend/test_program.py::TestSymbolicCapacityGuard`). Reusable
infrastructure for any future hint-baked kernel (e.g. a genuinely-fused smem-staged P@V, were it ever built).

## Step 2 — masked-K consumer alignment pad (LANDED, `05992058`)

The symbolic-seq P@V consumer stages its softmaxed `P` in a flat `[…, M, K]` smem slab read by `ldmatrix.x4`. With
K=512 (fp16) the M-row stride is 1024 B — a 128-byte bank multiple — so the ldmatrix M-row lanes all alias one 4-byte
bank: **3.67M load conflicts** (NCU 38 µs @ 26 % occ vs static's 16 µs).

`070_pad_smem` cannot fix it: it skips SYNC bundles (`kmask` pins masked-K to SYNC) **and** block-stamped MMA sources
(its `+1` pad breaks ldmatrix's 16 B alignment — its own comment names *"a future MMA-friendly swizzle"* as the answer).
A flat `[M][K]` slab can't reach static's 0-conflict floor anyway (that needs static's swizzle-atom-wide K-subtile
relayout); the deployable flat-slab fix is an **alignment-preserving inner pad**.

Implemented intrinsically on the `Source` at creation (`020_stage_inputs._masked_k_mma_pad`, where
`kmask`/addressing/`bytes_per_elem` converge): one 16-byte ldmatrix chunk (8 fp16 elems) on the innermost cache dim
steps the M-row stride off the 128-byte alias while keeping every row 16 B aligned. Intrinsic (not a `070` autotune
fork) because it's a near-strict win with no misalignment penalty — so **greedy deploys it without a re-tune**; `070`
then self-skips the already-padded source.

**Measured (RTX 5090, dynamic Qwen3-Embedding-0.6B layer 0, -O3, CUDA-graph captured):**

- P@V consumer ld-conflicts **3,670,114 → ~1,000**; NCU duration **38.2 µs → 27.4 µs** (−28 %).
- Numerically transparent — `max_diff` unchanged (0.000977), accuracy PASS (the pad only adds dead smem columns).
- Full-layer-0 e2e **340 µs → 329 µs**; the attention path's gap to static narrows ~51 µs → ~28 µs.

Tests: `tests/compiler/passes/test_masked_k_mma_pad.py` (gate + pad value + alignment + dtype-width cases). Full
compiler suite green (1576 passed).

## Producer step — cooperative softmax producer (VALIDATED via re-tune; deploy needs a re-tune)

The dynamic softmax producer deploys at low occupancy (greedy/learned-prior pick: BR=16, ~12–22 µs, 38 % occ; the
analytic/cold pick is worse still — block=2, FM=16, ~96 µs, 0 % occ). Forcing `BR=32` proves the cooperative config is
**reachable and ~2× faster** (the producer drops to ~6 µs at 100 % occ), and the enumeration already offers `BR=32+BM=8`
(head thread-bind) for the symbolic-seq reduce — so this is a **stale-prior artifact, not a structural lockout**: the
dynamic tune wedged on the prior commit (static-vs-dynamic findings, Finding 2 — now fixed on this branch), so this
producer was never tuned on current code. The deployed pick comes from the **learned** prior, which has no surgical
code override for a single op (the analytic/cold half can't change the learned deploy, and hand-editing the offline-fit
analytic weights risks regressing the other golden regimes). The in-scope mechanism is a **re-tune**.

**Validated** by a targeted re-tune of the P@V-cone reproducer (`tune <…a76a28.torch.json> --clean`, isolated DB +
prior, RTX 5090, 935 benches / ~18 min). Deploying with that prior:

- softmax producer `_xna`: **BR=16 / 38 % occ / 12.9 µs → BR=64 / 100 % occ / 7.3 µs** (cooperative warp-reduce, like
  static's 64-thread producer).
- P@V-cone reproducer e2e: **86 µs → 79 µs** (producer cooperativity on top of step 2's padded consumer); accuracy PASS.

**To deploy for default `run`:** a full dynamic re-tune (now that the #244 wedge is fixed) — `deplodock tune
Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench`. The isolated cone-only prior is not merged
into the shared `~/.cache/deplodock/prior.json` (it has evidence only for the P@V cone; replacing the default would lose
every other kernel's tuning). Step 2's consumer pad, by contrast, is intrinsic and deploys today with no re-tune.

A separate, optional code follow-up: the analytic (cold) prior's degenerate block=2 pick for this producer is a real
cold-start deficiency (helps `compile`/`run` before any tune, and re-tune cold-starts) — but the fix is to re-fit the
analytic weights via `scripts/golden_knob_heuristics.py` (jointly over all regimes), not a hand-edit, so it's left as
follow-up.

## Code map (as actually changed)

- `backend/cuda/program.py` — `symbolic_caps` + the `_resolve_symbolic` guard (step 1).
- `lowering/tile/020_stage_inputs.py` — `_masked_k_mma_pad` + its call at the masked-K `Source` construction (step 2).
- `lowering/tile/070_pad_smem.py` — unchanged; self-skips the already-padded source.

## Tests

- `tests/compiler/backend/test_program.py::TestSymbolicCapacityGuard` — guard (no GPU).
- `tests/compiler/passes/test_masked_k_mma_pad.py` — masked-K pad gate / value / alignment (no GPU).
- Accuracy + bench validated on the dynamic P@V reproducer and full dynamic layer 0 (RTX 5090).
