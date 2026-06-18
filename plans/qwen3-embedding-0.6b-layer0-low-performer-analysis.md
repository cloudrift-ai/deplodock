# Qwen3-Embedding-0.6B layer 0 — low-performer root-cause analysis

**Status:** focused follow-up to `plans/qwen3-embedding-0.6b-layer0-static-vs-dynamic-tune-findings.md`. Re-measured the
four flagged low performers on **current HEAD** and root-caused each. The headline result: **only one of the four is a
genuine loser** — the QK^T scores matmul, which is **locked out of the tensor-core tier**. The other three were
reproducer-overlap artifacts (the attention-tail reproducers each re-lower the *whole* attention cone and triple-count
that same QK^T). No re-tune was needed to establish this: the tier lockout is a code gate, not a search shortfall.

**Run commands** (RTX 5090, sm_120, HEAD `9717e493`, ncu 2025.3.1; reused the v2 dynamic reproducers + DB/prior under
`_tune/tune-model-qwen3-l0-v2/`, new artifacts under `_tune/tune-model-qwen3-l0-v3/`):

```bash
SRC=_tune/tune-model-qwen3-l0-v2/dyn-run-dump/07_lowering_cuda.kernels
export DEPLODOCK_PRIOR_FILE=_tune/tune-model-qwen3-l0-v2/dynamic-prior.json
export DEPLODOCK_TUNE_DB=_tune/tune-model-qwen3-l0-v2/dynamic.db
# refresh the four reproducers on HEAD (3-way)
deplodock run --ir $SRC/<kname>.torch.json --bench --bench-backends eager,tcompile,deplodock
# QK^T smem-layout A/B (refutes prior Finding 5) + NCU
deplodock run --ir $SRC/k_sdpa_reduce_77b0f0.torch.json --bench --ab "PAD_SMEM=1" --ab "PERMUTE_LANES=1"
deplodock run --ir $SRC/k_sdpa_reduce_77b0f0.torch.json --bench --profile --dump-dir _tune/tune-model-qwen3-l0-v3/ncu-qkt
```

**Date:** 2026-06-17.

**Run stats.** No new tune — this is a re-bench + A/B + NCU + code-read pass over the v2 dynamic artifacts. The greedy
picks read the v2 partial dynamic prior (`dynamic-prior.json`, ~2900 benches; the v2 dynamic tune wedged — see that
report's Finding 1). Triage used `eval variants` on `static.db` (5670 ok / 4 fail) and `dynamic.db` (2917 ok / 1 fail).

**Number-family disclaimer.** Every latency below is the deployable **-O3** re-bench (CUDA-graph captured), *not* the
tune-DB `-O1` ranking numbers. Where `-O1` ranks are quoted (`eval variants`) they are flagged ranking-only.

**Dynamic measurement semantics.** Symbolic `seq_len`, benched at `DEFAULT_SEQ_HINT=512`; the full-model torch closures
are tiled to 512 so eager / tcompile / deplodock are one shape. The masked-tile boundary guards are part of the cost.

**HEAD note.** `9717e493` is 3 commits past the v2 report's `a5a5690b` (incl. `0eb9dda9`, CUDA-backend liveness-based
scratch reuse). All four reproducer totals reproduced within noise (QK^T 64, MLP 57, o_proj 81, P@V 74 µs), so that
backend change did **not** move these kernels — the v2 numbers are stable on HEAD.

---

## Bench results

### Per-reproducer (current HEAD, dynamic, -O3 captured) — as the user's table sees them

The `depl (fix)` column is the re-bench **after** the transposed-B + causal-mask fix and the QK^T re-tune (the warp QK^T
deploys via greedy); `depl (v2)` is the original scalar-QK^T number. The QK^T scores kernel itself goes
**scalar → `mma_m16n8k16_f16`** in every reproducer.

| Reproducer (kernel)                       | eager | tc  | depl (v2) | depl (fix) | note                                   |
|-------------------------------------------|------:|----:|----------:|-----------:|----------------------------------------|
| `k_sdpa_reduce_77b0f0` (RoPE + QK^T)      |   148 |  25 |        64 |     **73** | QK^T consumer 34→**28.8 warp**; split rotary `xna` producer regressed (20µs) |
| `k_linear_mean_reduce_05d34c` (RMSNorm+MLP)|  119 |  57 |        57 |         57 | unchanged (QK^T not in this reproducer) |
| `k_linear_sdpa_reduce_43208b` (o_proj)    |    39 |  37 |        81 |     **70** | embedded QK^T 39.9→**29.3 warp**        |
| `k_sdpa_linear_reduce_a76a28` (P@V)       |    29 |  29 |        74 |     **64** | embedded QK^T 39.9→**29.3 warp**        |

### The same totals, decomposed into the *unique* kernels they actually run

Each attention-tail reproducer re-lowers the **whole attention dependency cone**, so the three "different" attention
reproducers (64 / 81 / 74 µs) are dominated by the **same** QK^T kernel. Summing them — as the per-kernel table in the
v2 report does — triple-counts it. The unique kernels and their real in-context costs:

The `µs (v2)` column is the original scalar-QK^T cost; `µs (fix)` is after the transposed-B + causal-mask fix + re-tune.

| Kernel                         | tier (fix)      | µs (v2) | µs (fix) | appears in reproducer(s)        | verdict                    |
|--------------------------------|-----------------|--------:|---------:|---------------------------------|----------------------------|
| QK^T scores `k_sdpa_reduce`    | **warp (mma)**  |    ~40  |  **~29** | all three (now `mma_m16n8k16_f16`) | tier-lockout removed; still > tc 25 |
| P@V split `k_sdpa_linear_reduce`| warp (mma)     |    ~23  |     ~23  | QK^T, P@V, o_proj               | competitive                |
| o_proj `k_linear_sdpa_reduce`  | warp (mma)      |    ~16  |     ~16  | o_proj repro                    | fine (eager 39, tc 37)     |
| v_proj tail `k_linear_reduce`  | scalar          |     ~9  |      ~9  | P@V repro                       | fine                       |
| MLP gate/up + down `k_linear_mean_reduce` | warp (mma) | ~54 |   ~54  | MLP repro (26.1 + 27.5)         | ties tc                    |

(The dedicated QK^T reproducer un-fuses the RoPE prologue into `xna`/`xnb` producers + the warp consumer, so its *total*
rose 64→73 despite the consumer getting faster — a producer-overhead artifact; in the o_proj/P@V reproducers the
embedded QK^T deploys warp as a single 29.3µs kernel and the totals drop. See Finding 1's re-tune addendum.)

**The real attention tail is ≈ QK^T 40 + P@V 23 + o_proj 16 + v_proj 9 ≈ 88 µs, not the ~150–195 µs the v2 report's
summed per-kernel rows imply.** And of that 88 µs, the **QK^T scores matmul (~40 µs) is the single dominant kernel and
the only one that loses to torch.compile** (which runs QK^T inside flash at ~25 µs on tensor cores). The o_proj and P@V
are individually competitive — their 81/74 µs "losses" in the user's table are the QK^T riding along inside their
reproducers. This corrects the v2 report's framing (its Findings 2/3, which read the o_proj and P@V as the ~2–2.5× loss).

---

## Finding 1 — QK^T scores matmul is locked out of the tensor-core tier (transposed-B layout)

**Severity: high — this is the dominant attention kernel and the only genuine loss vs torch.compile.**

**Symptom.** The QK^T scores consumer `k_sdpa_reduce_77b0f0` deploys at **scalar tier** (`MMA` blank, FM=4/FN=2 thread
tiles), ~34 µs captured (54% of its reproducer), and is the only attention kernel beaten by torch.compile (25 µs). The
P@V split-consumer right beside it (`k_sdpa_reduce_6e4bd6`) *does* reach `mma_m16n8k16_f16`.

**Evidence — the warp tier is never even enumerated for this kernel:**

```
eval variants --kernel k_sdpa_reduce  (static.db): 37 measured configs, columns BM BN BK BR FM FN FK SPLITK RING
                                                   STAGE FKWIN GROUP_M  — NO MMA / WM / WN column at all
                                       (dynamic.db): 31 configs, same scalar-only column set
```

Every enumerated config is a scalar split-K cooperative reduce. Contrast `k_sdpa_linear_reduce` (P@V): 243 configs, all
with `MMA=mma_m16n8k16_f16`. The QK^T enumeration omits the MMA tier entirely.

**Evidence — NCU, scalar QK^T vs the torch reference** (`_tune/tune-model-qwen3-l0-v3/ncu-qkt/61_ncu_metrics.json`):

| side      | kernel                              | dur ns |  occ% |   sm% | dram% | fma% |   lsu.inst | ld.cnflct | regs |
|-----------|-------------------------------------|-------:|------:|------:|------:|-----:|-----------:|----------:|-----:|
| deplodock | `k_sdpa_reduce_77b0f0` (scalar QK^T)| 57 248 |  70.4 |  62.1 |   4.4 | 48.0 | **4 718 592** |    15 045 |   48 |
| torch     | `flash_fwd_splitkv_kernel`          | 24 736 |   8.2 |  16.3 |   9.8 |  2.9 |    300 800 |         0 |  206 |

The scalar QK^T issues **4.7 M load/store instructions — 15× the flash kernel's 300 K** — and is instruction-bound
(`dram 4.4%` rules out memory-bound; `sm 62%` with `fma 48%` is scalar-FP saturation, not tensor cores). A tensor-core
`mma.sync` path would replace that scalar load/FMA storm with `ldmatrix` + `mma`, which is exactly what flash does.

**Root cause (code-located).** `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py:179-181`, the MMA eligibility
predicate:

```python
a_ld, b_ld = classify_matmul_operands(loads, K_name)
if a_ld is None or b_ld is None:
    return False
```

with the comment at `_atom.py:172-178` naming this exact case: *"a transposed-B Q @ K^T, where BOTH operands carry K in
their last dim — is never offered the mma tier … it falls to the scalar register-tile path instead."* The classifier
(`_atom.py:67-73`) tags `k_in_last and not k_in_first ⇒ A`; for QK^T **both** operands are K-in-last, so both are tagged
A and `b_load` stays `None`. Confirmed at the loop-IR level (`compile … --ir loop`): the reduce axis `a3` (head_dim)
sits in the last index dim of both operands —

```
in12 = load mul_3[0, a1, a0, a3]      # Q[query a1, head a0, dim a3]   → k_in_last
in8  = load mul_5[0, a2, (a0/2), a3]  # K[key a2, head a0/2, dim a3]   → k_in_last  ⇒ b_load = None
```

The `005_split_demoted` split *does* fire on the QK^T (un-fusing the RoPE prologue into `_xna`/`_xnb` producers), but the
resulting clean QK^T consumer **still has the transposed-B layout**, so the split un-fuses the prologue without unlocking
the tier. This is why re-tuning with more patience cannot help — the MMA atom is never offered, so no amount of search
surfaces it.

**Suggested fix (priority: high, hard).** Teach `classify_matmul_operands` + the `kernel/005` MMA emit the **NT
(transposed-B) layout**: when both operands are K-in-last, the M×K operand is A and the N×K operand is B-transposed —
loadable via `ldmatrix.trans` (or by treating K as the contiguous smem dim for B). This is the standard cuBLAS/flash NT
GEMM layout. It unlocks the single biggest attention kernel. A flash-style fully-fused symbolic-K attention (the v2
report's Finding 2 suggestion) is the larger win but strictly harder; just lifting the QK^T to mma.sync is the
self-contained first step and addresses the dominant cost directly.

**Repro.**
```bash
deplodock compile $SRC/k_sdpa_reduce_77b0f0.torch.json --ir loop      # shows both operands K-in-last (no GPU)
deplodock run --ir $SRC/k_sdpa_reduce_77b0f0.torch.json --bench --profile   # NCU: 4.7M lsu.inst, scalar
DEPLODOCK_TUNE_DB=…/static.db deplodock eval variants --kernel k_sdpa_reduce   # 0 MMA configs enumerated
```

**Update — fix landed (branch `feature/qkt-transposed-b-mma`).** Implementing the fix revealed the QK^T was gated by a
**chain** of three blockers, not one; the headline diagnosis above named only the first. All three are now resolved so
the QK^T reaches `mma_m16n8k16_f16` and is numerically correct (`run --ir` accuracy `max_diff=0.0020, PASS`):

1. **Transposed-B classifier** (this finding's named cause) — `classify_matmul_operands` now recovers A/B for a
   both-K-in-last cell from the output coordinates (`out_index`), and the emit reads B gmem-direct via a new
   `dpl_mma_load_b_gmem_trans` helper (`[N,K]` is the native `mma.row.col` col-major B — no `ldmatrix.trans`; carried on
   `Mma.b_trans`, the operand left unstaged by `020_stage_inputs`). Covered by
   `tests/compiler/test_matmul_mma_transposed_b.py`.
2. **RoPE prologue fusion** — turned out to be **already handled**: `tile/_split_demoted` is explicitly built for the
   rotary QK^T and the `SPLIT_CONE` cut materializes a clean **canonical** scores consumer (`xna[h,m,k] @ xnb[h,k,n]`).
   So the binding blocker was not transposed-B at all for this kernel (its split consumer is canonical).
3. **Causal-mask `Select` epilogue** (the *actual* binding blocker) — the clean consumer's epilogue is
   `(n<=m) ? mask_zero : mask_fill`, a coord-predicated `Select` that `classify_fragment_epilogue` rejected. It now folds
   into the fragment store as a per-element ternary (`__M__`/`__N__` placeholders → each element's absolute row/col).
   Covered by `tests/compiler/test_matmul_mma_causal_epilogue.py`.

**Greedy deployment after a re-tune (done).** Re-tuning the QK^T reproducer with the new capability (warm-started from
the v2 dynamic prior into `qkt-prior.json`, 4 fused terminals, ~19.5 min) trains the prior to deploy the warp tier:
greedy now picks the **split + warp QK^T** with no forcing. The scores consumer drops from **34.3 µs scalar → 28.8 µs
warp** (`mma_m16n8k16_f16`), and the full layer-0 dynamic e2e is **accuracy PASS** (`max_diff=0.0039`):

| Layer-0 dynamic e2e (seq 512) | eager | torch.compile | deplodock |
|-------------------------------|------:|--------------:|----------:|
| v2 (scalar QK^T)              |   219 |           147 |       217 |
| this branch (warp QK^T)       |   219 |           147 |   **210** |

So the warp QK^T deploys correctly e2e, but the **net e2e win is small (~3%)** and deplodock still loses to
torch.compile. Reason: un-fusing the RoPE prologue to reach the warp tier materializes rotated Q/K producers (`xna` /
`xnb`) whose cost (~20 µs for the rotary `xna` in this pick — itself a prior-reachability miss) eats most of the
28.8-vs-34.3 warp gain. **A flash-style *fused* tensor-core QK^T (softmax/RoPE on-chip, no materialization) is the real
win** and remains future work; this change removes the tier-lockout that blocked any tensor-core QK^T at all.

**Tuner note (corrects an earlier mis-call).** The re-tune was first reported as a "wedge" (the v2 report's Finding 1).
On re-investigation it was **not** a deadlock: the run was CPU-bound (~1800 % across ~66 parent threads, GPU mostly
idle) but **progressing** — the newly-enabled big warp variants are compile/lower-heavy. It completes if left alone
(~19.5 min for one reproducer). The genuine v2 0 %-CPU deadlock did **not** reproduce here; the SIGKILL-able bench-worker
child-recovery path works (a dirty child exits, a clean one respawns). The high per-variant tuning cost on this kernel
class is a real follow-up (tuner CPU-thrash), distinct from the deadlock.

---

## Finding 2 — Attention-tail per-kernel reproducers overlap (the v2 table triple-counts the QK^T)

**Severity: medium (methodology) — it changes which kernel you optimize.**

**Symptom.** The user's table lists three separate attention "losers" — o_proj (80 µs), P@V (74 µs), QK^T (64 µs) — as if
additive (~218 µs). They are not. Each reproducer's kernel breakdown (`run --ir … --bench`):

```
QK^T  repro (62.8): QK^T 34.2 + QK^T-producers 6.0 + P@V 15.6 + P@V-xn 7.0
P@V   repro (72.8): QK^T 39.9 + v_proj 9.1 + P@V 15.5 + P@V-producers 8.2
o_proj repro (79.4): QK^T 39.8 + P@V 21.8 + o_proj 16.2 + o_proj-xn 1.6
```

The QK^T (~40 µs) appears in **all three**. The reproducer slice is taken by op-provenance, but the o_proj depends on the
attn-out, which depends on P@V, which depends on QK^T — so the dependency cone pulls the whole attention into every
attention-tail reproducer. Summing the reproducer totals counts the QK^T three times.

**Consequence.** The o_proj kernel itself is only **16 µs** (eager 39, tc 37 — deplodock *wins* the isolated op) and the
P@V is **~23 µs** at warp tier. Neither is the ~2–2.5× loss the v2 report's Findings 2/3 attribute to them; that loss is
the QK^T riding inside their reproducers. **Optimize the QK^T (Finding 1), not the o_proj or P@V.**

**Suggested fix (priority: medium, tooling).** The per-kernel reproducer / `62_kernel_bench.json` should either (a) slice
to the *single* target kernel's ops (not the full cone) so the row is the kernel's own cost, or (b) print the
per-kernel breakdown table (which `run --ir --bench` already shows on screen) into the machine-readable JSON, so
`deplodock compare` and any summed "per-kernel total" stop double-counting shared upstream kernels. Today the breakdown
exists only in stdout; the JSON has one number per reproducer, which is the *cone* total.

---

## Finding 3 — PAD_SMEM / PERMUTE_LANES do **not** fix the QK^T bank conflicts (refutes v2 Finding 5)

**Severity: low (corrects a recommendation).**

**Symptom.** The v2 report's Finding 5 attributes the QK^T's tc loss to a smem bank-conflict storm and recommends
`PAD_SMEM` / `PERMUTE_LANES`. A/B on `k_sdpa_reduce_77b0f0` (consumer row), -O3 captured:

| variant                    | QK^T consumer µs | occ% | smem |
|----------------------------|-----------------:|-----:|-----:|
| greedy pick                |             34.2 |  100 | 8.0K |
| `PAD_SMEM=1`               |         **41.2** |   83 | 8.1K |
| `PERMUTE_LANES=1`          |             34.0 |  100 | 8.0K |
| `PAD_SMEM=1,PERMUTE_LANES=1`|        **41.2** |   83 | 8.1K |

`PAD_SMEM` **regresses** it (the extra smem drops occupancy 100→83%); `PERMUTE_LANES` is a no-op. The conflicts are
inherent to the scalar score-tile, not a paddable layout collision — and NCU shows the kernel is **instruction-bound
(4.7 M lsu.inst), not conflict-bound** (15 K ld-conflicts is small relative to 4.7 M LSU). The smem levers cannot fix
a scalar-tier kernel; the MMA tier (Finding 1) is the only fix. Drop the Finding-5 PAD_SMEM/PERMUTE recommendation.

---

## Finding 4 — o_proj's `-O1` search "shortfall" does not exist at `-O3` (ranking-family artifact)

**Severity: none (clears a false flag).**

**Symptom.** `eval variants --kernel k_linear_sdpa_reduce` (dynamic.db, v2 prior) flags the o_proj consumer
`k_linear_sdpa_reduce_43208b` as `pick: rank 5/8, 4.62x of best <-- misses best`. That looks like a search shortfall.

**Why it's not.** The 4.62× is the **-O1 tune-ranking** gap; the picked fork deploys at **16.2 µs -O3** (the o_proj
kernel row in its reproducer), which *beats* eager (39) and tc (37). The -O1 ranking ties/misorders configs that diverge
at -O3 (documented behaviour — `tune` re-benches near-best configs at -O3 precisely because of this). The o_proj needs
no action; the rank-5 flag is the -O1/-O3 family divergence, not a deployable miss. (A clean confirmation requires the
-O3 reservoir column, which the wedged v2 dynamic prior lacks for these rows — the deployed 16 µs is the truth.)

---

## Finding 5 — MLP and P@V are at the warp tier and competitive (not losers)

**Severity: none (scopes the work).** `k_linear_mean_reduce_05d34c` (post-attn RMSNorm + MLP gate/up) deploys two
`mma_m16n8k16_f16` kernels (gate/up `_mm1` 26.1 µs + down/consumer 27.5 µs) and **ties** torch.compile (57 vs 57 µs,
2× eager). The P@V split is warp-tier (~23 µs). Both reach the tensor-core tier the QK^T cannot, and `eval variants`
shows their greedy picks are well-reached (P@V consumer rank 2/243; MLP consumer rank 2/51). The v2 report's secondary
note on the P@V `_xna` softmax→gmem spill (dram-bound producer) still holds as a smaller residual, but P@V is not where
the layer's time is. **The whole attention/MLP tail reduces to one actionable kernel: the QK^T (Finding 1).**

---

## Repro / artifacts

Work dir: `_tune/tune-model-qwen3-l0-v3/` (gitignored). Reuses v2 reproducers/DB/prior under
`_tune/tune-model-qwen3-l0-v2/`. HEAD `9717e493`, RTX 5090 sm_120.

- Re-bench logs: `rebench_k_*.log` (3-way eager/tc/deplodock + per-kernel breakdown).
- QK^T smem A/B: `ab_qkt_smem.log`. QK^T NCU: `ncu-qkt/61_ncu_metrics.{csv,json}`, `ncu_qkt.log`.

```bash
SRC=_tune/tune-model-qwen3-l0-v2/dyn-run-dump/07_lowering_cuda.kernels
export DEPLODOCK_PRIOR_FILE=_tune/tune-model-qwen3-l0-v2/dynamic-prior.json
export DEPLODOCK_TUNE_DB=_tune/tune-model-qwen3-l0-v2/dynamic.db
# the decisive, GPU-free checks for Finding 1:
deplodock compile $SRC/k_sdpa_reduce_77b0f0.torch.json --ir loop   # both QK^T operands K-in-last
DEPLODOCK_TUNE_DB=_tune/tune-model-qwen3-l0-v2/static.db deplodock eval variants --kernel k_sdpa_reduce  # no MMA configs
# the decomposition (Finding 2) — read the per-kernel breakdown each prints:
for k in k_sdpa_reduce_77b0f0 k_sdpa_linear_reduce_a76a28 k_linear_sdpa_reduce_43208b; do
  deplodock run --ir $SRC/$k.torch.json --bench --bench-backends eager,tcompile,deplodock; done
```

---

## Workflow notes

For whoever maintains the deplodock CLI and the `tune-model` skill.

1. **The per-kernel reproducer overcounts shared upstream kernels (Finding 2) — the biggest analysis trap here.** Three
   "different" attention reproducers were really the same QK^T measured three times; the v2 report summed them into a
   ~150–195 µs "attention tail" that does not exist. I only caught it by reading the per-reproducer kernel breakdown
   tables by hand. *Proposed:* (a) write the per-kernel breakdown (already printed to stdout by `run --ir --bench`) into
   `62_kernel_bench.json`, and (b) have the reproducer slice target the single kernel's ops, not the full dependency
   cone — or at minimum label the reproducer total as a *cone* total in the table header. This would have made the
   correct decomposition a one-command read instead of four log reads + arithmetic.

2. **`eval variants` should mark `-O1`-only "shortfalls" that vanish at `-O3` (Finding 4).** The `<-- misses best` flag
   fires on the -O1 ranking; when the picked config has an -O3 reservoir row that is competitive, the flag is a false
   alarm. *Proposed:* suppress (or down-rank) the `misses best` flag when the pick's `-O3 us` is within tolerance of the
   best `-O3 us`, since -O3 is the deployable family. As-is it sent me chasing a non-existent o_proj search miss.

3. **No way to A/B a *tier* the way you A/B a knob.** Finding 1 is a tier lockout; I could prove the *current* tier is
   slow (NCU) and that the MMA tier is *never enumerated* (`eval variants`), but I could not bench "what the QK^T would
   cost on tensor cores" because the atom is gated off upstream — `--ab "MMA=mma_m16n8k16_f16"` can't override an
   eligibility predicate that returns False. *Proposed:* a `--force-atom` / `DEPLODOCK_FORCE_MMA` escape hatch that
   bypasses `is_atom_eligible` for an A/B (accepting it may crash in emit), so a tier-lockout finding can be quantified,
   not just argued. Today the fix has to be implemented before its payoff can be measured.

4. **Reusing the wedged v2 dynamic prior is a caveat I had to track manually.** The greedy picks read a partial prior
   (the v2 dynamic tune wedged ~2900 benches in). For the QK^T finding it doesn't matter (the tier is gated regardless of
   prior), but for any reachability claim it does. *Proposed:* the v2 report's Finding 1 wedge fix (recover the parent
   CUDA stream after a HungKernel SIGKILL) is still the prerequisite for a *clean* dynamic prior on this model — it has
   not landed on HEAD `9717e493` (verified: the three commits since `a5a5690b` are serving/backend, none touch the
   bench-worker recovery). Until then, dynamic tunes on this model stay salvage-only.

5. **Confirmed stable across the HEAD bump.** Re-benching the four reproducers on `9717e493` reproduced the v2 numbers
   within noise — `deplodock compare` would have shown this in one command, but the v2 dynamic run never wrote a
   `62_kernel_bench.json` (it wedged first), so the cross-run diff isn't available for the dynamic side. Reinforces note
   1 + the v2 report's workflow note 4 (give `run --bench --dump-dir` the `62_kernel_bench.json` writer).
