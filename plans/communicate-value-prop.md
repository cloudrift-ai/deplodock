# Plan: communicate deplodock's value-prop & prior-art positioning

**Goal.** Make any future agent or human "get the point" of deplodock in under a minute: *what it is, why it is different
from the obvious alternatives, and what not to over-claim.* Today the docs lead with deploy/benchmark and describe the
compiler as merely "a hackable PyTorch → CUDA compiler" — the actual differentiator (an **automated** kernel engineer
that **searches** the modern GPU scheduling space) and any comparison to prior art are absent. This plan adds a single,
consistent message across the README, a new positioning doc, the compiler `ARCHITECTURE.md`, and `CLAUDE.md`.

**Write for the target state, not the refactor.** Describe the end-state system (every regime lowered through the
block-DAG path: pointwise / reductions / matmul / attention, with tensor cores, async pipelines, warp specialization, TMA,
symbolic shapes). Do **not** describe the current half-recovered Tile-IR state. The *one* guardrail against dishonesty:
every doc that makes a capability or perf claim links to a single **status source of truth** (a short "what ships today"
note — see "Status guardrail") so a future agent knows the gap between the vision and `main`. Aspirational framing, one
honest pointer.

## Source material — align to these, don't re-invent (and don't out-claim them)

The team already has a **canonical, carefully-honest voice** in the CloudRift blog
(`/home/dikobraz/Projects/cloudrift-landing/content/blog`, public at cloudrift.ai/blog). The docs should *synthesize and
link* these, matching their framing and their numbers exactly — not invent a louder claim. Reading them is mandatory
before writing copy:

- **"A Principled ML Compiler Stack in 5,000 Lines of Python" (Parts 1–3)** — the core value-prop voice: a *hackable, fully
  inspectable, from-scratch* PyTorch→CUDA stack (six printable IRs, no library use). **Part 2** is the strongest honest
  headline: *"a manageable compiler stack — six IRs, sixteen Tile-IR rules, ~8K lines of Python — can match and on some
  shapes beat a production stack (PyTorch eager + Inductor)… geomean **1.11× vs eager**, **1.20× vs `torch.compile`**
  (per-kernel), full-block parity at TinyLlama-128 / Qwen-128 on FP32"* — and the key differentiator: *"the same sixteen
  rules… produce three different final kernels, with **no kernel-specific code anywhere**."* **Part 3** adds the autotuning
  result: SP-MCTS over Tile-IR rewrites → on RTX 5090 **geomean 0.96× vs eager end-to-end** (tuned; vs 0.87× heuristic),
  **32 of 84 kernel shapes beat PyTorch's hand-optimized kernels, max 5.6×**.
- **"Modern GPU Matmul Optimization: Tensor Cores, TMA, Warp Specialization"** — the proof the compiler *covers the modern
  decision space*: register tiling, vectorized loads, smem staging, cp.async double-buffer, **TMA, warp specialization,
  split-K, tensor cores**, each toggled on a real generated kernel. Reaches **96% of cuBLAS on 2048² fp32, 105% vs the
  fp16 tensor-core path**.
- **"Surfacing a 60% performance bug in cuBLAS"** (a.k.a. "Beating cuBLAS on RTX 5090") — **the honesty model.** The "beat
  cuBLAS" headline is explicitly a **cuBLAS dispatcher *bug*** (sm_120 batched FP32 picks a tiny `simt_128x32` kernel at
  ~42% FMA pipe); when cuBLAS dispatches correctly, deplodock's generated TMA kernel is **~90–95% of cuBLAS / ~93% of
  hand-tuned CUTLASS — in ~300 lines of generated C vs thousands of template lines**. Its own conclusion is "don't trust
  cuBLAS blindly," *not* "we are faster than NVIDIA."
- **Part 1's standing caveat (quote it):** *"Vendor kernels are still hard to beat at full prefill on the FFN-width
  matmuls, which is why every production stack falls back to cuBLAS/cuDNN/CUTLASS on the heavy hitters and code-generates
  everything around them."*

- **"Optimizing Qwen3 Coder for RTX 5090 and PRO 6000"** — evidence for deplodock's *second* pillar (the part the README
  already leads with): the **deploy + benchmark + config-tuning harness**. 277 → 1,207 tok/s on a PRO 6000 from
  *serving-config* tuning (not kernels). Keep this **separate** from the compiler claims — see "Two pillars" below.

**Two pillars — don't conflate them.** Deplodock is (1) a *deploy/benchmark/serve harness* (vLLM/SGLang recipes, cloud
provisioning, the `serve` embedding plugin, config sweeps — the Qwen3-Coder tok/s wins) **and** (2) the *automated CUDA
compiler* (the value-prop this doc work foregrounds). The README today leads with (1) and buries (2); the fix is to make
(2) legible *alongside* (1), not replace it. A perf number from one pillar must never be quoted as if it came from the
other (the 4× tok/s is config tuning; the 1.11× geomean is codegen).

The README already links Part 1; the docs should link the whole series. The **2×2 positioning below is the one thing the
blog does *not* make explicit** — that's the net-new contribution of this doc work.

---

## The canonical message (reuse verbatim — do not paraphrase per-file)

**One-liner.**
> **Deplodock is an automated CUDA engineer.** Hand it a PyTorch `nn.Module`; it traces, fuses, and **decides every
> scheduling choice itself** — tiling, shared-memory staging, async transport (cp.async/TMA), software pipelining, warp
> specialization, tensor-core atoms, split/cooperative-K — then emits optimized, **fully code-generated** CUDA. No
> hand-written kernels, no kernel DSL, no vendor library in the hot path.

**What it is (the three load-bearing bullets).**
- **Automated, not authored.** Unlike CUTLASS/CuTe, TileLang, or Triton — where a *human kernel engineer writes the
  kernel/schedule* — deplodock makes the decisions. You write a `torch` expression; it produces the kernel.
- **Covers the modern decision space as a *searchable* schedule.** Warp specialization, TMA, multi-stage async
  pipelining, and tensor-core atoms are first-class, *ranked* scheduling choices — not compiler-internal heuristics
  (Triton) and not hand-written templates (CUTLASS).
- **Learned + principled.** An MCTS over the schedule space is steered by a **learned cost prior**; move *legality*
  comes from **algebra** (carrier traits — associative / commutative / has-identity, SEMIRING / MONOID / TWISTED_MONOID),
  so one engine handles matmul, reductions, attention, and pointwise — including **symbolic (dynamic) shapes**.
- **Hackable and fully inspectable.** The whole stack is ~5,000 lines of Python (vs TVM's 500K C++ or the
  Dynamo/Inductor/Triton tower), with **every one of six IRs printable on demand** (`--ir torch|tensor|loop|tile|kernel|cuda`).
  You can read it end-to-end, diff a single optimization, and see the emitted CUDA — the appeal the blog series leads with,
  and the real differentiator vs the opaque production compilers.

**The positioning — a 2×2 (the heart of the prior-art comparison).** The right axes are *authored vs automated* ×
*covers the modern GPU decision space or not* — **not** "Halide-lineage" (Halide's schedule language predates and does
not express warp-spec / TMA / tensor cores at all).

| | **Authored** (human writes the kernel/schedule) | **Automated** (machine searches the schedule) |
|---|---|---|
| **Modern space** (warp-spec / TMA / async pipeline / tensor cores) | CUTLASS · CuTe · **TileLang** · Graphene · Triton | **▶ deplodock ◀** |
| **Classic space only** (tile / unroll / vectorize) | Halide | Ansor / MetaSchedule · Hidet · torch.compile (Inductor) |

> Every system that *expresses* warp-spec/TMA is in the **authored** column; every *automated* compiler is in the
> **classic-space** row. Deplodock targets the sparse **automated × modern** cell. Per framework: **Halide** — no
> warp-spec/TMA/tensor-cores. **TVM/Ansor** — tensorize + `software_pipeline`, but warp-spec/TMA are not searchable
> first-class primitives. **Triton** — does warp-spec/TMA, but compiler-internal and *auto*, not a searched user
> schedule. **CuTe/TileLang/Graphene** — express it all, but a human authors it. **torch.compile/Inductor** — automated,
> but matmul leans on cuBLAS/CUTLASS templates, not a from-scratch searched schedule.

**Honest delta (what is actually new).** Not "code-generated kernels" (Triton/TVM/Hidet do that) and not "an autotuner
beats cuBLAS on a shape" (established). The contribution is the **occupied cell + the unification**: a fully *automated*
PyTorch→CUDA compiler that *searches* the modern decision space with a *learned prior*, where move legality is *uniformly
licensed by carrier algebra* (the one genuinely uncommon design idea) across regimes and symbolic shapes — substantiated
by **vendor-competitive, fully-generated kernels end-to-end**.

---

## Messaging discipline (what NOT to claim — bake into every doc)

These keep the claim bulletproof under expert scrutiny (a reviewer will probe exactly here):

- **Don't headline "we beat cuBLAS" — the blog itself debunks that as a vendor *bug*.** The defensible headline is the
  blog's: *competitive with hand-tuned vendor/CUTLASS perf (~90–96%) from a fully code-generated, autotuned stack with
  ~10× less code* — and good enough as a measurement tool to *surface* a 60% cuBLAS dispatcher bug. The "1.4–1.7× cuBLAS
  on RTX 5090 batched FP32" number is a **cuBLAS dispatch bug on sm_120**, not deplodock superiority; always say so.
- **Perf claims carry arch + baseline + precision, always.** Name the GPU and the dtype path (FP32 SIMT vs FP16/BF16
  tensor-core — they dispatch entirely differently). Report the metric the blog uses (**FMA-pipe utilization / geomean vs
  eager / % of cuBLAS**), per-shape not cherry-picked. Compare against the *correctly-dispatched* cuBLAS / CUTLASS /
  `torch.compile(mode="max-autotune")`, not default torch (whose matmul is just cuBLAS). Carry Part 1's caveat: vendor
  kernels are still hard to beat at full prefill on FFN-width matmuls.
- **State the win/loss distribution honestly — it's more credible than a single number.** Per Part 2: deplodock **wins
  big on small kernels** (small RMSNorm/softmax/kv-proj, where Inductor's per-op launch overhead costs 4.5–7.1×),
  is **roughly at parity on the median shape**, and **loses on dense FFN-width matmuls at long sequence** (cuBLAS still
  wins the heavy hitters). Even the fp16 "beats cuBLAS at 105%" has the asterisk that cuBLAS dispatches an *Ampere-era*
  HGEMM on Blackwell. "Competitive overall, wins the small/fused tail, trails on heavy GEMM" is the true and durable
  shape of the result.
- **Numbers drift with versions — report the conservative read.** Part 2 notes the eager baseline swings (geomean 1.11×
  → 1.53× on PyTorch nightly) almost entirely from cuBLAS-heuristic drift on the *torch* side; deplodock's absolute
  timings stay within ~10%. Quote the conservative figure and pin the CUDA/cuBLAS/driver versions (as the cuBLAS post
  does).
- **Flash attention is the conspicuous gap** — the highest-value modern kernel and the one most defined by
  warp-spec/TMA/async. State it as not-yet-covered; don't imply "beats everything."
- **The moveset is human-designed.** Deplodock automates *search over an engineered move space* (like every
  auto-scheduler); it doesn't invent strategies from scratch. "Automated CUDA engineer" describes the workflow, not
  autonomy.
- **Separate design-novelty from the systems result.** The compute/schedule split is old; the contribution is the
  automated-modern cell + algebra-licensed unification + the empirical result. Don't sell the IR abstraction as the
  novelty.
- **Never anchor on "Halide."** It is the wrong reference for the modern-scheduling claim (see the 2×2). Anchor on
  CuTe / TileLang (authored-modern) and Ansor / Inductor (automated-classic).

---

## Files to change

### 1. `README.md` — add a tight "Why deplodock" section + the 2×2; keep it example-driven

The README convention is short / example-driven, so do **not** add narrative — add:
- A **one-paragraph "Why deplodock"** block right under the tagline (the one-liner + the three bullets, compressed).
- The **2×2 table** (verbatim above) with its one-line caption — this *is* the prior-art comparison; it reads in 10
  seconds.
- A one-line **Status** pointer (→ the status source of truth) so the vision/`main` gap is explicit.
- Leave the existing Compile/Benchmark/Deploy/Serve examples as-is; the RMSNorm `--ir cuda` example already *shows* the
  "fully code-generated" claim — reference it from the Why block ("see the emitted CUDA below").

Placement: directly after the `**Compile → Benchmark → Deploy ...**` tagline line, before `## Install`.

### 2. `docs/positioning.md` — NEW, the deep comparison (linked from README + the docs site)

The home for the *narrative* the README omits. Contents:
- The canonical message (one-liner + 3 bullets) expanded a paragraph each.
- The 2×2 + the **per-framework breakdown** (Halide / TVM-Ansor / Triton / CuTe / TileLang / Inductor — each: what it
  does and does not automate/express).
- The **honest delta** + the **messaging-discipline caveats** (above) as a "How we talk about performance" subsection.
- A short **"Design contribution"** paragraph: algorithm (invariant DAG) + Schedule (annotations) + one deterministic
  `assemble` + **algebra-licensed moves** + learned prior — and pointers to `plans/tile-ir-block-dag.md` (design) and
  `deplodock/compiler/ARCHITECTURE.md`.
- Wire it into the docusaurus site (`docs/` Docusaurus project: add to `sidebars.js` + a landing link) so it appears on
  the public docs, and link it from the README "Why" block.

### 3. `deplodock/compiler/ARCHITECTURE.md` — add a "Design philosophy" header

For contributors/agents who open the compiler. A short section near the top: **"Why this is an automated CUDA engineer,
not a kernel DSL."** State the strata (algorithm / Schedule / assemble), the *automated-search* model (enumeration forks +
learned prior + MCTS), and the algebra-licensed legality — and link `plans/tile-ir-block-dag.md`. This is where a future
agent learns the *mental model* before touching passes.

### 4. `CLAUDE.md` — fix the Project Overview lede

Today it opens "Deplodock is a Python tool for deploying and benchmarking LLM inference…" — burying the compiler, which is
the differentiator. Re-lead with the automated-CUDA-engineer framing (one sentence), then the deploy/bench capabilities.
Agents load `CLAUDE.md` first; the lede sets their model of the project.

### 5. (Optional) `docs/` Docusaurus landing — mirror the one-liner + 2×2 on the site's home/intro page

If the public docs site has an intro page, put the one-liner + 2×2 there too, so the web audience gets it without reading
the README.

---

## Status guardrail (the one honesty mechanism)

Future-state docs need exactly one place that tells the truth about *now*, or future agents will trust an aspirational
README against a half-recovered `main`. Pick one and link it from every capability/perf claim:
- **Preferred:** a short **"Status & roadmap"** subsection in `docs/positioning.md` — a capability matrix (regime ×
  {lowered? · tensor-core? · TMA? · perf-validated?}) + a one-line "tracked in `plans/tile-ir-block-dag.md` (recovery
  sequencing)." Update it as tiers land (it pairs naturally with the recovery phases R1–R7).
- The README "Status" line is just a pointer to it.

This lets the README/positioning speak in the present tense about the target system while keeping a single, greppable
source of ground truth.

---

## Reusable assets (so the executor doesn't re-derive)

The executor should lift these **verbatim** from this plan: the one-liner, the three bullets, the 2×2 table + caption,
the per-framework breakdown, and the messaging-discipline list. Consistency across files is the point — the same sentence
in the README, the docs site, and `CLAUDE.md` is what makes the message stick for the next reader.

## Sequencing & scope

1. `docs/positioning.md` first (it's the source the others quote) → 2. README "Why" block + 2×2 → 3. `CLAUDE.md` lede →
4. compiler `ARCHITECTURE.md` philosophy note → 5. (optional) docs-site landing.
- **In scope:** messaging + prior-art positioning + the status guardrail.
- **Out of scope:** the actual perf numbers / a Results section (gather those separately with the baselining discipline
  above before publishing any "beats X" figure), and the Tile-IR refactor itself (`plans/tile-ir-block-dag.md`).
- **Gate:** a newcomer (or a fresh agent) reading only the README "Why" block can state, in one sentence, what deplodock
  is and which 2×2 cell it occupies — and cannot mistake an aspirational claim for shipped behavior (the Status pointer).
</content>
</invoke>
