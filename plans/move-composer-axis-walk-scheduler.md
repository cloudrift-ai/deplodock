# Move composer: walk the annotated nest instead of matching three regimes

**Branch:** `feature/move-composer` (continues the hierarchical move composer)
**Status:** Phases 1 + 3 DONE (`partition/walk.py::walk_nest` replaces the three `lift_*`; the MONOID epilogue/map
envelope is dropped → RMSNorm + softmax compose with no new regime, whole-CTA geometry, accuracy verified). Phase 2
(ordered multi-transform `_assemble`) and Phase 4 (multi-different-axis reduce) folded forward: the single
`_replace_k_coop` already splits all same-extent K loops (reduce(s) + map) via `target_names`, so softmax's two
reduces + map work without the ordered-list generalization; a genuine multi-*different*-axis reduce (e.g.
multi-accumulator matmul) is still deferred. Phase 5 (TWISTED_MONOID / flash) remains blocked on the MMA-flash carry.

This plan replaces the three template-matchers with one **nest walk** that tags each axis algebraically and feeds
the already-shared `_assemble` emitter — unlocking the fused prologue / epilogue / multi-reduce cases that were
stranded on legacy, with no new "regime" added.

## Coverage status (autonomous gap-closing)

The composer now covers, opt-in behind `DEPLODOCK_MOVE_COMPOSER` (legacy is still the default + the fallthrough):
pointwise; matmul scalar + warp-MMA, with a MAP **epilogue** (QK^T scale, `matmul_add`) and **split-K**;
cooperative reduce + **RMSNorm / softmax / LayerNorm** (reduce → epilogue → second-pass map, multi-monoid);
**multi-accumulator** matmul (gated MLP / SwiGLU, via the cooperative multi-Accum path); **symbolic free axes**
(dynamic seq_len rows / M / N, masked ceil-div tiles); and **symbolic K** cooperative reduce (masked-K fill with
the carrier identity — the monoid-DAG mechanism). Forcing the composer on across the correctness-critical suite
now yields **31 failures (was 123)**, and *none* are composer correctness bugs — real models (`test_block`,
`test_attention_chains`, `test_dynamic_shapes`, the `cli` accuracy tests) PASS, because the composer covers most
kernels and cleanly declines the rest to legacy. The 31 are all legacy-structure / transport assertions:

- **Transports** (`test_matmul_mma_tma`, `test_matmul_mma_staged_pipelined`, RING / half2-window) — the composer
  stages SYNC; it does not yet emit TMA / cp.async / ring / fp16-half2-window. Perf parity, not correctness.
- **Warp masked-MMA** (`test_symbolic_m_masked_mma_*`) — symbolic-free matmul routes scalar; warp masked-MMA is
  declined to legacy.
- **Strided-cooperative rows** (`test_lowering_2d_coop_*`) — the composer does whole-CTA (correct, lower
  occupancy); the tests assert the strided multi-row structure.

**Retiring the legacy planner therefore still needs:** fused SDPA **P@V** / flash (`{MONOID, SEMIRING}` over one
K + online softmax — `TWISTED_MONOID`, blocked on the MMA-flash carrier, this plan's Phase 5 +
`algebraic-carrier-analysis` C1b–C5); the **matmul transports** (TMA / cp.async / ring — perf); **warp
masked-MMA** for symbolic-M; and **strided-cooperative rows**. None is a quick win; flash is the gating item.

## Framing — this is mostly inversion, not new machinery

Two facts about the current `try_compose` path make the walk tractable:

1. **The classification is already algebraic, not op-type pattern matching.** The three `lift_*` functions gate on the
   annotation the nest already carries — `Loop.algebra_kind`, computed bottom-up from the carrier
   (`compiler/ir/algebra.py::classify_algebra`): `lift_matmul` requires `AlgebraKind.SEMIRING`
   (`partition/skeleton.py:80`), `lift_coop_reduce` requires `AlgebraKind.MONOID` (`skeleton.py:150`), `lift_pointwise`
   requires "no reduce carrier anywhere" = `MAP` (`skeleton.py:196`). The loop nest **is** algebraically annotated and
   that annotation already drives the dispatch. What the `lift_*` functions add on top is *envelope recognition* — the
   rigid whole-kernel shape checks — and that is the part this plan removes.

2. **`_assemble` is already a regime-agnostic, axis-walking emitter.** `partition/materialize.py:66`. The three
   `build_*_tile` callers differ by exactly three things when they call it:
   - **free-axis tiling**: real thread/reg factors (pointwise, matmul) vs forced `(1, 1)` so rows go to grid
     (coop-reduce, `materialize.py:229`);
   - **the reduce-axis transform**, passed as the `k_transform` callback: `None` (pointwise), `_replace_k_scalar`
     (strip-mine + split-K), `_replace_k_coop` (cooperative lanes), `_replace_k_warp` (atom-strided);
   - **where extra axes land in the tower**: `coop_thread`, `extra_block`, `atom`.

   Everything else — the σ-split of each free axis (`_split_free_axis`), masked-store `Cond` guards, the
   innermost-first `_wrap_tower` — is already shared and already structured as "walk the axes, emit per axis." We are
   ~80% to a generic scheduler; the missing 20% is the recognition front-end and a per-axis dispatch table.

## The core idea

Invert the front-end. Instead of three lifts each asserting a whole-kernel template (`loops_in == [k_loop]`, no
prologue, no epilogue — `skeleton.py:104-111`, `:168-174`), **walk the nest once** and tag every loop axis with:

```
AxisPlan = (axis, role ∈ {PARALLEL, REDUCE}, algebra_kind ∈ {MAP, MONOID, SEMIRING, TWISTED_MONOID})
```

`role` is `REDUCE` iff `loop.is_reduce` (`ir/stmt/blocks.py`); `algebra_kind` is `loop.algebra_kind`. Then a per-axis
dispatcher picks that axis's transform and feeds the existing `_assemble`:

| role     | algebra_kind     | per-axis transform                               | today's equivalent                      |
|----------|------------------|--------------------------------------------------|-----------------------------------------|
| PARALLEL | (any)            | σ block/thread/reg split (`_split_free_axis`)    | free-axis path in `_assemble`           |
| REDUCE   | SEMIRING         | strip-K + split-K, or atom-strided if tensorized | `_replace_k_scalar` / `_replace_k_warp` |
| REDUCE   | MONOID           | cooperative-K lanes                              | `_replace_k_coop`                       |
| REDUCE   | TWISTED_MONOID   | coupled-carry transform (future — flash)         | (legacy / future)                       |

The schedule then *falls out of the walk* — one transform per axis, composed in tower order — rather than out of
matching one of three fixed shapes. The payoff is **composition**, which is exactly what the current envelope checks
reject to legacy:

- **fused prologue** (`skeleton.py:111`) and **epilogue** (`skeleton.py:174`): a `MAP` tail after a `REDUCE` axis is
  just a parallel-axis decision sequenced after a reduce-axis decision — no template needed.
- **multi-accumulator / multi-reduce** (`skeleton.py:82`, `:147`): the walk emits N reduce transforms; the single-`k`
  template structurally cannot.
- **TWISTED_MONOID (flash)** stops being "a regime the dispatcher lacks" and becomes "an axis transform not yet
  written" — it slots into the same table when the MMA-flash carry lands (see
  `plans/algebraic-carrier-analysis.md` C1b–C5, `plans/masked-gqa-mma-flash-attention.md`).

## What does NOT reduce to a local walk (keep it a search)

Be honest about the ceiling: a per-axis walk produces the **move space**, not the **schedule**. The decisions that
matter most are not local to one axis and must stay a search over that space — which is what the Fork tree already is
(`partition/tree.py`); `_assemble` materializes one point, the tree enumerates them:

- **Tier selection** (scalar vs warp-MMA) depends on operand dtypes + compute capability + the whole contraction —
  already global via `eligible_atoms(loop_op, ctx, graph)`. Cannot be read off the K axis alone.
- **Split-K vs cooperative-K** is an occupancy tradeoff (M·N tile count vs K length), not a K-local property.
- The knob/move **vocabulary** is itself regime-specific (the warp tier exists only for `SEMIRING`).

So the walk's job is to *enumerate the legal per-axis moves and their transforms*; the prior / MCTS still *picks*. Do
not try to make the walk deterministically choose a schedule.

## Proposed interface

A thin layer between the (removed) `lift_*` recognizers and `_assemble`:

```python
# partition/walk.py  (new)
@dataclass(frozen=True)
class AxisPlan:
    axis: Axis
    role: AxisRole              # PARALLEL | REDUCE
    algebra: AlgebraKind
    body_depth: int            # tower position, outer-first

def walk_nest(loop_op: LoopOp) -> list[AxisPlan] | None:
    """Pre-order walk of the loop nest, tagging each Loop axis with (role, algebra).
    Returns None only for shapes the composer still cannot emit (e.g. a live
    TWISTED_MONOID carry before the coupled transform lands) — NOT for mere
    prologue/epilogue/multi-reduce, which now compose."""
```

`try_compose` (`partition/compose.py:27`) collapses from three try/return blocks to: `walk_nest` → build the per-axis
transform list via the dispatch table → hand free-specs + `k_transform`s to a generalized `_assemble`. The Fork tree
(`tree.py`) is rebuilt over the *union* of per-axis move generators (`moves.py`) keyed by each axis's `(role, algebra)`
rather than over three hand-written per-regime trees.

`_assemble` needs one generalization: today it takes a single `k_transform`; the walk can yield several reduce axes, so
it takes an **ordered list** of axis transforms and applies them inside-out. The single-reduce path stays byte-identical
(a one-element list).

## First composition target — the RMSNorm reduce→epilogue

The cleanest proof that the walk *generalizes* rather than re-skins: a fused `reduce → pointwise-epilogue` kernel.
RMSNorm's `rsqrt` tail is explicitly deferred today (`skeleton.py:174`: "epilogue (e.g. RMSNorm rsqrt) — deferred").
Under the walk, the `K` mean-square reduce emits the coop-K transform and the trailing `mul`/`rsqrt` `MAP` statements
emit nothing structural (they ride the row's thread tile). Success criterion: `try_compose` stops returning `None` for
the RMSNorm `LoopOp`, the composer emits it, and `run --bench` / per-kernel accuracy matches eager — **with no new
regime function added**, only the epilogue statements surviving the walk.

## Phasing & the byte-identical gate

Land it behind the same discipline as the carrier-analysis work: each phase must be **byte-identical** on the kernels
already covered, verified by `deplodock compile` per-kernel compare under a fixed `PYTHONHASHSEED` (the
`62_kernel_bench.json` / `deplodock compare` path) plus green `make test`.

1. **Walk + dispatch table, three covered regimes only.** Build `walk_nest` and the per-axis transform dispatch;
   route `try_compose` through it but keep the envelope so only today's three shapes compose. Output must be
   byte-identical to the current `build_*_tile` for pointwise / scalar matmul / warp matmul / coop-reduce on
   Qwen3-Embedding-0.6B layer 0. Net: same kernels, new front-end.
2. **Generalize `_assemble` to an ordered transform list.** Single-reduce path stays byte-identical; adds the ability
   to sequence transforms. No new shapes yet.
3. **Drop the epilogue/prologue envelope checks; cover RMSNorm.** First genuinely new coverage — the reduce→epilogue
   target above. Verify accuracy + bench vs eager / `torch.compile`.
4. **Multi-reduce.** Drop the `len(reduce_loops) != 1` guards; walk emits N reduce transforms. Validate on a
   multi-accumulator matmul.
5. **TWISTED_MONOID slot.** Wire the flash coupled-carry transform into the table *with* the MMA-flash tier work
   (blocked on the consumer described in `plans/algebraic-carrier-analysis.md`); behind an `action = identity`
   byte-identical gate.

## Hard constraints

- **No regime regression.** Phases 1–2 are pure refactors; the per-kernel compare must be byte-identical or the phase
  is wrong.
- **The walk enumerates, the prior picks.** No schedule decision moves out of the Fork tree into the walk. Tier
  selection, split-K, and knob ranges stay a search (`tree.py` / `moves.py`).
- **`config.move_composer_enabled()` still gates the whole path** (`config.py`); legacy planner remains the default and
  the fallback for any shape `walk_nest` returns `None` on.
- **Symbolic axes stay on the documented paths.** The walk inherits the current static-extent restrictions per axis;
  symbolic-axis handling is not in scope for this plan (see the dynamic-shapes docs).

## Open questions

- **Move-generator keying.** `moves.py` generators are currently per-regime (`matmul_thread_offers`,
  `coop_reduce_offers`, …). Re-key them by `(role, algebra)` so the walk can assemble the per-axis move set — does any
  generator depend on *cross-axis* state (e.g. matmul reg offers reading the K `fk`, `materialize.py:177`) that breaks
  a clean per-axis split? If so those stay joint and the tree node spans both axes.
- **Tower ordering with multiple reduce axes.** `_wrap_tower` is innermost-first; with two reduce axes the relative
  STAGE_INNER / SERIAL_OUTER nesting becomes a choice, not a given — likely another Fork branch.
- **Cost of losing the templates.** The envelope checks also served as cheap fast-fail bails. Confirm the walk's
  `None`-return covers the same non-composable shapes without accidentally trying to emit something the downstream
  passes choke on.
