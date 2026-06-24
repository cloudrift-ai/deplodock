# `SPLIT_CONE` → `CUT`: per-edge cut knob + prior pricing

## Status

- **(a) Substrate — LANDED.** `cut_offers` now returns a ranked `offers` tuple (one whole-cone offer today →
  width-1 mask; `enumeration/_cut.py`), and `SPLIT_CONE: bool` is renamed to `CUT`, a width-1 `BINMASK` over those
  ranked edges (`split/005_split_demoted.py`). Behavior is **byte-identical**: `"0"` = keep fused (greedy cold pick),
  `"1"` = cut every ranked edge; `DEPLODOCK_SPLIT_CONE` still pins via the `CUT` alias. This is §1's knob, paid as one
  perf-DB re-key now, so the per-edge widening below is additive (no second re-key). The §1/§4 *enumeration* machinery
  (per-independent-edge `argmin`, full `2^k` masks for coupled cones) lands **with multi-producer `assemble_fused`** —
  that is the first graph that can expose a width>1 mask (today a multi-cone body is force-cut, not forked). Until
  then the mask stays width-1 and the boolean semantics are exact.
- **(b) Pricing — partially landed.** §2's indirect-control mechanism (`_pick_structural` Σ-pricing, the prior never
  reads `CUT`) is already live. §3's edge features: **`D_cut_roundtrip` landed** — a coarse, gated, re-key-free
  `knob_features` term (`knob.py::_cut_features`): positive on a cut fragment (`CUT` mask popcount > 0), **zero on the
  fused keep**, absent on never-offered kernels — the cost axis that discriminates the two realizations of one decision,
  on the live greedy per-kernel pricing path. Unit-pinned (`test_knob.py::test_knob_features_cut_roundtrip`).
  - **Feasibility finding (why the rest is deferred).** Literal §3 wants *precise per-kernel intermediate bytes* +
    cross-kernel `D_cone_fanout` / `D_recompute_flops`, all feeding greedy's per-kernel Σ. But the structural-feature
    vocabulary is deliberately coarse (`S_ext_*` extent **products** only — no per-operand shapes), so a cut fragment's
    free output (`S_ext_free_prod`) is the cleanest materialized-volume proxy available, and the cross-kernel
    fanout/recompute terms don't fit a single kernel's feature vector at all (greedy prices each kernel from its own
    knobs). Precise intermediate bytes (split from the consumer's own M·N output) need **per-operand shape stamping** —
    a separate change. `D_cone_fanout` / `D_recompute_flops` are decision-level (computable at the `_cut` fork site via
    `_fission`, but only the **decomposition-row** path — `two_level._decomposition_rows`, the outer-MCTS training
    signal, currently R7-quarantined — consumes decision-level features; greedy does not). Both are deferred with this
    finding rather than shipped as noisy proxies.
  - The end-to-end deploy (greedy picking the cut when its Σ is cheaper) stays the parent plan's "two-level pricing
    integration" bullet ([`dag-edge-placement-split-as-enumeration.md`](dag-edge-placement-split-as-enumeration.md)) —
    blocked on a *trained* prior + de-quarantining the structural-search tier (and the `_split_free_axis` `thread=0`
    pricing-candidate bug), not on these features.

When the structural split becomes a per-edge placement move
([`dag-edge-placement-split-as-enumeration.md`](dag-edge-placement-split-as-enumeration.md)), the single graph-level
`SPLIT_CONE: bool` no longer fits: one graph can have **N independently-cuttable edges**. This plan converts the boolean
into a per-edge knob and specifies how the learned prior **instructs** the cut — the key decision being that it does so
*indirectly* (by pricing kernel-sets), not via a "should-I-split" classifier.

## What `SPLIT_CONE` does today, and what must survive

The boolean serves three jobs (the considered-vs-declined idiom, `search/keys.py`): **(a)** idempotence guard (the rule
fires once), **(b)** perf-DB / prior variant identity (a fused-cone matmul vs its clean split twin), **(c)** outer-terminal
attribution (every split-fragment kernel carries `SPLIT_CONE: True`, the keep-fused carries `False`). All three must
survive; only the *shape* changes from scalar to per-edge.

## 1. The knob — a per-edge `BINMASK`, mirroring `STAGE`

`_cut.cut_offers(graph)` returns the **ranked** cuttable edges; the knob is a bitmask over them, exactly the `STAGE`
machinery (`compiler/pipeline/knob.py::KnobType.BINMASK`):

- **`CUT`** (renamed from `SPLIT_CONE`) — `BINMASK` of width `#cuttable-edges`; bit `i` = "cut ranked edge `i` (`GMEM`)
  vs keep it inline (`INLINE`)." Today's boolean is the **width-1 special case** (`"1"` = old `True`, `"0"` = old
  `False`).
- Reuses `STAGE`'s plumbing verbatim: `parse`/`pretty` with `width`, env pin (`DEPLODOCK_CUT=10` / `all` / `none`),
  idempotence guard = `CUT.name in op.knobs`, `off=""` (nothing cuttable).
- The mask is the **outer-terminal variant identity** — it selects the kernel set — and is **stamped on every resulting
  kernel** so each attributes to the right outer terminal. (a)/(b)/(c) all preserved, scalar→vector.

Ordering matches `STAGE`: offer most-cut-first (option-0 = cut all), env pin collapses the fork to one mask.

## 2. The prior instructs the cut **indirectly** — keep `_pick_structural`

The load-bearing decision: **do not add a knob the prior reads to output "cut: yes/no."** The existing mechanism is
better and composes. `policy/greedy._pick_structural` prices each candidate kernel-set's **Σ** and deploys the cut when
`Σ(cut) < Σ(keep)`; the prior is a per-kernel cost model, the structural decision is `argmin` over kernel-set Σ.

Why this is right, not a shortcut:

- After a cut the resulting kernels are **distinct ops with distinct `op_cache_key`s** — the cut consumer is a *clean
  canonical GEMM*, the keep-fused consumer is a *degenerate scalar matmul* (`MMA=0`, `FM=FN=1`). Those differences are
  already in `canonical(bodies+edges)` + the tile knobs, so the prior prices the clean GEMM cheaply (warp tier) and the
  degenerate fused matmul expensively (scalar tier) **from each kernel's own features** — no "split feature" needed.
- The cut is "instructed" because `Σ(clean-GEMM + producer) < Σ(degenerate-fused)` falls out of the *same* cost model
  that ranks tile knobs. One model, no separate split classifier.
- **Cross-graph transfer for free:** the clean-GEMM consumer keys structurally (`op_cache_key`), so "GEMM of shape X at
  f16 costs Y" is learned once and prices *every* cut that produces that GEMM, across models.

`CUT` therefore carries variant identity, not a prediction the prior reads. The prior never inputs `CUT`; it prices the
kernels the mask produces.

## 3. The `D_*` edge features the prior needs to price the trade

For the Σ comparison to be *accurate* the prior must price the **cut kernel set** well, and the cut's cost has two parts
the standard per-kernel features miss: the gmem round-trip it pays and the recompute it avoids. Add these as derived
**engineered edge features** (`D_*`, the way occupancy/CTA/reuse terms already enter `knob.knob_features`):

- `D_intermediate_bytes` — materialized + re-read size of the cut edge's buffer (why a cut *loses* on a huge intermediate
  — attention S/P).
- `D_cone_fanout` — number of consumers sharing the cone (dedup amortization; a shared cone pays more readily).
- `D_recompute_flops` — what `INLINE` would cost ×fanout (the cost the cut avoids).
- consumer matmul `S_ext_*` shape / dtype / tier — already present; the *gain* side (a big f16 GEMM → large tensor-core
  win).

These let the learned `CatBoostPrior` fit the actual tier-gain-vs-round-trip surface. (The `_cut` offer predicate decides
*reachability* — "a tier gain is available"; these features decide *profitability* — "does it pay." The necessary-
condition/profitability split is exactly the one built into `_cut`.)

## 4. Blowup control

`#cuttable-edges` is small by construction (tier-monotonicity → `O(#demoted operands)`, 1–3 per graph), so the `2^k`
mask space the outer MCTS enumerates is tiny. For greedy, avoid even that: decide each *independent* edge by its local
Σ-delta (per-edge `argmin`), reserving full-mask enumeration for `tune` where edges sharing a cone are coupled by dedup.
Same prefer-deeper-first ordering as `STAGE`.

## 5. Migration — perf DB and prior survival

The per-kernel perf history **transfers for free**, because kernel identity is structural, not knob-based: the cut
produces the *same* clean-GEMM and producer kernels regardless of which knob name drove the fork, and those kernels key
on `op_cache_key` (`canonical(bodies+edges)+Schedule`). So existing `perf` rows for clean GEMMs / producers keep serving
replay and prior training unchanged. Only the **outer-fork attribution knob** changes identity.

For that knob:

- The legacy `lowering/tile/split/005_split_demoted` pass (and its `SPLIT_CONE` rows) is being **deleted** with the
  edge-placement landing, so its outer-fork rows are for a *different cut-production mechanism* and are superseded, not
  migrated. The kernels they produced are re-keyed structurally and survive.
- Where strict continuity is wanted, the **width-1 `CUT` mask maps 1:1 to the old boolean** (`"1"`↔`True`, `"0"`↔`False`),
  so a `SPLIT_CONE`→`CUT` rename + bool→width-1-binmask shim can keep single-edge rows if profiling shows the re-tune is
  expensive. Default: rename cleanly and let the (cheap, structurally-keyed) per-kernel rows carry the value.
- The prior checkpoint re-fits on the structurally-keyed per-kernel rows; the `D_*` edge features are new columns
  (additive, the standard `knob_features` extension — no re-key of existing rows).

## Gate

- **Identity:** a `CUT` width-1 mask reproduces today's `SPLIT_CONE` keep/split kernel sets byte-identical; perf rows for
  the produced kernels are unchanged (`op_cache_key` stable).
- **Pricing:** `eval prior` / `eval variants` show `_pick_structural` deploys a cut iff `Σ(cut) < Σ(keep)` on the goldens
  that today split (rotary `QK^T`, SDPA `P@V`, `o_proj`, gated-MLP), and declines on `RMSNorm→linear` (which never
  drops a tier). The `D_*` features measurably improve cut-set ranking vs. without them.
- **Blowup:** the outer fork count stays `O(#demoted operands)` on every golden (logged, not silently capped).

## Future extension — two orthogonal axes, not one `CUT` ternary

Naive extension ("`CUT` ternary `INLINE`/`SMEM`/`GMEM`", or "`CUT` binary + `STAGE` binary") **conflates two orthogonal
axes** and cannot name all the realizations. There are **four** valid cells, from a product of:

- **`MATERIALIZE[edge] ∈ {INLINE, SMEM, GMEM}`** — *inner* (per-kernel): where the intermediate's buffer lives. `INLINE` =
  recompute in registers (no buffer); `SMEM` = a slab; `GMEM` = a global buffer.
- **`SPLIT[edge] ∈ {0,1}`** — *outer* (kernel-set-changing): is the materialized edge promoted to its own launch group
  (separate kernels) vs. kept in one kernel (an internal grid barrier).

The valid cells (constraints prune the cross-product):

| `MATERIALIZE` | `SPLIT=0` (one kernel)          | `SPLIT=1` (two kernels)     |
|---------------|--------------------------------|-----------------------------|
| `INLINE`      | INLINE                         | — (nothing to separate)     |
| `SMEM`        | SMEM (`__syncthreads`)         | **illegal** (smem can't cross a launch) |
| `GMEM`        | GMEM in-kernel (`grid.sync`)   | GMEM separate kernel        |

So `CUT`/`SPLIT_CONE` is **not** the location axis — it is the **grouping axis** (`SPLIT`), orthogonal to where the
buffer lives. And `STAGE` is **not** a peer binary of `CUT` — it is the **`SMEM` value of `MATERIALIZE`** (generalized
from "cache a gmem input `Load`" to "materialize a computed intermediate": the cooperative producer becomes a *compute*
that writes the slab, not a gmem→smem *copy*). The placement is one *inner ternary* + one *orthogonal outer binary*, with
explicit constraints (`SMEM ⟹ SPLIT=0`; `INLINE ⟹` no buffer):

```
PLACE(edge) = INLINE                        if MATERIALIZE = INLINE
            = SMEM                           if MATERIALIZE = SMEM            (⟹ one kernel)
            = GMEM_in_kernel (grid.sync)     if MATERIALIZE = GMEM, SPLIT = 0
            = GMEM_split     (two launches)  if MATERIALIZE = GMEM, SPLIT = 1
```

This is the edge-placement plan's `TileGraph.placement` as a **derived view** over `Schedule.staged` (the `SMEM` marker)
+ `Schedule.launch` (the grouping), now correctly two-axis rather than two conflated binaries.

**Why v1 *looks* like one binary.** v1 has no `grid.sync`, so `GMEM ⟹ SPLIT=1` necessarily — the `GMEM,SPLIT=0` cell is
unreachable, collapsing location-`GMEM` and `SPLIT=1` into one, which is exactly why a single `CUT` binary suffices
*today*. But the IR **must store `MATERIALIZE` and `SPLIT` separately anyway**: the moment `grid.sync` lands,
`GMEM,SPLIT=0` becomes a distinct reachable state, and a `CUT` binary that fused location with grouping cannot name it
(and would re-key the perf DB when you tried to add it). This is the concrete content of "model the barrier mechanism as
its own field, do not bake `GMEM ⇒ separate launch group`."

**Prior — zero new mechanism.** Each cell is a distinct kernel set priced by `argmin`-Σ; `GMEM_in_kernel` and
`GMEM_split` differ only in launch overhead vs. `grid.sync` cost, captured by the existing per-kernel pricing + `D_*`
features (add `D_smem_bytes` for the `SMEM` fit/occupancy cost).

**Sequencing & gating.** Ship `GMEM,SPLIT=1` (always-legal, what `split_demoted` does) first. Then **`SMEM`** (generalize
the `STAGE` mask) — offered only on **CTA-local** edges (derived dependency-span) that **fit the smem budget**; warp-tier
`SMEM` needs the **layout-choreography codegen** (write the cone into the slab in the `ldmatrix` layout — the deferred
fused-prologue frontier), so until then it serves only scalar-tier consumers. Then **`GMEM,SPLIT=0`** (`grid.sync`) with
the persistent/cooperative-launch frontier. The prior auto-prefers each cheaper cell as it becomes reachable, since each
is just another option in the same Σ comparison.

## Relationship to existing docs

- [`dag-edge-placement-split-as-enumeration.md`](dag-edge-placement-split-as-enumeration.md) — parent. This plan is its
  "Knobs & prior" detail: `CUT` is the variant identity for the `GMEM` placement; `_cut.cut_offers` supplies the ranked
  edges the mask indexes.
- [`tile-ir-block-dag.md`](tile-ir-block-dag.md) — the `BINMASK` knob type, `op_cache_key`, `_pick_structural`, and the
  considered-vs-declined idiom (`search/keys.py`) reused here all come from RF / the two-level search.
- The (deleted) perf backlog item "occupancy/reuse as engineered `D_*` features" is the same `knob_features` extension
  point the `D_*` edge features use.
