# Loop-body segmentation as the basis for the tile IR

Make **body segmentation** the composer's foundational primitive: split a `LoopOp` body into `Chunk`s, then *place*
each chunk at a scope by data-dependency. Placement subsumes today's hand-coded prologue / epilogue / `leading` / `mid`
handling, and — once chunks are first-class — generalizes to warp-specialization (producer vs consumer chunk) and
pipelining (chunk-to-stage assignment).

This is a *refactor toward a model the code already half-implements*, gated end-to-end by **byte-identical emitted
CUDA** — the same safety contract `plans/drop-overhang-knob-structural-masked-feature.md` uses.

## Why

The composer already segments the body — it just does it by hand, in five places, via structural pattern-matching
instead of dependency analysis:

- `iterdag.py` carries named segment fields: `leading` (per-CTA), `mid` (per-outer-axis), `prologue` (per-row,
  N-invariant), `inner_body` (the matmul reduce **+** the MAP epilogue — QK^T scale / `matmul_add` — as implicit
  sub-chunks).
- The segmentation rule is `_classify_fused_prologue` (`iterdag.py:190`): it hunts a specific statement shape
  (`[reduce…, assigns, exactly one non-reduce Loop containing a matmul reduce]`) and bails on anything off-pattern.
- Placement is hardcoded per builder. `_assemble_matmul_prologue` (`materialize.py:271-328`) is a ~60-line clone of
  `_assemble` whose entire reason to exist is to place ONE chunk (the prologue) at the `M_r` serial scope and keep it
  outside the `N_r` register tower.

Two observations make the abstraction worth extracting:

**Placement is loop-invariant code motion.** The prologue sits at `M_r`, outside `N_r`, *because it reads M and K but
not N* — "computed once per output row, must not be replicated per N register cell" (`materialize.py:315-317`). That is
exactly: **place a chunk at the outermost scope binding all the axes it reads.** `leading` (reads no axis-bound var →
per-CTA), the prologue (reads M, K → `M_r`), and the epilogue (reads M, N, the accumulator → innermost) are all the
same rule, not three special cases. The per-chunk σ-rewrite falls out of the same fact: a chunk is σ-rewritten by
exactly the axis-splits *in scope at its placement depth* — which is precisely why `_assemble_matmul_prologue` already
σ-rewrites the prologue separately from the matmul body (`materialize.py:299` vs `:305`).

**Placement is sometimes a search move, not a fixed rule.** `mid` is per-outer-axis but deliberately "recomputed per
inner element" rather than hoisted (`iterdag.py:101-104`) — a registers-vs-redundant-compute tradeoff. A chunk can be
legal at several depths; which one is a tuning decision, on the same footing as `TileMap` / `TileSerial`. The composer's
whole architecture (`moves.py` offers + `tree.py` Fork tree) exists to search moves, so placement becomes a move family
rather than a builder per case.

**Downstream, warp-spec and pipelining are already chunk-placement — but post-hoc.** `085_warp_specialize` rewrites a
post-`080` `ThreadTile`; `080_pipeline_stages` assigns `SerialTile(kind="pipeline")` (the `Role.PIPELINE` rung already
exists in `_tower.py`); `030_hoist_invariant_compute` is LICM on the materialized tower; `026_unify_sibling_stages`
merges stage siblings. All re-derive structure from already-placed towers. With first-class chunks carrying
read/write/carrier metadata, these become "partition the chunk DAG across warp-roles / assign chunks to stages,
respecting deps" — the CUTLASS mental model (mainloop = pipeline of stage-chunks, epilogue = a separate chunk,
warp-specialized = producer/consumer chunk partition).

## The `Chunk` abstraction

A chunk is a contiguous body segment plus the metadata placement needs — all derivable from primitives already present:

```
@dataclass(frozen=True)
class Chunk:
    stmts:   tuple[Stmt, ...]            # the contiguous segment
    reads:   frozenset[str]              # index-axis names it reads (free vars over nest axes)
    writes:  frozenset[str]              # names it defines (Stmt.defines())
    carrier: ReduceCarrier | None        # the monoid/algebra if it's a reduce (algebra_kind), else None
```

- `writes` ← `Stmt.defines()` (already used at `materialize.py:133`, `:326`).
- `reads` ← free-var walk over the segment intersected with the nest's axis names (the σ machinery already walks
  exprs; `Sigma.reduce` / `rewrite` give the var set).
- `carrier` ← `_carrier_of` / `Loop.algebra_kind` (already in `iterdag.py:153`).

No new IR node — `Chunk` is a *derived view*, like `IterDag` itself (computed on demand, never serialized, zero
`op_cache_key` surface). `IterDag.leading` / `mid` / `prologue` become a single `chunks: tuple[Chunk, ...]`.

## The placement rule

```
def place(chunk, scope_axes_outer_to_inner) -> depth:
    """Outermost scope binding every axis in chunk.reads (∅ reads → per-CTA, depth 0)."""
```

`scope_axes` is the realized tower's axis order (the `layers` list `_assemble` already builds: register → thread →
block → extra-outer, plus the K tower). A chunk lands just inside the innermost scope its reads require. The matmul body
chunk reads M, N, K → innermost; the prologue reads M, K → `M_r`; `leading` → per-CTA. The σ for that chunk is the
`Sigma` accumulated down to its depth.

**Recompute-vs-hoist as a move (later phase):** when a chunk is legal at depths `d_lo..d_hi`, emit one fork child per
candidate depth (bounded — only the scopes between its deepest-read and the consuming chunk). Reuses `moves.py` /
`tree.py` unchanged; the prior learns the register/compute tradeoff.

## Sequencing (each step gated on byte-identical CUDA + `make test`)

1. **Add the `Chunk` derived view + `place()`** alongside the existing `IterDag` fields (dual representation, no
   behavior change). Unit-test `place()` against the known placements (prologue → `M_r`, leading → per-CTA) on the
   existing golden shapes. → verify: no codegen change (nothing consumes `Chunk` yet).
2. **Collapse `_assemble_matmul_prologue` into `_assemble`** by representing the prologue as a `Chunk` placed by
   `place()` and σ-rewritten by its placement-depth `Sigma`. Delete the 60-line clone. → verify: the prologue-matmul
   goldens (RMSNorm-/softmax-prologue matmuls) emit **byte-identical CUDA**; `make test` green. *This is the
   proof-of-model step — it removes a special case at negative LOC.*
3. **Re-express `leading` / `mid` as chunks** through the same `place()` path; drop the bespoke `_split_leading_non_loops`
   threading in `_assemble` (`materialize.py:132-134`) in favor of per-chunk placement. → verify: byte-identical CUDA
   across every regime's goldens (pointwise / coop / matmul / flash).
4. **Lift the warp-tier fused-prologue gate.** With placement uniform, `build_warp_matmul_tile` can place a prologue
   chunk at `M_r` too; drop `scalar_only = … or bool(dag.prologue)` (`tree.py:363`). → verify: a fused-prologue matmul
   now offers a warp-tier terminal; accuracy holds; tune picks correctly.
5. **(Optional, separate plan) placement-as-move + downstream rewire.** Make recompute-vs-hoist a fork move; teach
   `080`/`085`/`026`/`030` to read `Chunk` metadata instead of re-deriving structure from the tower.

## Correctness invariant

Steps 1–3 must not move guard emission, σ semantics, or tower shape — only *where the placement decision is computed*.
The masked-axis `Cond` / `real_extent` / `gmem_extents` (`materialize.py:_split_free_axis`, `_warp_axis`) and the
K-transforms (`_replace_k_scalar` / `_coop` / `_warp`) are unchanged; chunks only decide *which scope* a segment's
already-correct rewrite wraps at. The safety margin: diff emitted CUDA before/after on every golden — it must be empty.
Interplay with `plans/drop-overhang-knob-structural-masked-feature.md`: both touch `materialize.py`'s assembly tail;
land whichever first and rebase the other onto the merged `_assemble`.

## Open / decided

- **Decided:** `Chunk` is a derived view (no new IR node, no cache-key surface), mirroring `IterDag`.
- **Decided:** placement = outermost scope binding `reads` (dependency-driven LICM), proven first on the prologue.
- **Open:** whether `reads` should track *register-tile* axes (`A_r`/`N_r`) distinctly from their parent free axis, so a
  chunk can be hoisted out of the register tier but not the thread tier (the prologue case needs exactly this — out of
  `N_r`, inside `M_r`). Lean: place against the *post-split* axis list (the `layers` entries), so `N_r` and `M_r` are
  separate scopes and the rule already expresses it.
- **Open:** placement-as-move's depth-candidate bounds (phase 5) — needs the prior to have a feature for recompute count;
  defer until steps 1–4 land.
