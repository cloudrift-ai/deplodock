# Retire the `Monoid` / `Semiring` op-tree node classes

**Branch:** `refactoring/tile-ir-rebuild` · **Status:** consumer-side decoupling LANDED; the
recognition / `030_split` build-side flip + the class deletion REMAIN (one atomic change).

## Goal

Eliminate the `Map` / `Monoid` / `Semiring` op-tree node classes as the stored tile-IR. A kernel's
per-cell compute becomes the **annotated loop nest** directly: `Kernel.op` is a `Body`, each reduce
`Loop` carries its `AxisRole` (`FREE` / `PLANAR` / `CONTRACTION` / `TWISTED`) + a `Carrier` (the
algebra payload). Dispatch reads the structure off the loop body, never a node type — the codebase's
settled "read the algebra off the body, no stored kind tag" philosophy
(git `c93b2c5a`), pushed all the way to the IR. The `Twist` carrier algebra (`ir/stmt/carrier.py`)
survives unchanged; only the node *wrappers* go.

## What has landed (this effort — all green, e2e contract held)

Dispatch + type consolidation:
- `dc2a54cc` dispatch on structural `axis_role(node)` (`ir/tile/ops.py`), not `isinstance(kernel,
  *Kernel)` — across `020_schedule`, `005_contract`, `010_materialize`, `030_split`.
- `0473d70d` collapse the `Map/Monoid/Semiring`-`Schedule`/`-Kernel` zoo (6 types) → one kind-free
  `TileSchedule` + `Kernel` (`ir/tile/schedule.py`).
- `d0a48c76` merge recognize + schedule into one pass: `010_recognize` calls `_schedule.schedule`
  inline (renamed from `020_schedule.py`); the rule matches `LoopOp` AND an unmapped `TileOp` so
  flash's graph-rewrite output gets scheduled on re-entry. `020_schedule` deleted.

Composition (Phase 4 first milestone):
- `ce46ecaa` a non-output-tiled contraction honours a coop/ILP/split `REDUCE` (was silently
  ignored): `_semiring_reduce_spec` passes `b`/`r`/`g`; the materialize gate routes a contraction
  with a cooperating `ReducePlan` into `_reduce`; `_reduce` folds it carrier-generically. Validated
  vs numpy (`test_matmul_reduce_partition`: `b4`/`r4`/`r2/b4`, emits `__shfl`).

Carrier decoupling (the read-side of the deletion):
- `f9a58eac` extract `Carrier(state, twist)` (the algebra payload — `merge` / `combine_states` /
  `as_accums` / `as_state_merge` / `dissolve`) out of `Monoid`; `Monoid` keeps a cached `.carrier`
  and delegates. Add a `StateMerge(Stmt)` — the renderable cross-partition combine
  `Carrier.as_state_merge` emits (so the combine no longer needs a `Monoid` node to render). The
  reduce `Loop`'s `carrier` field is now a `Carrier` (`ir/stmt/blocks.py`, `ir/axis.py:AxisRole`).
- `a7a749d7` `ops.lower` stamps `role` + `Carrier` onto each reduce `Loop` it emits
  (`_lower_monoid` → TWISTED/PLANAR; `_lower_semiring` → CONTRACTION + the degenerate fold carrier).
  The `_reduce` materializer reads the `Carrier` off the annotated loop, not `op.reduce_node` /
  `as_monoid`. `StateMerge` proven to render through the cooperative / REG-tree path.
- `9e3ac510` `_atomize.bind_contraction` binds the warp/scalar-tile A/B + fold off the lowered
  `CONTRACTION` loop body (operand `Load`s indexed over K, the fold `Accum`), not the `Semiring`
  node's `operands`/`fold`/`reduce_axis`/`out`.

**Net:** every consumer that *reads* structure is off the node classes. Stamping carriers on loops
does NOT change `op_cache_key` (it digests `Body.structural_key` = `pretty_body`, which ignores
`role`/`carrier`) — verified; this is the key enabler.

## What remains — the build-side flip + deletion (ONE atomic change)

The only remaining `Monoid`/`Semiring` users *build* or *dispatch on* the nodes; they must flip
together (the moment recognition stops emitting nodes, `lower`/`030_split` must stop expecting them
and `Kernel.op` becomes a `Body`):

1. **`ir/tile/ops.py`** — `lower(op)`: a `Map` returns its body's stmts; delete the
   `_lower_monoid`/`_lower_semiring` branches. `axis_role(op)`: scan the body for the outermost
   reduce `Loop`, return its `role` (FREE if none). `pretty(op)`: just `pretty_body`.
2. **`010_recognize` + `_schedule`** — `_lift_cell` returns a `Map(body=<annotated loop nest>)`
   (no `source`): annotate the reduce `Loop` with its role + `Carrier`, keep the projection stmts
   after it in the body. `_normalize` no longer converts `Accum`→`Monoid` (just annotate the plain
   reduce loop `PLANAR` + the `id` `Carrier`). `_lift_reduce`/`_clean_contraction` annotate the
   contraction loop `CONTRACTION` instead of building a `Semiring`. Drop the round-trip.
3. **`_flash.py` / `_softmax.py` / `_carrier.py`** — build the streaming-merge **loop body** +
   attach the `exp`-family `Carrier` to the kv/reduce loop, instead of returning a `Monoid` node
   (`flash_combine` / `online_softmax_combine` / `exp_family_twist` return a `Carrier`).
4. **`030_split.py`** — build the partial / finalize / atomic kernels as annotated **loop bodies**
   (`Map(body=…)`), slicing the `CONTRACTION`/reduce `Loop` directly (offset its operand `Load`s by
   `_ksplit·B`, shrink the axis) rather than `_slice_carrier` on an op-tree node. Read state names /
   identities / `as_state_merge` off the loop's `Carrier`. This is the most node-coupled file.
5. **Delete the classes** in `ir/stmt/algebra.py` (`Monoid`, `Semiring`, the `AlgebraNode` union;
   keep `Map` as the body wrapper OR replace `Kernel.op` with a bare `Body`, `Carrier`, `State`,
   `Twist`, `StateMerge`). Then the trailing cleanup: `blocks.py:_CARRIERS` (drop `Monoid`),
   `leaves.py:Accum.as_monoid` → `as_carrier` (returns a `Carrier`), `stmt/passes.py`
   (`substitute_axes` on the carrier), `loop/ir.py`, `kernel/ir.py` (the `ReduceOp` carrier reads),
   `provenance.py`. Update `005_contract`'s `node.reduce_node.reduce_axis` → the loop's axis.

Suggested sequence: make **all** edits, then run the full suite once and debug — there is no green
intermediate. Land it as one commit (or a tight series) only when `tests/compiler/e2e/` is green and
`grep -rn "Monoid\|Semiring" deplodock/` is clean except `carrier.py`'s "twisted monoid" prose.

## Key invariants / gotchas (don't relearn)

- **`op_cache_key` is safe under loop annotation** — it digests `pretty_body`, which omits
  `role`/`carrier`. Stamping carriers / changing `Kernel.op` to a `Body` keeps keys stable as long
  as the rendered loop text is unchanged.
- **`Carrier.as_state_merge` returns a `StateMerge` stmt** (renders via `render_merge_program`); it
  works through the cooperative / REG-tree / cross-CTA path (proven by `test_reduce_coverage`). A
  `StateMerge` is NOT a fold carrier — it must not make its enclosing loop `is_reduce`.
- **`is_reduce` already keys off `role`** (with a structural `_CARRIERS` fallback for un-annotated
  loops) — once recognition stamps roles, the fallback can drop `Monoid`.
- **Flash is a graph rewrite** → its `TileOp` is scheduled on pass re-entry (the `010_recognize`
  rule matches `LoopOp` + unmapped `TileOp`). Keep that two-step when flipping recognition.
- **Clean contractions only** reach `CONTRACTION` — a computed-cone / demoted-matmul operand stays a
  flat reduce (`_clean_contraction`), so the loop's K-indexed `Load`s ARE the gmem operands.

## Verification

- `make test` (correctness lane, `-O1`) green after the flip; `make lint`.
- e2e recovery contract: `tests/compiler/e2e/` green, `tests/xfail_registry.py` only shrinks.
- Delete the tile-IR **unit** tests that construct `Monoid`/`Semiring` objects (`test_op_tree`,
  `test_algebra_traits`, `test_monoid_forward`); `test_carrier_gen` keeps the `Twist`/`Carrier`-only
  cases. (e2e/accuracy tests are never weakened.)
- Spot-check: scalar/coop/warp matmul, plain + online-softmax reduce, scalar flash, split-K (`g2a`/
  `g2k`) all vs numpy/torch.

## Follow-ups (separate from the deletion)

- **Phase 4 remainder** — compose an output-tiled contraction's `TILE` × `REDUCE` (split-K on the
  warp tier; `test_mma_splitk_finalize[deferred]`). The non-tiled compose is done.
- **Phase 5** — multi-axis "planar" reductions: detect + annotate a planar axis *set* (drop the
  single-reduce bail in `_lift_cell`), materialize the joint partition; new e2e `sum` over `(H,W)`.
