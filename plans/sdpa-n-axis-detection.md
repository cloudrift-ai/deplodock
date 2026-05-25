# SDPA P@V N-axis detection — plan sketch

Lifts the partition planner past the `seq_q` chain stop so the P@V matmul gets register-tiled like
any other matmul. Goal: flip the three remaining SDPA-prologue xfails green —
`tests/compiler/test_torch_ops.py::test_sdpa_gqa[cuda]`,
`tests/compiler/test_block_accuracy.py::test_tinyllama_block_accuracy[cuda]`, and
`tests/compiler/test_block_accuracy.py::test_qwen_block_accuracy` — all marked
"M14: SDPA P@V N-axis sibling-to-reduces; planner detection pending".

The two `test_register_tile_rules` SDPA cases (`test_sdpa_qk_matmul_fires_register_tile`,
`test_sdpa_attention_kernel_fires_register_tile`) already turned green in commit `091ffec2`
once the planner reached P@V via `006a_register_tile_planned` for the *firing* check; we keep
them as regression coverage but they're not in the xfail set anymore.

Out of scope for this plan (mentioned only because they live in the same neighborhood):

- `tests/compiler/test_run_cli.py::test_run_code_sdpa_tinyllama_per_head` is xfailed under
  `_COOP_XFAIL` ("cooperative-reduce removed; planner-driven replacement pending"). It's a
  cooperative-K gap (single head, K=512, no parallel matmul N axis to tile), distinct from the
  prologue work here.
- `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py::test_compile_gated_mlp_with_sync_x_stage_does_not_dangle_smem`
  is xfailed because the planner rejects `BM*FM=128 > M=32` as non-divisible — unrelated to
  prologue detection.

## The framing

The planner already gives parallel reduces sharing the same K extent one common schedule. The
cooperative-K branch of `_split_kernel_fully` does it explicitly: collect every reduce Loop with
`extent == E_K`, drop those names into `target_names`, and `_replace_k_loops` (M10) σ-rewrites
each one through `Loop(K_o, SERIAL_OUTER) → Loop(K_i, STAGE_INNER, reduce)`. Plain softmax — two
reduces (max, sum) plus an output-Loop over the same K extent, all at one scope — flows through
this branch cleanly today.

SDPA P@V is the same idea, scaled up:

- It's a matmul (the K-reduce has 2 K-indexed Loads + an Accum), so the schedule we want is the
  *matmul* one (BN×BM partition with FN×FM register tile + BK chunk), not the cooperative-K one.
- The K reduces aren't all at one scope — softmax max + sum sit at the seq_q level, the matmul K
  sits inside the head_dim free Loop two levels deeper. But they all share the K extent (S_k) and
  the same dependency direction (sum after max, matmul after both), so the same K schedule
  applies to all three.
- The output axes (seq_q, head_dim) aren't both in the outer chain either — the head_dim Loop is
  buried inside seq_q's body as a sibling of the prologue reduces.

The plan is to teach the matmul branch what the cooperative branch already knows: gather *every*
K-reduce sharing the matmul's K extent, run one shared schedule across all of them, and at the
same time extend the output-axis walker so the head_dim Loop sitting one level below joins
`outer_n`.

## The shape (from `deplodock compile ... --ir loop`)

SDPA P@V kernel body, after fusion:

```
for a0 in 0..H_q:               # head             ← outer chain
    for a1 in 0..S_q:           # seq_q            ← outer chain stops here
        for a2 in 0..S_k:       # reduce — softmax-max         (prologue)
            ...
            acc0 <- maximum(acc0, ...)
        for a2 in 0..S_k:       # reduce — softmax-sum-exp     (prologue)
            ...
            acc1 <- add(acc1, ...)
        v6 = reciprocal(acc1)                                 # (prologue)
        for a3 in 0..D:         # head_dim — actual outer_n!
            for a4 in 0..S_k:   # reduce — matmul K
                ...
                acc2 <- add(acc2, ...)
            scaled_dot_product_attention[..., a3] = acc2
```

`_outer_free_loop_chain` stops at `a1` because the body has 4 siblings. So the planner currently
sees `outer_n = a1 (seq_q)`, scans `a1.body` for reduces and finds matmul + softmax mixed in,
with M = head and N = seq_q. The K-σ rewrites `a4` but never tile-splits the actual output N
(`a3`). `006a_register_tile_planned` sees no `Role.REGISTER` and skips. The kernel exits the tile
chain as `LoopOp` and `CudaBackend` rejects it.

## What's already in place

The softmax-sibling work (commits `0107e0ab`, `84d0cc89`, `2c4b6db2 — M10`) handles three pieces
the SDPA detection sits on top of:

- **`unify_sibling_reduce_axes`** (`deplodock/compiler/ir/stmt/normalize.py`) — when two sibling
  reduce Loops index overlapping `(Load.source, dim)` positions (softmax's max and sum both
  read `scaled[..., a2]`), rename to one canonical axis name. After it runs, both reduces use
  the same `a2` in the SDPA P@V body, which makes the planner's name-keyed σ-rewrite uniform
  across them.
- **`merge_sibling_reduce_loops`** (`normalize.py`) — concatenates sibling reduces with matching
  axis name + extent when the merge is semantically safe. For SDPA the sum reads `acc0` from the
  max, so the merge is correctly skipped. (Gated MLP's two reduces *do* merge — that's where this
  pass earns its keep.)
- **M10's `_replace_k_loops`** (`010_partition_loops.py`) — walks every reduce in the body
  whose axis name appears in `target_names` and σ-rewrites it to a `K_o → K_i` tower. Multiple
  reduces → multiple `K_o` / `K_i` axis pairs (named after the canonical K axis), but one shared
  K extent. Already the right primitive for "give every K-reduce the same schedule."

The recent **`2562448f planner refactor`** turned `_split_kernel_fully`'s three former
functions into a single unified entry-point with three elif arms (matmul / cooperative / pointwise)
driven by `is_matmul_reduce` and warp-size predicates. `KernelShape` is already a frozen
dataclass with `outer_n / outer_m / extra_outer / k_loop / target_names`, and `_enumerate_cartesian`
has env-pin narrowing via `Knob.narrow` plus a no-pin fallback. The plan below threads the new
`prologue` shape through that machinery; no parallel matmul/coop functions to keep in sync.

What's missing is the planner-level glue: chain walker bails before head_dim, and the matmul
branch's `target_names` only collects loops under `outer_n.body` — neither sees that the
prologue's K reduces should ride along.

## Detection rule (chain extension)

At the inner level where the chain stops, accept the "matmul-with-prologue" pattern: siblings
split cleanly into

- ≥ 0 `Loop`s with `is_reduce = True` (the prologue reduces — softmax max / sum,
  axis-name-unified by `unify_sibling_reduce_axes`),
- ≥ 0 leaf stmts (`Assign` / `Load` — e.g. `v6 = reciprocal(acc1)`),
- exactly **one** non-reduce `Loop` whose body transitively contains a `Write` (the output-N loop).

When matched, the inner Loop's axis is the true `outer_n`; the chain-last (`a1` here) is the true
`outer_m`; the prologue siblings ride along as `KernelShape.prologue`.

If anything else sits at this level (two non-reduce Loops, branching Conds, naked Writes), bail —
that's a shape the matmul tower can't represent.

## `_outer_free_loop_chain` extension

Today the helper returns `tuple[Loop, ...]` (single value). Switch to `(chain, prologue)`:

```python
def _outer_free_loop_chain(body) -> tuple[tuple[Loop, ...], tuple[Stmt, ...]]:
    """Walk the outer single-stmt chain of untagged free Loops, outermost-first.

    Returns ``(chain, prologue)``. ``prologue`` is empty for the plain
    matmul / pointwise / cooperative-reduce paths and non-empty only
    for the fused-prologue matmul (where the inner level has
    ``[reduces..., one Loop-with-Write]``)."""
    _, rest = _split_leading_non_loops(body)
    out: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        out.append(cur[0])
        cur = tuple(cur[0].body)

    # Fused-prologue extension: the chain stopped because the body has
    # siblings. If those siblings are [reduces..., one non-reduce Loop
    # with a Write], the matmul output-N is buried inside that inner
    # Loop. Pull it into the chain and stash the prologue.
    prologue, inner_loop = _classify_fused_prologue(cur)
    if inner_loop is not None:
        out.append(inner_loop)
        return tuple(out), prologue
    return tuple(out), ()


def _classify_fused_prologue(stmts):
    """Returns (prologue_stmts, inner_loop) if ``stmts`` has shape
    ``[reduces..., assigns..., one non-reduce Loop containing a Write]``;
    else ``((), None)``."""
    prologue: list[Stmt] = []
    inner: Loop | None = None
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            prologue.append(s)
        elif isinstance(s, Loop) and not s.is_reduce and inner is None and _contains_write(s):
            inner = s
        elif isinstance(s, (Loop, StridedLoop)):
            return (), None  # a second non-reduce Loop or unexpected wrapper — bail
        else:
            prologue.append(s)
    return (tuple(prologue), inner) if inner is not None else ((), None)
```

`_contains_write` already lives in `001_launch_geometry.py` (line 271). Import it (or inline —
it's 8 lines) rather than reinventing.

The single caller is `_split_kernel_fully` (line 203 of `010_partition_loops.py`); update the
tuple unpack accordingly.

## `target_names` expansion (matmul branch)

The matmul branch currently does (line 226):

```python
target_names = frozenset(lp.axis.name for lp in matmul_reduces)
```

`matmul_reduces` comes from `outer_n.body.iter_of_type(Loop)`. With chain extension,
`outer_n.body` is `head_dim.body` — only the matmul K reduce lives there; the prologue's
max/sum reduces are at the `outer_m` level (one scope up), so they fall outside the scan.

Collect over both the prologue and `outer_n.body`, gated by the matmul reduce's K extent:

```python
all_reduces: tuple[Loop, ...] = (
    *outer_n.body.iter_of_type(Loop),
    *Body(prologue).iter_of_type(Loop),
)
target_names = frozenset(
    lp.axis.name for lp in all_reduces
    if lp.is_reduce and int(lp.axis.extent) == E_K
)
```

(After `unify_sibling_reduce_axes` runs upstream, every prologue K-axis already shares a name
with the matmul K — so the frozenset typically has size 1 for SDPA. The extent filter guards
against unrelated reduces that happen to be co-located.)

`_replace_k_loops` already iterates `target_names` and σ-rewrites each match independently
(named K_o axes are derived per-source via M10's per-reduce K_o/K_i construction). The matmul
reduce gets `a4_o / a4_i`; the softmax reduces get `a2_o / a2_i`; all three share the same
`K_o_ext` and `K_i = BK` extents. The matmul tile knobs (BN/BM/FM/FN) drive the output tile
only — they don't touch the K schedule, which is what makes sharing safe.

## SPLITK gate when a prologue is present

The cooperative-K branch already forces `splitk_choices = (1,)` (line 372) because cross-CTA
partial-reduce + barrier-before-epilogue isn't wired up. The same problem hits the matmul-with-
prologue branch for a stricter reason: the prologue's max/sum/reciprocal are *consumed* by the
matmul reduce; running them under SPLITK > 1 means each K_s CTA sees a partial softmax stat and
the matmul produces garbage. Force SPLITK=1 when `prologue` is non-empty:

```python
if matmul_reduces:
    ...
    multi_accum = any(... )  # existing check
    has_prologue = bool(shape.prologue)
    splitk_disabled = multi_accum or has_prologue
    param_combos = _enumerate_cartesian(
        E_M=E_M, E_N=E_N, E_K=E_K, ctx=ctx, priority_mode="matmul", multi_accum=splitk_disabled,
    )
```

The `multi_accum` parameter name in `_enumerate_cartesian` is currently load-bearing for the
"force SPLITK=1" effect (line 364) — renaming it to something neutral like
`force_splitk_one` and threading both reasons through is the cleaner change. Either way, the
matmul-with-prologue search space is `splitk ∈ {1}`.

## `KernelShape` + `_split_kernel_fully`

Add `prologue: tuple[Stmt, ...] = ()` to the `KernelShape` dataclass (line 119). Update its
construction (line 264) to pass the prologue tuple through from the extended chain walker.
The cooperative and pointwise branches keep `prologue=()`; only matmul-with-prologue populates it.

For SDPA after the extension: chain is `(head, seq_q, head_dim)`; `outer_n = head_dim`,
`outer_m = seq_q`, `extra_outer = (head,)`, `prologue = (max_reduce, sum_reduce, reciprocal_assign)`.

## `_build_split_body`

The matmul tower (`_wrap_tower(layers, new_inner)` at line 569) currently wraps a body that's the
σ-rewritten output-Loop body. Extension: the prologue must run inside the M_r REGISTER scope so
that σ_outer's M-axis mapping resolves cleanly *and* softmax statistics are computed per-row
(each register cell along M owns its own seq_q value, hence its own max/sum/reciprocal).

```
Loop(M_b BLOCK):
  Loop(N_b BLOCK):
    Loop(M_t THREAD):
      Loop(N_t THREAD):
        Loop(M_r REGISTER):
          σ_outer + σ_k (prologue)     # max, sum, reciprocal — once per (M_b,M_t,M_r)
          Loop(N_r REGISTER):
            Init(acc)
            Loop(K_o SERIAL_OUTER):
              Loop(K_i STAGE_INNER reduce):
                a*b → Accum
            Write
```

Why inside M_r and not outside it: σ_outer maps the M-axis name to
`M_b·(BM·FM) + M_t·FM + M_r` (line 516). The prologue references the M-axis (e.g.
`scaled[..., seq_q, k]`), so M_r must be in scope when σ_outer rewrites it. Placing the
prologue outside M_r leaves `M_r` unbound. Semantically this is also correct — softmax stats
are per-row, and each M_r iteration is a distinct seq_q row.

This means each thread runs the prologue `FM` times. For FM > 1, smem traffic on `scaled[...]`
multiplies by FM — acceptable for correctness; a follow-up perf knob could hoist shared
prologue work to a separate kernel or share scaled[...] loads via Stages.

The N axis: prologue is N-invariant. Placing it *between* N_t and M_r means it executes once
per `N_t` thread group too, redundantly. A `Cond(N_t == 0)` gate + smem broadcast would remove
the duplication; deferred to the polish list — correctness first.

Concretely: split the tower's `layers` list at the M_r / N_r boundary, emit the inner tower,
prepend the σ-rewritten prologue, then emit the outer tower wrapping the prologue+matmul-tower:

```python
inner_layers = [(N_r, Role.REGISTER)]              # innermost — N_r
                                                   # M_r prepended below if outer_m present
outer_layers = []
if M_t is not None:
    outer_layers.append((M_t, Role.THREAD))
outer_layers.append((N_t, Role.THREAD))
if K_c is not None:
    outer_layers.append((K_c, Role.COOPERATIVE_STRIDE))
outer_layers.append((N_b, Role.BLOCK))
if M_b is not None:
    outer_layers.append((M_b, Role.BLOCK))
if K_s is not None:
    outer_layers.append((K_s, Role.SPLITK_BLOCK))
outer_layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))

if shape.prologue:
    # Prologue lives inside M_r so M_r is in scope for σ_outer's M-axis rewrite.
    matmul_inner_tower = _wrap_tower(inner_layers, new_inner)
    prologue_rewritten = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.prologue)
    prologue_rewritten, _ = _replace_k_loops(
        prologue_rewritten, target_names=shape.target_names, K_canonical_name=shape.k_loop.axis.name,
        K_s=K_s, K_c=K_c, br=params.br, bk=params.bk, K_o_ext=K_o_ext,
    )
    body_inside_mr = prologue_rewritten + matmul_inner_tower
    body_with_mr = (Loop(axis=M_r, role=Role.REGISTER, body=body_inside_mr),) if M_r is not None else body_inside_mr
    return _wrap_tower(outer_layers, body_with_mr)
else:
    # Existing path: M_r joins inner_layers, no prologue insertion.
    if M_r is not None:
        inner_layers.append((M_r, Role.REGISTER))
    return _wrap_tower(outer_layers, _wrap_tower(inner_layers, new_inner))
```

K_s = None in the prologue branch (SPLITK forced to 1 above), so the K_s SPLITK_BLOCK layer
collapses — keep the branch in `outer_layers` for symmetry / future-proofing.

## Per-pass compatibility

Most downstream passes don't care about the prologue's existence — they treat each Loop in the
Tile body as independent:

- **`006a_register_tile_planned`** — walks the BLOCK Tile body and folds REGISTER-tagged stmts.
  The M_r Loop now wraps `[prologue..., N_r tower]`; 006a sees the prologue as non-REGISTER
  sibling work and passes it through, replicating only the N_r-tagged inner tower. ✓
- **`020_stage_inputs`** — builds a Stage per (cache-axis-stable) Load. The prologue's Loads
  reference `scaled[..., a2]` — a different buffer + reduce axis than the matmul's V load. Each
  gets its own Stage. ✓
- **`040_use_ring_buffers` / `050_use_tma` / `060_use_async_copy`** — gate on Stage shape; both
  prologue and matmul Stages are eligible. ✓
- **`001_launch_geometry`** — descends into Loops looking for Writes for the atomic-lift
  rewrite. The prologue Loops have no Writes (only Accums), so `_rewrite_for_atomic_lift`
  ignores them. ✓
- **`materialize_tile`** — `57c1ef29`'s multi-Combine fix already handles sibling Combines from
  independent Accums. The prologue's `Combine(acc0)` / `Combine(acc1)` and the matmul's
  `Combine(acc2)` all land at the kernel scope but each comes immediately after its own reduce
  Loop, so the `pending_reduce` dict gets refreshed between them. ✓

## Risks / unknowns

- **σ_outer on the prologue.** The prologue uses M (`seq_q`) via the causal mask
  `(a2 <= a1)` and via the Load `scaled[..., a1, a2]`. After σ_outer rewrites
  `a1 → M_b·(BM·FM) + M_t·FM + M_r`, the mask compares a K-axis value to a thread-local M
  coord — should compile fine. Verify via `--ir loop` after the rewrite that no axis-name
  aliasing slips through `Sigma`.
- **Thread budget.** The prologue runs all `BN · BM` threads, gets a Combine, broadcasts. Total
  threads ≤ `max_threads_per_cta` — same constraint the planner already checks. Prologue smem
  (one Stage per reduce → 2 slabs of `scaled[S_q, S_k]`) could push the kernel over the static
  smem cap for large `S`. Add an explicit smem-cap check in `_enumerate_cartesian` if it bites;
  current sizes (S ≤ 128) are well below.
- **Broadcast across N threads.** With `BN > 1`, multiple threads share an `M_t` coord but
  differ on `N_t`. They all compute the prologue (same max, same sum) redundantly. A follow-up
  could `Cond(N_t == 0)`-gate the prologue and broadcast through smem, but that's a perf knob,
  not a correctness gate.
- **`extra_outer` collisions.** With the chain extended through `head_dim`, `extra_outer = (head,)`
  becomes a BLOCK axis. The launch geometry already supports multiple BLOCK axes via
  `_lift_output_loops` — should drop through. Verify via the tile-IR dump that the Tile's BLOCK
  list has `(head_b, seq_q_b, head_dim_b)` plus the SPLITK_BLOCK / cooperative axes (with
  SPLITK=1 forced, no `K_s` block axis).
- **Prologue inside M_r — Stage / smem cost.** When `FM > 1`, each M_r iteration owns a
  distinct seq_q row and its own max/sum stats. `020_stage_inputs` will produce a Stage scoped
  inside M_r for the prologue's `scaled[..., M_r]` Load — that's correct but means a fresh smem
  slab per register cell. Watch for ballooning smem allocation at high FM; the current
  `_enumerate_cartesian` priority already caps `FM·FN ≤ 32`, which limits the damage.

## MVP scope vs. polish

Minimal viable change to flip the xfails:

1. The walker extension + `_classify_fused_prologue` (~25 LoC in `010_partition_loops.py`,
   import `_contains_write` from `001_launch_geometry.py` or inline).
2. `target_names` expansion in the matmul branch (~6 LoC — union over `outer_n.body` and
   `prologue`, gated on `is_reduce and extent == E_K`).
3. SPLITK clamp when prologue is present (~3 LoC — extend the `multi_accum` flag or rename it).
4. `KernelShape.prologue` field + `_split_kernel_fully` wiring (~10 LoC).
5. `_build_split_body` outer/inner-layer split + σ_outer/σ_k prologue rewrite + M_r placement
   (~25 LoC).

Polish that can come later (not required for green tests):

- **`Cond(N_t == 0)`-gate the prologue** to remove the per-N_t broadcast redundancy.
- **Hoist the prologue out of M_r** when FM > 1 by running it over a separate M_r loop above
  the matmul tower; saves Stage replication. Requires careful σ_outer split.
- **Pin prologue staging.** Force the prologue's smem slab into `Role.STAGE_INNER` early so it
  doesn't fight the matmul K-stage for buffer-count budget.
- **Share staged K loads** between the prologue's max/sum and the matmul's K reduce when they
  read the same source (e.g. `scaled[...]` in SDPA).
- **Knob the prologue cooperative BR independently** from the matmul's BR — softmax max/sum
  benefit from warp-shuffle Combines even when the matmul wants BR=1.

## Validation plan

After the walker + body changes:

1. Re-run `tests/compiler/passes/test_register_tile_rules.py::test_sdpa_*` — should still see
   `register_tile_planned` fire (regression check; these already pass).
2. Re-run `tests/compiler/test_torch_ops.py::test_sdpa_gqa[cuda]` — accuracy vs eager; drop the
   M14 xfail.
3. Re-run `tests/compiler/test_block_accuracy.py::test_tinyllama_block_accuracy[cuda]` and
   `::test_qwen_block_accuracy` — full TinyLlama / Qwen blocks; drop both M14 xfails.
4. Re-run `tests/compiler/test_tune_accuracy.py::test_tuned_variant_matches_reference[sdpa]` —
   the tuner search now picks a properly-tiled variant; verify the chosen knobs end up faster
   than the current single-CTA-per-head fallback.
5. `make test` to catch any unintended fallout (the `extra_outer = (head,)` BLOCK path is
   already exercised by other tests, but the prologue-inside-M_r placement is new).
