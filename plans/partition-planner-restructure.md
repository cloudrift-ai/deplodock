# Tag-driven partition planner — M13–M16

## Context

`feature/partition-planner` HEAD `66528c36` (cooperative-reduce removed). The planner today already owns most axis
decisions but expresses them through four sequential forks (FM/FN, BK, SPLITK, BN/BM) each building a synthetic `Tile`
to call the heuristic. Parallelism (BLOCK / THREAD) lives on `BoundAxis.bind` in `Tile.axes`, only after `001_tileify`
runs — so `004_launch_geometry` still has to do the σ-split for BN/BM and pointwise `thread_tile_shape`, and
`008_register_tile` still has to do the matmul (1,1) post-blockify register-tile fallback.

The end state is a single planner pass that splits every output / reduce axis up front, stamping `Loop.role` with its
destiny: `BLOCK`, `THREAD`, `REGISTER`, `SERIAL_OUTER`, `STAGE_INNER`, `SPLITK_BLOCK`. Tileify becomes a tag-driven
lifter; `004` and `008` disappear. Heuristics in `tuning.py` become pure-numeric `(extents, body_info)` functions — no
synthetic Tiles, no post-blockify dependence, no fork-order coupling.

This is the right end state because:
- One place reasons about axis structure.
- Heuristics become extent-only and individually testable.
- `004` and `008` (and their idempotence guards) disappear entirely.
- The matmul `(1,1)`-fallback problem dissolves: the planner picks `(BN, BM)` first and `register_tile_shape` sees the
  post-blockify per-thread footprint by construction.

User decisions captured: **(a)** M14 enumerates the full pruned cartesian over `(BN, BM, FM, FN, BK, SPLITK)`;
**(b)** xfail temporarily-broken pointwise/reduce tests during M16 and recover at end.

## Tag semantics (final state)

Each pre-tileify free `Loop` carries one `Role`:

| Role | Lifted by tileify? | `BoundAxis.bind` after lift |
|---|---|---|
| `BLOCK` | yes | `BIND_BLOCK` |
| `THREAD` | yes | `BIND_THREAD` |
| `SPLITK_BLOCK` | yes | `BIND_BLOCK` |
| `REGISTER` | no — stays in body (006a unwraps) | — |
| `SERIAL_OUTER` | no — stays in body | — |
| `STAGE_INNER` | no — stays in body (reduce loop) | — |
| `PIPELINE` | no — stamped post-tile by `015` | — |

`COOPERATIVE_STRIDE` deleted (orphan after M12).

## Critical files

- `deplodock/compiler/ir/axis.py` — `Role` enum
- `deplodock/compiler/tuning.py` — heuristic signatures
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_planner.py` — unified split
- `deplodock/compiler/pipeline/passes/lowering/tile/001_tileify.py` — tag-driven lift
- `deplodock/compiler/pipeline/passes/lowering/tile/003_split_matmul_k.py` — shrinks
- `deplodock/compiler/pipeline/passes/lowering/tile/004_launch_geometry.py` — deleted in M15/M16
- `deplodock/compiler/pipeline/passes/lowering/tile/006a_register_tile_planned.py` — unchanged, validated
- `deplodock/compiler/pipeline/passes/lowering/tile/008_register_tile.py` — deleted in M14

---

## M13 — Role expansion + tuning.py numeric refactor (no behavior change)

**Mechanical refactor. End-to-end IR identical, every test stays green.**

### Changes

- `ir/axis.py`: add `BLOCK`, `THREAD`, `SPLITK_BLOCK` to `Role`. Drop `COOPERATIVE_STRIDE`.
- `tuning.py`: introduce `BodyInfo` (frozen dataclass) returned by `BodyInfo.of(body) -> BodyInfo`:
  ```
  has_matmul: bool
  has_fused_prologue: bool
  external_input_count: int
  ```
  Add `recover_logical_extents(body) -> tuple[int, ...]` helper that folds the σ-split sub-axes
  (`_o` / `_i` / `_reg` / `_b` / `_t` / `_r`) back into their parent extent — used by `_tile_class` to read logical
  extents after the planner has split.

  Heuristic signatures become:
  ```
  thread_tile_shape(output_extents, body_info) -> tuple[int, ...]
  register_tile_shape(output_extents, thread_extents, body_info) -> tuple[int, int]
  forced_bk(output_extents, body_info, static_smem_cap) -> int | None
  auto_splitk(output_extents, body_info, k_o_extent, thread_extents) -> int
  _tile_class(output_extents, body_info) -> str
  ```

  Tile-taking wrappers retained as thin adapters during M13 so callers can migrate one at a time.

- `000_partition_planner.py`: delete `_synthetic_tile_post_register` and the four inline synthetic-Tile builders; call
  the new pure-numeric heuristics directly. Compute `BodyInfo` once per LoopOp at the top of `rewrite`.
- `004_launch_geometry.py`, `008_register_tile.py`, `006a_register_tile_planned.py`, `003_split_matmul_k.py`: update
  call sites (still using Tile adapters where convenient).

### Invariants

Every existing test passes unchanged. No new Role tags stamped yet.

### Deleted

`_synthetic_tile_post_register` + four inline synthetic-Tile builders in the planner. `COOPERATIVE_STRIDE` enum value.
Tile-taking heuristic adapters once every caller migrated (end of M13).

---

## M14 — Joint matmul enumeration; planner emits BLOCK/THREAD/SPLITK_BLOCK on body

**Collapse the planner's five matmul forks into one `_split_matmul_fully`. Delete 008.**

### Per-axis split structure

For matmul output axis `M` (extent `E_M`) and factors `(BM, FM)`:
- `M_b` extent `E_M / (BM*FM)` — `Role.BLOCK`
- `M_t` extent `BM` — `Role.THREAD`
- `M_r` extent `FM` — `Role.REGISTER`

Same for `N` (using `BN`, `FN`). For reduce axis `K` with `(BK, SPLITK)`:
- `K_s` extent `SPLITK` — `Role.SPLITK_BLOCK` (omitted when `SPLITK=1`)
- `K_o` extent `E_K / (SPLITK * BK)` — `Role.SERIAL_OUTER`
- `K_i` extent `BK` — `Role.STAGE_INNER` (reduce loop)

### Literal Sigma map

```
σ: M → M_b*(BM*FM) + M_t*FM + M_r
   N → N_b*(BN*FN) + N_t*FN + N_r
   K → K_s*(K_o_count*BK) + K_o*BK + K_i
```

Factor=1 omits its sub-axis and contributes identity in that term (e.g. `FM=1` → no `M_r`, σ becomes
`M → M_b*BM + M_t`).

### Nesting (outermost first)

```
Loop(K_s, SPLITK_BLOCK,
  Loop(M_b, BLOCK,
    Loop(N_b, BLOCK,
      Loop(M_t, THREAD,
        Loop(N_t, THREAD,
          Loop(M_r, REGISTER,
            Loop(N_r, REGISTER,
              Loop(K_o, SERIAL_OUTER,
                Loop(K_i, STAGE_INNER, is_reduce=True,
                  σ(body))))))))))
```

### Variant enumeration (full pruned cartesian)

Cross-product over `_TUNE_AXIS_CHOICES × _TUNE_AXIS_CHOICES × _TUNE_F_CHOICES × _TUNE_F_CHOICES × _BK_CANDIDATES ×
_SPLITK_CANDIDATES`.

Prune in this order:
1. Divisibility: `E_M % (BM*FM) == 0`, `E_N % (BN*FN) == 0`, `E_K % (SPLITK*BK) == 0`.
2. Thread budget: `BN * BM ≤ 1024`.
3. Register budget: `FM * FN ≤ MAX_CELLS_PER_THREAD` (32 from `_helpers.py`).
4. Dedup after clamp (`BN = min(BN, E_N)` etc.).
5. Heuristic tuple first (computed via `register_tile_shape` + `thread_tile_shape` + `forced_bk` + `auto_splitk` on
   post-clamp extents) — variant 0 for greedy.

Typical viable count: 200–500 per matmul. Matches today's cross-pass cartesian envelope; autotune budget unaffected.

### 008 deletion safety

008 today only handles the matmul `(1,1)` fallback. With M14 the planner always stamps `FM`, `FN` (possibly =1).
`006a_register_tile_planned._replicate_register_loops` handles factor=1 correctly: a REGISTER `Loop` of extent 1
produces one replica with `σ: axis→Literal(0)`, identity to the original stmts. The previously-`(1,1)` SDPA inner
matmul therefore routes through 006a as a benign no-op REGISTER unwrap. Confirmed by re-reading
`006a_register_tile_planned.py:64-82`.

### Files

- `000_partition_planner.py`: new `_split_matmul_fully(loop_op, body_info, ctx)`, replaces
  `_try_matmul_register_tile`, `_try_matmul_k_chunk`, `_try_splitk`, `_try_matmul_bn_bm_fork`. Chain-reduce branch and
  pointwise branch stay deferred to M16.
- `008_register_tile.py`: **deleted**.
- `004_launch_geometry.py`: matmul branch (`_matmul_deterministic`, `_matmul_variants`) deleted — only pointwise branch
  remains. `BN`/`BM` knob declarations stay (planner still stamps them).
- `003_split_matmul_k.py`: idempotence guard on `Role.SPLITK_BLOCK` so it no-ops; planner already did the σ. Keep
  epilogue rewrite logic intact.
- `006a_register_tile_planned.py`: unchanged.

### Invariants

All existing matmul end-to-end tests stay green: `test_e2e_matmul*`, `test_run_code_matmul_*`,
`test_run_code_linear_blockify`, `test_run_code_matmul_k_chunked`, `test_run_code_sdpa_k_chunked` (the latter currently
xfailed due to cooperative removal — stays xfailed). `register_tile_rules`, `matmul_rules`, `partition_planner_rules`
tests update mechanically (snapshot diffs expected; not behavioral changes).

### Deleted at end of M14

`008_register_tile.py`. `004_launch_geometry`'s matmul branch (`_matmul_deterministic`, `_matmul_variants`). Planner's
`_try_matmul_register_tile`, `_try_matmul_k_chunk`, `_try_splitk`, `_try_matmul_bn_bm_fork`, `_split_register_outer_two`.
`TileOp.validate`'s thread-budget check (planner prunes pre-emit; validate becomes redundant).

---

## M15 — Tag-driven tileify; delete 004 matmul-tile lifting code

**Tileify becomes a mechanical lifter of tagged Loops.**

### New tileify logic

`001_tileify.py.rewrite` walks the body outside-in:
1. Collect Loops in encounter order whose `role in {BLOCK, THREAD, SPLITK_BLOCK}`. These become `Tile.axes` with
   `bind=BIND_BLOCK` (for BLOCK / SPLITK_BLOCK) or `bind=BIND_THREAD` (for THREAD). Same `Axis` object passes through —
   no rename.
2. Stop at any Loop whose role is in `{REGISTER, SERIAL_OUTER, STAGE_INNER, None}` or any non-Loop / multi-stmt level.
   Everything from that point down becomes `Tile.body`.
3. `_lift_output_loops` retained for sibling free Loops in the post-chain body (their tags must be set by the planner;
   M16 ensures this for pointwise).

The old `_strip_outer_free_chain` heuristic deleted. The "free outer Loop = thread axis" assumption is replaced by an
explicit tag check.

### Files

- `001_tileify.py`: rewrite of `rewrite()`, delete `_strip_outer_free_chain`. Keep `_lift_output_loops` (tag-gated).
- `003_split_matmul_k.py`: planner already did σ on K + stamped `SPLITK_BLOCK`. 003 keeps only the partial-sum +
  epilogue rewrite (still post-tile, since it depends on `Tile.axes` shape). Trim accordingly.
- Test fixtures in `tests/compiler/passes/test_tileify_rules.py`: update fixtures to stamp BLOCK/THREAD on their input
  loops, since they previously relied on tileify's inference. Mechanical — not xfail.

### Invariants

Matmul end-to-end stays green. `tileify_rules` and `matmul_rules` tests pass after fixture updates. Pointwise and
non-matmul reduce kernels still work because `004`'s pointwise branch is still live in M15.

### Deleted at end of M15

`001_tileify._strip_outer_free_chain`. `003_split_matmul_k`'s σ-split logic. `TileOp.validate`'s pre-tile pointwise
budget check (subsumed by planner prune).

---

## M16 — Pointwise + non-matmul reduce unification; delete 004 entirely

**Planner stamps BLOCK/THREAD on every kernel.**

### Pointwise σ structure

For pointwise output axes `(E_outer, …, E_innermost)` and `thread_tile_shape = (BN, …)`:

For each axis `A` with target `T`:
- `T == 1` or `E_A ≤ T`: stamp `Role.THREAD` directly on the existing Loop (no split).
- `E_A > T` and `E_A % T == 0`: σ-split into `A_b` (extent `E_A / T`, `Role.BLOCK`) over `A_t` (extent `T`,
  `Role.THREAD`), σ: `A → A_b * T + A_t`.

Outermost axes that have no target (rank-mismatch) bind to `Role.BLOCK` whole-extent.

### Non-matmul reduce

The existing `_try_chunk_reduce` already emits `SERIAL_OUTER` / `STAGE_INNER` on K and σ-substitutes. Extend it to also
stamp `BLOCK` / `THREAD` on the chain axes and lifted output axes, mirroring the pointwise stamping above. Combined
into one `_split_pointwise_or_reduce_fully(loop_op, body_info)` entry point.

### Files

- `000_partition_planner.py`: new `_split_pointwise_or_reduce_fully`. Merge `_try_chunk_reduce` into it. `rewrite()`
  dispatches on `body_info.has_matmul` to matmul / pointwise.
- `004_launch_geometry.py`: **deleted**. `BN`/`BM` knob declarations move into `_helpers.py` so the planner can still
  own them. `TileOp.validate` removed if no remaining caller.

### xfail policy (user-confirmed)

If the tag-driven pointwise path produces a launch geometry different from today's `thread_tile_shape` partition (e.g.
a different axis order or a missed split), the affected tests get
`pytest.mark.xfail(reason="M16 pointwise plumbing pending", strict=False)`. Target xfails to recover within M16:
- `test_tileify_rules.py` pointwise cases
- `test_reduction_rules.py::test_*chunk_reduce*` non-matmul cases
- Possibly some `test_e2e_*pointwise*` if extents miss
- Any block-test snapshot that depended on today's exact axis ordering

Recovery criterion before closing M16: all pointwise/reduce tests un-xfailed and green, OR explicit user decision to
keep an xfail with a tracked follow-up.

### Deleted at end of M16

`004_launch_geometry.py` entirely. `000_partition_planner._try_chunk_reduce`, `_predict_thread_extents`,
`_lifted_output_loops`, `_writes_with_axis`, `_chunk_reduces_in_body`, `_reduce_qualifies`, `_slab_geometry`,
`_chunk_reduce_loop`, `_pick_reduce_bk` (folded into the unified function).

---

## Verification

After each milestone:

1. **`make test`** — full suite. Track passed / xfailed / xpassed counts. M13 must match the pre-M13 baseline exactly.
   M14/M15 must keep matmul end-to-end green. M16 must end green (xfails un-xfailed or explicitly accepted).
2. **`make lint`** — `ruff check` and `ruff format --check` clean.
3. **End-to-end smoke on CUDA box (M14, M16):**
   ```
   ./venv/bin/python -m deplodock.deplodock run --code "torch.matmul(torch.randn(128,2048), torch.randn(2048,128))"
   ./venv/bin/python -m deplodock.deplodock run --code "torch.nn.RMSNorm(64)(torch.randn(1,8,64))"  # M16 only
   ```
   Both must `rc == 0` with reasonable accuracy.
4. **Autotune variant-count check (M14):** spot-check on `gate_proj.s512`, `sdpa.tinyllama.s32`, `kv_proj.s512` that
   the planner emits 200–500 variants post-prune. Out-of-band — won't fail tests but flags pruning bugs.
5. **`deplodock compile --code "<expr>" --ir loop`** roundtrip (M13): dump and reload a Loop IR with the new Role
   members to confirm serialization survives.

## Sequencing

- **M13** (~1 day) — refactor only, all green
- **M14** (~3 days) — highest risk; joint enumeration + 008 deletion
- **M15** (~1 day) — mechanical tileify rewrite
- **M16** (~2 days) — pointwise/reduce parity hunt + 004 deletion
