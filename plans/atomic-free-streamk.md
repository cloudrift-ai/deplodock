# Atomic-free Stream-K combine (Phase B4)

**Branch:** `feature/persistent-cta-streamk` (continues the adaptive Stream-K work)
**Status:** planned — adaptive Stream-K with the *atomic* boundary combine is built and correct (B1–B3b, B5-part-1);
this replaces the atomic combine with an atomic-free one.

## Context

Adaptive Stream-K (mid-tile K-splitting) is built and correct end-to-end on SYNC / BUFFERED / cp.async staging. Today
the boundary partials combine with **`atomicAdd`** into a pre-zeroed output — `kernel/098_persistent_streamk._adaptivize`
wraps each output `Write` in `Cond(is_full)` and the `else` branch is a `Write(atomic=True)`. The measured A/B **regresses**
vs plain matmul on the supported staging (18 vs 14µs; 55 vs 49µs): the atomic contention + per-segment bookkeeping cost
more than the wave tail they recover — exactly the plan's "Split-K atomicAdd cancels the win" prediction.

The fix: **atomic-free combine.** Each output cell is written exactly once. Boundary partials go to a scratch workspace;
a tiny separate combine kernel sums each boundary tile's partials and writes the output once. This is the only form that
can beat Split-K. It is orthogonal to B5 (TMA staging) and lands on the already-supported staging.

**Correctness foundation (already proven):** `kernel/_streamk.cta_segments` + its tests guarantee that for every output
tile, the K-sub-ranges touching it across all CTAs **partition `[0, K_blocks)` disjointly**, with **≤2 boundary partials
per CTA** (`test_each_tile_k_range_partitioned`, `test_boundary_partials_bounded`). So summing a boundary tile's partials
reconstructs the full K reduction, and the per-CTA partial count is bounded.

## Why this is NOT the generic atomic-free pass (and what is)

The atomic-free split-K transform (`017_atomic_free_splitk`) is generalizable: its real trigger is
`Body.coordination.atomic_axes` — *the enclosing block axes missing from a Write's index* — and the transform
(workspace `[prod(extent(atomic_axes)), *out]` + a sibling reduce over that leading dim) is mechanical. A **generic
`atomic_axes`-driven deferred-reduction tile-IR pass** could replace 017 and cover any small-static-axis atomic. That
refactor is worth doing on its own (see "Relationship" below) but it does **not** give efficient Stream-K:

- Stream-K's boundary atomic is **not axis-structured** — `(M_b, N_b)` *are* in the Write index, so `atomic_axes` is
  empty; the atomic is a runtime `Cond(is_full)` branch, not a missing axis.
- Forcing it into the generic mold makes the contender dimension the persistent CTA (`blockIdx.x`, extent `num_sms`),
  i.e. a dense `workspace[num_sms, M, N]` — ~2.8 GB at 2048², ~44 MB at 256², and the reduce reads `num_sms·M·N`
  (mostly zeros) → always *slower* than the atomics it replaces.

The generic pass can't exploit the one fact that makes Stream-K cheap: only the handful of **boundary** cells contend
(full tiles have one writer), with runtime contributor counts. That sparsity is Stream-K-specific and needs the
host-scheduled design below.

## Design (sparse two-kernel, host-scheduled combine)

Reuse the `017_atomic_free_splitk` *shape* (main kernel writes intermediates; a sibling kernel reduces them into the
real output; spliced as a Graph fragment) but with a sparse, boundary-only workspace and a host-computed combine
schedule.

**Main Stream-K kernel** (the existing adaptive kernel; two changes to its writes):
- Full-tile segment (`is_full`) → write `out_partial[m, n]` (an intermediate buffer, renamed from the real output).
- Boundary partial → write the `BM×BN` accumulator slab to `scratch[slab, m_local, n_local]`, where
  `slab = 2·blockIdx.x + partial_counter` (per-CTA counter, 0/1, incremented on each `!is_full` segment — bounded by 2,
  emitted alongside the MAC-walk in `PersistentTile.render`). No atomics.

**Scratch + metadata** (bounded by CTA count, tens of MB), passed as `int32` runtime arrays mirroring the existing
`streamk_work_start/end` plumbing:
- `scratch[2·num_sms, BM, BN]` — one slab per CTA boundary partial.
- Host-computed (Python, from `cta_segments` over all CTAs):
  - `slab_tile[2·num_sms]` — linear tile id each slab targets (`-1` if unused).
  - CSR grouping of slabs by **boundary tile**: `bnd_tiles[B]`, `bnd_off[B+1]`, `bnd_slabs[Σ]` — for boundary tile `b`,
    its contributing slabs are `bnd_slabs[bnd_off[b] : bnd_off[b+1]]`. Sized to the true total from `cta_segments` (so
    >2-contributor tiles, which occur when `per_cta < K_blocks`, are handled with no fixed cap).

**Combine kernel** (one CTA per output tile; each output cell written once):
- For tile `t = blockIdx.x`: if `t` is a boundary tile, gather its slabs (CSR) into a `BM×BN` accumulator and write
  `out[t]`; otherwise copy `out_partial[t] → out[t]`. Hand-built `TileOp` mirroring
  `017_atomic_free_splitk._build_reduce_tileop` (`GridTile → ThreadTile → inner reduce → bounded Write`), with the inner
  reduce a CSR gather over slabs instead of a static `SerialTile(K_s)`.

Both `out_partial` and `out` stay single-writer (no cross-kernel races, no atomics). Memory is bounded by
`2·num_sms·BM·BN` + one `out_partial`, not `K_blocks·M·N`.

### Rejected: dense workspace

`workspace[K_blocks, M, N]` (write each partial to `[k_lo, m, n]`, reduce along `K_blocks` — a near-verbatim 017 reuse)
is trivial but the reduce reads `K_blocks·M·N` (mostly zeros) → likely slower than the atomic version, and 2 GB at
2048². Use only as a throwaway correctness oracle during B4.3 if helpful.

## Key reuse points

- `pipeline/passes/lowering/tile/017_atomic_free_splitk.py` — `_build_reduce_tileop` (combine `TileOp` skeleton),
  `_build_atomic_free_fragment` (Graph fragment: `InputOp` aliases + intermediate node + sibling reduce node +
  `frag.outputs`). The engine splices a returned `Graph` via `Candidate.apply → Graph.splice(consumed=[root], output=root)`.
- `pipeline/passes/lowering/kernel/_streamk.py` — `cta_segments` / `cta_range` / `Segment` already give the exact
  boundary-partial schedule; reuse verbatim to build `slab_tile` + the CSR.
- `pipeline/passes/lowering/kernel/098_persistent_streamk.py` — `_adaptivize`: change the `Cond(is_full)` else branch
  from `Write(atomic=True)` to a plain `Write` into `scratch[slab,…]`; route the full-tile branch to `out_partial`.
- `backend/cuda/program.py` — `_allocate_streamk_work` / `_streamk_ranges` are the template for allocating + filling the
  scratch and the int32 metadata arrays at launch; `_buffers` already allocates any non-input/output graph node
  (`out_partial`, `scratch`) as zeroed scratch. `_Launch` carries `streamk_work_arrays` / `streamk_total_units` — add
  parallel fields for the combine metadata arrays.
- `ir/stmt/leaves.py` `Write.atomic` — the scratch write is a plain store; the boundary atomic goes away.

## Milestones (single branch, commit after each `make test` passes)

1. **B4.1 — scratch slab write.** In `_adaptivize`, full-tile → `out_partial`, boundary partial → `scratch[slab,…]` with
   a per-CTA partial counter emitted by `PersistentTile.render`. Add `out_partial` + `scratch` as main-kernel outputs.
   Validate structurally via `--ir cuda` (slab write present, no `atomicAdd`); not yet correct end-to-end.
2. **B4.2 — host combine schedule.** A `_allocate_streamk_combine` beside `_allocate_streamk_work` computes `slab_tile`
   + the boundary-tile CSR from `cta_segments`, allocates `scratch` + metadata arrays, and plumbs them through `_Launch`.
   Pure helper + unit tests (mirror `test_streamk_workdist.py`: every boundary tile's slabs reconstruct it, full tiles
   copied, one writer per cell).
3. **B4.3 — combine kernel + graph splice.** Build the combine `TileOp` (CSR gather → `out`); splice the
   `main → {out_partial, scratch}` + `combine → out` fragment like `_build_atomic_free_fragment`. Validate **accuracy vs
   numpy** on SYNC / async staging across the shapes the atomic path covers (`test_streamk_matmul.py`). Start with `FN=1`
   (scalar writes); generalize to vectorized (`float4`) slab write + gather after.
4. **B4.4 — make it the default; A/B.** Replace the atomic boundary combine with the atomic-free path (keep atomic
   reachable behind a knob for comparison if cheap). Re-run the A/B (128×K×256 fractional-wave shape) atomic-free vs
   plain vs atomic — record whether atomic-free now wins/ties. Update `plans/persistent-cta-streamk.md` + ARCHITECTURE
   docs.

## Risks / constraints

- **Launch overhead.** The combine is a second kernel launch; at tiny sizes it may erase the win. The B4.4 A/B is the
  go/no-go. If atomic-free still doesn't beat plain on the supported staging, the real win is gated behind B5 (TMA), and
  B4 stands as the correct mechanism awaiting B5.
- **Vectorized writes.** `FN>1` makes the output `Write` a `float4`; the slab write + combine gather must handle the
  vector width (pin `FN=1` for B4.3 first, then generalize).
- **Scope.** Lands on SYNC / BUFFERED / cp.async staging (where adaptive Stream-K runs today). Independent of B5.

## Verification

- `./venv/bin/python -m pytest tests/compiler/passes/test_streamk_workdist.py -q` — the combine-schedule helper (B4.2).
- Accuracy: `DEPLODOCK_KNOBS="BM=8,BN=16,FM=1,FN=1,BK=32,STREAMK=1,TMA=0,ASYNC_COPY=0,PIPELINE_STAGES=0" deplodock run
  --code "torch.matmul(torch.randn(256,256), torch.randn(256,256))" -v` → `Accuracy vs eager … PASS`; plus the
  parametrized `tests/compiler/e2e/test_streamk_matmul.py` (sync + async, multi-wave shapes).
- Structural: `… --ir cuda | grep -E "scratch\[|while \(__mac"` shows the slab write and no `atomicAdd`; the combine
  kernel appears as a second `__global__`.
- A/B (B4.4): `deplodock run --code "torch.matmul(torch.randn(128,512), torch.randn(512,256))" --bench` with `STREAMK=1`
  (atomic-free) vs without, on `…,ASYNC_COPY=1,…` — compare `k_matmul` µs against the recorded atomic (18/55µs) and plain
  (14/49µs) numbers.
- `make test` + `make lint` green at each milestone.

## Relationship to a generic atomic-free pass (separate follow-up)

The `017_atomic_free_splitk` transform should be generalized into a single **`atomic_axes`-driven deferred-reduction
tile-IR pass** (workspace `[prod(extent(atomic_axes)), *out]` + sibling reduce, keyed on `Body.coordination.atomic_axes`)
that replaces 017 and handles any small-static-axis atomic. That refactor and this plan **share the idea** (defer the
reduction to a second kernel) and the **plumbing** (`_build_atomic_free_fragment`, workspace tensors, scratch allocation)
— but not the **scheduling**: Stream-K's contender space is `num_sms` (dense is prohibitive) and only boundary cells
contend, so it needs the sparse host-scheduled combine above, which the generic axis-based pass cannot express. Do the
generic pass for its own sake; it does not subsume this one.
