"""Pair sibling f16 ``Accum``/``Init`` chains into ``__half2`` accumulators.

Register-tiled fp16 reductions (the FN=K knob in matmul / softmax /
RMSNorm) emit K independent scalar ``__half`` accumulators per thread.
With K=64 in the (32, 16384) fp16 softmax case this saturates the
register file and caps occupancy at 50%. Packing pairs of scalar
accumulators into ``__half2`` halves the register footprint of the
accumulator chain — accumulator updates use ``__hmax2`` / ``__hadd2``
instead of the scalar ``__hmax`` / ``__hadd``, two halves move per
shuffle, and NVRTC stops allocating a separate 32-bit slot for every
half.

## Pattern matched

Within a single scope (Tile body, or the body of a free Loop), the
register-tile reduction has the shape:

    Init(acc_0  :f16, op=...)
    Init(acc_1  :f16, op=...)
    ...
    Init(acc_K-1 :f16, op=...)
    [...other stmts...]
    Loop(k):
        body:
            ... Accum(acc_0, v_0, op, f16)
                Accum(acc_1, v_1, op, f16)
                ...
                Accum(acc_K-1, v_K-1, op, f16) ...
    [...uses of acc_*...]

The pass pairs adjacent Inits ``(acc_2k, acc_2k+1) → acc_pair_k`` at
:class:`F16x2`, rewrites each pair of Accums inside the Loop as
``[Pack(v_pair, v_2k, v_2k+1), Accum(acc_pair_k, v_pair, op, F16x2)]``,
and inserts ``Unpack(acc_2k, acc_2k+1, acc_pair_k)`` right after the
Loop in the outer scope so downstream stmts (``WarpShuffle``, inter-
warp Smem, etc.) still see the original scalar names.

## Scope of v1

- Pairs only ``Init`` + ``Accum`` (matching by name, same op, dtype
  F16). The downstream ``WarpShuffle`` / ``Smem`` reduction stays
  scalar — the Unpack restores the original names before they're
  consumed.
- Pairs only within the same Loop body. A pair is valid iff both
  Accums appear in the same nested Loop's body.
- Even-sized groups only. Odd-sized groups leave the last Accum
  scalar (the pair would have nothing to pair with).

## Not yet handled

- Packing the WarpShuffle / inter-warp Smem step (would extend the
  fp16x2 chain further; needs WarpShuffle.render at F16x2 + paired
  Smem decls).
- Cross-Loop pairings (Accums for the same pair-name spread across
  multiple Loops).
- Smarter pair packing for fp16 ops with no h2 form (would need to
  fall back to scalar at the boundary, which we don't yet detect).
"""

from __future__ import annotations

from emmy.compiler.dtype import F16, F16x2
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.stmt import Accum, Body, Cond, Init, Pack, Stmt, Unpack
from emmy.compiler.ir.tile.ir import GridTile, RegisterTile, SerialTile, StridedTile, ThreadTile, TileOp, WarpTile
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    top = root.op
    # The pass fires on MMA TileOps too: the MMA cell's C fragment is the
    # accumulator (no scalar f16 Accum/Init to pair *there*), but a
    # fused-prologue matmul (SDPA P@V) carries softmax max/sum/reciprocal
    # as scalar f16 Accum/Init siblings outside the MMA cell — those
    # still benefit from __half2 packing. ``_pack_body_recursive`` walks
    # past Mma* Stmts unchanged (they're neither Accum nor Init), so the
    # MMA cell stays intact; only the prologue/epilogue pairs at any
    # scope. When nothing is pairable, the natural ``did=False`` path
    # raises RuleSkipped — no special-case guard needed for MMA-only
    # kernels (plain matmul).
    new_body, did = _pack_body_recursive(top.body)
    if not did:
        raise RuleSkipped("no f16 Accum/Init groups to pair")
    return TileOp(body=new_body, name=top.name, knobs=dict(top.knobs))


def _pack_body_recursive(body: Body) -> tuple[Body, bool]:
    """Recurse into nested bodies first, then pair Inits+Accums at this
    scope. Returns ``(new_body, did_pair_anything)``."""
    did_any = False
    descended: list[Stmt] = []
    for s in body:
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            new_inner, d = _pack_body_recursive(s.body)
            did_any = did_any or d
            descended.append(s.with_bodies((new_inner,)))
        elif isinstance(s, Cond):
            nb, db = _pack_body_recursive(s.body)
            ne, de = _pack_body_recursive(s.else_body)
            did_any = did_any or db or de
            descended.append(Cond(cond=s.cond, body=nb, else_body=ne))
        elif isinstance(s, (GridTile, ThreadTile, WarpTile)):
            nb, db = _pack_body_recursive(s.body)
            did_any = did_any or db
            descended.append(s.with_bodies((nb,)))
        else:
            descended.append(s)
    new_stmts, did_local = _pair_at_scope(descended)
    return Body(tuple(new_stmts)), did_any or did_local


def _pair_at_scope(stmts: list[Stmt]) -> tuple[list[Stmt], bool]:
    """Pair sibling f16 ``Init`` Stmts at this scope, then rewrite
    matching Accums inside nested Loops + insert Unpack after the Loop.
    """
    # Step 1: identify candidate Init pairs.
    init_groups: dict[str, list[tuple[int, Init]]] = {}
    for i, s in enumerate(stmts):
        if isinstance(s, Init) and s.dtype.name == "f16":
            init_groups.setdefault(s.op.name, []).append((i, s))

    # Step 2: form candidate pairings (low_init, high_init, pair_name).
    candidates: list[tuple[Init, Init, str]] = []
    init_pair_id: dict[int, int] = {}
    for group in init_groups.values():
        # Pair only consecutive entries; odd tail stays scalar.
        for k in range(0, len(group) - 1, 2):
            (idx_a, init_a), (idx_b, init_b) = group[k], group[k + 1]
            pair_name = f"{init_a.name}_{init_b.name}_p"
            candidates.append((init_a, init_b, pair_name))
            pid = len(candidates) - 1
            init_pair_id[idx_a] = pid
            init_pair_id[idx_b] = pid

    if not candidates:
        return stmts, False

    # Step 3: validate — both Accums must appear in the SAME nested
    # Loop/StridedLoop body. Drop pairings that don't satisfy this.
    name_to_pair: dict[str, int] = {}
    for pid, (low, high, _) in enumerate(candidates):
        name_to_pair[low.name] = pid
        name_to_pair[high.name] = pid

    # Find Loops containing pairable Accums and verify each pid has
    # both Accums in the same Loop body.
    valid_pids: set[int] = set()
    for s in stmts:
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            pids_here = _pids_with_both_accums(s.body, name_to_pair, candidates)
            valid_pids |= pids_here

    if not valid_pids:
        return stmts, False

    # Filter candidates / name_to_pair / init_pair_id by valid_pids.
    valid_candidates: list[tuple[Init, Init, str]] = [candidates[pid] for pid in sorted(valid_pids)]
    pid_remap = {old_pid: new_pid for new_pid, old_pid in enumerate(sorted(valid_pids))}
    name_to_new_pid: dict[str, int] = {}
    for old_pid, new_pid in pid_remap.items():
        low, high, _ = candidates[old_pid]
        name_to_new_pid[low.name] = new_pid
        name_to_new_pid[high.name] = new_pid
    init_new_pair_id = {idx: pid_remap[pid] for idx, pid in init_pair_id.items() if pid in valid_pids}

    # Step 4: rebuild stmts.
    out: list[Stmt] = []
    emitted_pair_init: set[int] = set()
    for i, s in enumerate(stmts):
        if isinstance(s, Init) and i in init_new_pair_id:
            new_pid = init_new_pair_id[i]
            if new_pid in emitted_pair_init:
                continue
            emitted_pair_init.add(new_pid)
            low, _high, pair_name = valid_candidates[new_pid]
            out.append(Init(name=pair_name, op=low.op, dtype=F16x2))
            continue
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)) and _pids_with_both_accums(s.body, name_to_new_pid, valid_candidates):
            # Pair Accums inside.
            new_loop_body = _pair_accums_in_body(s.body, valid_candidates, name_to_new_pid)
            out.append(s.with_bodies((new_loop_body,)))
            # Unpack after the Loop to restore scalar names.
            for low, high, pair_name in valid_candidates:
                out.append(Unpack(low_name=low.name, high_name=high.name, value=pair_name, lane_dtype=F16))
            continue
        out.append(s)
    return out, True


def _pids_with_both_accums(
    body: Body,
    name_to_pair: dict[str, int],
    candidates: list[tuple[Init, Init, str]],
) -> set[int]:
    """For each candidate pid, check whether BOTH its Accums appear in
    this body. Return the set of pids with both Accums present."""
    seen_low: dict[int, bool] = {}
    seen_high: dict[int, bool] = {}
    for s in body.iter():
        if isinstance(s, Accum) and s.name in name_to_pair:
            pid = name_to_pair[s.name]
            low, high, _ = candidates[pid]
            if s.name == low.name:
                seen_low[pid] = True
            elif s.name == high.name:
                seen_high[pid] = True
    return {pid for pid in seen_low if seen_high.get(pid)}


def _pair_accums_in_body(
    body: Body,
    candidates: list[tuple[Init, Init, str]],
    name_to_pair: dict[str, int],
) -> Body:
    """Walk a body. When we see the two Accums of a candidate pair,
    replace the second with ``[Pack(v_pair, v_low, v_high),
    Accum(pair_name, v_pair, op, F16x2)]`` and drop the first. (Or
    rather: defer the first, emit nothing until both seen.) Pass
    everything else through. Recurses into nested bodies."""
    out: list[Stmt] = []
    pending_low: dict[int, Accum] = {}  # pid -> first-seen Accum

    for s in body:
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            out.append(s.with_bodies((_pair_accums_in_body(s.body, candidates, name_to_pair),)))
            continue
        if isinstance(s, Cond):
            out.append(
                Cond(
                    cond=s.cond,
                    body=_pair_accums_in_body(s.body, candidates, name_to_pair),
                    else_body=_pair_accums_in_body(s.else_body, candidates, name_to_pair),
                )
            )
            continue
        if isinstance(s, (GridTile, ThreadTile)):
            out.append(s.with_bodies((_pair_accums_in_body(s.body, candidates, name_to_pair),)))
            continue
        if isinstance(s, Accum) and s.name in name_to_pair:
            pid = name_to_pair[s.name]
            low_init, high_init, pair_name = candidates[pid]
            if s.name == low_init.name:
                # First Accum — defer emission until we see the high one.
                pending_low[pid] = s
                continue
            # high Accum of the pair.
            if pid not in pending_low:
                # high seen first or low missing — pass through as scalar
                # (shouldn't happen given the pre-validation, but
                # defensive).
                out.append(s)
                continue
            low_accum = pending_low.pop(pid)
            pack_name = f"{low_accum.value}_{s.value}_p"
            out.append(Pack(name=pack_name, low=low_accum.value, high=s.value, dtype=F16x2))
            # Carry forward axes from one of the paired Accums — the pair
            # reduces over the same axis set (they're sibling Accums of
            # the same reduce loop, hence the pre-validation check).
            out.append(Accum(name=pair_name, value=pack_name, op=s.op, dtype=F16x2, axes=s.axes))
            continue
        out.append(s)

    # Any pending low Accums (no matching high in this body) — pass
    # through as scalar. This shouldn't happen given validation, but
    # restores correctness if it does.
    for _pid, low_accum in pending_low.items():
        out.append(low_accum)

    return Body(tuple(out))
