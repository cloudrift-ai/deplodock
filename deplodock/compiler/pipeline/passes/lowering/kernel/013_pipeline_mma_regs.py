"""Register-tier software pipelining for the mma.sync K reduce — cross-K_o
operand prefetch (``plans/mma-register-pipeline.md``).

After ``080_pipeline_stages`` the matmul K_o main loop is the smem TMA ring::

    SerialTile(K_o, serial_outer):
        AsyncWait(phase, slot=K_o % bc)          # wait this tile's smem slot
        SerialTile(K_i, reduce): [ldmatrix×, mma×]
        StageBundle(issue TMA for slot (K_o + bc-1) % bc)

The ``AsyncWait`` materializes to ``MbarrierWait + __syncthreads``; that Sync is
the smem-ring WAR guard (slot ``s`` is overwritten one iteration after it's
consumed, and the per-iteration Sync brackets all reads of ``s`` before the
refill). At BC=4 — cuBLAS's config, 1 block/SM, shared-mem-bound — the
tensor-core pipeline *drains* at every tile boundary: warps hit the
``MbarrierWait`` and the pipe idles (ncu ``barrier`` 0.40 vs cuBLAS 0.05,
tensor-pipe 42% vs 47%). cuBLAS hides this by spending ~66 extra regs/thread
(232 vs our 166) on operand double-buffering — register pipelining that keeps
the pipe fed across the boundary. Register headroom is free here (occupancy is
smem-bound, not register-bound).

This pass prefetches each K_o tile's **first substep** operands one iteration
ahead, into a second ``__rp1`` fragment buffer, by **moving** the slot wait to
the iteration bottom (not adding one) so the Sync count — and the WAR guard —
stays one per iteration::

    SerialTile(K_o):
        Cond(K_o == 0):                          # iteration-0 prime (no prev
            AsyncWait(phase@0, slot 0)            #   iteration to prefetch from)
            ldmatrix(K_o=0, K_i=0) → __rp1
        mma(sub0) ← __rp1                         # current tile, issues immediately
        ldmatrix(K_i=1..) → main; mma             # current tile sub1.. (slot
                                                  #   waited at last prefetch)
        StageBundle(issue TMA)                    # unchanged
        AsyncWait(phase@K_o+1, slot (K_o+1)%bc)   # wait tile K_o+1's slot
        ldmatrix(K_o+1, K_i=0) → __rp1            # prefetch next tile's sub0

Each tile's first ``mma`` reads operands already in registers, so it issues
right after the boundary and the pipe refills without the ldmatrix-latency
bubble. The relocated Sync still brackets every slot's reads before its refill
(tile t's sub1 reads slot t before iter t's bottom Sync; iter t+1's issue-TMA
overwrites slot t only after it). The accumulator (``c`` role) never aliases.

**Loop-carried register / SSA note.** ``__rp1`` is written by the prefetch at
the bottom and read by the ``mma`` at the top of the *next* iteration — a
loop-carried value, which is not single-assignment within one loop body.
``normalize_body``'s ``topo_sort_siblings`` would otherwise bind the read to the
in-body prefetch write and sink the read below it (computing the wrong tile);
``_LOOP_CARRIED_MARK`` (``ir/stmt/normalize.py``) excludes ``__rp1`` names from
the topo dep graph so the emitted order is preserved. The TMA-group partitioner
(``_tma_groups``) likewise assigns the ``Cond``-nested prime wait to the loop's
group.

**The knob.** ``DEPLODOCK_REG_PIPELINE`` (BOOL, default off) gates the feature —
a measured autotune fork like ``PAD_SMEM`` / ``HOIST_COMPUTE``, off by default
so the greedy / DB-less path and every existing test stay byte-identical; a pin
(``DEPLODOCK_REG_PIPELINE=1``) forces it for bring-up / A-B benching.

Run order: AFTER ``012_fuse_sibling_register_cells`` (per-cell frags concrete)
and BEFORE ``020_place_inits`` / ``100_materialize_tile``. Falls back to identity
when the K_o loop isn't the TMA-pipelined shape (cp.async / sync-staged configs
keep their schedule). Idempotence: keyed on the ``REG_PIPELINE`` knob.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx, RegFragment
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Stmt
from deplodock.compiler.ir.tile.ir import AsyncWait, SerialTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

# Suffix for the second (loop-carried) operand-fragment buffer. Must contain the
# ``_LOOP_CARRIED_MARK`` that ``ir/stmt/normalize.py`` recognises so the topo
# sort leaves these reads/writes in source order. The accumulator (``c`` role)
# is never aliased — it accumulates in place across all K.
_BUF1 = "__rp1"

PATTERN = [Pattern("root", TileOp)]

# Default off (first hint): the greedy / DB-less path and every existing test
# stay byte-identical. The autotuner forks on (False, True) and measures both;
# a pin ``DEPLODOCK_REG_PIPELINE=1`` forces it on for bring-up / A-B benching.
REG_PIPELINE = Knob(
    "REG_PIPELINE",
    KnobType.BOOL,
    hints=(False, True),
    help=(
        "Software-pipeline the mma.sync K reduce: prefetch each K_o tile's first "
        "substep operands one iteration ahead (across the smem-stage boundary) "
        "into a second register buffer, so the next tile's mma issues without "
        "draining the tensor pipe. Costs ~32 regs/thread — measured fork, off by "
        "default. 1 forces it on."
    ),
)


def rewrite(root: Node) -> list[TileOp] | None:
    if REG_PIPELINE.name in root.op.knobs:
        raise RuleSkipped("reg-pipeline already applied (idempotence via knob)")
    if not _has_mma_reduce(root.op.body):
        raise RuleSkipped("no mma.sync K reduce to pipeline (ldmatrix + mma.sync chain absent)")

    variants: list[TileOp] = []
    for polarity in REG_PIPELINE.narrow((False, True)):
        # ``True`` prefetches each K_o tile's first substep one iteration ahead
        # across the smem-stage boundary; ``False`` leaves the body untouched
        # (byte-identical to the pre-pass path). The accumulator stays single.
        new_body = _pipeline(root.op.body) if polarity else root.op.body
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, REG_PIPELINE.name: polarity}))
    if not variants:
        raise RuleSkipped("REG_PIPELINE env pin produced no matching variants")
    return variants


# ---------------------------------------------------------------------------
# Cross-K_o boundary prefetch (loop-carried operand double-buffer)
# ---------------------------------------------------------------------------


def _pipeline(body: Body) -> Body:
    """Apply the cross-K_o prefetch to the TMA-pipelined main loop in ``body``
    and declare the second operand-fragment buffer. No-op (returns ``body``)
    when there's no TMA-pipelined ``serial_outer`` K_o loop to prefetch across —
    the within-tile boundary alone is one basic block ptxas already overlaps."""
    ko = _find_pipelined_ko(body)
    if ko is None:
        return body
    reduce = _reduce_of(ko)
    operand_frags = frozenset(s.frag for s in reduce.body if isinstance(s, LdmatrixLoad))
    if not operand_frags:
        return body
    alias = {f: f + _BUF1 for f in operand_frags}
    return _rewrite_body(body, alias)


def _find_pipelined_ko(body: Body) -> SerialTile | None:
    """The first ``serial_outer`` K_o ``SerialTile`` (static extent ≥ 2) carrying
    a TMA ``AsyncWait`` (phase + slot set) and a flat ldmatrix+mma reduce —
    ``080``'s pipelined main loop. The epilogue drain reduces are bare siblings
    (not ``serial_outer``), so they don't match."""
    for s in body.iter():
        if isinstance(s, SerialTile) and _is_pipelined_ko(s):
            return s
    return None


def _is_pipelined_ko(s: SerialTile) -> bool:
    if s.kind != "serial_outer" or not s.axis.extent.is_static or s.axis.extent.as_static() < 2:
        return False
    aw = _top_async_wait(s)
    return aw is not None and aw.phase is not None and aw.slot is not None and _reduce_of(s) is not None


def _top_async_wait(ko: SerialTile) -> AsyncWait | None:
    for s in ko.body:
        if isinstance(s, AsyncWait):
            return s
    return None


def _reduce_of(ko: SerialTile) -> SerialTile | None:
    """The K_i reduce directly inside the K_o body — a ``SerialTile`` whose flat
    body is only ``LdmatrixLoad`` + ``MmaSyncPtx`` (≥ 1 of each)."""
    for s in ko.body:
        if isinstance(s, SerialTile) and _is_reduce(s):
            return s
    return None


def _is_reduce(s: SerialTile) -> bool:
    if not s.axis.extent.is_static:
        return False
    has_ldm = has_mma = False
    for c in s.body:
        if isinstance(c, LdmatrixLoad):
            has_ldm = True
        elif isinstance(c, MmaSyncPtx):
            has_mma = True
        else:
            return False  # foreign stmt — not the clean ld+mma reduce shape
    return has_ldm and has_mma


def _rewrite_body(body: Body, alias: dict[str, str]) -> Body:
    """Recursively rewrite ``body``: inject the loop-carried twin decl after each
    operand ``RegFragment``, transform the pipelined K_o loop, recurse the rest."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, RegFragment) and s.name in alias:
            out.append(s)
            out.append(_replace(s, name=alias[s.name]))  # second operand buffer
            continue
        if isinstance(s, SerialTile) and _is_pipelined_ko(s):
            out.extend(_transform_ko(s, alias))
            continue
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_rewrite_body(b, alias) for b in nested))
        out.append(s)
    return Body(tuple(out))


def _alias_rename(alias: dict[str, str]):
    """Rename operand frags to their ``__rp1`` twin; leave the ``c`` accumulator
    (not in ``alias``) and everything else untouched."""
    return lambda nm: alias.get(nm, nm)


def _ident(nm: str) -> str:
    return nm


def _transform_ko(ko: SerialTile, alias: dict[str, str]) -> list[Stmt]:
    """Rewrite the pipelined K_o loop into the cross-K_o prefetch shape: tile
    K_o's *first* substep operands come from the ``__rp1`` buffer prefetched one
    iteration ago (so its mma issues immediately after the boundary); the
    remaining substeps load + compute from the current slot (already waited at
    the previous iteration's prefetch). Prefetching only the first substep keeps
    the register cost low (+~32 regs) — full-tile prefetch regressed (register
    pressure outweighs the benefit on this mma-pipeline-bound kernel). Returns
    the stmt list replacing ``ko`` (the iteration-0 prime lives inside it under
    ``Cond(K_o == 0)``)."""
    top_aw = _top_async_wait(ko)
    reduce = _reduce_of(ko)
    kvar_o = ko.axis.name
    kvar_i = reduce.axis.name
    n_i = reduce.axis.extent.as_static()
    lds = [c for c in reduce.body if isinstance(c, LdmatrixLoad)]
    mmas = [c for c in reduce.body if isinstance(c, MmaSyncPtx)]
    to_rp1 = _alias_rename(alias)

    new_reduce: list[Stmt] = [m.rewrite(to_rp1, Sigma({})) for m in mmas]  # sub0 ← __rp1
    for s in range(1, n_i):
        sig = Sigma({kvar_i: Literal(s, "int")})
        new_reduce += [ld.rewrite(_ident, sig) for ld in lds]
        new_reduce += [m.rewrite(_ident, Sigma({})) for m in mmas]

    # The issue-TMA StageBundle(s) — everything but the (relocated) wait + reduce.
    post = [s for s in ko.body if s is not top_aw and s is not reduce]

    # Bottom prefetch: wait tile K_o+1's slot, load its first substep → __rp1.
    # The wait is *moved* here from the loop top (not added) so the Sync count
    # stays one per iteration and the smem-ring WAR guard is preserved.
    sig_next = Sigma({kvar_o: Var(kvar_o) + Literal(1, "int")})
    prefetch_aw = _replace(top_aw, phase=sig_next.apply(top_aw.phase), slot=sig_next.apply(top_aw.slot))
    pf_sig = Sigma({kvar_o: Var(kvar_o) + Literal(1, "int"), kvar_i: Literal(0, "int")})
    prefetch_lds = [ld.rewrite(to_rp1, pf_sig) for ld in lds]

    # Iteration-0 prime: no previous iteration prefetched tile 0, so wait its slot
    # and load its first substep → __rp1 here. Kept *inside* the loop (a guarded
    # block, not a prologue before it) so the TMA-group partitioner doesn't split
    # the smem ring.
    sig0 = Sigma({kvar_o: Literal(0, "int")})
    prime_aw = _replace(top_aw, phase=sig0.apply(top_aw.phase), slot=sig0.apply(top_aw.slot))
    p0_sig = Sigma({kvar_o: Literal(0, "int"), kvar_i: Literal(0, "int")})
    prime_lds = [ld.rewrite(to_rp1, p0_sig) for ld in lds]
    prime = Cond(cond=BinaryExpr("==", Var(kvar_o), Literal(0, "int")), body=Body((prime_aw, *prime_lds)), else_body=())

    new_ko = ko.with_bodies((Body(tuple([prime, *new_reduce, *post, prefetch_aw, *prefetch_lds])),))
    return [new_ko]


def _has_mma_reduce(body: Body) -> bool:
    """True iff ``body`` holds a ``SerialTile`` whose (possibly nested) body
    contains both a ``LdmatrixLoad`` and a ``MmaSyncPtx`` — the K reduce the
    mma.sync path emits."""
    for s in body:
        if isinstance(s, SerialTile) and _chain_in(s.body):
            return True
        for sub in s.nested():
            if _has_mma_reduce(sub):
                return True
    return False


def _chain_in(body: Body) -> bool:
    """True iff ``body`` (recursively) contains both an ``LdmatrixLoad`` and a
    ``MmaSyncPtx``."""
    has_ldm = False
    has_mma = False
    for s in body.iter():
        if isinstance(s, LdmatrixLoad):
            has_ldm = True
        elif isinstance(s, MmaSyncPtx):
            has_mma = True
        if has_ldm and has_mma:
            return True
    return False
