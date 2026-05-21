"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, etc.) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes (``001_tileify``, ``006a_register_tile_planned``,
``007_stage_inputs``, ...) read the tags and skip their own equivalent
decisions, doing only the leftover rewrites (lift to ``Tile.axes``,
replicate stmts, build stages).

**M4 scope** — matmul register tile. Detect matmul-shaped LoopOps,
pre-split the outer M / N output Loops by ``(FM, FN)`` from
:func:`tuning.register_tile_shape`, tag the inner halves
``Role.REGISTER``, and σ-substitute the body. ``001_tileify`` lifts
M_o / N_o to ``Tile.axes`` and stops at the REGISTER tags;
``006a_register_tile_planned`` replicates the per-cell bodies *before*
``007_stage_inputs`` runs.

**M7 scope** — matmul K chunking. After the M/N register-tile decision
(if any), locate the matmul K reduce and pre-split it into
``Loop(K_o, SERIAL_OUTER) → Loop(K_i, reduce, STAGE_INNER)`` with
σ: K → K_o*BK + K_i. The planner forks over a ``BK`` knob (same
candidates as ``002_chunk_matmul_k``) so ``deplodock tune`` still
walks the same variant space; greedy callers pick variant 0
(the heuristic ``forced_bk`` value). ``002_chunk_matmul_k`` carries an
idempotence guard that no-ops once the planner has stamped
``Role.STAGE_INNER`` on the matmul reduce.

Gated by ``DEPLODOCK_PLANNER`` env var so each milestone can test the
new path against the legacy default (=0) for structural equivalence.
"""

from __future__ import annotations

import os
from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Tile
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

PATTERN = [Pattern("root", LoopOp)]

_ENABLE_ENV = "DEPLODOCK_PLANNER"
# Knob stamp signalling the planner produced output (for planner-side
# idempotence — re-firing on a planned LoopOp is a no-op). Downstream
# 006a uses ``Role.REGISTER`` presence + ``FN`` absence as its trigger.
_PLANNER_KNOB = "PLANNER"

# Matches ``002_chunk_matmul_k._BK_CANDIDATES`` so the planner-driven
# fork enumerates the same variant space as the legacy 002 fork did.
_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    if not os.environ.get(_ENABLE_ENV):
        raise RuleSkipped(f"{_ENABLE_ENV} not set")
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    after_reg = _try_matmul_register_tile(ctx, loop_op)
    base = after_reg if after_reg is not None else loop_op

    k_variants = _try_matmul_k_chunk(ctx, base)
    if k_variants is None and after_reg is None:
        raise RuleSkipped("no planner branch matched")

    if k_variants is None:
        return _stamp_planned(base)
    return [_stamp_planned(v) for v in k_variants]


def _stamp_planned(op: LoopOp) -> LoopOp:
    knobs = dict(op.knobs)
    knobs[_PLANNER_KNOB] = True
    return LoopOp(body=op.body, knobs=knobs)


# --- matmul register-tile branch -------------------------------------


def _try_matmul_register_tile(ctx: Context, loop_op: LoopOp) -> LoopOp | None:
    """Detect a matmul-shape LoopOp; if eligible, pre-split the outer two
    output Loops by ``(FM, FN)`` with ``Role.REGISTER`` on the inner
    halves and σ-substitute the body. Returns ``None`` when no
    transformation applies (legacy 008 then handles it post-staging)."""
    fm, fn = _pick_register_factors(loop_op)
    if fm <= 1 and fn <= 1:
        return None

    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None
    outer_m, outer_n = chain[0], chain[1]
    if int(outer_m.axis.extent) % fm != 0 or int(outer_n.axis.extent) % fn != 0:
        return None

    new_body = _split_register_outer_two(loop_op.body, outer_m.axis.name, outer_n.axis.name, fm, fn)
    return LoopOp(body=new_body, knobs=dict(loop_op.knobs))


def _pick_register_factors(loop_op: LoopOp) -> tuple[int, int]:
    """Heuristic ``(FM, FN)`` from :func:`tuning.register_tile_shape`
    against a synthetic Tile carrying the outer-chain axes as THREAD —
    same classification + small-tile guard 008 applies post-tileify."""
    from deplodock.compiler.tuning import _has_matmul_reduce, register_tile_shape  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body):
        return (1, 1)
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return (1, 1)
    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )
    fm, fn = register_tile_shape(synthetic)
    return int(fm), int(fn)


def _outer_free_loop_chain(body) -> tuple[Loop, ...]:
    """Walk the outer single-stmt chain of untagged free Loops outermost-
    first. Mirrors ``001_tileify._strip_outer_free_chain``."""
    out: list[Loop] = []
    cur = tuple(body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        out.append(cur[0])
        cur = tuple(cur[0].body)
    return tuple(out)


def _split_register_outer_two(body, m_name: str, n_name: str, fm: int, fn: int):
    """``Loop(M:E) → Loop(M_o:E/FM) → Loop(N_o:E'/FN) → Loop(M_i:FM, REG)
    → Loop(N_i:FN, REG) → σ(body)`` where σ maps M → M_o*FM+M_i and
    N → N_o*FN+N_i in one pass over the innermost body."""

    def _identity_rename(name: str) -> str:
        return name

    stmts = tuple(body)
    assert len(stmts) == 1 and isinstance(stmts[0], Loop) and stmts[0].axis.name == m_name
    m_loop = stmts[0]
    m_body = tuple(m_loop.body)
    assert len(m_body) == 1 and isinstance(m_body[0], Loop) and m_body[0].axis.name == n_name
    n_loop = m_body[0]
    inner = tuple(n_loop.body)

    m_o = Axis(f"{m_name}_o", int(m_loop.axis.extent) // fm)
    m_i = Axis(f"{m_name}_i", fm)
    n_o = Axis(f"{n_name}_o", int(n_loop.axis.extent) // fn)
    n_i = Axis(f"{n_name}_i", fn)

    sigma = Sigma(
        {
            m_name: Var(m_o.name) * Literal(fm, "int") + Var(m_i.name),
            n_name: Var(n_o.name) * Literal(fn, "int") + Var(n_i.name),
        }
    )
    inner_rewritten = tuple(s.rewrite(_identity_rename, sigma) for s in inner)

    rebuilt = Loop(
        axis=m_o,
        body=(
            Loop(
                axis=n_o,
                body=(
                    Loop(
                        axis=m_i,
                        role=Role.REGISTER,
                        body=(Loop(axis=n_i, role=Role.REGISTER, body=inner_rewritten),),
                    ),
                ),
            ),
        ),
    )
    return (rebuilt,)


# --- matmul K-chunk branch ------------------------------------------


def _try_matmul_k_chunk(ctx: Context, loop_op: LoopOp) -> list[LoopOp] | None:
    """Fork over BK for the matmul K reduce. Splits K → K_o
    (``Role.SERIAL_OUTER``) × K_i (``Role.STAGE_INNER``) with
    σ: K → K_o*BK + K_i. Returns one variant per BK candidate that
    divides K (with K > BK); variant 0 carries the heuristic
    ``forced_bk`` value so greedy callers (no autotune DB) reproduce
    the legacy 002 default. Returns ``None`` when no matmul K reduce
    is reachable or it's already chunked."""
    site = _find_first_matmul_reduce(loop_op.body)
    if site is None:
        return None
    if site.role is Role.STAGE_INNER:
        return None  # already chunked
    K = int(site.axis.extent)
    cands = _bk_candidates_for(ctx, loop_op, K)
    if not cands:
        return None
    variants: list[LoopOp] = []
    for bk in cands:
        new_body, changed = _chunk_first_matmul_k(loop_op.body, bk)
        if not changed:
            continue
        knobs = dict(loop_op.knobs)
        knobs[BK.name] = bk
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants or None


def _bk_candidates_for(ctx: Context, loop_op: LoopOp, K: int) -> tuple[int, ...]:
    """Filter ``_BK_CANDIDATES`` by ``K % bk == 0`` ∧ ``K > bk``, with
    the heuristic ``forced_bk`` value first when it qualifies (so
    variant 0 reproduces the legacy 002 default for greedy callers)."""
    from deplodock.compiler.tuning import forced_bk

    synthetic = _synthetic_tile_post_register(loop_op)
    forced = forced_bk(synthetic, ctx.static_smem_cap)
    base = [c for c in _BK_CANDIDATES if K % c == 0 and K > c]
    if forced is not None and K % forced == 0 and K > forced and forced not in base:
        base = [forced, *base]
    elif forced is not None and forced in base:
        base = [forced, *(c for c in base if c != forced)]
    return tuple(base)


def _synthetic_tile_post_register(loop_op: LoopOp) -> Tile | None:
    """Build a synthetic ``Tile`` reflecting the launch shape that
    ``001_tileify`` will produce — outer free Loops above the first
    REGISTER tag become ``Tile.axes`` (BIND_THREAD). Used to feed
    :func:`tuning.forced_bk` so the picker sees the post-register-tile
    extents."""
    chain: list[Loop] = []
    cur = tuple(loop_op.body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    if not chain:
        return None
    return Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )


def _find_first_matmul_reduce(stmts) -> Loop | None:
    """Locate the first matmul-shaped reduce ``Loop`` reachable by
    descent through Loops / StridedLoops / Conds. Mirrors 002's
    walker so the K we measure is the K the rewrite will hit."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_first_matmul_reduce(s.body)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_first_matmul_reduce(s.body) or _find_first_matmul_reduce(s.else_body)
            if found is not None:
                return found
    return None


def _chunk_first_matmul_k(stmts, bk: int) -> tuple[tuple, bool]:
    """Walk ``stmts``, replacing the first matmul-shaped reduce
    ``Loop(K, …)`` with ``Loop(K_o, SERIAL_OUTER, Loop(K_i, reduce,
    STAGE_INNER, σ(body)))``. Recurses through wrapper Loops /
    StridedLoops / Conds so a matmul nested under register-tile REG
    loops (the typical post-M4 shape) is reachable. Returns
    ``(new_body, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            chunked = _chunk_k_loop(s, bk)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_first_matmul_k(s.body, bk)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_first_matmul_k(s.body, bk)
            inner_e, ce = _chunk_first_matmul_k(s.else_body, bk)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _chunk_k_loop(loop: Loop, bk: int) -> Loop | None:
    K = int(loop.axis.extent)
    if K % bk != 0 or K <= bk:
        return None
    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // bk)
    K_i = Axis(f"{K_name}_i", bk)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_identity_rename_k, sigma) for s in loop.body)
    return Loop(
        axis=K_o,
        role=Role.SERIAL_OUTER,
        body=(Loop(axis=K_i, role=Role.STAGE_INNER, body=inner_body),),
    )


def _identity_rename_k(name: str) -> str:
    return name
