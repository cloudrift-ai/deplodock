"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, etc.) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes (``001_tileify``, ``006a_register_tile_planned``,
``007_stage_inputs``, ...) read the tags and skip their own equivalent
decisions, doing only the leftover rewrites (lift to ``Tile.axes``,
replicate stmts, build stages).

**M4 scope** — matmul register tile. The planner detects matmul-shaped
LoopOps, pre-splits the outer M / N output Loops by ``(FM, FN)`` from
:func:`tuning.register_tile_shape`, tags the inner halves with
``Role.REGISTER``, and σ-substitutes the body. ``001_tileify`` then
lifts M_o / N_o to ``Tile.axes`` and stops at the REGISTER tags;
``006a_register_tile_planned`` replicates the per-cell bodies *before*
``007_stage_inputs`` runs (so Stages see the F×F replicated Loads and
build a single coalesced cache slab).

Subsequent milestones populate the planner further: M5 = cooperative-
reduce + pipeline tags, M6 = cleanup + invariant assertion. Matmul K
chunking (currently 002) and SPLITK (currently 003) stay in their
existing passes for now — they don't suffer the staging-collision
issue because they don't introduce per-cell REGISTER axes.

Gated by ``DEPLODOCK_PLANNER`` env var so each milestone can test the
new path against the legacy default (=0) for structural equivalence.
"""

from __future__ import annotations

import os

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Loop
from deplodock.compiler.ir.tile.ir import Tile
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]

_ENABLE_ENV = "DEPLODOCK_PLANNER"
# Knob stamp signalling the planner produced output (for planner-side
# idempotence — re-firing on a planned LoopOp is a no-op). Downstream
# 006a uses ``Role.REGISTER`` presence + ``FN`` absence as its trigger.
_PLANNER_KNOB = "PLANNER"


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp:
    if not os.environ.get(_ENABLE_ENV):
        raise RuleSkipped(f"{_ENABLE_ENV} not set")
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    new_op = _try_matmul_register_tile(ctx, loop_op)
    if new_op is None:
        raise RuleSkipped("no planner branch matched (M4 scope: matmul-register-tile only)")
    return new_op


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
    knobs = dict(loop_op.knobs)
    knobs[_PLANNER_KNOB] = True
    return LoopOp(body=new_body, knobs=knobs)


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
