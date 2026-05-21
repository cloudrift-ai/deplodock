"""Planner-driven register tile — runs *before* ``007_stage_inputs``.

When ``000_partition_planner`` pre-splits a matmul's output Loops and
tags the inner halves ``Role.REGISTER``, this pass replicates those
loops per-cell **before** staging picks cache axes. That matters: if
the REGISTER tag survives into ``007_stage_inputs``, the Stage's cache
slab won't include the per-cell M_i / N_i axes (they aren't in
``Tile.axes``), and ``008_register_tile``'s post-staging replicate
would duplicate Stage stmts with name collisions, corrupting smem.

Running the per-cell replication here means ``007_stage_inputs`` sees
F×F copies of each body Load with distinct M_o*F+i / N_o*F+j
gmem indices. Staging then coalesces them through their shared source
buffer and the cache slab spans the full BM × BK / BK × BN — exactly
the layout today's post-staging ``008_register_tile`` produces in the
no-planner path.

When no REGISTER tags are present (default ``DEPLODOCK_PLANNER=0``),
the pass skips and the legacy ``008_register_tile`` fork still owns
the split-and-replicate work post-staging. Both stamp ``FN`` so the
existing idempotence marker prevents double-replication.
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Role
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Loop, Stmt
from deplodock.compiler.ir.tile.ir import Tile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import replicate_along_axis, single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, tile = single_tile(body)

    new_body, factors = _replicate_register_loops(tile.body)
    if not factors:
        # Idempotence + bail-out: once REGISTER Loops are gone (either
        # because the planner emitted a (FM=1, FN=1) variant or because
        # 006a already replicated), this pass is a no-op. In env=0 mode
        # no REGISTER tags ever appear so we bail to the legacy 008.
        raise RuleSkipped("no Role.REGISTER tags in body — legacy 008 owns this kernel")

    # FM/FN are stamped by the planner in env=1 mode; preserve them
    # rather than overwriting. In env=0 the planner never fired so
    # ``factors`` populates them on first run.
    knobs = dict(root.op.knobs)
    if len(factors) >= 1 and "FM" not in knobs:
        knobs["FM"] = factors[0]
    if len(factors) >= 2 and "FN" not in knobs:
        knobs["FN"] = factors[1]
    new_tile = Tile(axes=tile.axes, body=new_body)
    return TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=root.op.name, knobs=knobs)


def _replicate_register_loops(body: Body) -> tuple[Body, list[int]]:
    """Inside-out unwrap of ``Role.REGISTER`` Loops. For each tagged
    layer, recurse into nested REGISTER first, then replicate this
    layer's body by ``axis.extent`` with ``σ: axis → literal(i)``.
    Returns ``(new_body, factors)`` with factors in outermost-first
    order. Caller stamps factors[0] → FM, factors[1] → FN."""
    out: list[Stmt] = []
    factors: list[int] = []
    for s in body:
        if isinstance(s, Loop) and s.role is Role.REGISTER:
            factor = int(s.axis.extent)
            inner_unwrapped, inner_factors = _replicate_register_loops(s.body)
            replicated = replicate_along_axis(inner_unwrapped, s.axis.name, factor, _sigma_to_literal(s.axis.name))
            out.extend(replicated)
            factors.append(factor)
            factors.extend(inner_factors)
        else:
            out.append(s)
    return Body(out), factors


def _sigma_to_literal(axis: str) -> Callable[[int], Sigma]:
    """σ-factory: ``axis → Literal(i)``."""

    def _f(i: int) -> Sigma:
        return Sigma({axis: Literal(i, "int")})

    return _f
