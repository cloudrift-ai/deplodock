"""Planner-driven register tile — runs *before* ``007_stage_inputs``.

When ``000_partition_planner`` pre-splits a matmul's output Loops and
tags the inner halves ``Role.REGISTER``, this pass replicates those
loops per-cell **before** staging picks cache axes. That matters: if
the REGISTER tag survives into ``007_stage_inputs``, the Stage's cache
slab won't include the per-cell M_r / N_r axes (they aren't in
``Tile.axes``).

Running the per-cell replication here means ``007_stage_inputs`` sees
F×F copies of each body Load with distinct M_b*F+i / N_b*F+j gmem
indices. Staging then coalesces them through their shared source
buffer and the cache slab spans the full BM × BK / BK × BN.

When no REGISTER tags are present (non-matmul kernels), the pass
skips. Stamps ``FM`` / ``FN`` so the planner-stamped values persist
and the rule is idempotent on a second visit.
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
        # No REGISTER tags in body — non-matmul kernel (planner only
        # stamps REGISTER on matmul) or this rule has already run and
        # consumed the tags. Either way, nothing to do.
        raise RuleSkipped("no Role.REGISTER tags in body")

    # FM/FN are stamped by the planner; preserve them rather than
    # overwriting (factors carry the same values).
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
