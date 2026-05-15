"""Tile-IR Stmt registrations for the shared rewrite / simplify dispatch.

The Stage hierarchy (``Stage`` / ``BufferedStage`` / ``AsyncBufferedStage`` /
``TmaBufferedStage``) goes through one introspection-based handler — adding
a new ``Expr`` / ``Axis`` field to any subclass is picked up automatically
without an override. ``AsyncWait`` and ``Combine`` register their own
handlers because they're not dataclass-uniform with Stage.

Registration runs at module import (loaded from the bottom of
``tile/ir.py`` after class definitions, breaking the tile→stmt→tile
cycle).
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import SimplifyCtx
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.passes import AxisFn, Rename, _stage_kwargs, rewrite, simplify
from deplodock.compiler.ir.tile.ir import AsyncWait, Combine, Stage


@rewrite.register
def _(s: Stage, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # _stage_kwargs walks Expr/Axis fields but stops at Stmt boundaries — body
    # stmts need explicit recursion (parallel to Loop / StridedLoop handlers).
    kwargs = _stage_kwargs(s, on_expr=sigma.apply, on_axis=axis_fn)
    kwargs["body"] = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    return type(s)(**kwargs)


@simplify.register
def _(s: Stage, ctx: SimplifyCtx) -> Stmt:
    kwargs = _stage_kwargs(s, on_expr=lambda e: e.simplify(ctx), on_axis=lambda a: a)
    kwargs["body"] = tuple(simplify(c, ctx) for c in s.body)
    return type(s)(**kwargs)


@rewrite.register
def _(s: AsyncWait, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return AsyncWait(
        keep=s.keep,
        phase=sigma.apply(s.phase) if s.phase is not None else None,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@simplify.register
def _(s: AsyncWait, ctx: SimplifyCtx) -> Stmt:
    return AsyncWait(
        keep=s.keep,
        phase=s.phase.simplify(ctx) if s.phase is not None else None,
        slot=s.slot.simplify(ctx) if s.slot is not None else None,
    )


@rewrite.register
def _(s: Combine, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Combine(name=rename(s.name), op=s.op)


# Combine has no Expr fields — default ``simplify`` (identity) handles it.
