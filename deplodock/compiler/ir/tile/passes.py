"""Tile-IR Stmt registrations for the shared rewrite / simplify dispatch.

The Stage hierarchy (``Stage`` / ``BufferedStage`` / ``AsyncBufferedStage`` /
``TmaBufferedStage`` / ``ComputeStage``) goes through one introspection-based
handler — adding a new ``Expr`` / ``Axis`` field to any subclass (or to
``Source``) is picked up automatically without an override. ``Combine`` has
its own handler because it carries SSA names that need ``rename`` applied.

Stage's consumer ``body`` (and ``ComputeStage.compute``) are explicit-recursion
fields: ``_stage_kwargs`` walks ``Expr`` / ``Axis`` content but stops at
``Stmt`` boundaries, so we recurse into the body stmts ourselves.

Registration runs at module import (loaded from the bottom of
``tile/ir.py`` after class definitions, breaking the tile→stmt→tile cycle).
"""

from __future__ import annotations

from dataclasses import replace as _replace

from deplodock.compiler.ir.expr import Expr, Interval, SimplifyCtx
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.passes import AxisFn, Rename, _stage_kwargs, rewrite, simplify
from deplodock.compiler.ir.tile.ir import (
    AsyncWait,
    ComputeStage,
    GridTile,
    ParallelTile,
    RegisterTile,
    SerialTile,
    Stage,
    StridedTile,
    ThreadTile,
)


@rewrite.register
def _(s: Stage, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    kwargs = _stage_kwargs(s, on_expr=sigma.apply, on_axis=axis_fn)
    kwargs["body"] = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    if isinstance(s, ComputeStage):
        kwargs["compute"] = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.compute)
    return type(s)(**kwargs)


@simplify.register
def _(s: Stage, ctx: SimplifyCtx) -> Stmt:
    kwargs = _stage_kwargs(s, on_expr=lambda e: e.simplify(ctx), on_axis=lambda a: a)
    kwargs["body"] = tuple(simplify(c, ctx) for c in s.body)
    if isinstance(s, ComputeStage):
        kwargs["compute"] = tuple(simplify(c, ctx) for c in s.compute)
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


# Combine has no Expr fields — default ``simplify`` (identity) handles it.


# ---------------------------------------------------------------------------
# New tile flavor hierarchy (GridTile / ThreadTile / RegisterTile / SerialTile
# / StridedTile). Each carries a single body (or axes + body); rewrite descends
# into the body recursively, simplify extends the SimplifyCtx with axis ranges.
# ---------------------------------------------------------------------------


def _parallel_rewrite(s: ParallelTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(axis_fn(ax) for ax in s.axes)
    new_body = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    return _replace(s, axes=new_axes, body=new_body)


def _parallel_simplify(s: ParallelTile, ctx: SimplifyCtx) -> Stmt:
    inner = ctx
    for ax in s.axes:
        inner = inner.extend(ax.name, Interval(0, int(ax.extent) - 1))
    new_body = tuple(simplify(c, inner) for c in s.body)
    return _replace(s, body=new_body)


@rewrite.register
def _(s: GridTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(axis_fn(ax) for ax in s.axes)
    new_body = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    # Keep splitk_axes (axis names) in sync with any axis renames applied
    # via axis_fn. Build a name → renamed-name map from the (axis, axis_fn(axis))
    # pairs we just computed.
    name_map = {old.name: new.name for old, new in zip(s.axes, new_axes, strict=True)}
    new_splitk = tuple(name_map.get(n, n) for n in s.splitk_axes)
    return GridTile(axes=new_axes, body=new_body, splitk_axes=new_splitk)


@simplify.register
def _(s: GridTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: ThreadTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(axis_fn(ax) for ax in s.axes)
    new_body = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    # Mirror GridTile's splitk_axes propagation for cooperative_axes.
    name_map = {old.name: new.name for old, new in zip(s.axes, new_axes, strict=True)}
    new_coop = tuple(name_map.get(n, n) for n in s.cooperative_axes)
    return ThreadTile(axes=new_axes, body=new_body, cooperative_axes=new_coop)


@simplify.register
def _(s: ThreadTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: RegisterTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return _parallel_rewrite(s, rename, sigma, axis_fn)


@simplify.register
def _(s: RegisterTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: SerialTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return SerialTile(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        kind=s.kind,
        unroll=s.unroll,
    )


@simplify.register
def _(s: SerialTile, ctx: SimplifyCtx) -> Stmt:
    inner = ctx.extend(s.axis.name, Interval(0, int(s.axis.extent) - 1))
    return SerialTile(
        axis=s.axis,
        body=tuple(simplify(c, inner) for c in s.body),
        kind=s.kind,
        unroll=s.unroll,
    )


@rewrite.register
def _(s: StridedTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    step = sigma.apply(s.step) if isinstance(s.step, Expr) else s.step
    return StridedTile(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        start=sigma.apply(s.start),
        step=step,
        unroll=s.unroll,
    )


@simplify.register
def _(s: StridedTile, ctx: SimplifyCtx) -> Stmt:
    inner = ctx.extend(s.axis.name, Interval(0, int(s.axis.extent) - 1))
    step = s.step.simplify(ctx) if isinstance(s.step, Expr) else s.step
    return StridedTile(
        axis=s.axis,
        body=tuple(simplify(c, inner) for c in s.body),
        start=s.start.simplify(ctx),
        step=step,
        unroll=s.unroll,
    )
