"""Stmt rewrite + simplify, dispatched by type.

Replaces the per-class ``Stmt.rewrite`` overrides on body-carrying and
leaf stmts, and the ``_simplify_stmt`` if-ladder in ``normalize``.
The Stage hierarchy uses ``dataclasses.fields()`` introspection inside
the registered handler — adding a new ``Expr`` / ``Axis`` field on a
Stage subclass is picked up automatically (no override needed, no
silent-drop bug).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from functools import singledispatch

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Interval, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt, _axis_identity
from deplodock.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Pack, Select, SelectBranch, Unpack, Write

Rename = Callable[[str], str]
AxisFn = Callable[[Axis], Axis]


def _rename_ssa_vars_in_expr(e: Expr, rename: Rename) -> Expr:
    """Apply ``rename`` to every free ``Var`` leaf inside ``e``.

    Used by ``Load`` / ``Write`` rewriters so that *indirect* indices
    (gather: ``x[a0, (int)in0]``, scatter: ``out[(int)idx_v] = ...``)
    have their SSA-name references rewritten when the enclosing body
    is replicated. Without this, the register-tile replicator in
    ``010_split_register_axes`` suffixes the defining Load's name
    (``in0`` → ``in0_1``) but leaves dependent indirect Loads pointing
    at the original ``in0`` — silently dropping the cross-replica data
    dependency.

    Axis-name Vars (``a0``, ``M_b``, …) are never in the rename map
    (it only carries SSA defines), so ``rename(name) == name`` for
    them and they pass through unchanged.
    """
    mapping = {n: Var(rename(n)) for n in e.free_vars() if rename(n) != n}
    return e.substitute(mapping) if mapping else e


# ---------------------------------------------------------------------------
# Generic walker — recurses tuples + plain dataclasses (Addressing, BoundAxis,
# SelectBranch); applies ``on_expr`` to Expr leaves and ``on_axis`` to Axis.
# Stmt is excluded — Stmt traversal goes through the singledispatch handlers.
# ---------------------------------------------------------------------------


def _walk(value, *, on_expr, on_axis):
    if isinstance(value, Expr):
        return on_expr(value)
    if isinstance(value, Axis):
        return on_axis(value)
    if isinstance(value, tuple):
        return tuple(_walk(v, on_expr=on_expr, on_axis=on_axis) for v in value)
    if is_dataclass(value) and not isinstance(value, Stmt):
        return type(value)(**{f.name: _walk(getattr(value, f.name), on_expr=on_expr, on_axis=on_axis) for f in fields(value)})
    return value


def _stage_kwargs(stage, *, on_expr, on_axis):
    return {f.name: _walk(getattr(stage, f.name), on_expr=on_expr, on_axis=on_axis) for f in fields(stage)}


# ---------------------------------------------------------------------------
# rewrite — sigma + axis_fn + SSA renaming
# ---------------------------------------------------------------------------


@singledispatch
def rewrite(stmt: Stmt, rename: Rename, sigma: Sigma = Sigma.IDENTITY, axis_fn: AxisFn = _axis_identity) -> Stmt:
    raise NotImplementedError(f"rewrite not registered for {type(stmt).__name__}")


@rewrite.register
def _(s: Load, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Load(
        names=tuple(rename(n) for n in s.names),
        input=s.input,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        dtype=s.dtype,
    )


@rewrite.register
def _(s: Pack, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Pack(name=rename(s.name), low=rename(s.low), high=rename(s.high), dtype=s.dtype)


@rewrite.register
def _(s: Unpack, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Unpack(
        low_name=rename(s.low_name),
        high_name=rename(s.high_name),
        value=rename(s.value),
        lane_dtype=s.lane_dtype,
    )


@rewrite.register
def _(s: Assign, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Assign(name=rename(s.name), op=s.op, args=tuple(rename(a) for a in s.args), dtype=s.dtype)


@rewrite.register
def _(s: Accum, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(n for old in s.axes for n in _rewrite_axis_name(old, sigma))
    return Accum(
        name=rename(s.name),
        value=rename(s.value),
        op=s.op,
        dtype=s.dtype,
        axes=new_axes,
    )


def _rewrite_axis_name(name: str, sigma: Sigma) -> tuple[str, ...]:
    """Apply ``sigma`` to an axis name and return the resulting axis
    name(s). Handles three cases:

    - ``sigma`` doesn't touch ``name``: returns ``(name,)``.
    - Pure rename (``Var(old) → Var(new)``): returns ``(new,)``.
    - σ-split (``Var(K) → Var(K_o)*N + Var(K_i)``, etc.): returns the
      free-var names of the substitution expression. An Accum that
      reduced over the original axis now reduces over the split sub-
      axes.
    """
    replacement = sigma.mapping.get(name)
    if replacement is None:
        return (name,)
    return tuple(sorted(replacement.free_vars()))


@rewrite.register
def _(s: Init, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Init(name=rename(s.name), op=s.op, dtype=s.dtype)


@rewrite.register
def _(s: Write, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        values=tuple(rename(n) for n in s.values),
        value_dtype=s.value_dtype,
        atomic=s.atomic,
    )


@rewrite.register
def _(s: Select, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Select(
        name=rename(s.name),
        branches=tuple(SelectBranch(value=rename(b.value), select=sigma.apply(b.select)) for b in s.branches),
    )


@rewrite.register
def _(s: Loop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Loop(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
    )


@rewrite.register
def _(s: StridedLoop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    step = sigma.apply(s.step) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=axis_fn(s.axis),
        start=sigma.apply(s.start),
        step=step,
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
    )


@rewrite.register
def _(s: Cond, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Cond(
        cond=sigma.apply(s.cond),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        else_body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.else_body),
    )


# ---------------------------------------------------------------------------
# simplify — ctx-driven Expr simplification, threading axis ranges
# ---------------------------------------------------------------------------


@singledispatch
def simplify(stmt: Stmt, ctx: SimplifyCtx) -> Stmt:
    # Default: no Expr fields to simplify (Assign / Accum / Init / Combine).
    return stmt


@simplify.register
def _(s: Load, ctx: SimplifyCtx) -> Stmt:
    return Load(names=s.names, input=s.input, index=tuple(e.simplify(ctx) for e in s.index), dtype=s.dtype)


@simplify.register
def _(s: Write, ctx: SimplifyCtx) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(e.simplify(ctx) for e in s.index),
        values=s.values,
        value_dtype=s.value_dtype,
        atomic=s.atomic,
    )


@simplify.register
def _(s: Select, ctx: SimplifyCtx) -> Stmt:
    return Select(name=s.name, branches=tuple(SelectBranch(b.value, b.select.simplify(ctx)) for b in s.branches))


@simplify.register
def _(s: Loop, ctx: SimplifyCtx) -> Stmt:
    inner = ctx.extend(s.axis.name, Interval(0, s.axis.extent.as_static() - 1)) if s.axis.extent.is_static else ctx
    return Loop(axis=s.axis, body=tuple(simplify(c, inner) for c in s.body), unroll=s.unroll)


@simplify.register
def _(s: StridedLoop, ctx: SimplifyCtx) -> Stmt:
    inner = ctx.extend(s.axis.name, Interval(0, s.axis.extent.as_static() - 1)) if s.axis.extent.is_static else ctx
    step = s.step.simplify(ctx) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=s.axis,
        start=s.start.simplify(ctx),
        step=step,
        body=tuple(simplify(c, inner) for c in s.body),
        unroll=s.unroll,
    )


@simplify.register
def _(s: Cond, ctx: SimplifyCtx) -> Stmt:
    return Cond(
        cond=s.cond.simplify(ctx),
        body=tuple(simplify(c, ctx) for c in s.body),
        else_body=tuple(simplify(c, ctx) for c in s.else_body),
    )


# Tile-IR Stmt registrations (Stage / AsyncWait / Combine) live in
# ``deplodock.compiler.ir.tile.passes`` — that module is imported from the
# bottom of ``tile/ir.py`` so loading any Tile-IR symbol auto-registers the
# handlers without a circular import.
