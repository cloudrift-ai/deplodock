"""Consumer-driven recursive splicing of two adjacent ``LoopOp``s.

Fuses a producer/consumer pair by walking the consumer body and, at every
``Load`` that targets the producer's output, reverse-reconstructing the
producer expression that computed that value. Producer accumulators are
hoisted lazily to the consumer scope where they belong (the scope whose
enclosing axes match σ(enclosing producer axes)).

Algorithm outline:

- ``walk`` descends the consumer body. It emits rewritten stmts and returns
  a ``bubble`` of accumulators that still need to be materialized at an
  outer scope.
- At a consumer Load of the producer (and at each accumulator materialization),
  ``emit`` walks producer SSA from a target name back to its Loads via
  DFS and emits each stmt on the unwind. The result lands in forward
  dependency order, freshened and σ-applied. Accum references terminate
  the walk as ``PendingAccum`` entries.
- ``emit_accum_loops`` materializes a pending accumulator's reduce Loop
  at the current scope by calling ``emit`` on the Accum's value and
  appending a final ``Accum`` stmt that folds into the canonical binding.
  Scope-keyed: an accumulator needed at distinct consumer scopes gets
  separate emissions (SDPA's QK^T reduce inside softmax-max vs inside
  the output-K free loop).
- Consumer Loads whose source is not the producer get their source index
  bumped inline in ``walk`` (producer's inputs are prepended to the merged
  kernel's input list).

Sibling reduce axis unification is handled by
``ir/loop/normalize.py::unify_sibling_reduce_axes`` in
``LoopOp.__post_init__``, so the splicer doesn't need to track it here.

Returns ``None`` when the pattern isn't supported: σ fails, accumulator
scope can't be reached, or ``LoopOp`` validation fails on the merged body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.expr import Expr, Literal, Var, substitute
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Select,
    SelectBranch,
    Stmt,
    Write,
)
from deplodock.compiler.ir.tensor_ir import ElementwiseOp


class _NotSupported(Exception):
    """Pattern not handled — caller converts to ``None`` return."""


# ---------------------------------------------------------------------------
# Per-accumulator data extracted on demand from the producer
# ---------------------------------------------------------------------------


@dataclass
class _AccumDecl:
    """One producer accumulator, scoped enough for materialization.

    ``enclosing_axes`` is the free-axis chain outer to the reduce Loop; an
    accumulator can only be materialized at a consumer scope whose enclosing
    axes σ-match this tuple. ``accum_value`` is the SSA name whose value
    gets folded into the accumulator; ``accum_op`` is the fold op."""

    name: str
    reduce_axis: Axis
    enclosing_axes: tuple[Axis, ...]
    accum_value: str
    accum_op: ElementwiseOp


@dataclass
class _PendingAccum:
    """A producer accumulator whose reduce Loop still needs to be emitted.

    Created at each ``_emit`` site that visits an Accum. The
    ``required_c_axes`` tuple is σ(enclosing_axes) — the consumer axis
    names that must be in scope when we emit the Loop.
    """

    decl: _AccumDecl
    sigma: dict[str, Expr]
    bound_name: str
    required_c_axes: tuple[str, ...]


@dataclass
class _Ctx:
    """Per-splice state: one instance lives for the entire recursive walk.

    Holds the splice's invariants (producer, consumer, target source) plus
    the naming and materialization bookkeeping that must persist across
    every call in the recursion."""

    producer: LoopOp
    consumer: LoopOp
    source: int
    producer_inputs: list[str]
    consumer_inputs: list[str]
    used_names: set[str] = field(default_factory=set)
    # Scope-keyed: (producer_accum_name, required_consumer_axes) → consumer-side binding.
    # Same accum at different scopes gets different bindings because its value
    # varies per-outer-iteration (e.g. SDPA's QK^T reduce inside softmax's K loop
    # vs inside the output free loop).
    accum_name: dict[tuple[str, tuple[str, ...]], str] = field(default_factory=dict)
    consumer_source_remap: dict[int, int] = field(default_factory=dict)
    materialized: set[tuple[str, tuple[str, ...]]] = field(default_factory=set)

    def fresh(self, hint: str) -> str:
        name = hint
        i = 1
        while name in self.used_names:
            name = f"{hint}_s{i}"
            i += 1
        self.used_names.add(name)
        return name

    def name_for_accum(self, prod_name: str, scope: tuple[str, ...]) -> str:
        key = (prod_name, scope)
        existing = self.accum_name.get(key)
        if existing is not None:
            return existing
        new = self.fresh(prod_name)
        self.accum_name[key] = new
        return new


@dataclass(frozen=True)
class _Scope:
    """Per-call recursion state: the consumer axes enclosing the current
    walk position. Every recursive step into a ``Loop`` creates a new
    ``_Scope`` via :meth:`nest`; the pre-nest value stays alive in the
    caller's frame and is what we return to after the child call."""

    enclosing: tuple[Axis, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.enclosing)

    def nest(self, axis: Axis) -> _Scope:
        return _Scope(enclosing=self.enclosing + (axis,))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def splice_loop_ops(
    producer: LoopOp,
    consumer: LoopOp,
    source: int,
    *,
    producer_inputs: list[str] | None = None,
    consumer_inputs: list[str] | None = None,
) -> LoopOp | None:
    """Splice ``producer``'s body into every ``consumer`` Load that targets
    ``source``. Returns ``None`` when the pattern isn't supported."""
    try:
        p_inputs = list(producer_inputs or [])
        c_inputs = list(consumer_inputs or [])
        ctx = _Ctx(
            producer=producer,
            consumer=consumer,
            source=source,
            producer_inputs=p_inputs,
            consumer_inputs=c_inputs,
            used_names=_collect_names(producer) | _collect_names(consumer),
            consumer_source_remap=_build_consumer_source_remap(producer, consumer, source),
        )
        new_body, bubble = _walk(consumer.body, ctx, _Scope())
        if bubble:
            return None
        # Sibling reduce-axis unification runs in LoopOp.__post_init__ via
        # ir/loop/normalize.py::unify_sibling_reduce_axes.
        return LoopOp(body=new_body)
    except _NotSupported:
        return None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Recursion core
# ---------------------------------------------------------------------------


def _walk(
    c_stmts: tuple[Stmt, ...],
    ctx: _Ctx,
    scope: _Scope,
) -> tuple[tuple[Stmt, ...], list[_PendingAccum]]:
    """Walk consumer stmts at the current scope. Return (new_body, bubble).

    ``bubble`` lists accumulators that must be materialized at an outer
    scope (caller is responsible for flushing them at its level)."""
    out: list[Stmt] = []
    bubble: list[_PendingAccum] = []

    for s in c_stmts:
        if isinstance(s, Loop):
            inner_out, inner_bubble = _walk(s.body, ctx, scope.nest(s.axis))
            here, up = _partition_pendings(inner_bubble, scope)
            out.extend(_emit_accum_loops(here, ctx, scope))
            out.append(Loop(axis=s.axis, body=inner_out))
            bubble.extend(up)

        elif isinstance(s, Load) and s.source == ctx.source:
            sigma = _solve_sigma(_producer_write(ctx).index, s.index, _producer_axis_names(ctx))
            if sigma is None:
                raise _NotSupported
            prod_stmts, new_pending, final_name = _emit(_producer_write(ctx).value, ctx, sigma)
            here, up = _partition_pendings(new_pending, scope)
            out.extend(_emit_accum_loops(here, ctx, scope))
            out.extend(prod_stmts)
            # Alias consumer's load name to producer's final value via a
            # copy Assign. normalize's copy-elim pass will collapse it.
            # The consumer's load.name is already in used_names (consumer
            # SSA names were seeded up front) — that's fine, we're
            # replacing the Load that bound it with this Assign.
            out.append(Assign(name=s.name, op=ElementwiseOp("copy"), args=(final_name,)))
            bubble.extend(up)

        elif isinstance(s, Load):
            # Consumer-origin Load (not the spliced source). Remap source
            # index so it lands after the producer's inputs in the merged
            # kernel's input list.
            new_src = ctx.consumer_source_remap.get(s.source, s.source)
            out.append(Load(name=s.name, source=new_src, index=s.index))

        else:
            out.append(s)

    return tuple(out), bubble


# ---------------------------------------------------------------------------
# Demand-driven emission: produce a target SSA value from the producer
# ---------------------------------------------------------------------------


def _emit(
    target: str,
    ctx: _Ctx,
    sigma: dict[str, Expr],
) -> tuple[list[Stmt], list[_PendingAccum], str]:
    """Emit the producer-side SSA chain needed to compute ``target``.

    Each recursive step works in two beats:

    1. **Requirements** — ask the recursion for the freshened name of each
       SSA value this stmt reads. The call returns once those upstream
       stmts are in ``emitted`` and their names are in ``ssa_rename``.
    2. **Emit on unwind** — with all requirements resolved, freshen this
       stmt's own name, construct the rewritten stmt (σ on Exprs, resolved
       names on SSA refs), and append it to ``emitted``.

    The Accum branch short-circuits both beats: Accums live at an outer
    scope (they're the only SSA values that cross Loop boundaries), so we
    record a ``_PendingAccum`` for later materialization and return the
    canonical consumer-side binding. The reduce Loop body itself is built
    by a separate ``_emit`` call targeting the Accum's folded value.
    """
    defs = _producer_defs(ctx)
    emitted: list[Stmt] = []
    pending: list[_PendingAccum] = []
    ssa_rename: dict[str, str] = {}

    def resolve(n: str) -> str:
        if n in ssa_rename:
            return ssa_rename[n]
        entry = defs.get(n)
        if entry is None:
            return n
        defn, enclosing = entry

        if isinstance(defn, Accum):
            bound = _record_accum_pending(defn, enclosing, sigma, ctx, pending)
            ssa_rename[n] = bound
            return bound

        # Beat 1: resolve each requirement (SSA dep) recursively.
        deps = _ssa_deps(defn)
        resolved = {d: resolve(d) for d in deps}

        # Beat 2: emit on unwind. All deps are now in ssa_rename + emitted.
        new_name = ctx.fresh(defn.name)
        ssa_rename[n] = new_name
        emitted.append(_build_rewritten_stmt(defn, new_name, resolved, sigma))
        return new_name

    return emitted, pending, resolve(target)


def _ssa_deps(defn: Stmt) -> tuple[str, ...]:
    """SSA names the stmt reads — its 'requirements'."""
    if isinstance(defn, Load):
        return ()
    if isinstance(defn, Assign):
        return defn.args
    if isinstance(defn, Select):
        return tuple(b.value for b in defn.branches)
    raise _NotSupported


def _build_rewritten_stmt(
    defn: Stmt,
    new_name: str,
    resolved: dict[str, str],
    sigma: dict[str, Expr],
) -> Stmt:
    """Construct the rewritten form of ``defn``: ``new_name`` as the SSA
    binding, ``sigma`` applied to every Expr, and each SSA reference
    remapped via ``resolved``."""
    if isinstance(defn, Load):
        return Load(
            name=new_name,
            source=defn.source,
            index=tuple(substitute(e, sigma) for e in defn.index),
        )
    if isinstance(defn, Assign):
        return Assign(name=new_name, op=defn.op, args=tuple(resolved[a] for a in defn.args))
    if isinstance(defn, Select):
        return Select(
            name=new_name,
            branches=tuple(SelectBranch(value=resolved[b.value], select=substitute(b.select, sigma)) for b in defn.branches),
        )
    raise _NotSupported


def _record_accum_pending(
    defn: Accum,
    enclosing: tuple[Axis, ...],
    sigma: dict[str, Expr],
    ctx: _Ctx,
    pending: list[_PendingAccum],
) -> str:
    """Stash a ``_PendingAccum`` for later reduce-Loop materialization;
    return the canonical consumer-side binding name."""
    if not enclosing:
        raise _NotSupported
    reduce_axis = enclosing[-1]
    free_axes = enclosing[:-1]
    decl = _AccumDecl(
        name=defn.name,
        reduce_axis=reduce_axis,
        enclosing_axes=free_axes,
        accum_value=defn.value,
        accum_op=defn.op,
    )
    required_c_axes = tuple(_c_axis_name(sigma.get(a.name), ctx) for a in free_axes)
    if any(a is None for a in required_c_axes):
        raise _NotSupported
    required_c_axes_t: tuple[str, ...] = required_c_axes  # type: ignore[assignment]
    bound = ctx.name_for_accum(defn.name, required_c_axes_t)
    pending.append(_PendingAccum(decl=decl, sigma=dict(sigma), bound_name=bound, required_c_axes=required_c_axes_t))
    return bound


# ---------------------------------------------------------------------------
# Accumulator materialization
# ---------------------------------------------------------------------------


def _emit_accum_loops(
    pendings: list[_PendingAccum],
    ctx: _Ctx,
    scope: _Scope,
) -> list[Stmt]:
    """Build reduce Loops for each unique pending accumulator at this scope.

    Dedupes by (decl name, required scope). Sibling-reduce axis unification
    runs later in ``LoopOp.__post_init__`` — here we always pick fresh axis
    names and let the normalize pass collapse collisions."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[_PendingAccum] = []
    for p in pendings:
        key = (p.decl.name, p.required_c_axes)
        if key in seen or key in ctx.materialized:
            continue
        seen.add(key)
        unique.append(p)

    out: list[Stmt] = []
    for p in unique:
        reduce_axis_name = ctx.fresh(p.decl.reduce_axis.name)
        extended_sigma: dict[str, Expr] = dict(p.sigma)
        extended_sigma[p.decl.reduce_axis.name] = Var(reduce_axis_name)
        # Emit the SSA chain that computes the Accum's folded value.
        body_stmts, nested_pending, value_fresh = _emit(p.decl.accum_value, ctx, extended_sigma)
        nested_here, nested_up = _partition_pendings(nested_pending, scope)
        if nested_up:
            raise _NotSupported
        out.extend(_emit_accum_loops(nested_here, ctx, scope))
        # Append the Accum stmt that folds value_fresh into the canonical
        # consumer-side binding name.
        body_stmts.append(Accum(name=p.bound_name, value=value_fresh, op=p.decl.accum_op))
        out.append(Loop(axis=Axis(name=reduce_axis_name, extent=p.decl.reduce_axis.extent), body=tuple(body_stmts)))
        ctx.materialized.add((p.decl.name, p.required_c_axes))
    return out


# ---------------------------------------------------------------------------
# Consumer source remap
# ---------------------------------------------------------------------------


def _build_consumer_source_remap(producer: LoopOp, consumer: LoopOp, source: int) -> dict[int, int]:
    """Consumer-origin Load sources shift past producer's inputs. The
    spliced source itself is dropped from the map — those Loads never
    survive the splice. Producer-origin Loads keep their original sources
    because producer inputs come first in the merged kernel."""
    n_prod = producer.num_inputs
    remap: dict[int, int] = {}
    next_src = n_prod
    for i in range(consumer.num_inputs):
        if i == source:
            continue
        remap[i] = next_src
        next_src += 1
    return remap


# ---------------------------------------------------------------------------
# σ solver
# ---------------------------------------------------------------------------


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> dict[str, Expr] | None:
    """Solve the per-dim pairing writer[k] == reader[k]. Supported writer
    forms: ``Var(a)`` (``a`` in producer_axes) → bind a→reader[k];
    ``Literal(c)`` → no binding. Anything else → None."""
    if len(writer) != len(reader):
        return None
    sigma: dict[str, Expr] = {}
    for w, r in zip(writer, reader, strict=True):
        if isinstance(w, Literal):
            continue
        if isinstance(w, Var) and w.name in producer_axes:
            existing = sigma.get(w.name)
            if existing is not None and existing != r:
                return None
            sigma[w.name] = r
            continue
        return None
    return sigma


# ---------------------------------------------------------------------------
# Producer inspection helpers — on-demand walks, no precomputed analysis
# ---------------------------------------------------------------------------


def _producer_write(ctx: _Ctx) -> Write:
    for s in _iter_all(ctx.producer.body):
        if isinstance(s, Write) and s.output == 0:
            return s
    raise _NotSupported


def _producer_axis_names(ctx: _Ctx) -> set[str]:
    return {a.name for a in ctx.producer.axes}


def _producer_defs(ctx: _Ctx) -> dict[str, tuple[Stmt, tuple[Axis, ...]]]:
    """Collect producer SSA definitions with their enclosing Loop axes.

    Returns ``{ssa_name: (defining_stmt, enclosing_axes)}``. The enclosing
    chain lets ``_emit`` classify an Accum's scope — its last axis is the
    reduce axis; prior entries are the free axes outer to the reduce Loop.
    Producer SSA names are unique across the tree (enforced by LoopOp
    validator), so a flat dict is sound."""
    defs: dict[str, tuple[Stmt, tuple[Axis, ...]]] = {}

    def walk(stmts: tuple[Stmt, ...], enclosing: tuple[Axis, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                walk(s.body, enclosing + (s.axis,))
            elif isinstance(s, (Load, Assign, Select, Accum)):
                defs[s.name] = (s, enclosing)

    walk(ctx.producer.body, ())
    return defs


def _iter_all(stmts: tuple[Stmt, ...]):
    """Iterate every stmt in the tree (depth-first, no scope tracking)."""
    for s in stmts:
        yield s
        if isinstance(s, Loop):
            yield from _iter_all(s.body)


# ---------------------------------------------------------------------------
# Name collection
# ---------------------------------------------------------------------------


def _collect_names(op: LoopOp) -> set[str]:
    """All SSA names plus all axis names in ``op``."""
    names: set[str] = set()
    for s in _iter_all(op.body):
        if isinstance(s, (Load, Assign, Select, Accum)):
            names.add(s.name)
        if isinstance(s, Loop):
            names.add(s.axis.name)
    return names


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _partition_pendings(
    pendings: list[_PendingAccum],
    scope: _Scope,
) -> tuple[list[_PendingAccum], list[_PendingAccum]]:
    """Split pendings into (flush-here, bubble-up).

    - ``required_c_axes == scope.names`` → flush here.
    - ``required_c_axes`` is a strict prefix of ``scope.names`` → bubble up.
    - otherwise (longer, or non-prefix mismatch) → NotSupported.
    """
    names = scope.names
    here: list[_PendingAccum] = []
    up: list[_PendingAccum] = []
    for p in pendings:
        if p.required_c_axes == names:
            here.append(p)
        elif _is_strict_prefix(p.required_c_axes, names):
            up.append(p)
        else:
            raise _NotSupported
    return here, up


def _is_strict_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return len(a) < len(b) and b[: len(a)] == a


def _c_axis_name(r: Expr | None, ctx: _Ctx) -> str | None:
    """Extract a consumer axis name from a σ value. Only bare ``Var``s are
    supported (affine binding forms would need more work)."""
    if r is None:
        return None
    if isinstance(r, Var):
        return r.name
    return None
