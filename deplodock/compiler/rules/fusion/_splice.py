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

from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Select,
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
class _AccumRef:
    """A producer accumulator referenced by a consumer splice site.

    Bundles everything needed to later materialize the reduce Loop: the
    producer Accum's identity, its enclosing chain (reduce axis is the
    last element), the SSA name of the folded value, the fold op, and the
    σ active at the reference site. ``required_c_axes`` is the consumer
    scope that must be in scope when we finally emit the Loop."""

    name: str
    enclosing: tuple[Axis, ...]
    value: str
    op: ElementwiseOp
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
    """An axis chain used two ways:

    - As consumer walk position: every recursive step into a ``Loop``
      produces a new ``_Scope`` via :meth:`nest`; the pre-nest value
      stays alive in the caller and is what we return to.
    - As producer enclosing, σ-remapped into consumer namespace via
      :meth:`remap`, so it can be compared against a consumer scope.
    """

    enclosing: tuple[Axis, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.enclosing)

    def nest(self, axis: Axis) -> _Scope:
        return _Scope(enclosing=self.enclosing + (axis,))

    def remap(self, sigma: dict[str, Expr]) -> _Scope:
        """σ-remap each axis name"""
        out: list[Axis] = []
        for a in self.enclosing:
            r = sigma.get(a.name)
            out.append(Axis(name=r.name, extent=a.extent))
        return _Scope(enclosing=tuple(out))

    def __eq__(self, other):
        return frozenset(a.name for a in self.enclosing) == frozenset(a.name for a in other.enclosing)


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
) -> tuple[tuple[Stmt, ...], list[_AccumRef]]:
    """Walk consumer stmts at the current scope.

    Shape at every step: rewrite a stmt → for each accum reference it
    produced, ask ``_emit_accum`` to materialize at this scope or bubble
    outward → emit the materializations before the rewritten stmt →
    bubble the rest upward.

    Rewrite cases:
    - ``Loop``: recurse into the body at a nested scope; its inner bubble
      becomes this stmt's accum refs.
    - ``Load`` targeting the spliced source: replace with the producer
      expression that computes the value, aliased via copy (normalize's
      copy-elim collapses the alias).
    - ``Load`` from any other source: remap the source past the producer's
      inputs.
    - anything else: keep as-is.
    """

    def visit(stmts: tuple[Stmt, ...], scope: _Scope) -> tuple[tuple[Stmt, ...], list[_AccumRef]]:
        out: list[Stmt] = []
        bubble: list[_AccumRef] = []
        for s in stmts:
            if isinstance(s, Loop):
                inner_out, new_refs = visit(s.body, scope.nest(s.axis))
                new_stmts: list[Stmt] = [Loop(axis=s.axis, body=inner_out)]
            elif isinstance(s, Load) and s.source == ctx.source:
                sigma = _solve_sigma(_producer_write(ctx).index, s.index, _producer_axis_names(ctx))
                if sigma is None:
                    raise _NotSupported
                prod_stmts, new_refs, final_name = _emit(_producer_write(ctx).value, ctx, sigma, scope)
                new_stmts = [*prod_stmts, Assign(name=s.name, op=ElementwiseOp("copy"), args=(final_name,))]
            elif isinstance(s, Load):
                new_src = ctx.consumer_source_remap.get(s.source, s.source)
                new_stmts = [Load(name=s.name, source=new_src, index=s.index)]
                new_refs = []
            else:
                new_stmts = [s]
                new_refs = []
            for ref in new_refs:
                ref_stmts, ref_bubble, _ = _emit(ref.name, ctx, ref.sigma, scope)
                out.extend(ref_stmts)
                bubble.extend(ref_bubble)
            out.extend(new_stmts)
        return tuple(out), bubble

    return visit(c_stmts, scope)


# ---------------------------------------------------------------------------
# Demand-driven emission: produce a target SSA value from the producer
# ---------------------------------------------------------------------------


class _Bubble(Exception):
    """Early exit from ``_emit``'s walk: target accum bubbles to outer scope."""

    def __init__(self, ref: _AccumRef) -> None:
        self.ref = ref


class _Skip(Exception):
    """Early exit from ``_emit``'s walk: target accum was already materialized."""

    def __init__(self, bound: str) -> None:
        self.bound = bound


def _emit(
    target: str,
    ctx: _Ctx,
    sigma: dict[str, Expr],
    scope: _Scope,
) -> tuple[list[Stmt], list[_AccumRef], str]:
    """Emit the consumer-side stmts that bind ``target``.

    Walks the producer tree in reverse program order carrying a live set.
    When a visited stmt's name is in the live set, freshen it, cache the
    rename, and add its SSA deps to the live set — so defs earlier in
    program order (visited later in reverse) get pulled in transitively.
    Producer Accum defs encountered via deps are packaged as ``_AccumRef``
    for the caller to bubble back into ``_emit`` at the right outer scope.
    ``scope`` is the consumer scope these stmts will land in; each emitted
    stmt's producer enclosing must σ-resolve to exactly that (effective)
    scope.

    Accum-as-target: we don't know up front whether ``target`` names an
    Accum — the walk discovers it. When ``visit`` hits the target Accum,
    it does a partition-by-scope check (bubble to outer, skip if already
    materialized, else materialize). Materializing = freshen reduce axis,
    extend the effective σ with it, nest the effective scope, add the
    Accum's value to ``needed``. The remaining walk pulls in the value's
    SSA chain under the new σ/scope. A postamble wraps that chain in the
    reduce Loop and appends the Accum fold. Nested accums encountered
    inside the body bubble back through ``_emit`` at the outer scope.
    """
    refs: list[_AccumRef] = []
    ssa_rename: dict[str, str] = {}
    needed: set[str] = {target}
    collected: list[tuple[Stmt, str]] = []  # (original_stmt, fresh_name), reverse order
    outer_scope = scope
    eff_sigma: dict[str, Expr] = dict(sigma)
    eff_scope = scope
    wrapper: tuple[Axis, str, ElementwiseOp, str] | None = None  # (fresh_axis, bound, op, value)

    def visit(stmts: tuple[Stmt, ...], producer_scope=_Scope()) -> None:
        nonlocal eff_scope, wrapper
        for s in reversed(stmts):
            if isinstance(s, Loop):
                visit(s.body, producer_scope.nest(s.axis))
                continue
            if not isinstance(s, (Load, Assign, Select, Accum)):
                continue
            if s.name not in needed:
                continue
            needed.discard(s.name)
            if isinstance(s, Accum):
                if s.name == target:
                    enclosing = producer_scope.enclosing
                    if not enclosing:
                        raise _NotSupported
                    required = _Scope(enclosing=enclosing[:-1]).remap(eff_sigma).names
                    bound = ctx.name_for_accum(s.name, required)
                    names = eff_scope.names
                    if not set(required).issubset(set(names)):
                        raise _NotSupported
                    if set(required).issubset(set(names[:-1]) if names else set()):
                        raise _Bubble(_AccumRef(s.name, enclosing, s.value, s.op, dict(eff_sigma), bound, required))
                    key = (s.name, required)
                    if key in ctx.materialized:
                        raise _Skip(bound)
                    ctx.materialized.add(key)
                    fresh = Axis(name=ctx.fresh(enclosing[-1].name), extent=enclosing[-1].extent)
                    eff_sigma[enclosing[-1].name] = Var(fresh.name)
                    eff_scope = eff_scope.nest(fresh)
                    ssa_rename[s.name] = bound
                    wrapper = (fresh, bound, s.op, s.value)
                    needed.add(s.value)
                else:
                    ssa_rename[s.name] = _register_accum_ref(s, producer_scope.enclosing, eff_sigma, ctx, refs)
                continue
            if producer_scope.remap(eff_sigma) != eff_scope:
                raise _NotSupported
            new_name = ctx.fresh(s.name)
            ssa_rename[s.name] = new_name
            collected.append((s, new_name))
            needed.update(s.deps())

    try:
        visit(ctx.producer.body)
    except _Bubble as b:
        return [], [b.ref], b.ref.bound_name
    except _Skip as sk:
        return [], [], sk.bound

    emitted: list[Stmt] = []
    for s, new_name in reversed(collected):
        resolved = {d: ssa_rename.get(d, d) for d in s.deps()}
        emitted.append(s.rewrite(new_name, resolved, eff_sigma))

    if wrapper is None:
        return emitted, refs, ssa_rename.get(target, target)

    reduce_fresh, bound, op, value = wrapper
    value_fresh = ssa_rename.get(value, value)
    out: list[Stmt] = []
    bubble: list[_AccumRef] = []
    for sub in refs:
        sub_stmts, sub_bubble, _ = _emit(sub.name, ctx, sub.sigma, outer_scope)
        out.extend(sub_stmts)
        bubble.extend(sub_bubble)
    if bubble:
        raise _NotSupported
    emitted.append(Accum(name=bound, value=value_fresh, op=op))
    out.append(Loop(axis=reduce_fresh, body=tuple(emitted)))
    return out, [], bound


def _register_accum_ref(
    defn: Accum,
    enclosing: tuple[Axis, ...],
    sigma: dict[str, Expr],
    ctx: _Ctx,
    refs: list[_AccumRef],
) -> str:
    """Record an ``_AccumRef`` for later reduce-Loop materialization;
    return the canonical consumer-side binding name."""
    if not enclosing:
        raise _NotSupported
    required_c_axes = _Scope(enclosing=enclosing[:-1]).remap(sigma).names
    bound = ctx.name_for_accum(defn.name, required_c_axes)
    refs.append(
        _AccumRef(
            name=defn.name,
            enclosing=enclosing,
            value=defn.value,
            op=defn.op,
            sigma=dict(sigma),
            bound_name=bound,
            required_c_axes=required_c_axes,
        )
    )
    return bound


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


