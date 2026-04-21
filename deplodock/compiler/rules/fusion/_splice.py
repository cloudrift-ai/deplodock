"""Worklist-driven splicer for adjacent ``LoopOp``s.

Builds the merged body one statement at a time. Seed: every consumer
``Write``. Each iteration pops one pending dep from ``LoopBuilder`` and
emits its def, queueing that def's own deps in turn. Three dep shapes:

- **Consumer Load from the producer** — emit a copy alias at the demand
  scope; queue the producer's ``Write.value`` under σ = solve(writer,
  reader). The producer's expression chain lands piecemeal over
  subsequent iterations.
- **Accum (producer or consumer)** — freshen the reduce axis, place a
  new ``Loop(reduce_axis, Accum(...))`` at ``_accum_enclosure`` of the
  demand scope, and queue the Accum's ``value``. Scope-keyed dedup:
  the same Accum demanded at different ``required_c_axes`` gets two
  separate materializations (SDPA's QK^T at softmax-max vs output).
- **Plain Assign / Select / Load(non-producer source)** — ``rewrite``
  the original stmt with fresh names and σ-substituted Exprs; insert at
  the stmt's natural scope (consumer: σ-remapped enclosing, which for
  consumer origin is usually identity; producer: σ-remapped enclosing).

``LoopBuilder.insert`` is pure tree-splicing: descend the body along
the enclosure path, creating ``Loop`` nodes if missing, prepend at the
leaf. Always-prepend yields defined-before-use ordering naturally,
because the worklist resolves deps in reverse-topological order —
consumers demand before producers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Literal, Var, render, substitute
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopMeta,
    LoopOp,
    Scope,
    Select,
    Stmt,
    Write,
)
from deplodock.compiler.ir.tensor_ir import ElementwiseOp


class _NotSupported(Exception):
    """Pattern not handled — caller converts to ``None`` return."""


# ---------------------------------------------------------------------------
# Demand — one pending worklist entry
# ---------------------------------------------------------------------------


@dataclass
class _Demand:
    """A pending dep in the worklist.

    ``bound_as`` is the fresh name the dep's def will bind in the merged
    body — allocated at queue time so callers can reference it without
    waiting for resolution.
    """

    name: str
    origin: str  # "producer" | "consumer"
    sigma: dict[str, Expr]
    demand_scope: Scope
    bound_as: str


def _sigma_key(sigma: dict[str, Expr]) -> tuple[tuple[str, str], ...]:
    """Hashable key for σ — (k, render(v)) pairs sorted by k."""
    return tuple(sorted((k, render(v)) for k, v in sigma.items()))


# ---------------------------------------------------------------------------
# LoopBuilder — the merged body + the worklist
# ---------------------------------------------------------------------------


class LoopBuilder:
    """Accumulates the merged body by inserting stmts one at a time.

    ``insert`` is pure tree-splicing: given an enclosure path, descend
    into the body creating ``Loop`` nodes as needed, and prepend the
    stmt at the leaf. Worklist dep-resolution is reverse-topological —
    producers demanded after consumers — so prepend always places
    definitions before uses.
    """

    def __init__(self, used_names: set[str]) -> None:
        self._body: tuple[Stmt, ...] = ()
        self._used: set[str] = set(used_names)
        self._pending: deque[_Demand] = deque()
        # Key: (origin, name, σ-key). Plain deps dedupe by σ — two refs to
        # the same producer name under different σs emit twice.
        self._plain_binding: dict[tuple[str, str, tuple[tuple[str, str], ...]], str] = {}
        # Key: (origin, name, required_c_axes). Accums dedupe by the
        # σ-mapped enclosing axes — same accum at different scopes emits
        # twice (SDPA's QK^T at softmax-max vs softmax-output).
        self._accum_binding: dict[tuple[str, str, tuple[str, ...]], str] = {}

    def fresh(self, hint: str) -> str:
        if hint not in self._used:
            self._used.add(hint)
            return hint
        i = 1
        while f"{hint}_s{i}" in self._used:
            i += 1
        name = f"{hint}_s{i}"
        self._used.add(name)
        return name

    def queue(self, demand: _Demand) -> None:
        self._pending.append(demand)

    def take(self) -> _Demand | None:
        return self._pending.popleft() if self._pending else None

    def has_deps(self) -> bool:
        return bool(self._pending)

    def seen_plain(self, origin: str, name: str, sigma: dict[str, Expr]) -> str | None:
        return self._plain_binding.get((origin, name, _sigma_key(sigma)))

    def bind_plain(self, origin: str, name: str, sigma: dict[str, Expr], bound_as: str) -> None:
        self._plain_binding[(origin, name, _sigma_key(sigma))] = bound_as

    def seen_accum(self, origin: str, name: str, required: tuple[str, ...]) -> str | None:
        return self._accum_binding.get((origin, name, required))

    def bind_accum(self, origin: str, name: str, required: tuple[str, ...], bound_as: str) -> None:
        self._accum_binding[(origin, name, required)] = bound_as

    def insert(self, stmt: Stmt, enclosure: Scope) -> None:
        self._body = _prepend_at(self._body, enclosure.enclosing, stmt)

    def finish(self) -> tuple[Stmt, ...]:
        return self._body


def _prepend_at(body: tuple[Stmt, ...], path: tuple[Axis, ...], stmt: Stmt) -> tuple[Stmt, ...]:
    """Descend ``body`` following ``path``; create missing ``Loop`` nodes;
    prepend ``stmt`` at the leaf."""
    if not path:
        return (stmt,) + tuple(body)
    head, rest = path[0], path[1:]
    for i, s in enumerate(body):
        if isinstance(s, Loop) and s.axis == head:
            new_inner = _prepend_at(s.body, rest, stmt)
            return tuple(body[:i]) + (Loop(axis=head, body=new_inner),) + tuple(body[i + 1 :])
    return (Loop(axis=head, body=_prepend_at((), rest, stmt)),) + tuple(body)


# ---------------------------------------------------------------------------
# Context — per-splice frozen analysis
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    loops: dict[str, LoopMeta]
    source: int

    @classmethod
    def build(cls, producer: LoopOp, consumer: LoopOp, source: int) -> _Ctx:
        return cls(
            loops={"producer": producer.analyze(), "consumer": consumer.analyze()},
            source=source,
        )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def splice_loop_ops(
    producer: LoopOp,
    consumer: LoopOp,
    source: int,
) -> LoopOp | None:
    """Splice ``producer``'s expression into every consumer ``Load`` that
    targets ``source``. Returns ``None`` when the pattern isn't supported."""
    # Refuse when the producer has an ``Accum`` nested inside another
    # ``Accum``'s reduce Loop. Splicing such a producer would produce a
    # merged body where the inner Accum's reduce depends on a fresh axis
    # introduced by the outer Accum's materialization — valid in scalar-
    # loop semantics but the whole-tensor numpy backend (``execute_loop_op``)
    # reduces every Accum over *all* reduce axes at once, collapsing
    # dimensions that were meant to vary. The old splicer hit this same
    # limitation via a different code path (``required ⊆ names`` check on
    # a bubbled sub-ref); match it here by rejecting the pattern up front.
    if _has_nested_accums(producer.body):
        return None
    try:
        ctx = _Ctx.build(producer, consumer, source)
        b = LoopBuilder(used_names=_collect_names(producer) | _collect_names(consumer))
        _seed(ctx, b)
        while b.has_deps():
            d = b.take()
            _resolve(d, ctx, b)
        return LoopOp(body=b.finish())
    except _NotSupported:
        return None
    except ValueError:
        return None


def _has_nested_accums(stmts: tuple[Stmt, ...], inside_accum_loop: bool = False) -> bool:
    """True if any ``Accum``'s reduce Loop sits inside another ``Accum``'s
    reduce Loop. A Loop's immediate body Accum is "this Loop's" — only
    deeper Loops with Accums count as nested."""
    for s in stmts:
        if isinstance(s, Loop):
            wraps_accum = any(isinstance(x, Accum) for x in s.body)
            if inside_accum_loop and wraps_accum:
                return True
            if _has_nested_accums(s.body, inside_accum_loop or wraps_accum):
                return True
    return False


# ---------------------------------------------------------------------------
# Seed: every consumer Write, with its value queued
# ---------------------------------------------------------------------------


def _seed(ctx: _Ctx, b: LoopBuilder) -> None:
    for w, scope in ctx.loops["consumer"].writes:
        v_bound = _ensure_dep(w.value, "consumer", {}, scope, ctx, b)
        b.insert(Write(output=w.output, index=w.index, value=v_bound), scope)


# ---------------------------------------------------------------------------
# Dep binding: look up or queue
# ---------------------------------------------------------------------------


def _ensure_dep(
    name: str,
    origin: str,
    sigma: dict[str, Expr],
    ref_scope: Scope,
    ctx: _Ctx,
    b: LoopBuilder,
) -> str:
    """Return the merged-body name bound to ``(origin, name, σ)``. Queue a
    new demand if this is the first reference.

    ``ref_scope`` is where the reference sits in the merged body; used only
    to compute an Accum's enclosure. Plain deps take their scope from the
    def's own enclosing (σ-remapped).
    """
    meta = ctx.loops[origin]
    def_stmt = meta.defs.get(name)
    if def_stmt is None:
        raise _NotSupported

    if isinstance(def_stmt, Accum):
        full_enc = meta.scopes[name]
        required = tuple(_remap_axis_name(a, sigma) for a in full_enc.enclosing[:-1])
        existing = b.seen_accum(origin, name, required)
        if existing is not None:
            return existing
        bound = b.fresh(name)
        b.bind_accum(origin, name, required, bound)
        accum_enc = _scope_for_axes(ref_scope, required)
        b.queue(_Demand(name=name, origin=origin, sigma=dict(sigma), demand_scope=accum_enc, bound_as=bound))
        return bound

    existing = b.seen_plain(origin, name, sigma)
    if existing is not None:
        return existing
    bound = b.fresh(name)
    b.bind_plain(origin, name, sigma, bound)
    required_axes = tuple(_remap_axis_name(a, sigma) for a in meta.scopes[name].enclosing)
    # Producer plain stmts: require the σ-mapped enclosing to axis-set-equal
    # ``ref_scope``. With whole-tensor backend semantics Accums reduce over
    # every reduce axis at once, so emitting a producer stmt at a scope that
    # has extra reduce axes would collapse dimensions that were meant to
    # vary — e.g. softmax's sum accum would fold matmul@V's k along with
    # softmax's k. Consumer stmts are fine at a shorter prefix (the
    # original outer-scope binding seen from a nested reference).
    if origin == "producer":
        if set(required_axes) != set(a.name for a in ref_scope.enclosing):
            raise _NotSupported
        def_scope = ref_scope
    else:
        def_scope = _scope_for_axes(ref_scope, required_axes)
    b.queue(_Demand(name=name, origin=origin, sigma=dict(sigma), demand_scope=def_scope, bound_as=bound))
    return bound


# ---------------------------------------------------------------------------
# Resolution dispatch
# ---------------------------------------------------------------------------


def _resolve(d: _Demand, ctx: _Ctx, b: LoopBuilder) -> None:
    stmt = ctx.loops[d.origin].defs[d.name]

    if isinstance(stmt, Load):
        if d.origin == "consumer" and stmt.source == ctx.source:
            _resolve_producer_load(stmt, d, ctx, b)
        else:
            _resolve_plain_load(stmt, d, ctx, b)
    elif isinstance(stmt, Accum):
        _resolve_accum(stmt, d, ctx, b)
    elif isinstance(stmt, (Assign, Select)):
        _resolve_plain(stmt, d, ctx, b)
    else:
        raise _NotSupported


def _resolve_plain(stmt: Stmt, d: _Demand, ctx: _Ctx, b: LoopBuilder) -> None:
    """Generic Assign / Select emission — rewrite the stmt with fresh args
    and σ-substituted Exprs, insert at ``d.demand_scope``."""
    resolved = {arg: _ensure_dep(arg, d.origin, d.sigma, d.demand_scope, ctx, b) for arg in stmt.deps()}
    b.insert(stmt.rewrite(d.bound_as, resolved, d.sigma), d.demand_scope)


def _resolve_plain_load(stmt: Load, d: _Demand, ctx: _Ctx, b: LoopBuilder) -> None:
    """A Load that doesn't target the producer — remap source, σ-sub index.
    Producer inputs occupy [0, n_prod) in the merged kernel; consumer's
    surviving inputs shift to [n_prod, ...), skipping the spliced slot."""
    if d.origin == "consumer":
        n_prod = ctx.loops["producer"].op.num_inputs
        s = stmt.source
        new_src = n_prod + (s if s < ctx.source else s - 1)
    else:
        new_src = stmt.source
    new_index = tuple(substitute(e, d.sigma) for e in stmt.index)
    b.insert(Load(name=d.bound_as, source=new_src, index=new_index), d.demand_scope)


def _resolve_producer_load(stmt: Load, d: _Demand, ctx: _Ctx, b: LoopBuilder) -> None:
    """A consumer Load that targets the producer — emit a copy alias and
    queue the producer's ``Write.value`` under the solved σ. The producer's
    expression chain reconstructs piecemeal in subsequent iterations."""
    producer = ctx.loops["producer"]
    prod_write = next((w for w, _ in producer.writes if w.output == 0), None)
    if prod_write is None:
        raise _NotSupported
    effective_index = tuple(substitute(e, d.sigma) for e in stmt.index)
    sigma = _solve_sigma(prod_write.index, effective_index, {a.name for a in producer.op.axes})
    if sigma is None:
        raise _NotSupported
    v_bound = _ensure_dep(prod_write.value, "producer", sigma, d.demand_scope, ctx, b)
    b.insert(Assign(name=d.bound_as, op=ElementwiseOp("copy"), args=(v_bound,)), d.demand_scope)


def _resolve_accum(stmt: Accum, d: _Demand, ctx: _Ctx, b: LoopBuilder) -> None:
    """Emit ``Loop(fresh_reduce_axis, [Accum(bound, value_bound, op)])`` at
    ``d.demand_scope``. The Accum's value is queued under σ extended with
    the fresh reduce-axis binding."""
    full_enc = ctx.loops[d.origin].scopes[stmt.name]
    orig_axis = full_enc.enclosing[-1]
    fresh_name = b.fresh(orig_axis.name)
    reduce_axis = Axis(name=fresh_name, extent=orig_axis.extent)
    inner_sigma = dict(d.sigma)
    inner_sigma[orig_axis.name] = Var(fresh_name)
    inner_scope = Scope(enclosing=d.demand_scope.enclosing + (reduce_axis,))
    value_bound = _ensure_dep(stmt.value, d.origin, inner_sigma, inner_scope, ctx, b)
    b.insert(Accum(name=d.bound_as, value=value_bound, op=stmt.op), inner_scope)


# ---------------------------------------------------------------------------
# σ and scope helpers
# ---------------------------------------------------------------------------


def _scope_for_axes(ref_scope: Scope, required: tuple[str, ...]) -> Scope:
    """Shortest prefix of ``ref_scope`` whose axis set contains ``required``.

    Used two ways:
    - For Accums: places the reduce ``Loop`` at the innermost consumer
      scope where all σ-mapped producer enclosing axes are visible (today's
      behavior — further hoisting is left to later passes).
    - For plain producer stmts: picks the emit scope from the consumer's
      nest, tolerating producer's free-axis order differing from consumer's.
      A matmul producer ``(a0, a1, a2)`` σ-maps to consumer ``(a0, a2, a1)``
      (shuffled), but the consumer's scope ``(a0, a1, a2)`` covers the same
      axis set; emitting at the consumer's nest avoids a duplicate Loop tree.
    """
    names = tuple(a.name for a in ref_scope.enclosing)
    remaining = set(required)
    k = 0
    while remaining and k < len(names):
        remaining.discard(names[k])
        k += 1
    if remaining:
        raise _NotSupported
    return Scope(enclosing=ref_scope.enclosing[:k])


def _remap_axis_name(axis: Axis, sigma: dict[str, Expr]) -> str:
    target = sigma.get(axis.name)
    if target is None:
        return axis.name
    if isinstance(target, Var):
        return target.name
    raise _NotSupported


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> dict[str, Expr] | None:
    """Solve per-dim pairing ``writer[k] == reader[k]``. Supported writer
    forms: ``Var(a)`` (``a`` in ``producer_axes``) → bind ``a → reader[k]``;
    ``Literal(c)`` → no binding. Anything else → ``None``."""
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
# Name collection
# ---------------------------------------------------------------------------


def _collect_names(op: LoopOp) -> set[str]:
    """All SSA names plus all axis names used anywhere in ``op``."""
    names: set[str] = set()

    def walk(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                names.add(s.axis.name)
                walk(s.body)
            elif isinstance(s, (Load, Assign, Select, Accum)):
                names.add(s.name)

    walk(op.body)
    return names
