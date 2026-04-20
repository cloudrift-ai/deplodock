"""Consumer-driven recursive splicing of two adjacent ``LoopOp``s.

Fuses a producer/consumer pair by walking the consumer body and, at every
``Load`` that targets the producer's output, reverse-reconstructing the
producer expression that computed that value. Producer accumulators are
hoisted lazily to the consumer scope where they belong (the scope whose
enclosing axes match σ(enclosing producer axes)). Sibling reduce loops that
sweep the same external buffer at the same dim share an axis name so the
post-fusion kernel carries a single reduce axis.

Algorithm outline:

- ``walk`` descends the consumer body. It emits rewritten stmts and returns
  a ``bubble`` of accumulators that still need to be materialized at an
  outer scope.
- At a consumer Load of the producer, ``dfs_build`` walks the producer's
  SSA from ``write.value`` back to its Loads, emitting freshened
  ``Load | Assign | Select`` stmts with σ applied to every Expr.
  References to producer Accums become ``PendingAccum`` records.
- ``emit_accum_loops`` materializes a pending accumulator's reduce Loop
  at the current scope. It runs a mini-dfs over the accum body and checks
  for a sibling reduce Loop in the current-scope output that already
  reads the same external buffer at the same dim — if so, it reuses that
  sibling's axis name so both reduces end up sharing one axis.
- Consumer Loads whose source is not the producer get their source index
  bumped in a final pass (producer's inputs are prepended to the merged
  kernel's input list).

Returns ``None`` when the pattern isn't supported: σ fails, consumer
introduces an axis outside σ's image, accumulator scope can't be reached,
or ``LoopOp`` validation fails on the merged body.
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

    ``body`` is the immediate body of the reduce Loop — the stmts that fold
    into the accumulator (Loads + Assigns + Selects + the Accum itself).
    ``enclosing_axes`` is the free-axis chain outer to the reduce Loop; an
    accumulator can only be materialized at a consumer scope whose enclosing
    axes σ-match this tuple.
    """

    name: str
    reduce_axis: Axis
    enclosing_axes: tuple[Axis, ...]
    body: tuple[Stmt, ...]


@dataclass
class _PendingAccum:
    """A producer accumulator whose reduce Loop still needs to be emitted.

    Created at each ``dfs_build`` site that references an Accum. The
    ``required_c_axes`` tuple is σ(enclosing_axes) — the consumer axis
    names that must be in scope when we emit the Loop.
    """

    decl: _AccumDecl
    sigma: dict[str, Expr]
    bound_name: str
    required_c_axes: tuple[str, ...]


@dataclass
class _Ctx:
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
    merged_inputs: list[str] = field(default_factory=list)
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
        merged_inputs = list(p_inputs) + [c for i, c in enumerate(c_inputs) if i != source]
        ctx = _Ctx(
            producer=producer,
            consumer=consumer,
            source=source,
            producer_inputs=p_inputs,
            consumer_inputs=c_inputs,
            used_names=_collect_names(producer) | _collect_names(consumer),
            consumer_source_remap=_build_consumer_source_remap(producer, consumer, source),
            merged_inputs=merged_inputs,
        )
        new_body, bubble = _walk(consumer.body, ctx, enclosing=())
        if bubble:
            return None
        new_body = _unify_sibling_reduce_axes(new_body, ctx.merged_inputs)
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
    enclosing: tuple[Axis, ...],
) -> tuple[tuple[Stmt, ...], list[_PendingAccum]]:
    """Walk consumer stmts at the current scope. Return (new_body, bubble).

    ``bubble`` lists accumulators that must be materialized at an outer
    scope (caller is responsible for flushing them at its level)."""
    out: list[Stmt] = []
    bubble: list[_PendingAccum] = []
    enclosing_names = tuple(a.name for a in enclosing)

    for s in c_stmts:
        if isinstance(s, Loop):
            inner_out, inner_bubble = _walk(s.body, ctx, enclosing + (s.axis,))
            here, up = _partition_pendings(inner_bubble, enclosing_names)
            out.extend(_emit_accum_loops(here, ctx, enclosing, current_scope_out=out))
            out.append(Loop(axis=s.axis, body=inner_out))
            bubble.extend(up)

        elif isinstance(s, Load) and s.source == ctx.source:
            sigma = _solve_sigma(_producer_write(ctx).index, s.index, _producer_axis_names(ctx))
            if sigma is None:
                raise _NotSupported
            prod_stmts, new_pending, final_name = _dfs_build(_producer_write(ctx).value, sigma, ctx, enclosing_names)
            here, up = _partition_pendings(new_pending, enclosing_names)
            out.extend(_emit_accum_loops(here, ctx, enclosing, current_scope_out=out))
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
# Reverse-reconstruction of producer expression at a Load site
# ---------------------------------------------------------------------------


def _dfs_build(
    root_ssa: str,
    sigma: dict[str, Expr],
    ctx: _Ctx,
    enclosing_names: tuple[str, ...],
) -> tuple[list[Stmt], list[_PendingAccum], str]:
    """Walk producer SSA from ``root_ssa`` back to Loads / Accums, emitting
    freshened stmts in reverse-topological (i.e. definition) order.

    Returns (emitted_stmts, pending_accums, final_name) where final_name is
    the freshened SSA name that holds root_ssa's value at the emission site.
    """
    defs = _producer_defs(ctx)
    emitted: list[Stmt] = []
    pending: list[_PendingAccum] = []
    ssa_rename: dict[str, str] = {}

    def rename(n: str) -> str:
        return ssa_rename.get(n, n)

    visited: set[str] = set()

    def visit(n: str) -> str:
        if n in ssa_rename:
            return ssa_rename[n]
        if n in visited:
            # Cycle (shouldn't happen in SSA) — return tentative name.
            return ssa_rename.get(n, n)
        visited.add(n)

        defn = defs.get(n)
        if defn is None:
            # Not a producer def — an external name (shouldn't arrive here).
            return n

        if isinstance(defn, Accum):
            decl = _find_accum_scope(ctx.producer, n)
            if decl is None:
                raise _NotSupported
            required_c_axes = tuple(_c_axis_name(sigma.get(ax.name), ctx) for ax in decl.enclosing_axes)
            if any(a is None for a in required_c_axes):
                raise _NotSupported
            required_c_axes_t: tuple[str, ...] = required_c_axes  # type: ignore[assignment]
            bound = ctx.name_for_accum(n, required_c_axes_t)
            ssa_rename[n] = bound
            pending.append(
                _PendingAccum(
                    decl=decl,
                    sigma=dict(sigma),
                    bound_name=bound,
                    required_c_axes=required_c_axes_t,
                )
            )
            return bound

        if isinstance(defn, Load):
            # Recurse into any ssa inputs first (Loads don't have any), then
            # emit the Load with sigma-substituted index.
            new_name = ctx.fresh(defn.name)
            ssa_rename[n] = new_name
            emitted.append(
                Load(
                    name=new_name,
                    source=defn.source,
                    index=tuple(substitute(e, sigma) for e in defn.index),
                )
            )
            return new_name

        if isinstance(defn, Assign):
            resolved_args = tuple(visit(a) for a in defn.args)
            new_name = ctx.fresh(defn.name)
            ssa_rename[n] = new_name
            emitted.append(Assign(name=new_name, op=defn.op, args=resolved_args))
            return new_name

        if isinstance(defn, Select):
            resolved_branches = tuple(SelectBranch(value=visit(b.value), select=substitute(b.select, sigma)) for b in defn.branches)
            new_name = ctx.fresh(defn.name)
            ssa_rename[n] = new_name
            emitted.append(Select(name=new_name, branches=resolved_branches))
            return new_name

        raise _NotSupported

    final_name = visit(root_ssa)
    return emitted, pending, final_name


# ---------------------------------------------------------------------------
# Accumulator materialization
# ---------------------------------------------------------------------------


def _emit_accum_loops(
    pendings: list[_PendingAccum],
    ctx: _Ctx,
    enclosing: tuple[Axis, ...],
    current_scope_out: list[Stmt],
) -> list[Stmt]:
    """Build reduce Loops for each unique pending accumulator at this scope.

    Dedupes by decl name. For each pending, checks whether an already-emitted
    sibling Loop at this scope reduces over the same external buffer at the
    same dim — if so, reuses that sibling's reduce axis name.
    """
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
        reduce_axis_name = _find_sibling_reduce_alias(p, current_scope_out + out, ctx)
        if reduce_axis_name is None:
            reduce_axis_name = ctx.fresh(p.decl.reduce_axis.name)
        extended_sigma: dict[str, Expr] = dict(p.sigma)
        extended_sigma[p.decl.reduce_axis.name] = Var(reduce_axis_name)
        body_stmts, nested_pending, _ = _dfs_build_accum_body(p, extended_sigma, ctx)
        # Nested accumulators that match THIS scope are emitted before.
        nested_here, nested_up = _partition_pendings(nested_pending, tuple(a.name for a in enclosing))
        if nested_up:
            raise _NotSupported
        out.extend(_emit_accum_loops(nested_here, ctx, enclosing, current_scope_out=current_scope_out + out))
        out.append(Loop(axis=Axis(name=reduce_axis_name, extent=p.decl.reduce_axis.extent), body=body_stmts))
        ctx.materialized.add((p.decl.name, p.required_c_axes))
    return out


def _dfs_build_accum_body(
    p: _PendingAccum,
    sigma: dict[str, Expr],
    ctx: _Ctx,
) -> tuple[tuple[Stmt, ...], list[_PendingAccum], str]:
    """Materialize a producer accumulator's reduce-body under ``sigma``.

    Rewrites the decl body's Loads and Assigns/Selects with freshened SSA
    names and σ-substituted Exprs. The trailing Accum stmt binds to the
    accumulator's canonical consumer-side name (``ctx.name_for_accum``).
    """
    emitted: list[Stmt] = []
    pending: list[_PendingAccum] = []
    ssa_rename: dict[str, str] = {}

    def rename(n: str) -> str:
        return ssa_rename.get(n, n)

    for s in p.decl.body:
        if isinstance(s, Load):
            new_name = ctx.fresh(s.name)
            ssa_rename[s.name] = new_name
            emitted.append(
                Load(
                    name=new_name,
                    source=s.source,
                    index=tuple(substitute(e, sigma) for e in s.index),
                )
            )
        elif isinstance(s, Assign):
            new_name = ctx.fresh(s.name)
            ssa_rename[s.name] = new_name
            emitted.append(Assign(name=new_name, op=s.op, args=tuple(rename(a) for a in s.args)))
        elif isinstance(s, Select):
            new_name = ctx.fresh(s.name)
            ssa_rename[s.name] = new_name
            emitted.append(
                Select(
                    name=new_name,
                    branches=tuple(SelectBranch(value=rename(b.value), select=substitute(b.select, sigma)) for b in s.branches),
                )
            )
        elif isinstance(s, Accum):
            # Bind the reduce to the canonical consumer-side accum name.
            emitted.append(Accum(name=p.bound_name, value=rename(s.value), op=s.op))
        elif isinstance(s, Loop):
            # Nested reduce/free loop inside the accum body — not yet supported.
            raise _NotSupported
        elif isinstance(s, Write):
            raise _NotSupported
        else:
            raise _NotSupported
    return tuple(emitted), pending, p.bound_name


# ---------------------------------------------------------------------------
# Sibling-reduce alias detection
# ---------------------------------------------------------------------------


def _find_sibling_reduce_alias(
    p: _PendingAccum,
    emitted_so_far: list[Stmt],
    ctx: _Ctx,
) -> str | None:
    """Scan already-emitted siblings at the current scope for a reduce Loop
    that reads the same external buffer at the same dim that p's reduce
    axis would read. Return that Loop's axis name for reuse, or ``None``.

    The match is tight: both the pending's reduce and the sibling's reduce
    must contain a bare ``Var(axis)`` at the same (buffer, dim) position.
    """
    p_patterns = _reduce_axis_buffer_positions(p.decl.body, p.decl.reduce_axis.name, ctx.producer_inputs)
    if not p_patterns:
        return None

    for stmt in emitted_so_far:
        if not isinstance(stmt, Loop):
            continue
        if not _loop_is_reduce(stmt):
            continue
        # Sibling Loops in current_scope_out have already been source-
        # remapped into the merged kernel's input space, so look up buffers
        # via merged_inputs.
        sib_axis = stmt.axis.name
        sib_patterns = _reduce_axis_buffer_positions(stmt.body, sib_axis, ctx.merged_inputs)
        if p_patterns & sib_patterns:
            if int(p.decl.reduce_axis.extent) != int(stmt.axis.extent):
                continue
            return sib_axis

    return None


def _reduce_axis_buffer_positions(
    body: tuple[Stmt, ...],
    reduce_axis_name: str,
    inputs: list[str],
) -> set[tuple[str, int]]:
    """Set of (external_buffer_name, dim) positions where a bare
    ``Var(reduce_axis_name)`` appears in a Load's index inside ``body``."""
    positions: set[tuple[str, int]] = set()

    def walk(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Load):
                if 0 <= s.source < len(inputs):
                    buf = inputs[s.source]
                    for dim, e in enumerate(s.index):
                        if isinstance(e, Var) and e.name == reduce_axis_name:
                            positions.add((buf, dim))
            elif isinstance(s, Loop):
                walk(s.body)

    walk(body)
    return positions


def _loop_is_reduce(loop: Loop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


def _unify_sibling_reduce_axes(body: tuple[Stmt, ...], merged_inputs: list[str]) -> tuple[Stmt, ...]:
    """Post-pass: at every scope, find sibling reduce Loops whose reduce
    axes index the same (external_buffer, dim) position, and rename them
    to share a single canonical axis name.

    Example: after splicing max + sub + exp into sum + div, the merged
    body has both max's and sum's reduce loops as siblings, each loading
    x at dim 1. This pass collapses their axis names so the final kernel
    reports one reduce axis (per LoopOp.axes dedup-by-name)."""
    # Recurse into nested Loops first.
    new_body: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop):
            new_body.append(Loop(axis=s.axis, body=_unify_sibling_reduce_axes(s.body, merged_inputs)))
        else:
            new_body.append(s)

    # Group sibling reduce Loops by their (buffer, dim) pattern.
    groups: dict[frozenset[tuple[str, int]], list[int]] = {}
    for i, s in enumerate(new_body):
        if isinstance(s, Loop) and _loop_is_reduce(s):
            positions = _reduce_axis_buffer_positions(s.body, s.axis.name, merged_inputs)
            if positions:
                groups.setdefault(frozenset(positions), []).append(i)

    for _, indices in groups.items():
        if len(indices) < 2:
            continue
        first_loop = new_body[indices[0]]
        assert isinstance(first_loop, Loop)
        canonical_axis = first_loop.axis.name
        canonical_extent = int(first_loop.axis.extent)
        for idx in indices[1:]:
            loop = new_body[idx]
            assert isinstance(loop, Loop)
            if int(loop.axis.extent) != canonical_extent:
                continue
            if loop.axis.name == canonical_axis:
                continue
            renamed_body = _rename_axis_in_body(loop.body, loop.axis.name, canonical_axis)
            new_body[idx] = Loop(axis=Axis(name=canonical_axis, extent=canonical_extent), body=renamed_body)

    return tuple(new_body)


def _rename_axis_in_body(body: tuple[Stmt, ...], old_name: str, new_name: str) -> tuple[Stmt, ...]:
    mapping = {old_name: Var(new_name)}
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop):
            # Shadow check: if s.axis.name == old_name, the rename is at this
            # exact Loop's scope and we're already handling it at the caller.
            # Otherwise descend.
            if s.axis.name == old_name:
                out.append(Loop(axis=Axis(name=new_name, extent=s.axis.extent), body=_rename_axis_in_body(s.body, old_name, new_name)))
            else:
                out.append(Loop(axis=s.axis, body=_rename_axis_in_body(s.body, old_name, new_name)))
        elif isinstance(s, Load):
            out.append(Load(name=s.name, source=s.source, index=tuple(substitute(e, mapping) for e in s.index)))
        elif isinstance(s, Write):
            out.append(Write(output=s.output, index=tuple(substitute(e, mapping) for e in s.index), value=s.value))
        elif isinstance(s, Select):
            out.append(
                Select(
                    name=s.name,
                    branches=tuple(SelectBranch(value=b.value, select=substitute(b.select, mapping)) for b in s.branches),
                )
            )
        else:
            out.append(s)
    return tuple(out)


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


def _check_no_extra_consumer_axes(
    sigma: dict[str, Expr],
    enclosing_names: tuple[str, ...],
    ctx: _Ctx,
) -> None:
    """Reject when the consumer introduces axes outside σ's image — that
    would mean the producer is being asked to redo the same work for
    unrelated consumer iterations. See design clarification #3."""
    producer_output_coord_axes = {e.name for e in _producer_write(ctx).index if isinstance(e, Var)}
    mapped_consumer_axes: set[str] = set()
    for p_name, r in sigma.items():
        if p_name not in producer_output_coord_axes:
            continue
        if isinstance(r, Var):
            mapped_consumer_axes.add(r.name)
    # Every axis currently in the consumer's enclosing chain must either be
    # mapped from a producer output-coord axis OR be literal-pinned in the
    # producer's Write (i.e. producer iterates over it implicitly by having
    # extent 1).
    for c_axis in enclosing_names:
        if c_axis in mapped_consumer_axes:
            continue
        # The axis may correspond to a producer axis via a pinned Literal;
        # that's fine. If there's no correspondence at all, the consumer is
        # introducing extra iteration.
        if _axis_is_broadcastable(c_axis, ctx):
            continue
        raise _NotSupported


def _axis_is_broadcastable(c_axis: str, ctx: _Ctx) -> bool:
    """True if ``c_axis`` is a consumer axis with extent 1 or only used for
    the spliced Load (which we're replacing anyway)."""
    for a in ctx.consumer.axes:
        if a.name == c_axis and int(a.extent) == 1:
            return True
    return False


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


def _producer_defs(ctx: _Ctx) -> dict[str, Stmt]:
    """Collect producer SSA definitions (flat, ignoring Loop scope).

    This is safe for our use because the producer's SSA names are unique
    across the tree (enforced by LoopOp validator)."""
    defs: dict[str, Stmt] = {}
    for s in _iter_all(ctx.producer.body):
        if isinstance(s, (Load, Assign, Select, Accum)):
            defs[s.name] = s
    return defs


def _find_accum_scope(producer: LoopOp, accum_name: str) -> _AccumDecl | None:
    """Walk the producer body to find the Accum named ``accum_name``,
    returning its scoping information (reduce axis, enclosing free axes,
    and the immediate reduce-Loop body)."""
    result: list[_AccumDecl] = []

    def walk(stmts: tuple[Stmt, ...], enclosing: tuple[Axis, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                is_reduce = any(isinstance(ss, Accum) for ss in s.body)
                if is_reduce:
                    for ss in s.body:
                        if isinstance(ss, Accum) and ss.name == accum_name:
                            result.append(
                                _AccumDecl(
                                    name=accum_name,
                                    reduce_axis=s.axis,
                                    enclosing_axes=enclosing,
                                    body=s.body,
                                )
                            )
                            return
                    # Nested reduce scanning: not currently emitted, but
                    # walk in case future producers nest.
                    walk(s.body, enclosing + (s.axis,))
                else:
                    walk(s.body, enclosing + (s.axis,))
            # ignore non-Loop sibling stmts (Accum should always be inside a Loop)

    walk(producer.body, ())
    return result[0] if result else None


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
    enclosing_names: tuple[str, ...],
) -> tuple[list[_PendingAccum], list[_PendingAccum]]:
    """Split pendings into (flush-here, bubble-up).

    - ``required_c_axes == enclosing_names`` → flush here.
    - ``required_c_axes`` is a strict prefix of enclosing_names → bubble up.
    - otherwise (longer, or non-prefix mismatch) → NotSupported.
    """
    here: list[_PendingAccum] = []
    up: list[_PendingAccum] = []
    for p in pendings:
        if p.required_c_axes == enclosing_names:
            here.append(p)
        elif _is_strict_prefix(p.required_c_axes, enclosing_names):
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
