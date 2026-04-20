"""Tree-splicing merge for two adjacent ``LoopOp``s — parallel tree walk.

Both producer and consumer are already nested optimally by their lifts, so
merging is a pairwise tree walk: at each scope, match producer's Loops with
consumer's Loops by σ-bound axis identity, recurse into merged bodies,
emit unmatched producer Loops (e.g. reduce sweeps) as siblings in the
current scope, and at the leaf level splice producer leaves in front of
the consumer's target-Load-rewritten leaves.

σ is a minimal positional solver: producer.Write.index[k] paired with the
consumer Load.index[k] gives ``{producer_axis: reader_expr}`` entries for
each ``Var(p)`` writer position, with ``Literal(c)`` writer positions
adding no binding. Anything else in the writer → no merge.

Under the lift convention each kernel iterates its own output shape with
identity Write, so σ values are typically ``Var(consumer_axis)`` identity
maps. A reduce-axis alias detector feeds σ when producer and consumer
both have reduce axes over a shared external buffer.

Not yet supported (returns ``None``):

- Multi-read (consumer reads the producer at more than one distinct index).
- Non-trivial writer forms (``Var(p) ± c``, ``Cast``, multiplicative).
- Producer/consumer with differently-ordered Loop nests (e.g. producer's
  σ-matched Loop is at a different depth than the consumer's matching one).
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Expr, Literal, Var, substitute
from deplodock.compiler.ir.loop_ir import (
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
    flatten_body,
)

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
    """Splice ``producer``'s body into every ``consumer`` Load with matching
    ``source``. Returns ``None`` when the pattern isn't supported yet."""

    writes = [s for s in flatten_body(producer.body) if isinstance(s, Write) and s.output == 0]
    if len(writes) != 1:
        return None
    write = writes[0]

    target_loads = [ld for ld in consumer.loads if ld.source == source]
    if not target_loads:
        return None
    # Multi-read with distinct reader indices is deferred.
    reader_index = target_loads[0].index
    for ld in target_loads[1:]:
        if tuple(ld.index) != tuple(reader_index):
            return None

    producer_axis_names = {a.name for a in producer.axes}
    sigma = _solve_sigma(write.index, reader_index, producer_axis_names)
    if sigma is None:
        return None

    if producer_inputs is not None and consumer_inputs is not None:
        aliases = _detect_reduce_axis_aliases(producer, consumer, sigma, source, producer_inputs, consumer_inputs)
        for p_name, c_var in aliases.items():
            sigma.setdefault(p_name, c_var)

    reduce_names = producer.reduce_axis_names
    for ax in producer.axes:
        if ax.name in sigma:
            continue
        if ax.name in reduce_names:
            continue
        if int(ax.extent) == 1:
            continue
        return None

    # --- Renames ---
    consumer_ssa = _collect_ssa_names(consumer)
    consumer_axis_names = {a.name for a in consumer.axes}
    producer_ssa = _collect_ssa_names(producer)

    # Producer axes surviving as Loops: reduce (not σ-bound) with extent > 1.
    surviving_axes = [a for a in producer.axes if a.name in reduce_names and a.name not in sigma and int(a.extent) > 1]
    axis_rename = _fresh_names({a.name for a in surviving_axes}, consumer_axis_names | consumer_ssa)
    ssa_rename = _fresh_names(producer_ssa, consumer_ssa | consumer_axis_names | set(axis_rename.values()))

    full_sigma: dict[str, Expr] = dict(sigma)
    for old, new in axis_rename.items():
        full_sigma[old] = Var(new)

    # --- Source renumbering ---
    consumer_source_remap: dict[int, int] = {}
    next_src = 0
    for i in range(consumer.num_inputs):
        if i == source:
            continue
        consumer_source_remap[i] = next_src
        next_src += 1
    producer_source_remap: dict[int, int] = {j: next_src + j for j in range(producer.num_inputs)}

    bridge_value = ssa_rename.get(write.value, write.value)
    load_alias = {ld.name: bridge_value for ld in target_loads}

    # --- Parallel tree walk ---
    merged_body = _merge_trees(
        producer.body,
        consumer.body,
        sigma=sigma,
        full_sigma=full_sigma,
        ssa_rename=ssa_rename,
        axis_rename=axis_rename,
        producer_source_remap=producer_source_remap,
        consumer_source_remap=consumer_source_remap,
        target_source=source,
        load_alias=load_alias,
        skip_write=write,
    )

    try:
        return LoopOp(body=merged_body)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Parallel tree walk
# ---------------------------------------------------------------------------


def _merge_trees(
    p_stmts: tuple[Stmt, ...],
    c_stmts: tuple[Stmt, ...],
    *,
    sigma: dict[str, Expr],
    full_sigma: dict[str, Expr],
    ssa_rename: dict[str, str],
    axis_rename: dict[str, str],
    producer_source_remap: dict[int, int],
    consumer_source_remap: dict[int, int],
    target_source: int,
    load_alias: dict[str, str],
    skip_write: Write,
) -> tuple[Stmt, ...]:
    """Walk producer and consumer bodies at the current scope.

    Match any producer Loop whose axis is σ-bound to a consumer axis via
    ``Var`` identity with a consumer Loop of that axis at this scope; merge
    them recursively. Unmatched producer Loops / leaves / non-target Writes
    get rewritten and emitted in producer order, then consumer's remaining
    stmts follow (with target Loads dropped and aliases applied)."""

    # Index consumer Loops by axis name at THIS scope.
    c_loop_by_axis: dict[str, Loop] = {s.axis.name: s for s in c_stmts if isinstance(s, Loop)}
    consumed_c_loops: set[str] = set()

    def rewrite_producer(stmt: Stmt) -> Stmt | None:
        """Rewrite a producer stmt that's emitted at this scope as-is. Loops
        recurse into their own body (with no consumer to merge — they're
        unmatched at this scope)."""
        if stmt is skip_write:
            return None
        if isinstance(stmt, Loop):
            axis_name = stmt.axis.name
            new_axis_name = axis_rename.get(axis_name, axis_name)
            new_body = tuple(s for s in (rewrite_producer(c) for c in stmt.body) if s is not None)
            return Loop(axis=Axis(name=new_axis_name, extent=stmt.axis.extent), body=new_body)
        if isinstance(stmt, Load):
            return Load(
                name=ssa_rename.get(stmt.name, stmt.name),
                source=producer_source_remap[stmt.source],
                index=tuple(substitute(e, full_sigma) for e in stmt.index),
            )
        if isinstance(stmt, Assign):
            return Assign(
                name=ssa_rename.get(stmt.name, stmt.name),
                op=stmt.op,
                args=tuple(ssa_rename.get(a, a) for a in stmt.args),
            )
        if isinstance(stmt, Accum):
            return Accum(
                name=ssa_rename.get(stmt.name, stmt.name),
                value=ssa_rename.get(stmt.value, stmt.value),
                op=stmt.op,
            )
        if isinstance(stmt, Select):
            return Select(
                name=ssa_rename.get(stmt.name, stmt.name),
                branches=tuple(
                    SelectBranch(value=ssa_rename.get(b.value, b.value), select=substitute(b.select, full_sigma)) for b in stmt.branches
                ),
            )
        if isinstance(stmt, Write):
            return Write(
                output=stmt.output,
                index=tuple(substitute(e, full_sigma) for e in stmt.index),
                value=ssa_rename.get(stmt.value, stmt.value),
            )
        return stmt

    def ren(name: str) -> str:
        return load_alias.get(name, name)

    out: list[Stmt] = []

    # Walk producer stmts in order. σ-matched Loops defer to consumer walk;
    # everything else rewrites and emits here.
    for p_stmt in p_stmts:
        if p_stmt is skip_write:
            continue
        if isinstance(p_stmt, Loop):
            bound = sigma.get(p_stmt.axis.name)
            if isinstance(bound, Var) and bound.name in c_loop_by_axis:
                c_loop = c_loop_by_axis[bound.name]
                merged_inner = _merge_trees(
                    p_stmt.body,
                    c_loop.body,
                    sigma=sigma,
                    full_sigma=full_sigma,
                    ssa_rename=ssa_rename,
                    axis_rename=axis_rename,
                    producer_source_remap=producer_source_remap,
                    consumer_source_remap=consumer_source_remap,
                    target_source=target_source,
                    load_alias=load_alias,
                    skip_write=skip_write,
                )
                out.append(Loop(axis=c_loop.axis, body=merged_inner))
                consumed_c_loops.add(bound.name)
                continue
        rewritten = rewrite_producer(p_stmt)
        if rewritten is not None:
            out.append(rewritten)

    # Walk consumer stmts in order; skip the ones consumed by σ matching.
    for c_stmt in c_stmts:
        if isinstance(c_stmt, Loop):
            if c_stmt.axis.name in consumed_c_loops:
                continue
            # Consumer's own Loop (unmatched) — recurse with empty producer.
            merged_inner = _merge_trees(
                (),
                c_stmt.body,
                sigma=sigma,
                full_sigma=full_sigma,
                ssa_rename=ssa_rename,
                axis_rename=axis_rename,
                producer_source_remap=producer_source_remap,
                consumer_source_remap=consumer_source_remap,
                target_source=target_source,
                load_alias=load_alias,
                skip_write=skip_write,
            )
            out.append(Loop(axis=c_stmt.axis, body=merged_inner))
            continue
        if isinstance(c_stmt, Load):
            if c_stmt.source == target_source:
                continue  # dropped; name resolves via load_alias
            out.append(Load(name=c_stmt.name, source=consumer_source_remap[c_stmt.source], index=c_stmt.index))
            continue
        if isinstance(c_stmt, Assign):
            out.append(Assign(name=c_stmt.name, op=c_stmt.op, args=tuple(ren(a) for a in c_stmt.args)))
            continue
        if isinstance(c_stmt, Accum):
            out.append(Accum(name=c_stmt.name, value=ren(c_stmt.value), op=c_stmt.op))
            continue
        if isinstance(c_stmt, Write):
            out.append(Write(output=c_stmt.output, index=c_stmt.index, value=ren(c_stmt.value)))
            continue
        if isinstance(c_stmt, Select):
            out.append(
                Select(
                    name=c_stmt.name,
                    branches=tuple(SelectBranch(value=ren(b.value), select=b.select) for b in c_stmt.branches),
                )
            )
            continue
        out.append(c_stmt)

    return tuple(out)


# ---------------------------------------------------------------------------
# σ solver
# ---------------------------------------------------------------------------


def _solve_sigma(writer: tuple[Expr, ...], reader: tuple[Expr, ...], producer_axes: set[str]) -> dict[str, Expr] | None:
    """``writer[k] == reader[k]`` per dim. Supported writer forms:

    - ``Var(a)`` with ``a`` in ``producer_axes`` → bind ``a → reader[k]``.
    - ``Literal(c)`` → no binding (writer pins a constant).

    Anything else → unsupported (returns ``None``)."""
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
# Reduce-axis alias (sibling reductions)
# ---------------------------------------------------------------------------


def _detect_reduce_axis_aliases(
    producer: LoopOp,
    consumer: LoopOp,
    sigma: dict[str, Expr],
    target_source: int,
    producer_inputs: list[str],
    consumer_inputs: list[str],
) -> dict[str, Expr]:
    """Producer reduce axes that unify with consumer reduce axes.

    A producer reduce axis ``p_ax`` aliases to a consumer reduce axis ``c_ax``
    when both index the same external buffer at the same dim as a bare
    ``Var``, the extents match, and the consumer's Load exposing ``c_ax``
    does *not* target the producer."""
    reduce_names = producer.reduce_axis_names
    candidate_p_axes = [a for a in producer.axes if a.name in reduce_names and a.name not in sigma]
    if not candidate_p_axes:
        return {}

    consumer_reduce = {a.name: a for a in consumer.axes if a.name in consumer.reduce_axis_names}
    if not consumer_reduce:
        return {}

    aliases: dict[str, Expr] = {}
    for p_ax in candidate_p_axes:
        alias = _find_alias(p_ax, producer, consumer, target_source, producer_inputs, consumer_inputs, consumer_reduce)
        if alias is not None:
            aliases[p_ax.name] = Var(alias)
    return aliases


def _find_alias(
    p_ax: Axis,
    producer: LoopOp,
    consumer: LoopOp,
    target_source: int,
    producer_inputs: list[str],
    consumer_inputs: list[str],
    consumer_reduce: dict[str, Axis],
) -> str | None:
    for p_load in producer.loads:
        if p_load.source >= len(producer_inputs):
            continue
        p_buf = producer_inputs[p_load.source]
        for dim, expr in enumerate(p_load.index):
            if not (isinstance(expr, Var) and expr.name == p_ax.name):
                continue
            for c_load in consumer.loads:
                if c_load.source == target_source:
                    continue
                if c_load.source >= len(consumer_inputs):
                    continue
                if consumer_inputs[c_load.source] != p_buf:
                    continue
                if dim >= len(c_load.index):
                    continue
                c_expr = c_load.index[dim]
                if not isinstance(c_expr, Var):
                    continue
                c_axis = consumer_reduce.get(c_expr.name)
                if c_axis is None:
                    continue
                if int(c_axis.extent) != int(p_ax.extent):
                    continue
                return c_axis.name
    return None


# ---------------------------------------------------------------------------
# Name collection + fresh-name tables
# ---------------------------------------------------------------------------


def _collect_ssa_names(op: LoopOp) -> set[str]:
    names: set[str] = set()
    for stmt in flatten_body(op.body):
        if isinstance(stmt, (Assign, Load, Select, Accum)):
            names.add(stmt.name)
    return names


def _fresh_names(to_rename: set[str], taken: set[str]) -> dict[str, str]:
    """Map original→fresh for names in ``to_rename`` that collide with ``taken``.
    Names without collision get no entry (identity)."""
    result: dict[str, str] = {}
    used = set(taken)
    for name in to_rename:
        if name not in used:
            used.add(name)
            continue
        candidate = name
        suffix = 1
        while candidate in used:
            candidate = f"{name}_p{suffix}"
            suffix += 1
        result[name] = candidate
        used.add(candidate)
    return result
