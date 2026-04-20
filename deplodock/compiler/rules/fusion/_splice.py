"""Tree-splicing merge for two adjacent ``LoopOp``s — parallel tree walk.

The producer is the host namespace: its body passes through unchanged
(axes, SSA names, Load sources all retained). The consumer adapts to fit:
σ-matched axes are rewritten to the producer's axis name, colliding axis
and SSA names are freshened, Load sources are remapped to the merged
layout, and Loads targeting the producer are dropped with their downstream
references rewired to the producer's Write value.

σ is a minimal positional solver: producer.Write.index[k] paired with the
consumer Load.index[k] gives ``{producer_axis: reader_expr}`` entries for
each ``Var(p)`` writer position, with ``Literal(c)`` writer positions
adding no binding. Anything else in the writer → no merge.

Under the lift convention each kernel iterates its own output shape with
identity Write, so σ values are typically ``Var(consumer_axis)`` identity
maps. A reduce-axis alias detector feeds σ when producer and consumer
both have reduce axes over a shared external buffer. The consumer-side
substitution is built by inverting σ: for every ``σ[p] = Var(c)`` entry,
``consumer_sigma[c] = Var(p)``.

Not yet supported (returns ``None``):

- Multi-read (consumer reads the producer at more than one distinct index).
- Non-trivial writer forms (``Var(p) ± c``, ``Cast``, multiplicative).
- Producer/consumer with differently-ordered Loop nests (e.g. producer's
  σ-matched Loop is at a different depth than the consumer's matching one).
"""

from __future__ import annotations

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

    # --- Consumer adaptation tables ---
    producer_ssa_names = _collect_ssa_names(producer)
    consumer_ssa_names = _collect_ssa_names(consumer)
    consumer_axis_names = {a.name for a in consumer.axes}

    # Consumer axes σ-matched to producer axes (Var-valued σ entries only).
    # These vanish in the merged kernel: their refs are rewritten to the
    # producer's axis name, and their Loops fold into producer's Loops.
    sigma_matched_c_axes = {b.name for b in sigma.values() if isinstance(b, Var)}

    # Non-σ-matched consumer axes: freshen any that collide with producer.
    non_matched_c_axes = consumer_axis_names - sigma_matched_c_axes
    taken_by_producer = producer_axis_names | producer_ssa_names
    consumer_axis_rename = _fresh_names(non_matched_c_axes, taken_by_producer)

    # Consumer SSA names: freshen collisions with producer names or the
    # (already-allocated) consumer axis renames.
    consumer_ssa_rename = _fresh_names(
        consumer_ssa_names,
        taken_by_producer | set(consumer_axis_rename.values()),
    )

    # consumer_sigma: substitution applied to axis refs inside consumer Expr
    # trees. Inverts σ's Var-valued entries ({p → Var(c)} ⟹ {c → Var(p)})
    # and adds consumer axis renames as identity-shaped entries.
    consumer_sigma: dict[str, Expr] = {}
    for p_name, reader_expr in sigma.items():
        if isinstance(reader_expr, Var):
            consumer_sigma[reader_expr.name] = Var(p_name)
    for old, new in consumer_axis_rename.items():
        consumer_sigma[old] = Var(new)

    # Producer axis name → consumer axis name, for Loop matching during the walk.
    producer_to_consumer_axis = {p_name: r.name for p_name, r in sigma.items() if isinstance(r, Var)}

    # --- Source renumbering ---
    # Layout: [producer.inputs] ++ [consumer.inputs \ source]. Producer Loads
    # keep their source unchanged; consumer Loads shift past producer's inputs
    # and compact over the dropped slot.
    consumer_source_remap: dict[int, int] = {}
    next_src = producer.num_inputs
    for i in range(consumer.num_inputs):
        if i == source:
            continue
        consumer_source_remap[i] = next_src
        next_src += 1

    # Bridge value is producer's Write value verbatim — producer stays in its
    # own namespace, so no rename composition is needed.
    consumer_load_alias = {ld.name: write.value for ld in target_loads}

    # --- Parallel tree walk ---
    merged_body = _merge_trees(
        producer.body,
        consumer.body,
        consumer_sigma=consumer_sigma,
        consumer_ssa_rename=consumer_ssa_rename,
        consumer_axis_rename=consumer_axis_rename,
        consumer_source_remap=consumer_source_remap,
        consumer_load_alias=consumer_load_alias,
        producer_to_consumer_axis=producer_to_consumer_axis,
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
    consumer_sigma: dict[str, Expr],
    consumer_ssa_rename: dict[str, str],
    consumer_axis_rename: dict[str, str],
    consumer_source_remap: dict[int, int],
    consumer_load_alias: dict[str, str],
    producer_to_consumer_axis: dict[str, str],
) -> tuple[Stmt, ...]:
    """Walk producer and consumer bodies at the current scope.

    Producer stmts emit verbatim (skipping the producer's single Write to
    output 0 — its role is subsumed by the consumer rewires). For each
    producer Loop whose axis is σ-bound to a consumer Loop at this scope,
    merge the two bodies recursively and emit a single Loop under the
    producer's axis. Consumer stmts then follow, rewritten: σ-matched Loops
    consumed above are skipped, the target Load is dropped, and remaining
    consumer stmts get axis substitution, SSA rename, source remap, and
    dropped-Load aliasing applied."""

    # Index consumer Loops by axis name at THIS scope.
    c_loop_by_axis: dict[str, Loop] = {s.axis.name: s for s in c_stmts if isinstance(s, Loop)}
    consumed_c_loops: set[str] = set()

    def ren(name: str) -> str:
        """Resolve a consumer SSA reference: dropped-Load names → producer
        bridge value; otherwise → fresh name (or identity)."""
        if name in consumer_load_alias:
            return consumer_load_alias[name]
        return consumer_ssa_rename.get(name, name)

    def rewrite_consumer(stmt: Stmt) -> Stmt | None:
        """Rewrite a consumer stmt at this scope. Loops recurse into their
        body (nested consumer stmts receive the same adaptations)."""
        if isinstance(stmt, Loop):
            axis_name = stmt.axis.name
            new_axis_name = consumer_axis_rename.get(axis_name, axis_name)
            new_body = tuple(s for s in (rewrite_consumer(c) for c in stmt.body) if s is not None)
            return Loop(axis=Axis(name=new_axis_name, extent=stmt.axis.extent), body=new_body)
        if isinstance(stmt, Load):
            new_source = consumer_source_remap.get(stmt.source)
            if new_source is None:
                return None  # target Load dropped; name resolves via consumer_load_alias
            return Load(
                name=consumer_ssa_rename.get(stmt.name, stmt.name),
                source=new_source,
                index=tuple(substitute(e, consumer_sigma) for e in stmt.index),
            )
        if isinstance(stmt, Assign):
            return Assign(
                name=consumer_ssa_rename.get(stmt.name, stmt.name),
                op=stmt.op,
                args=tuple(ren(a) for a in stmt.args),
            )
        if isinstance(stmt, Accum):
            return Accum(
                name=consumer_ssa_rename.get(stmt.name, stmt.name),
                value=ren(stmt.value),
                op=stmt.op,
            )
        if isinstance(stmt, Select):
            return Select(
                name=consumer_ssa_rename.get(stmt.name, stmt.name),
                branches=tuple(
                    SelectBranch(
                        value=ren(b.value),
                        select=substitute(b.select, consumer_sigma),
                    )
                    for b in stmt.branches
                ),
            )
        if isinstance(stmt, Write):
            return Write(
                output=stmt.output,
                index=tuple(substitute(e, consumer_sigma) for e in stmt.index),
                value=ren(stmt.value),
            )
        return stmt

    out: list[Stmt] = []

    # Producer pass: emit verbatim, except σ-matched Loops merge with their
    # consumer counterpart here (under the producer's axis name).
    for p_stmt in p_stmts:
        if isinstance(p_stmt, Write) and p_stmt.output == 0:
            continue
        if isinstance(p_stmt, Loop):
            matched_c_axis = producer_to_consumer_axis.get(p_stmt.axis.name)
            if matched_c_axis is not None and matched_c_axis in c_loop_by_axis:
                c_loop = c_loop_by_axis[matched_c_axis]
                merged_inner = _merge_trees(
                    p_stmt.body,
                    c_loop.body,
                    consumer_sigma=consumer_sigma,
                    consumer_ssa_rename=consumer_ssa_rename,
                    consumer_axis_rename=consumer_axis_rename,
                    consumer_source_remap=consumer_source_remap,
                    consumer_load_alias=consumer_load_alias,
                    producer_to_consumer_axis=producer_to_consumer_axis,
                )
                out.append(Loop(axis=p_stmt.axis, body=merged_inner))
                consumed_c_loops.add(matched_c_axis)
                continue
        out.append(p_stmt)

    # Consumer pass: skip σ-consumed Loops; rewrite and emit the rest.
    for c_stmt in c_stmts:
        if isinstance(c_stmt, Loop) and c_stmt.axis.name in consumed_c_loops:
            continue
        rewritten = rewrite_consumer(c_stmt)
        if rewritten is not None:
            out.append(rewritten)

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
