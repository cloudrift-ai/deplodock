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

    producer_axis_names = {a.name for a in producer.axes}

    # Per-Load σ. Each consumer Load that targets the producer gets its own
    # σ: writer.index vs that Load's reader index. Single-read is the common
    # case; softmax-style multi-read (two sweeps over producer at distinct
    # indices) exercises the per-Load path.
    per_load_sigma: dict[str, dict[str, Expr]] = {}
    for ld in target_loads:
        s = _solve_sigma(write.index, ld.index, producer_axis_names)
        if s is None:
            return None
        per_load_sigma[ld.name] = s

    # Common σ: bindings that agree across every per-Load σ. Drives
    # producer/consumer Loop folding during the tree walk.
    common_sigma: dict[str, Expr] = {}
    first_sigma = next(iter(per_load_sigma.values()))
    for p_name, r in first_sigma.items():
        if all(per_load_sigma[lname].get(p_name) == r for lname in per_load_sigma):
            common_sigma[p_name] = r

    if producer_inputs is not None and consumer_inputs is not None:
        aliases = _detect_reduce_axis_aliases(producer, consumer, common_sigma, source, producer_inputs, consumer_inputs)
        for p_name, c_var in aliases.items():
            common_sigma.setdefault(p_name, c_var)
        bridge_aliases = _detect_output_coord_reduce_aliases(producer, consumer, per_load_sigma, common_sigma, write, producer_inputs)
        for p_name, c_var in bridge_aliases.items():
            common_sigma.setdefault(p_name, c_var)

    # Parametric axes: producer axes bound by some per-Load σ but not
    # common — their binding varies per Load (the element-space replay
    # substitutes them per-site). Fine: they don't need a producer Loop
    # in the row-space output.
    parametric_axes: set[str] = set()
    for ps in per_load_sigma.values():
        for p_name in ps:
            if p_name not in common_sigma:
                parametric_axes.add(p_name)

    reduce_names = producer.reduce_axis_names
    for ax in producer.axes:
        if ax.name in common_sigma or ax.name in parametric_axes or ax.name in reduce_names:
            continue
        if int(ax.extent) == 1:
            continue
        return None

    # --- Consumer adaptation tables (built against common_sigma) ---
    producer_ssa_names = _collect_ssa_names(producer)
    consumer_ssa_names = _collect_ssa_names(consumer)
    consumer_axis_names = {a.name for a in consumer.axes}

    sigma_matched_c_axes = {b.name for b in common_sigma.values() if isinstance(b, Var)}
    non_matched_c_axes = consumer_axis_names - sigma_matched_c_axes
    taken_by_producer = producer_axis_names | producer_ssa_names
    consumer_axis_rename = _fresh_names(non_matched_c_axes, taken_by_producer)

    consumer_ssa_rename = _fresh_names(
        consumer_ssa_names,
        taken_by_producer | set(consumer_axis_rename.values()),
    )

    consumer_sigma: dict[str, Expr] = {}
    for p_name, reader_expr in common_sigma.items():
        if isinstance(reader_expr, Var):
            consumer_sigma[reader_expr.name] = Var(p_name)
    for old, new in consumer_axis_rename.items():
        consumer_sigma[old] = Var(new)

    producer_to_consumer_axis = {p_name: r.name for p_name, r in common_sigma.items() if isinstance(r, Var)}

    consumer_source_remap: dict[int, int] = {}
    next_src = producer.num_inputs
    for i in range(consumer.num_inputs):
        if i == source:
            continue
        consumer_source_remap[i] = next_src
        next_src += 1

    # --- Split producer body into row-space + element-space template ---
    # Row-space: stmts that feed the producer's Accums (scope-crossing
    # values like softmax's acc_max) — emitted once at the merged scope.
    # Element-space template: stmts feeding the Write, with the Write
    # elided and output-coord Loops unwrapped. Replayed per consumer Load
    # at the Load site with that Load's σ.
    row_stmts, element_template = _split_producer_body(producer.body, write)

    # Multi-read safety checks.
    if len(target_loads) > 1:
        # (1) Every parametric axis (σ-bound per-Load but not common) must
        # appear inside the element-space template. Otherwise different
        # consumer Loads would inline an expression that ignores their
        # distinct σ bindings — or, when the template is empty, Write.value
        # is just an Accum name reachable only via scope propagation, not
        # re-runnable per Load.
        template_axes = _axes_in_template(element_template)
        if not parametric_axes.issubset(template_axes):
            return None

        # (2) Any producer Accum referenced by the element-space template
        # must be defined at a scope whose enclosing row-space Loops are
        # all bound by common σ. If the Accum sits under a parametric
        # producer Loop (e.g. SDPA's score reduce nested inside the
        # output K axis), its value varies per-parametric-iteration and
        # can't be shared across distinct consumer reads — one
        # ``acc`` in row-space, many consumer reads with different σ
        # would all see the last written value.
        common_axis_names = {v.name for v in common_sigma.values() if isinstance(v, Var)} | set(common_sigma)
        referenced = _referenced_ssa_names(element_template)
        accum_scopes = _accum_defining_scopes(row_stmts)
        for acc_name in referenced:
            scope = accum_scopes.get(acc_name)
            if scope is None:
                continue  # not an Accum (e.g. external SSA); no scope constraint
            if not scope.issubset(common_axis_names):
                return None

    # --- Per-Load replay ---
    consumer_load_alias: dict[str, str] = {}
    replay_stmts_per_load: dict[str, tuple[Stmt, ...]] = {}
    for i, ld in enumerate(target_loads):
        load_sigma = per_load_sigma[ld.name]
        # σ values use consumer's original axis names; push through
        # consumer_sigma so renames are applied before replay substitutes.
        axis_sub = {p_name: substitute(r, consumer_sigma) for p_name, r in load_sigma.items()}
        suffix = f"_r{i}"
        replay_stmts, final_name = _replay_element_template(element_template, axis_sub, suffix, write.value)
        replay_stmts_per_load[ld.name] = replay_stmts
        consumer_load_alias[ld.name] = final_name

    # --- Align producer's free-loop chain to consumer's depths ---
    aligned_row_stmts = _align_producer_free_chain(
        row_stmts,
        consumer.body,
        producer_to_consumer_axis,
        producer.reduce_axis_names,
    )

    # --- Parallel tree walk ---
    merged_body = _merge_trees(
        aligned_row_stmts,
        consumer.body,
        consumer_sigma=consumer_sigma,
        consumer_ssa_rename=consumer_ssa_rename,
        consumer_axis_rename=consumer_axis_rename,
        consumer_source_remap=consumer_source_remap,
        consumer_load_alias=consumer_load_alias,
        replay_stmts_per_load=replay_stmts_per_load,
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
    replay_stmts_per_load: dict[str, tuple[Stmt, ...]],
    producer_to_consumer_axis: dict[str, str],
) -> tuple[Stmt, ...]:
    """Walk producer (row-space) and consumer bodies at the current scope.

    Producer's row-space stmts emit verbatim; where a row-space Loop's
    axis is σ-bound to a consumer Loop at this scope, the two bodies
    merge recursively. Consumer stmts then follow, rewritten: σ-matched
    Loops consumed above are skipped; a target Load is replaced by the
    pre-computed element-space replay for that Load (may be multiple
    stmts); remaining consumer stmts get axis substitution, SSA rename,
    source remap, and load aliasing."""

    c_loop_by_axis: dict[str, Loop] = {s.axis.name: s for s in c_stmts if isinstance(s, Loop)}
    consumed_c_loops: set[str] = set()

    def ren(name: str) -> str:
        """Resolve a consumer SSA reference: dropped-Load names → replay
        final-value name; otherwise → fresh name (or identity)."""
        if name in consumer_load_alias:
            return consumer_load_alias[name]
        return consumer_ssa_rename.get(name, name)

    def rewrite_consumer(stmt: Stmt) -> Stmt | tuple[Stmt, ...] | None:
        """Rewrite a consumer stmt. Loops recurse; target Loads expand to
        their replay stmt tuple; other stmts rewrite in place."""
        if isinstance(stmt, Loop):
            axis_name = stmt.axis.name
            # Use consumer_sigma for the axis rename so the Loop header stays
            # consistent with body Expr substitution. consumer_sigma contains
            # both σ-matched entries (→ producer's axis name) and non-matched
            # rename entries. When a σ-matched Loop folds with a producer
            # Loop, this rewrite doesn't run (the Loop is in consumed_c_loops);
            # when the producer counterpart was stripped into element-space,
            # applying the same mapping here keeps axis name + body refs aligned.
            mapped = consumer_sigma.get(axis_name)
            new_axis_name = mapped.name if isinstance(mapped, Var) else consumer_axis_rename.get(axis_name, axis_name)
            new_body: list[Stmt] = []
            for c in stmt.body:
                rewritten = rewrite_consumer(c)
                if rewritten is None:
                    continue
                if isinstance(rewritten, tuple):
                    new_body.extend(rewritten)
                else:
                    new_body.append(rewritten)
            return Loop(axis=Axis(name=new_axis_name, extent=stmt.axis.extent), body=tuple(new_body))
        if isinstance(stmt, Load):
            new_source = consumer_source_remap.get(stmt.source)
            if new_source is None:
                # Target Load: emit the pre-computed element-space replay.
                return replay_stmts_per_load.get(stmt.name, ())
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

    # Producer pass over row-space stmts (Writes already elided by the
    # splitter). σ-matched row-space Loops fold with their consumer Loops
    # — except when both sides are reduce loops, in which case they emit
    # as siblings under the shared axis name. Folding their bodies would
    # interleave accumulators (sum seeing max mid-sweep), breaking the
    # softmax pattern where the second reduce reads the first's finalized
    # value.
    for p_stmt in p_stmts:
        if isinstance(p_stmt, Loop):
            matched_c_axis = producer_to_consumer_axis.get(p_stmt.axis.name)
            if matched_c_axis is not None and matched_c_axis in c_loop_by_axis:
                c_loop = c_loop_by_axis[matched_c_axis]
                p_is_reduce = any(isinstance(s, Accum) for s in p_stmt.body)
                c_is_reduce = any(isinstance(s, Accum) for s in c_loop.body)
                if p_is_reduce and c_is_reduce:
                    # Sibling reduce Loops share axis; consumer's Loop
                    # emits in the consumer pass (not consumed here).
                    out.append(p_stmt)
                    continue
                merged_inner = _merge_trees(
                    p_stmt.body,
                    c_loop.body,
                    consumer_sigma=consumer_sigma,
                    consumer_ssa_rename=consumer_ssa_rename,
                    consumer_axis_rename=consumer_axis_rename,
                    consumer_source_remap=consumer_source_remap,
                    consumer_load_alias=consumer_load_alias,
                    replay_stmts_per_load=replay_stmts_per_load,
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
        if rewritten is None:
            continue
        if isinstance(rewritten, tuple):
            out.extend(rewritten)
        else:
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
    paired: set[str] = set()

    for p_ax in candidate_p_axes:
        alias = _find_alias(p_ax, producer, consumer, target_source, producer_inputs, consumer_inputs, consumer_reduce)
        if alias is not None and alias not in paired:
            aliases[p_ax.name] = Var(alias)
            paired.add(alias)

    return aliases


def _detect_output_coord_reduce_aliases(
    producer: LoopOp,
    consumer: LoopOp,
    per_load_sigma: dict[str, dict[str, Expr]],
    common_sigma: dict[str, Expr],
    write: Write,
    producer_inputs: list[str],
) -> dict[str, Expr]:
    """Alias producer reduce axes via the output-coord bridge.

    When the consumer reads the producer's *output* (not a shared external
    buffer), the strict buffer-match check misses aliases. But the
    producer's element-space and its reduce loads often both index the
    same external buffer at the same dim — via different producer axes
    (the element-space uses an output-coord; the reduce uses a reduce
    axis). In softmax, producer's ``a1`` (max reduce) and ``a1_p1``
    (exp's output-coord) both index ``input`` at dim 1: that's what makes
    them the "same K dim". When the consumer's reduce axis ``c_ax`` is
    σ-bound to the producer's output-coord ``p_out`` in any per-Load σ,
    and a producer reduce axis ``p_ax`` indexes the same (buffer, dim) as
    ``p_out``, alias ``p_ax → Var(c_ax)`` so both reduce sweeps merge
    into sibling loops sharing an axis name.

    This is strictly tighter than matching by extent alone: patterns like
    SDPA (producer reduces over D, consumer reduces over K; both extent
    but different physical dims) get no alias here.
    """
    output_coord_axes = {e.name for e in write.index if isinstance(e, Var)}

    reduce_names = producer.reduce_axis_names
    candidate_p_axes = [a for a in producer.axes if a.name in reduce_names and a.name not in common_sigma]
    if not candidate_p_axes:
        return {}

    consumer_reduce = {a.name: a for a in consumer.axes if a.name in consumer.reduce_axis_names}
    if not consumer_reduce:
        return {}

    # Load fingerprints: which axis names appear at each (buffer, dim).
    load_patterns: dict[tuple[str, int], set[str]] = {}
    for ld in producer.loads:
        if ld.source >= len(producer_inputs):
            continue
        buf = producer_inputs[ld.source]
        for dim, expr in enumerate(ld.index):
            if isinstance(expr, Var):
                load_patterns.setdefault((buf, dim), set()).add(expr.name)

    aliases: dict[str, Expr] = {}
    paired_c: set[str] = set()

    for p_ax in candidate_p_axes:
        p_ax_positions = {pos for pos, axes in load_patterns.items() if p_ax.name in axes}
        if not p_ax_positions:
            continue
        for ps in per_load_sigma.values():
            for p_out_name, reader_expr in ps.items():
                if p_out_name not in output_coord_axes:
                    continue
                if not isinstance(reader_expr, Var):
                    continue
                c_ax_name = reader_expr.name
                if c_ax_name not in consumer_reduce or c_ax_name in paired_c:
                    continue
                if int(consumer_reduce[c_ax_name].extent) != int(p_ax.extent):
                    continue
                p_out_positions = {pos for pos, axes in load_patterns.items() if p_out_name in axes}
                if p_ax_positions & p_out_positions:
                    aliases[p_ax.name] = Var(c_ax_name)
                    paired_c.add(c_ax_name)
                    break
            if p_ax.name in aliases:
                break

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


# ---------------------------------------------------------------------------
# Producer body split: row-space vs element-space template
# ---------------------------------------------------------------------------


def _split_producer_body(
    body: tuple[Stmt, ...],
    write: Write,
) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Split producer's body into ``(row_stmts, element_template)``.

    The Write is elided. Stmts that feed the Write's value (via SSA dep
    flooding inside the Write's scope, stopping at Accum names which
    cross scopes as accumulator bindings) form the element-space
    template. Stmts that feed Accums (or don't feed the Write) form
    row-space.

    Loops that iterate output-coord axes (axes appearing as ``Var`` in
    the Write's index) are unwrapped in the element-space template —
    the replay substitutes those axes per-site. Loops whose bodies are
    purely row-space stay wrapped in row_stmts.
    """
    output_coord_axes: set[str] = set()
    for e in write.index:
        if isinstance(e, Var):
            output_coord_axes.add(e.name)

    def split(stmts: tuple[Stmt, ...]) -> tuple[list[Stmt], list[Stmt]]:
        write_idx: int | None = None
        for i, s in enumerate(stmts):
            if isinstance(s, Write) and s.output == 0 and s is write:
                write_idx = i
                break

        row_out: list[Stmt] = []
        element_flat: list[Stmt] = []

        if write_idx is not None:
            # This scope owns the Write. Flood locally from write.value
            # via SSA args of Assign/Select; stop at Loads (leaves) and at
            # names not defined in this scope (Accum names from outer).
            local_defs: dict[str, Stmt] = {}
            for s in stmts:
                if isinstance(s, (Assign, Load, Select)):
                    local_defs[s.name] = s

            element_names: set[str] = set()
            to_visit = [write.value]
            while to_visit:
                n = to_visit.pop()
                if n in element_names or n not in local_defs:
                    continue
                element_names.add(n)
                defn = local_defs[n]
                if isinstance(defn, Assign):
                    for a in defn.args:
                        to_visit.append(a)
                elif isinstance(defn, Select):
                    for b in defn.branches:
                        to_visit.append(b.value)

            for i, s in enumerate(stmts):
                if i == write_idx:
                    continue  # elide Write
                if isinstance(s, Loop):
                    inner_row, inner_element = split(s.body)
                    if inner_row:
                        row_out.append(Loop(axis=s.axis, body=tuple(inner_row)))
                    if inner_element:
                        if s.axis.name in output_coord_axes:
                            element_flat.extend(inner_element)
                        else:
                            element_flat.append(Loop(axis=s.axis, body=tuple(inner_element)))
                elif isinstance(s, (Assign, Load, Select)) and s.name in element_names:
                    element_flat.append(s)
                else:
                    row_out.append(s)
            return row_out, element_flat

        # No Write at this scope: recurse into Loops to locate it below.
        for s in stmts:
            if isinstance(s, Loop):
                inner_row, inner_element = split(s.body)
                if inner_row:
                    row_out.append(Loop(axis=s.axis, body=tuple(inner_row)))
                if inner_element:
                    if s.axis.name in output_coord_axes:
                        element_flat.extend(inner_element)
                    else:
                        element_flat.append(Loop(axis=s.axis, body=tuple(inner_element)))
            else:
                row_out.append(s)
        return row_out, element_flat

    row_stmts, element_template = split(body)
    return tuple(row_stmts), tuple(element_template)


# ---------------------------------------------------------------------------
# Element-space replay (per consumer Load)
# ---------------------------------------------------------------------------


def _referenced_ssa_names(stmts: tuple[Stmt, ...]) -> set[str]:
    """Every SSA name read inside ``stmts`` (Assign.args, Accum.value,
    Select.branches.value, Write.value)."""
    names: set[str] = set()
    for s in stmts:
        if isinstance(s, Loop):
            names |= _referenced_ssa_names(s.body)
        elif isinstance(s, Assign):
            names |= set(s.args)
        elif isinstance(s, Accum):
            names.add(s.value)
        elif isinstance(s, Select):
            for b in s.branches:
                names.add(b.value)
        elif isinstance(s, Write):
            names.add(s.value)
    return names


def _accum_defining_scopes(row_stmts: tuple[Stmt, ...]) -> dict[str, set[str]]:
    """For each ``Accum`` name defined in ``row_stmts``, return the set of
    enclosing Loop axis names **outside** the Accum's own reduce Loop —
    i.e. the scope in which the finalized accumulator is usable."""
    result: dict[str, set[str]] = {}

    def walk(stmts: tuple[Stmt, ...], enclosing: tuple[str, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                walk(s.body, enclosing + (s.axis.name,))
            elif isinstance(s, Accum):
                # The Accum is finalized after its enclosing reduce Loop
                # exits — that's one axis up from here.
                scope = set(enclosing[:-1]) if enclosing else set()
                result.setdefault(s.name, scope)

    walk(row_stmts, ())
    return result


def _axes_in_template(template: tuple[Stmt, ...]) -> set[str]:
    """Collect every axis name referenced via ``Var`` inside the template's
    Expr fields (Load indices, Select predicates, Loop axis names)."""
    axes: set[str] = set()

    def walk_expr(expr: object) -> None:
        if isinstance(expr, Var):
            axes.add(expr.name)
            return
        for attr in ("left", "right", "cond", "if_true", "if_false", "expr"):
            child = getattr(expr, attr, None)
            if child is not None:
                walk_expr(child)
        children = getattr(expr, "args", None)
        if isinstance(children, (list, tuple)):
            for c in children:
                walk_expr(c)

    def walk_stmts(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Load):
                for e in s.index:
                    walk_expr(e)
            elif isinstance(s, Select):
                for b in s.branches:
                    walk_expr(b.select)
            elif isinstance(s, Loop):
                axes.add(s.axis.name)
                walk_stmts(s.body)

    walk_stmts(template)
    return axes


def _replay_element_template(
    template: tuple[Stmt, ...],
    axis_sub: dict[str, Expr],
    suffix: str,
    write_value_name: str,
) -> tuple[tuple[Stmt, ...], str]:
    """Emit a per-Load replay of the element-space template.

    Substitutes producer axis Vars per ``axis_sub`` (post-consumer_sigma
    — so consumer renames are already applied in the mapped exprs) and
    freshens every SSA name with ``suffix``. Returns the replay stmts
    plus the final (freshened) SSA name that the dropped target Load
    should alias to.
    """
    ssa_rename: dict[str, str] = {}

    def rn(name: str) -> str:
        return ssa_rename.get(name, name)

    def walk(stmts: tuple[Stmt, ...]) -> list[Stmt]:
        result: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Load):
                new_name = s.name + suffix
                ssa_rename[s.name] = new_name
                result.append(
                    Load(
                        name=new_name,
                        source=s.source,
                        index=tuple(substitute(e, axis_sub) for e in s.index),
                    )
                )
            elif isinstance(s, Assign):
                new_name = s.name + suffix
                ssa_rename[s.name] = new_name
                result.append(Assign(name=new_name, op=s.op, args=tuple(rn(a) for a in s.args)))
            elif isinstance(s, Select):
                new_name = s.name + suffix
                ssa_rename[s.name] = new_name
                result.append(
                    Select(
                        name=new_name,
                        branches=tuple(SelectBranch(value=rn(b.value), select=substitute(b.select, axis_sub)) for b in s.branches),
                    )
                )
            elif isinstance(s, Loop):
                result.append(Loop(axis=s.axis, body=tuple(walk(s.body))))
            else:
                result.append(s)
        return result

    out = tuple(walk(template))
    final_name = ssa_rename.get(write_value_name, write_value_name)
    return out, final_name


# ---------------------------------------------------------------------------
# Producer free-loop reorder (pre-walk alignment)
# ---------------------------------------------------------------------------


def _consumer_depths(body: tuple[Stmt, ...]) -> dict[str, int]:
    """Record the nesting depth of every ``Loop`` in the consumer body
    (pre-order). Duplicate axis names — rare under normal lift/merge —
    resolve to the outermost occurrence."""
    depths: dict[str, int] = {}

    def walk(stmts: tuple[Stmt, ...], depth: int) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                depths.setdefault(s.axis.name, depth)
                walk(s.body, depth + 1)

    walk(body, 0)
    return depths


def _align_producer_free_chain(
    producer_body: tuple[Stmt, ...],
    consumer_body: tuple[Stmt, ...],
    producer_to_consumer_axis: dict[str, str],
    reduce_axis_names: frozenset[str],
) -> tuple[Stmt, ...]:
    """Permute producer's outer free-loop chain so σ-matched axes sit at
    the same depth as their consumer partners.

    The chain is the top run of singleton-body Loops that aren't reduce
    loops. For each axis in the chain, the target depth is the consumer
    depth of its σ partner (producer axes without a Var-valued σ entry
    sort to the end). Stable sort keeps the relative order of axes with
    the same target.
    """
    chain: list[Axis] = []
    current = producer_body
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.axis.name in reduce_axis_names:
            break
        chain.append(loop.axis)
        current = loop.body

    if len(chain) < 2:
        return producer_body

    consumer_depth = _consumer_depths(consumer_body)
    sentinel = len(consumer_depth) + len(chain)

    def target(axis: Axis) -> int:
        c = producer_to_consumer_axis.get(axis.name)
        if c is None:
            return sentinel
        return consumer_depth.get(c, sentinel)

    reordered = sorted(chain, key=target)
    if [a.name for a in reordered] == [a.name for a in chain]:
        return producer_body

    result: tuple[Stmt, ...] = current
    for axis in reversed(reordered):
        result = (Loop(axis=axis, body=result),)
    return result
