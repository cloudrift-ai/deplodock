"""Core helpers for merging two adjacent ``LoopOp``s.

Given ``producer`` (writes a buffer via a ``Write`` statement) and ``consumer``
(reads that buffer via one or more ``Port``s), ``merge_loop_ops`` produces a
single ``LoopOp`` whose iteration space is the consumer's axes plus any
producer-only reduce axes that survive after axis alignment.

The substitution σ maps each producer axis to an ``Expr`` over consumer axes.
For each consumer port that reads the producer, σ is solved from the equation
``writer.index[k] == reader.index[k]`` at each output dim. Supported writer
forms: direct ``Var(a)``, ``Var(a) ± c``, and ``Literal(0)`` broadcast slots.

When the consumer reads the producer at *multiple* distinct reader indices
(e.g. softmax, where the sum-reduce kernel reads the max+sub+exp kernel at
``[row, col]`` for the post-reduce divide and ``[row, k]`` for the sum
sweep), each port gets its own σ_k. Producer axes bound identically across
all σ_k become part of a shared σ_common — the producer's pre-reduce body
(up to and including the last ``Update``) is emitted once using σ_common.
The producer's post-reduce body is instantiated *per σ_k*, with distinct
SSA names and a distinct bridge value that the matching consumer port ref
is rewritten to.

Legality:
- every unbound producer axis must be kind ``"reduce"``;
- the merged kernel must have at most one reduce axis (single-reduce CUDA
  backend);
- the producer's pre-reduce statements must not reference axes that are
  bound differently across σ_k (doing so would require replicating the
  Update, which is semantically ambiguous).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.expr import BinOp, Cast, Expr, Literal, Var, substitute
from deplodock.compiler.ir.loop import (
    Assign,
    Axis,
    LocalBuffer,
    LoopOp,
    Port,
    Select,
    SelectBranch,
    Stmt,
    Update,
    Write,
)
from deplodock.compiler.ir.tensor import ElementwiseOp

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_loop_ops(
    producer: LoopOp,
    producer_output: int,
    consumer: LoopOp,
    consumer_port: int | list[int],
    axis_aliases: dict[str, str] | None = None,
) -> LoopOp | None:
    """Merge ``producer`` into ``consumer`` across the connecting buffer.

    ``consumer_port`` is either a single port index (one consumer read of the
    producer) or a list of indices (consumer reads the producer at several
    distinct reader patterns — e.g. softmax's sum+div kernel).

    ``axis_aliases`` optionally identifies unbound producer reduce axes with
    existing consumer reduce axes. If ``axis_aliases[p_name] == c_name``,
    ``p_name`` is treated as an alias of ``c_name`` during σ substitution.

    Returns ``None`` when the merge is illegal.
    """
    consumer_ports = [consumer_port] if isinstance(consumer_port, int) else list(consumer_port)
    if not consumer_ports:
        return None
    if len(set(consumer_ports)) != len(consumer_ports):
        return None
    for cp in consumer_ports:
        if cp < 0 or cp >= len(consumer.inputs):
            return None

    # Flatten nested Loop blocks — merge operates on the linear stmt sequence.
    # The output body gets re-nested via flat_body_to_nested before return.
    from deplodock.compiler.ir.loop import flatten_body

    producer_flat = tuple(flatten_body(producer.body))
    consumer_flat = tuple(flatten_body(consumer.body))

    writes = [s for s in producer_flat if isinstance(s, Write) and s.output == producer_output]
    if len(writes) != 1:
        return None
    write = writes[0]

    producer_axis_names = {a.name for a in producer.axes}
    consumer_axis_by_name = {a.name: a for a in consumer.axes}

    # Solve σ per consumer port, applying axis aliases.
    sigmas: dict[int, dict[str, Expr]] = {}
    for cp in consumer_ports:
        s = _solve_sigma(write.index, consumer.inputs[cp].index, producer_axis_names)
        if s is None:
            return None
        if axis_aliases:
            for p_name, c_name in axis_aliases.items():
                if p_name in s:
                    continue
                if p_name not in producer_axis_names:
                    continue
                if c_name not in consumer_axis_by_name:
                    continue
                s[p_name] = Var(c_name)
        sigmas[cp] = s

    # Classify producer axes: bound_common (agreed across σ_k), specific
    # (bound by every σ_k but not agreeing), unbound (not bound by some σ_k).
    sigma_list = [sigmas[cp] for cp in consumer_ports]
    bound_in_all = set(sigma_list[0].keys())
    for s in sigma_list[1:]:
        bound_in_all &= set(s.keys())

    bound_common: dict[str, Expr] = {}
    specific_names: set[str] = set()
    for name in bound_in_all:
        first_expr = sigma_list[0][name]
        if all(s[name] == first_expr for s in sigma_list[1:]):
            bound_common[name] = first_expr
        else:
            specific_names.add(name)

    unbound_axes = [a for a in producer.axes if a.name not in bound_in_all]
    if any(a.kind != "reduce" for a in unbound_axes):
        return None

    consumer_reduce_count = sum(1 for a in consumer.axes if a.kind == "reduce")
    if consumer_reduce_count + len(unbound_axes) > 1:
        return None

    # Split producer body at the last Update. Pre-reduce includes the Update.
    last_update_idx = -1
    for i, stmt in enumerate(producer_flat):
        if isinstance(stmt, Update):
            last_update_idx = i
    pre_reduce_stmts = producer_flat[: last_update_idx + 1]
    post_reduce_stmts = producer_flat[last_update_idx + 1 :]

    # Pre-reduce stmts may only reference bound_common / aliased / unbound axes —
    # never specific_names (which disagree across σ_k). Enforcing this keeps the
    # Update emission unambiguous. "Reference" here includes the axes that
    # appear in the index Expr of any Port the stmt reads via `$N`, because
    # port accesses materialize under σ_k substitution.
    if specific_names:
        # Collect per-port axis dependencies.
        port_axes: dict[int, set[str]] = {}
        for j, port in enumerate(producer.inputs):
            used: set[str] = set()
            for e in port.index:
                _collect_expr_axes(e, used)
            port_axes[j] = used
        for stmt in pre_reduce_stmts:
            if _stmt_refs_axes(stmt, specific_names):
                return None
            if _stmt_port_refs_axes(stmt, port_axes, specific_names):
                return None

    axis_rename = _fresh_axis_names(unbound_axes, consumer.axes)
    local_rename = _fresh_local_names(producer.locals, consumer.locals)
    ssa_rename_common = _fresh_ssa_map(producer, consumer, local_rename)

    # Augment each σ_k with bindings for unbound axes (they survive as new axes).
    unbound_binding = {a.name: Var(axis_rename[a.name]) for a in unbound_axes}
    for s in sigma_list:
        s.update(unbound_binding)
    # Include unbound bindings in the shared σ used for pre-reduce emission.
    pre_reduce_sigma = dict(bound_common)
    pre_reduce_sigma.update(unbound_binding)

    merged_axes = tuple(consumer.axes) + tuple(Axis(name=axis_rename[a.name], extent=a.extent, kind=a.kind) for a in unbound_axes)
    merged_locals = _merge_locals(consumer.locals, producer.locals, local_rename, pre_reduce_sigma)

    # Build merged ports: consumer ports (minus consumer_ports) first, then
    # a set of producer ports per σ_k (σ_k-substituted).
    merged_ports: list[Port] = []
    consumer_port_remap: dict[int, int] = {}
    consumer_port_set = set(consumer_ports)
    for i, p in enumerate(consumer.inputs):
        if i in consumer_port_set:
            continue
        merged_ports.append(p)
        consumer_port_remap[i] = len(merged_ports) - 1

    producer_port_remap_per_cp: dict[int, dict[int, int]] = {}
    for cp in consumer_ports:
        s = sigmas[cp]
        remap: dict[int, int] = {}
        for j, p in enumerate(producer.inputs):
            merged_ports.append(Port(index=tuple(substitute(e, s) for e in p.index)))
            remap[j] = len(merged_ports) - 1
        producer_port_remap_per_cp[cp] = remap

    # Track all defined names in the merged kernel for fresh-name generation.
    taken_names = _all_defined_names(producer, consumer, local_rename, ssa_rename_common)

    # Emit producer pre-reduce body once using σ_common + unbound bindings.
    # Any σ_k has matching values for bound_common axes (by definition), so
    # picking the first σ_k's port remap is safe.
    first_cp = consumer_ports[0]
    body: list[Stmt] = []
    pre_reduce_rewritten = _rewrite_body(
        pre_reduce_stmts,
        sigma=pre_reduce_sigma,
        port_remap=producer_port_remap_per_cp[first_cp],
        local_rename=local_rename,
        ssa_rename=ssa_rename_common,
    )
    body.extend(pre_reduce_rewritten)
    for stmt in pre_reduce_rewritten:
        if isinstance(stmt, (Assign, Select)):
            taken_names.add(stmt.name)

    # Lazily instantiate the producer's post-reduce body per σ_k when the
    # consumer first references the matching $cp. The bridge SSA name is
    # returned so the caller can rewrite the consumer reference.
    instantiated_bridges: dict[int, str] = {}

    def instantiate_for_cp(cp: int) -> str:
        if cp in instantiated_bridges:
            return instantiated_bridges[cp]
        s = sigmas[cp]
        # Fresh SSA rename for this instantiation, inheriting common collisions.
        ssa_rename_k: dict[str, str] = dict(ssa_rename_common)
        for stmt in post_reduce_stmts:
            if isinstance(stmt, (Assign, Select)):
                base = stmt.name
                renamed_common = ssa_rename_common.get(base, base)
                fresh = _fresh_name(f"{renamed_common}_cp{cp}", taken_names)
                ssa_rename_k[base] = fresh
                taken_names.add(fresh)

        bridge = _fresh_name(f"v_bridge_cp{cp}", taken_names)
        taken_names.add(bridge)
        bridge_value = _rename_ssa_arg(write.value, producer_port_remap_per_cp[cp], local_rename, ssa_rename_k)
        post_rewritten = _rewrite_body(
            post_reduce_stmts,
            sigma=s,
            port_remap=producer_port_remap_per_cp[cp],
            local_rename=local_rename,
            ssa_rename=ssa_rename_k,
            replace_write={id(write): Assign(name=bridge, op=ElementwiseOp("copy"), args=(bridge_value,))},
        )
        body.extend(post_rewritten)
        instantiated_bridges[cp] = bridge
        return bridge

    # Walk consumer body, rewriting $i references. For consumed ports, trigger
    # σ_k instantiation (which appends stmts to body before the current one).
    def rewrite_arg(arg: str) -> str:
        if not arg.startswith("$"):
            return arg
        try:
            idx = int(arg[1:])
        except ValueError:
            return arg
        if idx in consumer_port_set:
            return instantiate_for_cp(idx)
        if idx in consumer_port_remap:
            return f"${consumer_port_remap[idx]}"
        return arg

    for stmt in consumer_flat:
        if isinstance(stmt, Assign):
            new_args = tuple(rewrite_arg(a) for a in stmt.args)
            body.append(Assign(name=stmt.name, op=stmt.op, args=new_args))
        elif isinstance(stmt, Update):
            new_value = rewrite_arg(stmt.value)
            body.append(Update(target=stmt.target, value=new_value))
        elif isinstance(stmt, Write):
            body.append(Write(output=stmt.output, index=stmt.index, value=rewrite_arg(stmt.value)))
        elif isinstance(stmt, Select):
            new_branches = tuple(SelectBranch(value=rewrite_arg(br.value), select=br.select) for br in stmt.branches)
            body.append(Select(name=stmt.name, branches=new_branches))

    # Re-nest the flat merged body into the nested Loop-block form.
    from deplodock.compiler.ir.loop import flat_body_to_nested

    nested = flat_body_to_nested(tuple(merged_axes), tuple(body))
    return LoopOp(
        inputs=tuple(merged_ports),
        locals=tuple(merged_locals),
        body=nested,
    )


# ---------------------------------------------------------------------------
# σ solver
# ---------------------------------------------------------------------------


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> dict[str, Expr] | None:
    """Solve ``writer[k] == reader[k]`` for producer axes at each dim k."""
    if len(writer) != len(reader):
        return None
    sigma: dict[str, Expr] = {}
    for w, r in zip(writer, reader, strict=True):
        binding = _bind_axis(w, r, producer_axes)
        if binding is None:
            return None
        axis_name, expr = binding
        if axis_name is None:
            continue
        if axis_name in sigma:
            if sigma[axis_name] != expr:
                return None
        else:
            sigma[axis_name] = expr
    return sigma


def _bind_axis(
    writer: Expr,
    reader: Expr,
    producer_axes: set[str],
) -> tuple[str | None, Expr] | None:
    """Return ``(axis_name, reader_expr)`` binding, or ``(None, _)`` if the
    writer entry fixes a constant (no binding needed), or ``None`` if the
    writer form is unsupported."""
    if isinstance(writer, Literal):
        return (None, writer)
    if isinstance(writer, Var) and writer.name in producer_axes:
        return (writer.name, reader)
    if isinstance(writer, Cast):
        return _bind_axis(writer.expr, reader, producer_axes)
    if isinstance(writer, BinOp) and writer.op in ("+", "-"):
        left, right = writer.left, writer.right
        if isinstance(left, Var) and left.name in producer_axes and isinstance(right, Literal):
            # Var ± c == reader  =>  Var = reader ∓ c
            inverse = "-" if writer.op == "+" else "+"
            return (left.name, BinOp(inverse, reader, right))
        if isinstance(right, Var) and right.name in producer_axes and isinstance(left, Literal):
            if writer.op == "+":
                # c + Var == reader  =>  Var = reader - c
                return (right.name, BinOp("-", reader, left))
            # c - Var == reader  =>  Var = c - reader
            return (right.name, BinOp("-", left, reader))
    return None


# ---------------------------------------------------------------------------
# Name hygiene
# ---------------------------------------------------------------------------


def _fresh_axis_names(to_rename: list[Axis], taken: tuple[Axis, ...]) -> dict[str, str]:
    used = {a.name for a in taken}
    result: dict[str, str] = {}
    for a in to_rename:
        name = a.name if a.name not in used else _fresh_name(a.name, used)
        result[a.name] = name
        used.add(name)
    return result


def _fresh_local_names(
    to_rename: tuple[LocalBuffer, ...],
    taken: tuple[LocalBuffer, ...],
) -> dict[str, str]:
    used = {lb.name for lb in taken}
    result: dict[str, str] = {}
    for lb in to_rename:
        name = lb.name if lb.name not in used else _fresh_name(lb.name, used)
        result[lb.name] = name
        used.add(name)
    return result


def _fresh_ssa_map(
    producer: LoopOp,
    consumer: LoopOp,
    local_rename: dict[str, str],
) -> dict[str, str]:
    """Rename producer SSA names (Assign / Select outputs) that collide with
    consumer names. Does not rename locals — those are handled separately."""
    from deplodock.compiler.ir.loop import flatten_body

    producer_ssa: set[str] = set()
    for stmt in flatten_body(producer.body):
        if isinstance(stmt, (Assign, Select)):
            producer_ssa.add(stmt.name)

    consumer_names: set[str] = {lb.name for lb in consumer.locals}
    for stmt in flatten_body(consumer.body):
        if isinstance(stmt, (Assign, Select)):
            consumer_names.add(stmt.name)

    # Renamed producer locals also occupy space in the merged namespace.
    taken = producer_ssa | consumer_names | set(local_rename.values())
    result: dict[str, str] = {}
    for name in producer_ssa:
        if name in consumer_names:
            fresh = _fresh_name(f"{name}_p", taken)
            result[name] = fresh
            taken.add(fresh)
    return result


def _fresh_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    for suffix in range(1, 10000):
        cand = f"{base}_{suffix}"
        if cand not in taken:
            return cand
    raise ValueError(f"Could not find fresh name for {base!r}")


def _all_defined_names(
    producer: LoopOp,
    consumer: LoopOp,
    local_rename: dict[str, str],
    ssa_rename: dict[str, str],
) -> set[str]:
    from deplodock.compiler.ir.loop import flatten_body

    names: set[str] = set()
    for op in (producer, consumer):
        for lb in op.locals:
            names.add(lb.name)
        for stmt in flatten_body(op.body):
            if isinstance(stmt, (Assign, Select)):
                names.add(stmt.name)
    names.update(local_rename.values())
    names.update(ssa_rename.values())
    return names


# ---------------------------------------------------------------------------
# Port / local merging
# ---------------------------------------------------------------------------


def _merge_locals(
    consumer_locals: tuple[LocalBuffer, ...],
    producer_locals: tuple[LocalBuffer, ...],
    local_rename: dict[str, str],
    sigma: dict[str, Expr],
) -> list[LocalBuffer]:
    result: list[LocalBuffer] = list(consumer_locals)
    for lb in producer_locals:
        new_name = local_rename.get(lb.name, lb.name)
        new_init = substitute(lb.init, sigma) if lb.init is not None else None
        result.append(replace(lb, name=new_name, init=new_init))
    return result


# ---------------------------------------------------------------------------
# Axis-reference helpers
# ---------------------------------------------------------------------------


def _stmt_refs_axes(stmt: Stmt, axis_names: set[str]) -> bool:
    """Return True if ``stmt`` references any axis name in ``axis_names``."""
    if isinstance(stmt, Write):
        return any(_expr_refs_axes(e, axis_names) for e in stmt.index)
    if isinstance(stmt, Select):
        return any(_expr_refs_axes(br.select, axis_names) for br in stmt.branches)
    # Assign and Update don't reference axes directly — their args are SSA
    # names. Axis references only live in Port index exprs and in Write /
    # Select expressions.
    return False


def _expr_refs_axes(expr: Expr, axis_names: set[str]) -> bool:
    if isinstance(expr, Var):
        return expr.name in axis_names
    if isinstance(expr, BinOp):
        return _expr_refs_axes(expr.left, axis_names) or _expr_refs_axes(expr.right, axis_names)
    if isinstance(expr, Cast):
        return _expr_refs_axes(expr.expr, axis_names)
    return False


def _collect_expr_axes(expr: Expr, out: set[str]) -> None:
    if isinstance(expr, Var):
        out.add(expr.name)
    elif isinstance(expr, BinOp):
        _collect_expr_axes(expr.left, out)
        _collect_expr_axes(expr.right, out)
    elif isinstance(expr, Cast):
        _collect_expr_axes(expr.expr, out)


def _stmt_port_refs_axes(stmt: Stmt, port_axes: dict[int, set[str]], axis_names: set[str]) -> bool:
    """Return True if any ``$N`` reference in ``stmt`` reads a Port whose
    index expressions include any axis in ``axis_names``."""
    args: tuple[str, ...] = ()
    if isinstance(stmt, Assign):
        args = stmt.args
    elif isinstance(stmt, Update):
        args = (stmt.value,)
    elif isinstance(stmt, Write):
        args = (stmt.value,)
    elif isinstance(stmt, Select):
        args = tuple(br.value for br in stmt.branches)
    for arg in args:
        if not arg.startswith("$"):
            continue
        try:
            idx = int(arg[1:])
        except ValueError:
            continue
        axes_used = port_axes.get(idx, set())
        if axes_used & axis_names:
            return True
    return False


# ---------------------------------------------------------------------------
# Body rewrite
# ---------------------------------------------------------------------------


def _rewrite_body(
    body: tuple[Stmt, ...],
    *,
    sigma: dict[str, Expr],
    port_remap: dict[int, int],
    local_rename: dict[str, str],
    ssa_rename: dict[str, str],
    replace_write: dict[int, Stmt] | None = None,
) -> list[Stmt]:
    """Apply σ to every Expr, remap ``$N`` refs, rename locals and SSA names.

    ``replace_write`` maps ``id(Write)`` → replacement statement; used to swap
    the connecting Write for an Assign binding the bridge SSA value.
    """
    rep = replace_write or {}
    result: list[Stmt] = []

    def rn(arg: str) -> str:
        return _rename_ssa_arg(arg, port_remap, local_rename, ssa_rename)

    for stmt in body:
        if isinstance(stmt, Assign):
            result.append(
                Assign(
                    name=ssa_rename.get(stmt.name, stmt.name),
                    op=stmt.op,
                    args=tuple(rn(a) for a in stmt.args),
                )
            )
        elif isinstance(stmt, Update):
            result.append(
                Update(
                    target=local_rename.get(stmt.target, stmt.target),
                    value=rn(stmt.value),
                )
            )
        elif isinstance(stmt, Write):
            if id(stmt) in rep:
                result.append(rep[id(stmt)])
            else:
                result.append(
                    Write(
                        output=stmt.output,
                        index=tuple(substitute(e, sigma) for e in stmt.index),
                        value=rn(stmt.value),
                    )
                )
        elif isinstance(stmt, Select):
            result.append(
                Select(
                    name=ssa_rename.get(stmt.name, stmt.name),
                    branches=tuple(SelectBranch(value=rn(br.value), select=substitute(br.select, sigma)) for br in stmt.branches),
                )
            )
    return result


def _rename_ssa_arg(
    arg: str,
    port_remap: dict[int, int],
    local_rename: dict[str, str],
    ssa_rename: dict[str, str],
) -> str:
    """Rewrite a body-statement argument reference under the merge's rename maps."""
    if arg in ssa_rename:
        return ssa_rename[arg]
    if arg.startswith("$"):
        try:
            idx = int(arg[1:])
        except ValueError:
            return arg
        if idx in port_remap:
            return f"${port_remap[idx]}"
        return arg
    if arg in local_rename:
        return local_rename[arg]
    return arg
