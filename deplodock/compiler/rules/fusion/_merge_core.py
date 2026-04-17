"""Core helpers for merging two adjacent ``LoopOp``s.

Given ``producer`` (writes a buffer via a ``Write`` statement) and ``consumer``
(reads that buffer via a ``Port``), ``merge_loop_ops`` produces a single
``LoopOp`` whose iteration space is the consumer's axes plus any producer-only
reduce axes that survive after axis alignment.

The substitution σ maps each producer axis to an ``Expr`` over consumer axes.
It is solved from the equation ``writer.index[k] == reader.index[k]`` at each
output dim. Supported forms: direct ``Var(a)``, ``Var(a) ± c``, and
``Literal(0)`` broadcast slots (no binding). Anything else returns ``None``.

Legality:
- every unbound producer axis must be kind ``"reduce"`` (unbound free axes would
  require replicating the producer per consumer slot — we refuse);
- the merged kernel must have at most one reduce axis (matching today's
  single-reduce CUDA backend).
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
    consumer_port: int,
) -> LoopOp | None:
    """Merge ``producer`` into ``consumer`` across the connecting buffer.

    ``producer_output`` is the ``Write.output`` slot on the producer side (v1:
    always 0 for single-output LoopOps). ``consumer_port`` is the index into
    ``consumer.inputs`` of the Port that reads the producer's buffer.

    Returns ``None`` when the merge is illegal (axis alignment fails, a free
    producer axis would leak, or the merged kernel would have ≥2 reduce axes).
    """
    writes = [s for s in producer.body if isinstance(s, Write) and s.output == producer_output]
    if len(writes) != 1:
        return None
    write = writes[0]

    reader_index = consumer.inputs[consumer_port].index
    producer_axis_names = {a.name for a in producer.axes}
    sigma = _solve_sigma(write.index, reader_index, producer_axis_names)
    if sigma is None:
        return None

    unbound = [a for a in producer.axes if a.name not in sigma]
    if any(a.kind != "reduce" for a in unbound):
        return None

    consumer_reduce = sum(1 for a in consumer.axes if a.kind == "reduce")
    producer_surviving_reduce = sum(1 for a in unbound if a.kind == "reduce")
    if consumer_reduce + producer_surviving_reduce > 1:
        return None

    axis_rename = _fresh_axis_names(unbound, consumer.axes)
    local_rename = _fresh_local_names(producer.locals, consumer.locals)
    ssa_rename = _fresh_ssa_map(producer, consumer, local_rename)

    sigma = {**sigma, **{a.name: Var(axis_rename[a.name]) for a in unbound}}

    merged_axes = tuple(consumer.axes) + tuple(Axis(name=axis_rename[a.name], extent=a.extent, kind=a.kind) for a in unbound)

    merged_inputs, consumer_port_remap, producer_port_remap = _merge_ports(consumer.inputs, consumer_port, producer.inputs, sigma)

    merged_locals = _merge_locals(consumer.locals, producer.locals, local_rename, sigma)

    bridge = _fresh_name("v_bridge", _all_defined_names(producer, consumer, local_rename, ssa_rename))

    # Producer body: apply σ, remap ports, rename locals + SSA; replace the
    # connecting Write with an Assign that binds the bridge SSA name.
    bridge_value = _rename_ssa_arg(write.value, producer_port_remap, local_rename, ssa_rename)
    prod_body = _rewrite_body(
        producer.body,
        sigma=sigma,
        port_remap=producer_port_remap,
        local_rename=local_rename,
        ssa_rename=ssa_rename,
        replace_write={id(write): Assign(name=bridge, op=ElementwiseOp("copy"), args=(bridge_value,))},
    )

    # Consumer body: remap ports, rewrite the consumed Port reference to the bridge.
    cons_body = _rewrite_body(
        consumer.body,
        sigma={},
        port_remap=consumer_port_remap,
        local_rename={},
        ssa_rename={f"${consumer_port}": bridge},
    )

    return LoopOp(
        axes=merged_axes,
        inputs=tuple(merged_inputs),
        locals=tuple(merged_locals),
        body=tuple(prod_body) + tuple(cons_body),
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
    producer_ssa: set[str] = set()
    for stmt in producer.body:
        if isinstance(stmt, (Assign, Select)):
            producer_ssa.add(stmt.name)

    consumer_names: set[str] = {lb.name for lb in consumer.locals}
    for stmt in consumer.body:
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
    names: set[str] = set()
    for op in (producer, consumer):
        for lb in op.locals:
            names.add(lb.name)
        for stmt in op.body:
            if isinstance(stmt, (Assign, Select)):
                names.add(stmt.name)
    names.update(local_rename.values())
    names.update(ssa_rename.values())
    return names


# ---------------------------------------------------------------------------
# Port / local merging
# ---------------------------------------------------------------------------


def _merge_ports(
    consumer_inputs: tuple[Port, ...],
    consumer_port: int,
    producer_inputs: tuple[Port, ...],
    sigma: dict[str, Expr],
) -> tuple[list[Port], dict[int, int], dict[int, int]]:
    merged: list[Port] = []
    consumer_remap: dict[int, int] = {}
    for i, p in enumerate(consumer_inputs):
        if i == consumer_port:
            continue
        merged.append(p)
        consumer_remap[i] = len(merged) - 1

    producer_remap: dict[int, int] = {}
    for j, p in enumerate(producer_inputs):
        merged.append(Port(index=tuple(substitute(e, sigma) for e in p.index)))
        producer_remap[j] = len(merged) - 1

    return merged, consumer_remap, producer_remap


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
