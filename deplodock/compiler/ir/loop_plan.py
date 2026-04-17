"""LoopOp → KernelPlan: explicit nested-loop view of a flat SSA body.

Analyzes a flat SSA body to determine iteration spaces, segment
boundaries, and rematerialization sets. The result is a ``KernelPlan``
that codegen walks mechanically (no re-derivation needed) and that the
pretty-printer renders as the loop nest a kernel will execute.

The analysis is backend-agnostic (no CUDA imports).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.loop import Assign, LoopOp, Port, Select, Update, Write

# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Accum:
    """Reduction accumulator emitted outside the K-loop."""

    var: str  # variable name ("acc0")
    fn: str  # combine op name ("add", "max", "mul", "min")
    identity: float
    src: str  # SSA name that feeds the accumulator
    result: str  # LocalBuffer name (= this is what post-reduce code reads)


@dataclass(frozen=True)
class Loop:
    """A K-loop segment: iterates over the reduction dimension."""

    recompute: tuple[Assign, ...]  # prior element-space assigns to re-eval
    body: tuple[Assign, ...]  # this segment's elementwise assigns
    accum: Accum | None  # accumulator (reduce segments)
    stores_output: bool  # True for trailing element segments (store inside loop)
    store_index: tuple[Expr, ...] = ()  # write index when stores_output
    store_value: str = ""  # SSA name to store when stores_output
    store_output_idx: int = 0  # Write.output
    iter_shape: tuple = ()
    reduce_axis: int = 0
    k_size: int = 0


@dataclass(frozen=True)
class Inline:
    """Per-row assigns emitted without a loop."""

    body: tuple[Assign | Select, ...]


type Step = Loop | Inline


@dataclass(frozen=True)
class TrailingWrite:
    """A post-reduce Write emitted once per thread (free point)."""

    output: int
    index: tuple[Expr, ...]
    value: str


@dataclass(frozen=True)
class KernelPlan:
    """Everything codegen needs to emit a reduce/contraction kernel."""

    steps: tuple[Step, ...]
    per_elem_ports: frozenset[str]
    n_output: int  # thread count: elements for flat, rows for reduce
    trailing_writes: tuple[TrailingWrite, ...] = ()


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------


def analyze_kernel(kernel: LoopOp, shapes: dict[str, tuple], out_shape: tuple) -> KernelPlan:
    """Analyze a kernel's SSA body and produce a KernelPlan."""
    input_ports = _collect_input_ports(kernel)
    has_reduce = any(isinstance(s, Update) for s in kernel.body)

    # --- Reduction geometry (None for flat) ---
    if has_reduce:
        pre_shape = tuple(a.extent for a in kernel.axes)
        reduce_axis_idx = next((i for i, a in enumerate(kernel.axes) if a.kind == "reduce"), 0)
        k_size = int(pre_shape[reduce_axis_idx]) if pre_shape else 1
        n_rows = _n_rows(pre_shape, reduce_axis_idx)
        n_output = n_rows
        reduce_axis = reduce_axis_idx
    else:
        pre_shape = ()
        reduce_axis = 0
        k_size = 0
        n_output = _numel(out_shape)

    # --- Classify ports as per-element (references reduce axis) vs per-row ---
    if has_reduce:
        reduce_axis_names = {a.name for a in kernel.axes if a.kind == "reduce"}
        per_elem_ports = frozenset(name for name, port in input_ports.items() if _port_references_axis(port, reduce_axis_names))
    else:
        per_elem_ports = frozenset()  # flat: all ports loaded at idx

    # Filter body: Writes that appear *after* the last Update become trailing_writes;
    # Writes that appear inside the per-reduce-iter region are stores_output flags
    # on the Loop step they trail.
    trailing_writes: list[TrailingWrite] = []
    last_update_idx = -1
    for i, stmt in enumerate(kernel.body):
        if isinstance(stmt, Update):
            last_update_idx = i
    for i, stmt in enumerate(kernel.body):
        if isinstance(stmt, Write):
            if last_update_idx >= 0 and i < last_update_idx:
                # Writes inside the reduce-sweep region are folded into the
                # enclosing Loop step below; don't emit as trailing.
                continue
            trailing_writes.append(TrailingWrite(output=stmt.output, index=stmt.index, value=stmt.value))

    # --- Classify all SSA values as element-space or row-space ---
    accumulator_names = {lb.name for lb in kernel.locals if lb.combine is not None}
    row_space: set[str] = {name for name in input_ports if name not in per_elem_ports}
    row_space |= accumulator_names
    elem_space: set[str] = set(per_elem_ports)

    for stmt in kernel.body:
        if isinstance(stmt, Assign):
            if any(a in elem_space for a in stmt.args):
                elem_space.add(stmt.name)
            else:
                row_space.add(stmt.name)
        elif isinstance(stmt, Select):
            # Select output is row-space when all branch values are row-space
            # (the typical case for absorbed multi-source IndexMapOps, whose
            # branches read row-space ports).
            if all(br.value in row_space for br in stmt.branches):
                row_space.add(stmt.name)
            else:
                elem_space.add(stmt.name)
        # Update/Write don't define new names in our SSA sense.

    # --- Split body into segments at Update boundaries ---
    if not has_reduce:
        # Flat: Writes are trailing; Assigns and Selects fold into one Inline step.
        inline_body = [s for s in kernel.body if isinstance(s, Assign | Select)]
        steps: list[Step] = [Inline(body=tuple(inline_body))] if inline_body else []
        return KernelPlan(
            steps=tuple(steps),
            per_elem_ports=per_elem_ports,
            n_output=n_output,
            trailing_writes=tuple(trailing_writes),
        )

    acc_map = {lb.name: lb for lb in kernel.locals}

    # Hoist leading row-space Select statements (emitted by rule 001 when
    # absorbing multi-source IndexMapOps) into an Inline prelude — computed
    # once per output coord, before any K-loop. They land in ``row_space``
    # above so downstream Loop segments see their SSA name as row-space.
    body_list = list(kernel.body)
    prelude_selects: list[Select] = []
    while body_list and isinstance(body_list[0], Select) and body_list[0].name in row_space:
        prelude_selects.append(body_list[0])
        body_list.pop(0)

    steps: list[Step] = []
    if prelude_selects:
        steps.append(Inline(body=tuple(prelude_selects)))

    segments = _split_at_updates(tuple(body_list))
    acc_count = 0
    prior_ew: list[Assign] = []

    for seg in segments:
        # A segment is a sequence of statements terminating at Update or at end.
        last = seg[-1]
        ew = [s for s in seg[:-1] if isinstance(s, Assign)]
        if isinstance(last, Update):
            remat = _remat_set(seg, prior_ew, row_space)
            remat_assigns = tuple(a for a in prior_ew if a.name in remat)
            lb = acc_map[last.target]
            combine_fn = lb.combine.fn if lb.combine is not None else "add"
            identity = _literal_value(lb.init)
            accum = Accum(
                var=f"acc{acc_count}",
                fn=combine_fn,
                identity=identity,
                src=last.value,
                result=last.target,
            )
            steps.append(
                Loop(
                    recompute=remat_assigns,
                    body=tuple(ew),
                    accum=accum,
                    stores_output=False,
                    iter_shape=pre_shape,
                    reduce_axis=reduce_axis,
                    k_size=k_size,
                )
            )
            acc_count += 1
            prior_ew.extend(ew)
        else:
            # Tail segment after the last Update.
            # If any Assign in the tail references an elem_space value, emit a Loop
            # so the per-element recomputation can reach it.
            assigns = [s for s in seg if isinstance(s, Assign)]
            tail_writes = [s for s in seg if isinstance(s, Write)]
            if assigns and _touches_elem_space(assigns, elem_space):
                remat = _remat_set(seg, prior_ew, row_space)
                remat_assigns = tuple(a for a in prior_ew if a.name in remat)
                # If there are Writes in the tail, emit them inside the Loop body.
                store_idx: tuple[Expr, ...] = ()
                store_val = ""
                store_out = 0
                if tail_writes:
                    w = tail_writes[-1]
                    store_idx = w.index
                    store_val = w.value
                    store_out = w.output
                    # The Write is now inside the loop; remove it from trailing.
                    trailing_writes = [t for t in trailing_writes if not (t.index == w.index and t.value == w.value)]
                steps.append(
                    Loop(
                        recompute=remat_assigns,
                        body=tuple(assigns),
                        accum=None,
                        stores_output=bool(tail_writes),
                        store_index=store_idx,
                        store_value=store_val,
                        store_output_idx=store_out,
                        iter_shape=pre_shape,
                        reduce_axis=reduce_axis,
                        k_size=k_size,
                    )
                )
                prior_ew.extend(assigns)
            elif assigns:
                steps.append(Inline(body=tuple(assigns)))
                prior_ew.extend(assigns)

    return KernelPlan(
        steps=tuple(steps),
        per_elem_ports=per_elem_ports,
        n_output=n_output,
        trailing_writes=tuple(trailing_writes),
    )


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------


def _collect_input_ports(kernel: LoopOp) -> dict[str, Port]:
    """Build $N → Port mapping for the kernel's top-level Port inputs."""
    return {f"${i}": inp for i, inp in enumerate(kernel.inputs)}


def _port_references_axis(port: Port, axis_names: set[str]) -> bool:
    """Does any Expr in ``port.index`` reference a Var with a name in ``axis_names``?"""
    return any(_expr_references_any(e, axis_names) for e in port.index)


def _expr_references_any(expr, names: set[str]) -> bool:
    if isinstance(expr, Var):
        return expr.name in names
    children = []
    for attr in ("left", "right", "cond", "if_true", "if_false"):
        c = getattr(expr, attr, None)
        if c is not None:
            children.append(c)
    for attr in ("args",):
        c = getattr(expr, attr, None)
        if isinstance(c, (list, tuple)):
            children.extend(c)
    return any(_expr_references_any(c, names) for c in children)


def _split_at_updates(body: tuple) -> list[list]:
    """Split SSA body into segments, cutting after each Update."""
    segments: list[list] = []
    current: list = []
    for stmt in body:
        current.append(stmt)
        if isinstance(stmt, Update):
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _touches_elem_space(assigns: list[Assign], elem_space: set[str]) -> bool:
    """Does any assign in the list reference an element-space value?"""
    local = {a.name for a in assigns}
    return any(arg in elem_space and arg not in local for a in assigns for arg in a.args)


# ---------------------------------------------------------------------------
# Worklist rematerialization
# ---------------------------------------------------------------------------


def _remat_set(segment: list, prior_ew: list[Assign], row_space: set[str]) -> set[str]:
    """Compute the set of prior element-space assigns to recompute.

    Backward reachability from segment args (Assign or Update), stopping
    at row-space values and input ports. Each name enters the worklist at
    most once.
    """
    prior_args: dict[str, tuple[str, ...]] = {a.name: a.args for a in prior_ew}

    needed: set[str] = set()
    worklist: deque[str] = deque()

    def _seed_from_args(args: tuple[str, ...]) -> None:
        for arg in args:
            if arg not in row_space and arg in prior_args and arg not in needed:
                needed.add(arg)
                worklist.append(arg)

    for stmt in segment:
        if isinstance(stmt, Assign):
            _seed_from_args(stmt.args)
        elif isinstance(stmt, Update):
            _seed_from_args((stmt.value,))
        elif isinstance(stmt, Write):
            _seed_from_args((stmt.value,))

    while worklist:
        name = worklist.popleft()
        for arg in prior_args.get(name, ()):
            if arg not in needed and arg not in row_space and arg in prior_args:
                needed.add(arg)
                worklist.append(arg)

    return needed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _literal_value(expr) -> float:
    """Extract a scalar value from an Expr (Literal expected) or 0.0."""
    from deplodock.compiler.ir.expr import Literal

    if isinstance(expr, Literal):
        return float(expr.value)
    return 0.0


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _n_rows(pre_reduce_shape: tuple, reduce_axis: int) -> int:
    outer = list(pre_reduce_shape)
    if outer:
        del outer[reduce_axis]
    return _numel(tuple(outer)) if outer else 1


__all__ = [
    "Accum",
    "Inline",
    "KernelPlan",
    "Loop",
    "TrailingWrite",
    "analyze_kernel",
    "pretty_print_plan",
]


# ---------------------------------------------------------------------------
# Pretty-printing: plan → nested-loop text
# ---------------------------------------------------------------------------


def pretty_print_plan(
    loop: LoopOp,
    plan: KernelPlan,
    port_buffers: list[str] | None = None,
    indent: str = "",
) -> str:
    """Render a ``LoopOp`` + ``KernelPlan`` as an explicit nested-loop program.

    ``port_buffers`` maps ``$N → external buffer name`` (from
    ``LoopLaunch.input_names``); when provided, port loads render as
    ``buf[i, j]`` instead of ``$N``.
    """
    accum_by_result = {step.accum.result: step.accum for step in plan.steps if isinstance(step, Loop) and step.accum is not None}

    free_axes = tuple(a for a in loop.axes if a.kind == "free")
    reduce_axes = tuple(a for a in loop.axes if a.kind == "reduce")
    ports = loop.inputs
    buffers = port_buffers or [f"${i}" for i in range(len(ports))]

    def render_arg(name: str) -> str:
        """Render an SSA arg: ``$N`` → ``buf[...]``; accumulator name → acc var; else passthrough."""
        if name.startswith("$"):
            try:
                pi = int(name[1:])
            except ValueError:
                return name
            if 0 <= pi < len(ports):
                idx = ", ".join(render_expr(e) for e in ports[pi].index)
                return f"{buffers[pi]}[{idx}]" if idx else buffers[pi]
            return name
        if name in accum_by_result:
            return accum_by_result[name].var
        return name

    def render_assign(a: Assign) -> str:
        args = ", ".join(render_arg(arg) for arg in a.args)
        return f"{a.name} = {a.op.fn}({args})"

    def render_accum_update(accum: Accum) -> str:
        return f"{accum.var} = {accum.fn}({accum.var}, {render_arg(accum.src)})"

    def render_write(output: int, index: tuple[Expr, ...], value: str) -> str:
        idx = ", ".join(render_expr(e) for e in index)
        # Prefer the real output buffer name if we can map it (same convention as ports
        # but there's no direct mapping here — codegen uses LoopLaunch.output_name).
        # For clarity render as ``out<N>[...]`` which matches the Write IR.
        return f"out{output}[{idx}] = {render_arg(value)}"

    lines: list[str] = []
    lines.append(f"{indent}# iter: {_format_axes(free_axes, reduce_axes)}")
    lines.append("")

    body_indent = indent + "    "
    for fa in free_axes:
        lines.append(f"{indent}for {fa.name} in 0..{fa.extent}:")

    # Collect accumulator inits (from LoopOp.locals), emitted once at the top of
    # the free-axis body in plan order.
    init_lines: list[str] = []
    acc_seen: set[str] = set()
    for step in plan.steps:
        if isinstance(step, Loop) and step.accum is not None and step.accum.var not in acc_seen:
            acc = step.accum
            acc_seen.add(acc.var)
            init_lines.append(f"{body_indent}{acc.var} = {_fmt_identity(acc.identity)}  # {acc.result} via {acc.fn}")
    if init_lines:
        lines.extend(init_lines)

    for step in plan.steps:
        lines.append("")
        if isinstance(step, Loop):
            if step.recompute:
                # Comment marking the recompute block; statements are elided into the loop body.
                pass
            header = f"{body_indent}for {reduce_axes[0].name} in 0..{step.k_size}:" if reduce_axes else f"{body_indent}# inline step"
            if isinstance(step, Loop) and not reduce_axes:
                # Degenerate: Loop step but no reduce axis; fall back to labeled block.
                header = f"{body_indent}# elem step"
            lines.append(header)
            inner = body_indent + "    "
            for a in step.recompute:
                lines.append(f"{inner}{render_assign(a)}  # recompute")
            for a in step.body:
                lines.append(f"{inner}{render_assign(a)}")
            if step.accum is not None:
                lines.append(f"{inner}{render_accum_update(step.accum)}  # accum {step.accum.result} via {step.accum.fn}")
            if step.stores_output:
                lines.append(f"{inner}{render_write(step.store_output_idx, step.store_index, step.store_value)}")
        elif isinstance(step, Inline):
            for stmt in step.body:
                if isinstance(stmt, Assign):
                    lines.append(f"{body_indent}{render_assign(stmt)}")
                else:  # Select
                    for bi, br in enumerate(stmt.branches):
                        kw = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                        lines.append(f"{body_indent}{kw} {render_arg(br.value)} when ({render_expr(br.select)})")

    for tw in plan.trailing_writes:
        lines.append("")
        lines.append(f"{body_indent}{render_write(tw.output, tw.index, tw.value)}")

    return "\n".join(lines)


def _format_axes(free_axes: tuple, reduce_axes: tuple) -> str:
    parts = [f"{a.name}:{a.extent} free" for a in free_axes]
    parts.extend(f"{a.name}:{a.extent} reduce" for a in reduce_axes)
    return ", ".join(parts) if parts else "(none)"


def _fmt_identity(v: float) -> str:
    if v == float("inf"):
        return "+inf"
    if v == float("-inf"):
        return "-inf"
    return repr(v)
