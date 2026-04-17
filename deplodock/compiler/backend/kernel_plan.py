"""Pre-codegen analysis: LoopOp → KernelPlan.

Analyzes a flat SSA body to determine iteration spaces, segment
boundaries, and rematerialization sets. The result is a ``KernelPlan``
that codegen walks mechanically — no re-derivation needed.

The analysis is backend-agnostic (no CUDA imports).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.loop import Assign, LoopOp, Port, Update, Write

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

    body: tuple[Assign, ...]


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
        # Update/Write don't define new names in our SSA sense.

    # --- Split body into segments at Update boundaries ---
    if not has_reduce:
        # Flat: Writes are trailing, everything else is Inline.
        assigns_only = [s for s in kernel.body if isinstance(s, Assign)]
        steps: list[Step] = [Inline(body=tuple(assigns_only))] if assigns_only else []
        return KernelPlan(
            steps=tuple(steps),
            per_elem_ports=per_elem_ports,
            n_output=n_output,
            trailing_writes=tuple(trailing_writes),
        )

    acc_map = {lb.name: lb for lb in kernel.locals}
    segments = _split_at_updates(kernel.body)
    steps = []
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


__all__ = ["Accum", "Inline", "KernelPlan", "Loop", "TrailingWrite", "analyze_kernel"]
