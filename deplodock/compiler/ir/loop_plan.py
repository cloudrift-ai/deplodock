"""LoopOp → KernelPlan: explicit nested-loop view of a nested Loop-tree body.

Walks a ``LoopOp``'s nested ``Loop`` tree to determine iteration spaces,
segment boundaries, and rematerialization sets. The result is a
``KernelPlan`` that codegen walks mechanically (no re-derivation needed)
and that the pretty-printer renders as the loop nest a kernel will execute.

The analysis is backend-agnostic (no CUDA imports).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.loop_ir import Assign, LoopOp, Port, Select, Stmt, Update, Write
from deplodock.compiler.ir.loop_ir import Loop as LoopStmt

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
    body: tuple[Assign | Select, ...]  # this segment's elementwise assigns/selects
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
    """Analyze a kernel's nested-``Loop`` body and produce a ``KernelPlan``.

    Walks the body tree: descends outer free ``Loop`` blocks to find the
    "inner body" (the first level that either has reduce ``Loop`` children
    or no Loop children at all), then classifies each inner-body child —
    reduce ``Loop``s become ``plan.Loop`` steps, loose stmts become prelude
    / post-reduce Inline or Loop-with-stores, direct ``Write`` children
    become ``TrailingWrite``s.
    """
    input_ports = _collect_input_ports(kernel)
    inner_body = _descend_free_loops(kernel.body)
    reduce_loops = [s for s in inner_body if isinstance(s, LoopStmt) and s.axis.kind == "reduce"]
    has_reduce = bool(reduce_loops)

    # --- Reduction geometry (None for flat) ---
    if has_reduce:
        pre_shape = tuple(a.extent for a in kernel.axes)
        reduce_axis_idx = next((i for i, a in enumerate(kernel.axes) if a.kind == "reduce"), 0)
        k_size = int(pre_shape[reduce_axis_idx]) if pre_shape else 1
        n_output = _n_rows(pre_shape, reduce_axis_idx)
    else:
        pre_shape = ()
        reduce_axis_idx = 0
        k_size = 0
        n_output = _numel(out_shape)

    # --- Classify ports as per-element (references reduce axis) vs per-row ---
    if has_reduce:
        reduce_axis_names = {a.name for a in kernel.axes if a.kind == "reduce"}
        per_elem_ports = frozenset(name for name, port in input_ports.items() if _port_references_axis(port, reduce_axis_names))
    else:
        per_elem_ports = frozenset()

    # --- Classify all SSA values as element-space or row-space (tree-wide) ---
    from deplodock.compiler.ir.loop_ir import flatten_body

    flat_all = tuple(flatten_body(kernel.body))
    accumulator_names = {lb.name for lb in kernel.locals if lb.combine is not None}
    row_space: set[str] = {name for name in input_ports if name not in per_elem_ports}
    row_space |= accumulator_names
    elem_space: set[str] = set(per_elem_ports)

    for stmt in flat_all:
        if isinstance(stmt, Assign):
            if any(a in elem_space for a in stmt.args):
                elem_space.add(stmt.name)
            else:
                row_space.add(stmt.name)
        elif isinstance(stmt, Select):
            if all(br.value in row_space for br in stmt.branches):
                row_space.add(stmt.name)
            else:
                elem_space.add(stmt.name)

    # --- Trailing writes: direct Write children of the inner body ---
    trailing_writes: list[TrailingWrite] = []
    for stmt in inner_body:
        if isinstance(stmt, Write):
            trailing_writes.append(TrailingWrite(output=stmt.output, index=stmt.index, value=stmt.value))

    # --- Pointwise: no reduce Loops; collect leaf Assign/Select as one Inline ---
    if not has_reduce:
        inline_body = [s for s in flatten_body(inner_body) if isinstance(s, (Assign, Select))]
        steps: list[Step] = [Inline(body=tuple(inline_body))] if inline_body else []
        return KernelPlan(
            steps=tuple(steps),
            per_elem_ports=per_elem_ports,
            n_output=n_output,
            trailing_writes=tuple(trailing_writes),
        )

    # --- Reduce: build steps by walking inner_body left-to-right. ---
    acc_map = {lb.name: lb for lb in kernel.locals}
    steps = []
    prior_ew: list[Assign] = []
    acc_count = 0

    # Hoist leading row-space Selects (from absorbed multi-source IndexMapOps)
    # into an Inline prelude — computed once per output coord, before any K-loop.
    idx = 0
    prelude_selects: list[Select] = []
    while idx < len(inner_body) and isinstance(inner_body[idx], Select) and inner_body[idx].name in row_space:
        prelude_selects.append(inner_body[idx])
        idx += 1
    if prelude_selects:
        steps.append(Inline(body=tuple(prelude_selects)))

    # Walk remaining children: reduce Loops → Loop steps; trailing stmts → Inline/Loop-with-stores.
    tail_stmts: list[Assign | Select] = []
    tail_writes: list[Write] = []
    for stmt in inner_body[idx:]:
        if isinstance(stmt, LoopStmt) and stmt.axis.kind == "reduce":
            step = _reduce_loop_to_step(stmt, acc_map, acc_count, prior_ew, row_space, pre_shape, reduce_axis_idx, k_size)
            if step is None:
                continue
            steps.append(step)
            if step.accum is not None:
                acc_count += 1
            # Bring this reduce Loop's element-space Assigns into scope for later remat.
            prior_ew.extend(s for s in stmt.body if isinstance(s, Assign))
        elif isinstance(stmt, (Assign, Select)):
            tail_stmts.append(stmt)
        elif isinstance(stmt, Write):
            tail_writes.append(stmt)
        # Non-reduce Loop children at this level — pass through silently; free-axis
        # tiling is a future story and won't happen in current producers.

    # Tail after last reduce Loop.
    if tail_stmts:
        if _touches_elem_space(tail_stmts, elem_space):
            remat = _remat_set(tail_stmts, prior_ew, row_space)
            remat_assigns = tuple(a for a in prior_ew if a.name in remat)
            store_idx: tuple[Expr, ...] = ()
            store_val = ""
            store_out = 0
            if tail_writes:
                w = tail_writes[-1]
                store_idx = w.index
                store_val = w.value
                store_out = w.output
                trailing_writes = [t for t in trailing_writes if not (t.index == w.index and t.value == w.value)]
            steps.append(
                Loop(
                    recompute=remat_assigns,
                    body=tuple(tail_stmts),
                    accum=None,
                    stores_output=bool(tail_writes),
                    store_index=store_idx,
                    store_value=store_val,
                    store_output_idx=store_out,
                    iter_shape=pre_shape,
                    reduce_axis=reduce_axis_idx,
                    k_size=k_size,
                )
            )
        else:
            steps.append(Inline(body=tuple(tail_stmts)))

    return KernelPlan(
        steps=tuple(steps),
        per_elem_ports=per_elem_ports,
        n_output=n_output,
        trailing_writes=tuple(trailing_writes),
    )


def _descend_free_loops(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Descend through outer free ``Loop`` wrappers to the innermost body.

    Keeps descending as long as ``body`` is a single free ``Loop`` — the free
    axes are grid dims already captured in ``LoopOp.axes`` via the property
    walk. Stops at the first level with either multiple children or a non-
    free-Loop child (a reduce Loop, an Assign, a Write, etc.).
    """
    while len(body) == 1 and isinstance(body[0], LoopStmt) and body[0].axis.kind == "free":
        body = body[0].body
    return body


def _reduce_loop_to_step(
    loop: LoopStmt,
    acc_map: dict,
    acc_count: int,
    prior_ew: list[Assign],
    row_space: set[str],
    pre_shape: tuple,
    reduce_axis_idx: int,
    k_size: int,
) -> Loop | None:
    """Convert a reduce ``Loop`` block into a ``plan.Loop`` step.

    Two shapes supported:
    - Body ends in ``Update``: classic reduce segment with an accumulator.
    - Body ends in ``Write`` with no ``Update``: per-K elementwise pass that
      stores each element (``stores_output=True``). Used by softmax's
      writeback sweep.
    """
    body = loop.body
    if not body:
        return None

    updates = [s for s in body if isinstance(s, Update)]
    writes = [s for s in body if isinstance(s, Write)]
    ew = [s for s in body if isinstance(s, (Assign, Select))]

    if updates:
        # Classic reduce segment: last Update drives the accumulator.
        last_update = updates[-1]
        # Elementwise stmts are those before the last Update.
        last_idx = body.index(last_update)
        ew = [s for s in body[:last_idx] if isinstance(s, (Assign, Select))]
        remat = _remat_set(list(body), prior_ew, row_space)
        remat_assigns = tuple(a for a in prior_ew if a.name in remat)
        lb = acc_map[last_update.target]
        combine_fn = lb.combine.fn if lb.combine is not None else "add"
        identity = _literal_value(lb.init)
        accum = Accum(
            var=f"acc{acc_count}",
            fn=combine_fn,
            identity=identity,
            src=last_update.value,
            result=last_update.target,
        )
        return Loop(
            recompute=remat_assigns,
            body=tuple(ew),
            accum=accum,
            stores_output=False,
            iter_shape=pre_shape,
            reduce_axis=reduce_axis_idx,
            k_size=k_size,
        )

    # No Update — per-K elementwise pass. Must have Writes to justify iteration.
    if writes:
        remat = _remat_set(list(body), prior_ew, row_space)
        remat_assigns = tuple(a for a in prior_ew if a.name in remat)
        w = writes[-1]
        return Loop(
            recompute=remat_assigns,
            body=tuple(ew),
            accum=None,
            stores_output=True,
            store_index=w.index,
            store_value=w.value,
            store_output_idx=w.output,
            iter_shape=pre_shape,
            reduce_axis=reduce_axis_idx,
            k_size=k_size,
        )

    return None


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


def _touches_elem_space(stmts: list, elem_space: set[str]) -> bool:
    """Does any stmt in the list reference an element-space value?"""
    local = {s.name for s in stmts}

    def refs(stmt) -> tuple[str, ...]:
        if isinstance(stmt, Assign):
            return stmt.args
        if isinstance(stmt, Select):
            return tuple(br.value for br in stmt.branches)
        return ()

    return any(arg in elem_space and arg not in local for s in stmts for arg in refs(s))


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
        elif isinstance(stmt, Select):
            _seed_from_args(tuple(br.value for br in stmt.branches))
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
]
