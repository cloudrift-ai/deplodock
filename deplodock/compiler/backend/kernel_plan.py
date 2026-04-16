"""Pre-codegen analysis: KernelOp → KernelPlan.

Analyzes a flat SSA body to determine iteration spaces, segment
boundaries, and rematerialization sets. The result is a ``KernelPlan``
that codegen walks mechanically — no re-derivation needed.

The analysis is backend-agnostic (no CUDA imports).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.block import Assign, Combine, KernelOp, Mux, Port
from deplodock.compiler.ir.tensor import ReduceOp

# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Accum:
    """Reduction accumulator emitted outside the K-loop."""

    var: str  # variable name ("acc0")
    fn: str  # "sum" | "max" | "prod"
    identity: float  # 0.0 | -1e30 | 1.0
    src: str  # SSA name that feeds the accumulator
    result: str  # SSA name of the ReduceOp Assign


@dataclass(frozen=True)
class Loop:
    """A K-loop segment: iterates over the reduction dimension."""

    recompute: tuple[Assign, ...]  # prior element-space assigns to re-eval
    body: tuple[Assign, ...]  # this segment's elementwise assigns
    accum: Accum | None  # accumulator (reduce segments)
    stores_output: bool  # True for trailing element segments (store inside loop)
    iter_shape: tuple  # shape of the iteration space (pre-reduce broadcast shape)
    reduce_axis: int  # which dim of iter_shape the loop iterates over
    k_size: int  # trip count of the loop


@dataclass(frozen=True)
class Inline:
    """Per-row assigns emitted without a loop."""

    body: tuple[Assign, ...]


type Step = Loop | Inline


@dataclass(frozen=True)
class KernelPlan:
    """Everything codegen needs to emit a reduce/contraction kernel."""

    steps: tuple[Step, ...]
    per_elem_ports: frozenset[str]
    n_output: int  # thread count: elements for flat, rows for reduce
    stores_final: bool  # store last value after all loops


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------


def analyze_kernel(kernel: KernelOp, shapes: dict[str, tuple], out_shape: tuple) -> KernelPlan:
    """Analyze a kernel's SSA body and produce a KernelPlan.

    Works for both flat (pointwise) and reduce kernels.
    """
    # Build $N → Port mapping.
    input_ports: dict[str, Port] = {}
    port_idx = 0
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            input_ports[f"${port_idx}"] = inp
            port_idx += 1
    has_reduce = any(isinstance(a.op, ReduceOp) for a in kernel.body)

    # --- Reduction geometry (None for flat) ---
    if has_reduce:
        ssa_shapes = kernel.infer_shapes(shapes)
        first_reduce = next(a for a in kernel.body if isinstance(a.op, ReduceOp))
        pre_shape = tuple(int(d) for d in ssa_shapes.get(first_reduce.args[0], ()))
        reduce_axis = first_reduce.op.axis % len(pre_shape) if pre_shape else 0
        k_size = int(pre_shape[reduce_axis]) if pre_shape else 1
        n_rows = _n_rows(pre_shape, reduce_axis)
        n_output = n_rows
    else:
        pre_shape = ()
        reduce_axis = 0
        k_size = 0
        n_rows = 0
        n_output = _numel(out_shape)

    # --- Classify ports ---
    if has_reduce:
        per_elem_ports = frozenset(name for name, port in input_ports.items() if _numel(_dollar_port_shape(name, port, shapes)) > n_rows)
    else:
        per_elem_ports = frozenset()  # flat: all ports loaded at idx

    # --- Classify all SSA values as element-space or row-space ---
    row_space: set[str] = {name for name in input_ports if name not in per_elem_ports}
    elem_space: set[str] = set(per_elem_ports)

    for assign in kernel.body:
        if isinstance(assign.op, ReduceOp):
            row_space.add(assign.name)
        elif any(a in elem_space for a in assign.args):
            elem_space.add(assign.name)
        else:
            row_space.add(assign.name)

    # --- Split body at ReduceOps and build steps ---
    if not has_reduce:
        # Flat: single inline step with the entire body.
        steps: list[Step] = [Inline(body=tuple(kernel.body))] if kernel.body else []
        stores_final = True
        return KernelPlan(
            steps=tuple(steps),
            per_elem_ports=per_elem_ports,
            n_output=n_output,
            stores_final=stores_final,
        )

    segments = _split_at_reduces(kernel.body)
    steps: list[Step] = []
    acc_count = 0
    prior_ew: list[Assign] = []

    for seg in segments:
        last = seg[-1]
        has_reduce = isinstance(last.op, ReduceOp)
        ew = seg[:-1] if has_reduce else seg

        if has_reduce:
            remat = _remat_set(seg, prior_ew, row_space)
            remat_assigns = tuple(a for a in prior_ew if a.name in remat)
            accum = Accum(
                var=f"acc{acc_count}",
                fn=last.op.fn,
                identity=_identity(last.op.fn),
                src=last.args[0],
                result=last.name,
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
        elif _touches_elem_space(ew, elem_space):
            remat = _remat_set(seg, prior_ew, row_space)
            remat_assigns = tuple(a for a in prior_ew if a.name in remat)
            steps.append(
                Loop(
                    recompute=remat_assigns,
                    body=tuple(ew),
                    accum=None,
                    stores_output=True,
                    iter_shape=pre_shape,
                    reduce_axis=reduce_axis,
                    k_size=k_size,
                )
            )
        else:
            steps.append(Inline(body=tuple(ew)))

        prior_ew.extend(ew)

    # Does the last SSA value need a per-row store after all loops?
    last_assign = kernel.body[-1] if kernel.body else None
    stores_final = last_assign is not None and last_assign.name in row_space

    return KernelPlan(
        steps=tuple(steps),
        per_elem_ports=per_elem_ports,
        n_output=n_output,
        stores_final=stores_final,
    )


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------


def _split_at_reduces(body: tuple[Assign, ...]) -> list[list[Assign]]:
    """Split SSA body into segments, cutting after each ReduceOp."""
    segments: list[list[Assign]] = []
    current: list[Assign] = []
    for assign in body:
        current.append(assign)
        if isinstance(assign.op, ReduceOp):
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


def _remat_set(segment: list[Assign], prior_ew: list[Assign], row_space: set[str]) -> set[str]:
    """Compute the set of prior element-space assigns to recompute.

    Backward reachability from segment args, stopping at row-space
    values and input ports. Each name enters the worklist at most once.
    """
    prior_args: dict[str, tuple[str, ...]] = {a.name: a.args for a in prior_ew}

    needed: set[str] = set()
    worklist: deque[str] = deque()

    # Seed: direct args of this segment that are prior element-space values.
    for assign in segment:
        for arg in assign.args:
            if arg not in row_space and arg in prior_args and arg not in needed:
                needed.add(arg)
                worklist.append(arg)

    # Expand backward through use-def chains.
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


def _identity(fn: str) -> float:
    return {"sum": 0.0, "max": -1e30, "prod": 1.0}.get(fn, 0.0)


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _n_rows(pre_reduce_shape: tuple, reduce_axis: int) -> int:
    outer = list(pre_reduce_shape)
    if outer:
        del outer[reduce_axis]
    return _numel(tuple(outer)) if outer else 1


def _dollar_port_shape(key: str, port, shapes: dict) -> tuple:
    """Get port shape for a $N-keyed port."""
    if isinstance(port, Port) and port.indexmap is not None:
        return tuple(port.indexmap.out_shape)
    return tuple(shapes.get(key, ()))


def _port_shape(inp, shapes: dict) -> tuple:
    """Get port shape from indexmap or shapes dict (keyed by $N)."""
    if isinstance(inp, Port):
        if inp.indexmap is not None:
            return tuple(inp.indexmap.out_shape)
        # Without buffer_id, the caller must have populated shapes with $N keys.
        return ()
    if isinstance(inp, Combine) and inp.sources:
        return _port_shape(inp.sources[0], shapes)
    if isinstance(inp, Mux) and inp.branches:
        return _port_shape(inp.branches[0].input, shapes)
    return ()
