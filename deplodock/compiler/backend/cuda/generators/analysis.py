"""Tile analysis: classify a FusedRegionOp's computation pattern.

Walks the ops, identifies reduction axes, op phases, input access patterns,
and classifies the region as one of: pointwise, row_reduce,
reduce_broadcast, or contraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReduceOp

# Type alias for region op tuples: (node_id, op, input_ids)
RegionEntry = tuple[str, object, list[str]]


@dataclass
class OpPhases:
    """Ops split into prologue (before first reduce), reduces, epilogue (after last reduce)."""

    prologue: list[RegionEntry]
    reduces: list[RegionEntry]
    epilogue: list[RegionEntry]


@dataclass
class AccessPattern:
    """How a single input tensor is accessed within the kernel."""

    shape: tuple[int, ...]
    size: int  # total elements
    is_scalar: bool  # size == 1 (broadcast everywhere)
    is_row_vector: bool  # 1D, indexed by column only
    is_2d: bool  # indexed by both row and column


@dataclass
class TileAnalysis:
    """Analysis result for a FusedRegionOp.

    Captures everything needed to choose a tiling strategy and generate
    the kernel, without re-walking the ops.
    """

    pattern: str  # "pointwise" | "row_reduce" | "reduce_broadcast" | "contraction"
    op_phases: OpPhases
    output_shape: tuple[int, ...]
    reduce_fns: list[str]  # ["sum"], ["max", "sum"], etc.
    input_access: dict[str, AccessPattern]  # per external input
    # Dimensions (concrete ints, derived from shapes).
    rows: int  # product of non-reduced dims (M for contraction, rows for reduce)
    cols: int  # last dim of the pre-reduction tensor (N for contraction, cols for reduce)
    k_dim: int  # shared/reduced dimension (K for contraction, same as cols for reduce)
    # For contraction only: names of the two matmul operands.
    contraction_a: str | None = None
    contraction_b: str | None = None
    # Whether the epilogue needs a second per-element pass over inputs.
    epilogue_needs_per_element: bool = False


def analyze(region: FusedRegionOp, shapes: dict[str, tuple]) -> TileAnalysis:
    """Analyze a FusedRegionOp and classify its computation pattern.

    Args:
        region: The fused region containing primitive ops in topo order.
        shapes: Map of node_id/buffer_name -> shape tuple.

    Returns:
        TileAnalysis with pattern classification and metadata.
    """
    # Split ops into phases.
    phases = _split_phases(region.region_ops)

    # Determine output shape.
    out_id = region.output_names[0]
    out_shape = shapes.get(out_id, (1,))

    # Collect reduce function names.
    reduce_fns = [op.fn for _, op, _ in phases.reduces]

    # Build input access patterns.
    input_access = {}
    for inp in region.input_names:
        inp_shape = shapes.get(inp, (1,))
        inp_size = math.prod(d for d in inp_shape if isinstance(d, int))
        has_symbolic = any(isinstance(d, str) for d in inp_shape)
        input_access[inp] = AccessPattern(
            shape=inp_shape,
            size=inp_size,
            is_scalar=(inp_size == 1 and not has_symbolic),
            is_row_vector=(len(inp_shape) == 1 and (inp_size > 1 or has_symbolic)),
            is_2d=(len(inp_shape) >= 2 and (inp_size > 1 or has_symbolic)),
        )

    # No reduces → pointwise.
    if not phases.reduces:
        total = math.prod(d for d in out_shape if isinstance(d, int))
        return TileAnalysis(
            pattern="pointwise",
            op_phases=phases,
            output_shape=out_shape,
            reduce_fns=[],
            input_access=input_access,
            rows=1,
            cols=total,
            k_dim=0,
        )

    # Has reduces — determine the pre-reduction tensor shape.
    first_reduce_input = phases.reduces[0][2][0]  # first reduce's first input_id
    pre_shape = shapes.get(first_reduce_input, out_shape)

    # Check for contraction pattern: exactly 2 ops (mul + sum), two 2D inputs
    # sharing a dimension that gets reduced, producing a 2D output.
    is_contraction, a_name, b_name, m, n, k = _detect_contraction(region, phases, shapes, input_access)

    if is_contraction:
        epilogue_per_elem = _epilogue_needs_per_element(region, phases, shapes, input_access)
        return TileAnalysis(
            pattern="contraction",
            op_phases=phases,
            output_shape=out_shape,
            reduce_fns=reduce_fns,
            input_access=input_access,
            rows=m,
            cols=n,
            k_dim=k,
            contraction_a=a_name,
            contraction_b=b_name,
            epilogue_needs_per_element=epilogue_per_elem,
        )

    # Row reduction patterns — extract rows/cols from pre-reduction shape.
    if len(pre_shape) >= 2:
        rows = math.prod(d for d in pre_shape[:-1] if isinstance(d, int))
        cols = pre_shape[-1] if isinstance(pre_shape[-1], int) else 1
    else:
        rows = 1
        cols = math.prod(d for d in pre_shape if isinstance(d, int))

    epilogue_per_elem = _epilogue_needs_per_element(region, phases, shapes, input_access)

    if epilogue_per_elem:
        pattern = "reduce_broadcast"
    else:
        pattern = "row_reduce"

    return TileAnalysis(
        pattern=pattern,
        op_phases=phases,
        output_shape=out_shape,
        reduce_fns=reduce_fns,
        input_access=input_access,
        rows=rows,
        cols=cols,
        k_dim=cols,  # for row reductions, k == cols
        epilogue_needs_per_element=epilogue_per_elem,
    )


def _split_phases(region_ops: list) -> OpPhases:
    """Split region ops into prologue, reduces, epilogue."""
    prologue = []
    reduces = []
    epilogue = []
    phase = "prologue"
    for entry in region_ops:
        _, op, _ = entry
        if isinstance(op, ReduceOp):
            reduces.append(entry)
            phase = "epilogue"
        elif phase == "prologue":
            prologue.append(entry)
        else:
            epilogue.append(entry)
    return OpPhases(prologue=prologue, reduces=reduces, epilogue=epilogue)


def _needed_by(ops: list) -> set[str]:
    """Return set of node_ids referenced as inputs by the given ops."""
    needed = set()
    for _, _, input_ids in ops:
        needed.update(input_ids)
    return needed


def _detect_contraction(
    region: FusedRegionOp,
    phases: OpPhases,
    shapes: dict[str, tuple],
    input_access: dict[str, AccessPattern],
) -> tuple[bool, str | None, str | None, int, int, int]:
    """Detect if region is a contraction (matmul-like) pattern.

    A contraction requires:
    - Prologue has exactly one binary ElementwiseOp (mul) with two 2D inputs
    - First reduce is sum
    - The two inputs share a dimension (K) that gets reduced
    - Output is 2D (M, N)

    Returns: (is_contraction, a_name, b_name, M, N, K)
    """
    if not phases.reduces:
        return False, None, None, 0, 0, 0

    # The first reduce must be sum.
    first_reduce = phases.reduces[0]
    _, reduce_op, _ = first_reduce
    if reduce_op.fn != "sum":
        return False, None, None, 0, 0, 0

    # Find the binary mul in prologue that feeds the reduce.
    mul_entry = None
    for entry in phases.prologue:
        _, op, input_ids = entry
        if isinstance(op, ElementwiseOp) and op.fn == "mul" and len(input_ids) == 2:
            mul_entry = entry
            break

    if mul_entry is None:
        return False, None, None, 0, 0, 0

    _, _, mul_inputs = mul_entry
    a_id, b_id = mul_inputs[0], mul_inputs[1]

    # Both inputs must be external (in region.input_names) and 2D.
    if a_id not in input_access or b_id not in input_access:
        return False, None, None, 0, 0, 0

    a_acc = input_access[a_id]
    b_acc = input_access[b_id]

    if not a_acc.is_2d or not b_acc.is_2d:
        return False, None, None, 0, 0, 0

    # A(M, K) @ B(K, N): A's last dim == B's first dim.
    a_shape = a_acc.shape
    b_shape = b_acc.shape

    if len(a_shape) < 2 or len(b_shape) < 2:
        return False, None, None, 0, 0, 0

    a_k = a_shape[-1]
    b_k = b_shape[0]

    # K dimension must match (both int or both same symbolic string).
    if a_k != b_k:
        return False, None, None, 0, 0, 0

    m = math.prod(d for d in a_shape[:-1] if isinstance(d, int)) if any(isinstance(d, int) for d in a_shape[:-1]) else 1
    k = a_k if isinstance(a_k, int) else 1
    n = b_shape[-1] if isinstance(b_shape[-1], int) else 1

    return True, a_id, b_id, m, n, k


def _epilogue_needs_per_element(
    region: FusedRegionOp,
    phases: OpPhases,
    shapes: dict[str, tuple],
    input_access: dict[str, AccessPattern],
) -> bool:
    """Check if the epilogue requires a second per-element pass.

    This is true when epilogue ops (or prologue ops they depend on) need
    to read per-element values from 2D inputs — e.g., rmsnorm epilogue
    needs the original x values to multiply by the normalization factor.
    """
    if not phases.epilogue:
        return False

    epilogue_needs = _needed_by(phases.epilogue) | _needed_by(
        phases.prologue if any(node_id in _needed_by(phases.epilogue) for node_id, _, _ in phases.prologue) else []
    )

    # Check if any external input needed by epilogue is non-scalar.
    for inp in region.input_names:
        if inp in epilogue_needs:
            acc = input_access.get(inp)
            if acc and acc.size > 1:
                return True

    # Check if any prologue op needed by epilogue (transitive 2D dependency).
    for node_id, _, _ in phases.prologue:
        if node_id in _needed_by(phases.epilogue):
            return True

    return False
