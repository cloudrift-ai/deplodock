"""Tile analysis: classify a KernelOp's computation pattern.

Walks the ops, identifies reduction axes, op phases, input access patterns,
and classifies the region as one of: pointwise, row_reduce,
reduce_broadcast, or contraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.ops import AccessPattern, ContractionCore, KernelOp

# Type alias for region op tuples: (node_id, op, input_ids)
RegionEntry = tuple[str, object, list[str]]


@dataclass
class OpPhases:
    """Ops split into prologue (before first reduce), reduces, epilogue (after last reduce).

    For multi-reduce patterns (e.g. softmax max+sum), inter_reduce[i] holds the
    ops between reduces[i] and reduces[i+1].
    """

    prologue: list[RegionEntry]
    reduces: list[RegionEntry]
    epilogue: list[RegionEntry]
    inter_reduce: list[list[RegionEntry]] = field(default_factory=list)


@dataclass
class TileAnalysis:
    """Analysis result for a KernelOp.

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
    # Batch dimensions for batched contractions (e.g. multi-head attention).
    batch_dims: tuple[int, ...] = ()
    batch_size: int = 1
    # GQA / broadcast batch: when one operand has fewer batch elements,
    # its batch index is divided by this factor.  E.g. 28 Q heads / 4 KV heads = 7.
    # "b_batch_group" means B's batch index = batch // b_batch_group.
    # 1 means both operands use the same batch index (no broadcast).
    a_batch_group: int = 1
    b_batch_group: int = 1
    # Per-input indexmaps carried on Port.indexmap. When set, the load path
    # substitutes the placeholder coord_map with the kernel's runtime
    # indices to build the actual input address (transpose-into-matmul).
    port_indexmaps: dict = field(default_factory=dict)


def flat_region_ops(kernel: KernelOp) -> list:
    """Flat (id, op, inputs) body walk — delegates to ``KernelOp.body_ops()``."""
    return kernel.body_ops()


def analyze(region: KernelOp, shapes: dict[str, tuple]) -> TileAnalysis:
    """Analyze a KernelOp and classify its computation pattern.

    Args:
        region: The fused region containing primitive ops in topo order.
        shapes: Map of node_id/buffer_name -> shape tuple.

    Returns:
        TileAnalysis with pattern classification and metadata.
    """
    # Split ops into phases directly from structured kernel fields.
    phases = _split_phases(region)

    # Determine output shape.
    out_id = [p.buffer_id for p in region.outputs][0]
    out_shape = shapes.get(out_id, (1,))

    reduce_fns = region.reduce_fn_names()
    port_indexmaps = region.port_indexmaps()
    input_access = region.input_accesses(shapes, out_shape)

    # No reduces → pointwise.
    if not phases.reduces:
        total = math.prod(d for d in out_shape if isinstance(d, int))
        return TileAnalysis(
            port_indexmaps=port_indexmaps,
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
    is_contraction, a_name, b_name, m, n, k, batch_dims, batch_size, a_bg, b_bg = _detect_contraction(region, phases, shapes, input_access)

    if is_contraction:
        epilogue_per_elem = _epilogue_needs_per_element(region, phases, shapes, input_access)
        return TileAnalysis(
            port_indexmaps=port_indexmaps,
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
            batch_dims=batch_dims,
            batch_size=batch_size,
            a_batch_group=a_bg,
            b_batch_group=b_bg,
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
        port_indexmaps=port_indexmaps,
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


def _split_phases(kernel: KernelOp) -> OpPhases:
    """Wrap ``KernelOp.phases()`` as an ``OpPhases`` dataclass.

    The phase decomposition lives on ``KernelOp`` itself; this function
    just names the returned fields for consumer convenience.
    """
    prologue, reduces, inter_reduce, epilogue = kernel.phases()
    return OpPhases(
        prologue=prologue,
        reduces=reduces,
        epilogue=epilogue,
        inter_reduce=inter_reduce,
    )


def _needed_by(ops: list) -> set[str]:
    """Return set of node_ids referenced as inputs by the given ops."""
    needed = set()
    for _, _, input_ids in ops:
        needed.update(input_ids)
    return needed


def _detect_contraction(
    region: KernelOp,
    phases: OpPhases,
    shapes: dict[str, tuple],
    input_access: dict[str, AccessPattern],
) -> tuple[bool, str | None, str | None, int, int, int, tuple[int, ...], int]:
    """Determine matmul metadata from a KernelOp's structured ``core``.

    If ``kernel.core`` is a ``ContractionCore``, read a/b buffer IDs from
    its Ports and derive M, N, K, batch_dims, batch_size (plus GQA batch
    groups) from the Port shapes — no scanning. Otherwise not a contraction.

    Returns: (is_contraction, a_name, b_name, M, N, K, batch_dims, batch_size,
              a_batch_group, b_batch_group).
    """
    if not isinstance(region.core, ContractionCore):
        return False, None, None, 0, 0, 0, (), 1, 1, 1
    a_id = region.core.a.buffer_id
    b_id = region.core.b.buffer_id

    if a_id not in input_access or b_id not in input_access:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    a_acc = input_access[a_id]
    b_acc = input_access[b_id]
    if not a_acc.is_2d or not b_acc.is_2d:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    a_shape = a_acc.shape
    b_shape = b_acc.shape
    if len(a_shape) < 2 or len(b_shape) < 2:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    # Detect batch dimensions: leading dims that match or broadcast between A and B.
    batch_dims: tuple[int, ...] = ()
    batch_size = 1
    a_batch_group = 1
    b_batch_group = 1
    if len(a_shape) > 2 and len(b_shape) > 2:
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        if a_batch == b_batch:
            batch_dims = a_batch
        else:
            # Broadcast batch dims (e.g. GQA: 28 Q heads vs 4 KV heads).
            # Pad the shorter batch tuple with leading 1s so both have the same ndim,
            # then check each dim matches or one divides the other.
            max_len = max(len(a_batch), len(b_batch))
            a_padded = (1,) * (max_len - len(a_batch)) + a_batch
            b_padded = (1,) * (max_len - len(b_batch)) + b_batch
            merged_batch: list[int] = []
            for ad, bd in zip(a_padded, b_padded, strict=True):
                if not isinstance(ad, int) or not isinstance(bd, int):
                    return False, None, None, 0, 0, 0, (), 1, 1, 1
                if ad == bd:
                    merged_batch.append(ad)
                elif ad > bd and bd > 0 and ad % bd == 0:
                    merged_batch.append(ad)
                elif bd > ad and ad > 0 and bd % ad == 0:
                    merged_batch.append(bd)
                else:
                    return False, None, None, 0, 0, 0, (), 1, 1, 1
            batch_dims = tuple(merged_batch)
            a_batch_size = math.prod(d for d in a_padded if isinstance(d, int))
            b_batch_size = math.prod(d for d in b_padded if isinstance(d, int))
            if a_batch_size >= b_batch_size and b_batch_size > 0:
                b_batch_group = a_batch_size // b_batch_size
            elif a_batch_size > 0:
                a_batch_group = b_batch_size // a_batch_size
        batch_size = math.prod(d for d in batch_dims if isinstance(d, int))
        # Batched: A(B..., M, K) @ B(B..., K, N)
        a_k = a_shape[-1]
        b_k = b_shape[-2]
    else:
        # 2D: A(M, K) @ B(K, N)
        a_k = a_shape[-1]
        b_k = b_shape[0]

    # K dimension must match (both int or both same symbolic string).
    if a_k != b_k:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    m = a_shape[-2] if isinstance(a_shape[-2], int) else 1
    if not batch_dims:
        # 2D: M = product of all dims except last
        m = math.prod(d for d in a_shape[:-1] if isinstance(d, int)) if any(isinstance(d, int) for d in a_shape[:-1]) else 1
    k = a_k if isinstance(a_k, int) else 1
    n = b_shape[-1] if isinstance(b_shape[-1], int) else 1

    return True, a_id, b_id, m, n, k, batch_dims, batch_size, a_batch_group, b_batch_group


def _epilogue_needs_per_element(
    region: KernelOp,
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

    # Check if any external input needed by epilogue requires per-element
    # access (is_2d). Per-row and scalar inputs are available during the
    # reduce pass and don't need a second per-element loop.
    for inp in [p.buffer_id for p in region.inputs]:
        if inp in epilogue_needs:
            acc = input_access.get(inp)
            if acc and acc.is_2d:
                return True

    # Check if any prologue op needed by epilogue (transitive 2D dependency).
    for node_id, _, _ in phases.prologue:
        if node_id in _needed_by(phases.epilogue):
            return True

    return False
