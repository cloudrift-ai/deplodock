"""Tile analysis: classify a KernelOp's computation pattern.

Walks the ops, identifies reduction axes, op phases, input access patterns,
and classifies the region as one of: pointwise, row_reduce,
reduce_broadcast, or contraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.ops import AccessPattern, KernelOp, _needed_by_ids

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
    first_reduce_input = phases.reduces[0][2][0]
    pre_shape = shapes.get(first_reduce_input, out_shape)

    # Contraction pattern (matmul, possibly batched / GQA-broadcast).
    cinfo = region.contraction_info(shapes)
    if cinfo is not None:
        a_acc = input_access.get(cinfo.a_id)
        b_acc = input_access.get(cinfo.b_id)
        if a_acc and b_acc and a_acc.is_2d and b_acc.is_2d:
            return TileAnalysis(
                port_indexmaps=port_indexmaps,
                pattern="contraction",
                op_phases=phases,
                output_shape=out_shape,
                reduce_fns=reduce_fns,
                input_access=input_access,
                rows=cinfo.m,
                cols=cinfo.n,
                k_dim=cinfo.k,
                contraction_a=cinfo.a_id,
                contraction_b=cinfo.b_id,
                epilogue_needs_per_element=region.epilogue_needs_per_element(shapes, out_shape),
                batch_dims=cinfo.batch_dims,
                batch_size=cinfo.batch_size,
                a_batch_group=cinfo.a_batch_group,
                b_batch_group=cinfo.b_batch_group,
            )

    # Row reduction patterns — extract rows/cols from pre-reduction shape.
    if len(pre_shape) >= 2:
        rows = math.prod(d for d in pre_shape[:-1] if isinstance(d, int))
        cols = pre_shape[-1] if isinstance(pre_shape[-1], int) else 1
    else:
        rows = 1
        cols = math.prod(d for d in pre_shape if isinstance(d, int))

    epilogue_per_elem = region.epilogue_needs_per_element(shapes, out_shape)
    pattern = "reduce_broadcast" if epilogue_per_elem else "row_reduce"

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


def _needed_by(ops: list) -> set:
    """Backward-compat alias for ``_needed_by_ids`` (used by codegen)."""
    return _needed_by_ids(ops)
