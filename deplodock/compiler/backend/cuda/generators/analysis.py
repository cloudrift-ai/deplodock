"""Tile analysis: classify a KernelOp's computation pattern.

Walks the ops, identifies reduction axes, op phases, input access patterns,
and classifies the region as one of: pointwise, row_reduce,
reduce_broadcast, or contraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ops import KernelOp


@dataclass
class TileAnalysis:
    """Analysis result for a KernelOp.

    Captures everything needed to choose a tiling strategy and generate
    the kernel, without re-walking the ops. Phase decomposition is NOT
    stored here — consumers call ``KernelOp.phases()`` directly.
    """

    pattern: str  # "pointwise" | "row_reduce" | "reduce_broadcast" | "contraction"
    output_shape: tuple[int, ...]
    reduce_fns: list[str]  # ["sum"], ["max", "sum"], etc.
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


def analyze(region: KernelOp, shapes: dict[str, tuple]) -> TileAnalysis:
    """Build a ``TileAnalysis`` snapshot from KernelOp accessors.

    All derivation logic lives on ``KernelOp`` (Layer 1). This function
    is pure assembly: it calls the accessors with ``shapes`` and packs
    the results into the codegen-facing struct.
    """
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    rows, cols, k_dim = region.tile_dims(shapes, out_shape)
    cinfo = region.contraction_info(shapes)

    return TileAnalysis(
        pattern=region.tile_pattern(shapes, out_shape),
        output_shape=out_shape,
        reduce_fns=region.reduce_fn_names(),
        rows=rows,
        cols=cols,
        k_dim=k_dim,
        contraction_a=cinfo.a_id if cinfo is not None else None,
        contraction_b=cinfo.b_id if cinfo is not None else None,
        epilogue_needs_per_element=region.epilogue_needs_per_element(shapes, out_shape),
        batch_dims=cinfo.batch_dims if cinfo is not None else (),
        batch_size=cinfo.batch_size if cinfo is not None else 1,
        a_batch_group=cinfo.a_batch_group if cinfo is not None else 1,
        b_batch_group=cinfo.b_batch_group if cinfo is not None else 1,
    )
