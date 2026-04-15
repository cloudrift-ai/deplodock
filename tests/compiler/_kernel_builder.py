"""Shared test helper: build a KernelOp from flat region_ops tuples.

Mirrors what ``FusedRegionOp(region_ops=..., input_names=..., output_names=...)``
used to produce, but constructs a KernelOp so tests exercise the same IR
the production pipeline uses.
"""

from __future__ import annotations

from deplodock.compiler.ir import Node, Tensor
from deplodock.compiler.ops import KernelOp, Port


def build_kernel(
    region_ops: list,
    input_names: list,
    output_names: list,
    shapes: dict | None = None,
) -> KernelOp:
    """Construct a KernelOp with flat prologue from legacy-shape region ops.

    ``region_ops`` is a list of ``(node_id, op, input_ids)`` tuples in topo
    order — the same shape ``FusedRegionOp.region_ops`` used to hold.
    ``shapes`` maps node_id → output shape; falls back to () if absent.
    """
    shapes = shapes or {}
    body = tuple(
        Node(
            id=nid,
            op=op_obj,
            inputs=list(inp_ids),
            output=Tensor(name=nid, shape=tuple(shapes.get(nid, ())), dtype="f32"),
        )
        for nid, op_obj, inp_ids in region_ops
    )
    external_shapes = {n: tuple(shapes.get(n, ())) for n in input_names}
    return KernelOp(
        inputs=[Port(buffer_id=n) for n in input_names],
        outputs=[Port(buffer_id=n) for n in output_names],
        prologue=body,
        core=None,
        epilogue=(),
        external_shapes=external_shapes,
    )
