"""Shared test helper: build a KernelOp from flat region_ops tuples.

Mirrors what ``FusedRegionOp(region_ops=..., input_names=..., output_names=...)``
used to produce, but constructs a KernelOp so tests exercise the same IR
the production pipeline uses.
"""

from __future__ import annotations

from deplodock.compiler.ir import Node, Tensor
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    KernelOp,
    Port,
    ReduceOp,
    ReduceStage,
)


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

    Also infers a structured ``core`` annotation from the op sequence so
    kernels built by tests look the same as kernels built by the fusion
    rules — a ContractionCore for ``sum(mul(a, b))`` shapes, a
    tuple[ReduceStage, ...] for single or multi-reduce chains, else None.
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
    core = _infer_core(body, input_names)
    return KernelOp(
        inputs=[Port(buffer_id=n) for n in input_names],
        outputs=[Port(buffer_id=n) for n in output_names],
        prologue=body,
        core=core,
        epilogue=(),
        external_shapes=external_shapes,
    )


def _infer_core(body: tuple, input_names: list) -> object:
    """Classify body ops into ContractionCore / tuple[ReduceStage] / None."""
    reduces = [n for n in body if isinstance(n.op, ReduceOp)]
    if not reduces:
        return None
    # Matmul shape: first reduce is sum, fed directly by an Elementwise mul
    # whose two inputs are external (both in input_names).
    first = reduces[0]
    if first.op.fn == "sum" and len(first.inputs) == 1:
        feeder = next((n for n in body if n.id == first.inputs[0]), None)
        if (
            feeder is not None
            and isinstance(feeder.op, ElementwiseOp)
            and feeder.op.fn == "mul"
            and len(feeder.inputs) == 2
            and feeder.inputs[0] in input_names
            and feeder.inputs[1] in input_names
        ):
            k_axis = first.op.axis if isinstance(first.op.axis, int) else len(feeder.output.shape) - 1
            if k_axis < 0:
                k_axis = len(feeder.output.shape) + k_axis
            return ContractionCore(
                a=Port(buffer_id=feeder.inputs[0]),
                b=Port(buffer_id=feeder.inputs[1]),
                k_axis=k_axis,
                mul=feeder,
                reduce=first,
            )
    # Generic reduce chain: one stage per reduce, pre_ops are the
    # elementwise nodes chained between reduces (or between the kernel's
    # external inputs and the first reduce, for a single-reduce stage).
    id_to_node = {n.id: n for n in body}
    stages: list[ReduceStage] = []
    prev_reduce_id: str | None = None
    for red in reduces:
        pre_chain: list = []
        cur = red.inputs[0] if red.inputs else None
        while cur in id_to_node:
            node = id_to_node[cur]
            if isinstance(node.op, ReduceOp):
                break
            if node.id == prev_reduce_id:
                break
            if node.inputs and any(inp in id_to_node and not isinstance(id_to_node[inp].op, ReduceOp) for inp in node.inputs):
                pre_chain.append(node)
                cur = node.inputs[0]
            else:
                pre_chain.append(node)
                break
        stages.append(ReduceStage(pre_ops=tuple(reversed(pre_chain)), reduce=red))
        prev_reduce_id = red.id
    # First stage's pre_ops go into the "flowing into reduce" slot; the
    # tuple convention is that only stages[1:] own their pre_ops chain
    # (stage[0]'s pre_ops live in the prologue proper in production).
    if stages:
        stages[0] = ReduceStage(pre_ops=(), reduce=stages[0].reduce)
    return tuple(stages) if stages else None
