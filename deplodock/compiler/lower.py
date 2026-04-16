"""Graph -> list[KernelOp] direct structural lowering.

Walks a traced ``Graph`` and emits one ``KernelOp`` per compute unit, with
external buffers connected via bare ``Port`` leaves. No rewriter, no
fixed-point — a single linear pass grouped by a handful of structural
patterns.

Conceptual framing (see ``ops.py``): the output is a sequence of tiled
dataflow pipelines. Each ``KernelOp`` describes one pipeline with its
own input-assembly tree; adjacent kernels communicate through external
buffers named by ``Port.buffer_id`` that correspond to graph node ids.

Scope: pointwise primitives, single-reduce kernels, and the mul+sum
matmul pattern. Decomposition of ``LinearOp`` / ``MatmulOp`` / ``MeanOp``
/ ``SdpaOp`` and layout-op folding into ``Port.indexmap`` are added
incrementally as later commits drive them through tracer and E2E tests.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Node
from deplodock.compiler.ops import (
    Combine,
    ConstantOp,
    ContractionCore,
    ElementwiseOp,
    InputOp,
    KernelOp,
    Port,
    ReduceOp,
    ReduceStage,
)


def lower(graph: Graph) -> list[KernelOp]:
    """Lower a traced ``Graph`` into a list of structural ``KernelOp``.

    Groups are identified by a greedy walk over topo-sorted nodes:

    - ``ReduceOp(sum)`` whose sole input is a fan-out-1 ``ElementwiseOp(mul)``
      forms a ``ContractionCore`` (matmul pattern).
    - Every other non-trivial compute node becomes its own singleton kernel
      (pointwise or single-reduce).

    ``InputOp`` and ``ConstantOp`` nodes don't emit kernels — they are
    referenced as external ``Port.buffer_id`` leaves by downstream
    kernels.
    """
    groups = _identify_groups(graph)
    return [_build_kernel(graph, group) for group in groups]


# ---------------------------------------------------------------------------
# Grouping — walk topo order, greedy pattern match
# ---------------------------------------------------------------------------


def _identify_groups(graph: Graph) -> list[list[str]]:
    """Partition compute nodes into groups, one ``KernelOp`` per group.

    Two-pass approach so pattern matches don't lose to topo order:
      1. Precompute the set of absorbed producer ids — nodes that belong
         to a matmul pair (mul collapses into the reduce's group).
      2. Walk topo order, emit each matmul group at its reduce node and
         everything else as a singleton (skipping absorbed nodes).
    """
    absorbed: set[str] = set()
    matmul_map: dict[str, list[str]] = {}  # reduce_id -> [mul_id, reduce_id]

    for nid in graph.nodes:
        pair = _match_matmul(graph, nid)
        if pair is not None and pair[0] not in absorbed:
            absorbed.add(pair[0])
            matmul_map[pair[1]] = pair

    groups: list[list[str]] = []
    for nid in graph.topological_order():
        if nid in absorbed:
            continue
        node = graph.nodes[nid]
        if isinstance(node.op, (InputOp, ConstantOp)):
            continue
        if nid in matmul_map:
            groups.append(matmul_map[nid])
        else:
            groups.append([nid])
    return groups


def _match_matmul(graph: Graph, reduce_nid: str) -> list[str] | None:
    """Match ``ReduceOp(sum)`` with a fan-out-1 ``ElementwiseOp(mul)`` producer."""
    node = graph.nodes.get(reduce_nid)
    if node is None:
        return None
    if not isinstance(node.op, ReduceOp) or node.op.fn != "sum":
        return None
    if len(node.inputs) != 1:
        return None
    mul_id = node.inputs[0]
    mul = graph.nodes.get(mul_id)
    if mul is None or not isinstance(mul.op, ElementwiseOp) or mul.op.fn != "mul":
        return None
    if len(graph.consumers(mul_id)) != 1:
        return None
    return [mul_id, reduce_nid]


# ---------------------------------------------------------------------------
# KernelOp construction from a group of node ids
# ---------------------------------------------------------------------------


def _build_kernel(graph: Graph, group: list[str]) -> KernelOp:
    """Construct a ``KernelOp`` from a list of node ids in topo order."""
    if len(group) == 2:
        mul_id, red_id = group
        if _is_matmul_pair(graph, mul_id, red_id):
            return _build_matmul(graph, mul_id, red_id)

    assert len(group) == 1, f"unknown group shape: {group}"
    nid = group[0]
    node = graph.nodes[nid]
    if isinstance(node.op, ElementwiseOp):
        return _build_pointwise(graph, node)
    if isinstance(node.op, ReduceOp):
        return _build_reduce(graph, node)
    raise NotImplementedError(f"lowering for {type(node.op).__name__} not implemented yet")


def _is_matmul_pair(graph: Graph, mul_id: str, red_id: str) -> bool:
    mul = graph.nodes.get(mul_id)
    red = graph.nodes.get(red_id)
    if mul is None or red is None:
        return False
    return isinstance(mul.op, ElementwiseOp) and mul.op.fn == "mul" and isinstance(red.op, ReduceOp) and red.op.fn == "sum"


def _build_pointwise(graph: Graph, node: Node) -> KernelOp:
    """Singleton pointwise kernel: body is a Combine on external ports."""
    src_ports = tuple(Port(buffer_id=inp) for inp in node.inputs)
    shapes = {inp: tuple(graph.nodes[inp].output.shape) for inp in node.inputs}
    body = Combine(sources=src_ports, ops=(node,))
    return KernelOp(
        inputs=(body,),
        outputs=(Port(node.id),),
        external_shapes=shapes,
    )


def _build_reduce(graph: Graph, node: Node) -> KernelOp:
    """Singleton reduce kernel: reduce_stages=(one stage with empty pre_ops,)."""
    src_ports = tuple(Port(buffer_id=inp) for inp in node.inputs)
    shapes = {inp: tuple(graph.nodes[inp].output.shape) for inp in node.inputs}
    stage = ReduceStage(pre_ops=(), reduce=node)
    return KernelOp(
        inputs=src_ports,
        outputs=(Port(node.id),),
        reduce_stages=(stage,),
        external_shapes=shapes,
    )


def _build_matmul(graph: Graph, mul_id: str, red_id: str) -> KernelOp:
    """Matmul kernel: mul+sum collapsed into a single ContractionCore."""
    mul = graph.nodes[mul_id]
    red = graph.nodes[red_id]
    src_ports = tuple(Port(buffer_id=inp) for inp in mul.inputs)
    shapes = {inp: tuple(graph.nodes[inp].output.shape) for inp in mul.inputs}
    operand = Combine(sources=src_ports, ops=(mul,))
    assert isinstance(red.op, ReduceOp)
    k_axis = red.op.axis if isinstance(red.op.axis, int) else -1
    contraction = ContractionCore(operand=operand, k_axis=k_axis, reduce=red)
    return KernelOp(
        inputs=(),
        outputs=(Port(red.id),),
        contraction=contraction,
        external_shapes=shapes,
    )
