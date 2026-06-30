"""Single-kernel structural slice: isolate one finalized kernel node into a
standalone graph.

Used in two places:

- the **dump** sink writes one ``<kname>.json`` reproducer per kernel
  (``CompilerDump._dump_kernel_subgraphs``), and
- the **two-level tuner** (`search.two_level`) slices each post-fusion kernel
  into its own graph so the inner per-op search explores only that op's forks.

The slice keeps the root kernel node plus its transitive ``ConstantOp`` /
``InputOp`` producers (so scalar-constant inlining and load-op replay behave
identically) and replaces every *compute* ancestor — another kernel feeding
this one — with a synthetic ``InputOp`` boundary, so the result is standalone.
The root op is shared **by reference**: its body (and therefore
:func:`op_cache_key`) is byte-for-byte the full-graph op's, which is what lets
inner-tuned ``perf`` / ``lowering`` rows transfer back to the assembled graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


def _kernel_compute_types() -> tuple[type, ...]:
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileGraphOp, TileOp  # noqa: PLC0415

    # ``TileGraphOp`` is a kernel-bearing op too: an outer two-level terminal slices
    # partially-tiled ``TileGraphOp`` seeds (the structural-fork producer/consumer), so a
    # ``TileGraphOp`` ancestor must become a synthetic ``InputOp`` boundary like any kernel.
    return (LoopOp, TileOp, TileGraphOp, KernelOp, CudaOp)


def collect_kernel_ancestors(graph: Graph, root_id: str, compute_types: tuple[type, ...]) -> tuple[set[str], set[str]]:
    """Collect ``root_id`` + its transitive ``ConstantOp`` / ``InputOp``
    ancestors. Compute-op ancestors (another kernel feeding this one) are
    returned in the ``synthetic`` set — they become synthetic ``InputOp``
    boundaries in the slice and their own producers are NOT walked."""
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415

    keep: set[str] = {root_id}
    synthetic: set[str] = set()
    stack = list(graph.nodes[root_id].inputs)
    while stack:
        cur = stack.pop()
        if cur in keep:
            continue
        keep.add(cur)
        node = graph.nodes.get(cur)
        if node is None:
            continue
        if isinstance(node.op, compute_types):
            synthetic.add(cur)
            continue
        if isinstance(node.op, (ConstantOp, InputOp)):
            stack.extend(node.inputs)
    return keep, synthetic


def topo_order(graph: Graph, keep: set[str]) -> list[str]:
    """Topo-sorted node ids restricted to ``keep`` (producers first)."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(nid: str) -> None:
        if nid in visited or nid not in keep:
            return
        visited.add(nid)
        for dep in graph.nodes[nid].inputs:
            visit(dep)
        order.append(nid)

    for nid in keep:
        visit(nid)
    return order


def single_node_graph(graph: Graph, node_id: str) -> Graph:
    """Slice ``graph`` to the single kernel node ``node_id`` plus its
    leaf-op closure, with every compute-op input turned into a synthetic
    ``InputOp``. Returns a standalone :class:`Graph` whose sole output is
    ``node_id`` and whose ``inputs`` list its synthetic boundaries + real
    graph inputs — sized identically to the full graph (partition
    enumeration depends on the producers' extents)."""
    from emmy.compiler.graph import Graph as _Graph  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415

    keep, synthetic = collect_kernel_ancestors(graph, node_id, _kernel_compute_types())
    sub = _Graph()
    for kid in topo_order(graph, keep):
        src = graph.nodes[kid]
        if kid in synthetic:
            sub.add_node(InputOp(), [], src.output, node_id=src.id)
            sub.inputs.append(kid)
        else:
            sub.add_node(src.op, list(src.inputs), src.output, node_id=src.id)
            if isinstance(src.op, InputOp) and kid in graph.inputs:
                sub.inputs.append(kid)
    sub.outputs.append(node_id)
    return sub
