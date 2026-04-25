"""Lower each ``TileOp`` node to a ``CudaOp``.

Renders the ``TileOp`` body to a ``__global__`` CUDA source string and
mutates the node's op payload in place. Grid / block geometry is derived
from the ``Enclosure.thread_axes`` of the body (a single 1D grid over
``BLOCK_SIZE`` threads, sized to cover the product of thread-axis extents).
``arg_order`` is ``tile_op.inputs + tile_op.outputs`` — matches the kernel
signature emitted by ``render_tileop``.
"""

from __future__ import annotations

from math import prod

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.tile import Enclosure, TileOp
from deplodock.compiler.ir.tile.render import render_tileop
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

_BLOCK = 256


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    shapes: dict[str, tuple[int, ...]] = {bid: tuple(graph.nodes[bid].output.shape) for bid in tile_op.inputs}
    for out in tile_op.outputs:
        shapes[out] = tuple(graph.nodes[out].output.shape) if out in graph.nodes else tuple(node.output.shape)

    source = render_tileop(tile_op, shapes=shapes)
    grid, block = _launch_geometry(tile_op)
    arg_order = (*tile_op.inputs, *tile_op.outputs)

    node.op = CudaOp(
        kernel_source=source,
        kernel_name=tile_op.name,
        arg_order=arg_order,
        grid=grid,
        block=block,
    )
    return None


def _launch_geometry(tile_op: TileOp) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Pick (grid, block) from the first ``Enclosure`` in the body.

    Single-thread fallback (no Enclosure) launches one thread; the renderer
    emits no tid guard in that case so this matches the kernel shape.
    """
    for s in tile_op.body:
        if isinstance(s, Enclosure):
            n_threads = max(prod(int(a.extent) for a in s.thread_axes), 1)
            grid = ((n_threads + _BLOCK - 1) // _BLOCK, 1, 1)
            return grid, (_BLOCK, 1, 1)
    return (1, 1, 1), (1, 1, 1)
