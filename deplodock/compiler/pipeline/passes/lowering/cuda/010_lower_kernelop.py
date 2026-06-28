"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` source string and derives the
launch geometry. The first ``KernelOp`` shape the skeleton produces is a single
:class:`Tile` (one thread per output cell), so the grid is sized from the tile's
element count over the render's fixed 256-thread block. Other launch geometries
arrive as the skeleton grows (see ``plans/tile-ir-rebuild.md``).
"""

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.kernel.render import _BLOCK_SIZE, render_kernelop
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(match: Match, root: Node) -> CudaOp | None:
    kernel: KernelOp = root.op
    tiles = [s for s in kernel.body if isinstance(s, Tile)]
    if len(tiles) != 1:
        raise RuleSkipped("only single-Tile KernelOps lower so far")
    (tile,) = tiles

    # The kernel function name doubles as the CudaOp's launch name — keep them
    # identical. Loop naming always stamps a label, but fall back to the node id.
    name = kernel.name or f"k_{root.id}"
    if name != kernel.name:
        kernel = replace(kernel, name=name)

    # Buffer shapes / dtypes for the renderer come from the op's I/O Tensors
    # (snapped to the graph by the matcher's populate_io).
    tensors = {**kernel.inputs, **kernel.outputs}
    source = render_kernelop(kernel, tensors=tensors)

    # A cooperative tile fixes the per-CTA thread count (``coop · ∏block-cells``): one CTA
    # per output-cell group, ``blockDim = block_threads``, ``gridDim = N / block_threads``
    # (the linear ``_gid`` decode groups ``block_threads`` consecutive cells per CTA). The
    # scalar tier is one thread per cell over the fixed ``_BLOCK_SIZE`` block.
    n = tile.n_elements
    if tile.block_threads is not None:
        blockdim = tile.block_threads
        blocks = (n + blockdim - 1) // blockdim
    else:
        blockdim = _BLOCK_SIZE
        blocks = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    grid = ((blocks,), (1,), (1,))
    block = ((blockdim,), (1,), (1,))
    arg_order = (*kernel.inputs, *kernel.outputs)
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=arg_order,
        grid=grid,
        block=block,
        smem_bytes=kernel.smem_bytes(),
        comment=name,
    )
