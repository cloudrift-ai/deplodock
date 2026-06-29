"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` source string and derives the
launch geometry. The body is a single :class:`Tile` (the thread-grid decode), so the grid
is sized from the tile's element count over the block (``_BLOCK_SIZE`` scalar tier, the
cooperative ``block_threads`` otherwise). A **symbolic reduce / output-sweep axis** (a
dynamic ``seq_len`` inside the body's loops) becomes a runtime ``int`` arg: the kernel
signature gains one ``int <name>`` per symbolic ``Dim`` and the launch resolves it from the
input array shapes (``sym_values`` → ``runtime_args``). The free (grid) axes are static
here (``010_recognize`` defers a symbolic free axis), so the grid stays a static int.
"""

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.kernel.render import _BLOCK_SIZE, render_kernelop
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def _symbolic_runtime_args(kernel: KernelOp) -> tuple[str, ...]:
    """Every symbolic ``Dim`` name referenced by an axis anywhere in the body, in first-seen
    order (a dynamic reduce / output-sweep ``seq_len`` — the kernel takes one ``int`` arg per
    name, the launch resolves it from the input shapes). Dict-keyed for stable ordering."""
    seen: dict[str, None] = {}
    for s in kernel.body.iter():
        axes = s.axes if isinstance(s, Tile) else ((s.axis,) if hasattr(s, "axis") else ())
        for ax in axes:
            if isinstance(ax, Axis) and not ax.extent.is_static:
                for nm in ax.extent.expr.free_vars():
                    seen.setdefault(nm, None)
    return tuple(seen)


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
    # (snapped to the graph by the matcher's populate_io). A symbolic reduce axis adds an
    # ``int <name>`` runtime arg to the signature, threaded through to the CudaOp.
    runtime_args = _symbolic_runtime_args(kernel)
    tensors = {**kernel.inputs, **kernel.outputs}
    source = render_kernelop(kernel, tensors=tensors, runtime_args=runtime_args)

    # A cooperative tile fixes the per-CTA thread count (``coop · ∏block-cells``): one CTA
    # per output-cell group, ``blockDim = block_threads``, ``gridDim = N / block_threads``
    # (the linear ``_gid`` decode groups ``block_threads`` consecutive cells per CTA). The
    # scalar tier is one thread per cell over the fixed ``_BLOCK_SIZE`` block.
    blockdim = tile.block_threads if tile.block_threads is not None else _BLOCK_SIZE
    if tile.is_static_grid:
        n = tile.n_elements
        grid = (((n + blockdim - 1) // blockdim,), (1,), (1,))
    else:
        # Symbolic grid (a dynamic free axis): ``ceil(∏extents / blockDim)`` CTAs, the symbolic
        # factor resolved from ``sym_values`` at launch (the ``Expr`` grid factor).
        grid = ((tile.n_dim.ceil_div(blockdim).expr,), (1,), (1,))
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
        runtime_args=runtime_args,
    )
