"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` CUDA source string
and mutates the node's op payload in place. Grid / block geometry is
derived from the ``Tile`` in the body; ``smem_bytes`` is summed
over ``Smem`` decls. ``arg_order`` is ``kernel_op.inputs +
kernel_op.outputs`` — matches the kernel signature emitted by
``render_kernelop``.
"""

from __future__ import annotations

from math import prod

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.kernel import KernelOp, Smem, Tile
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", KernelOp)]

_BLOCK = 256

_DTYPE_BYTES: dict[str, int] = {"float": 4, "double": 8, "int": 4, "half": 2}


def rewrite(graph: Graph, root: Node) -> Graph | None:
    shapes: dict[str, tuple[int, ...]] = {bid: tuple(graph.nodes[bid].output.shape) for bid in root.op.inputs}
    for out in root.op.outputs:
        shapes[out] = tuple(graph.nodes[out].output.shape) if out in graph.nodes else tuple(root.output.shape)

    grid, block = _launch_geometry(root.op)
    root.op = CudaOp(
        kernel_source=render_kernelop(root.op, shapes=shapes),
        kernel_name=root.op.name,
        arg_order=(*root.op.inputs, *root.op.outputs),
        grid=grid,
        block=block,
        smem_bytes=_smem_bytes(root.op),
    )
    return None


def _launch_geometry(kernel_op: KernelOp) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Pick (grid, block) from the first ``Tile`` in the body."""
    for s in kernel_op.body:
        if isinstance(s, Tile):
            if s.block_axes:
                grid_total = max(prod(int(a.extent) for a in s.block_axes), 1)
                block_total = max(prod(int(a.extent) for a in s.thread_axes), 1)
                return (grid_total, 1, 1), (block_total, 1, 1)
            n_threads = max(prod(int(a.extent) for a in s.thread_axes), 1)
            grid = ((n_threads + _BLOCK - 1) // _BLOCK, 1, 1)
            return grid, (_BLOCK, 1, 1)
    return (1, 1, 1), (1, 1, 1)


def _smem_bytes(kernel_op: KernelOp) -> int:
    """Sum ``prod(extents) * sizeof(dtype)`` across all ``Smem`` decls."""
    total = 0
    for s in kernel_op:
        if isinstance(s, Smem):
            elements = prod(int(e) for e in s.extents) if s.extents else 1
            total += elements * _DTYPE_BYTES.get(s.dtype, 4)
    return total
