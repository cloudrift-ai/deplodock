"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` CUDA source string
and mutates the node's op payload in place. Grid / block geometry is
derived from the ``Tile`` in the body; ``smem_bytes`` is summed
over ``Smem`` decls. ``arg_order`` is ``kernel_op.body_inputs +
kernel_op.body_outputs`` — matches the kernel signature emitted by
``render_kernelop``.
"""

from __future__ import annotations

from math import prod

from deplodock.compiler.backend.cuda.dtype import nbytes_of as _nbytes_of
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.cuda import CudaOp, TmaDescMeta
from deplodock.compiler.ir.kernel import KernelOp, Smem, Tile
from deplodock.compiler.ir.kernel.ir import TmaDescriptor
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", KernelOp)]

_BLOCK = 256


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph

    # Per-buffer Tensor descriptors (shape + dtype) for the kernel
    # signature and the renderer's index flattening — read from the
    # surrounding graph using the body-derived buffer names. Falls back
    # to ``root.output`` for any body-output name that isn't a graph
    # node (legacy KernelOps constructed bare in tests).
    body_inputs = root.op.body_inputs
    body_outputs = root.op.body_outputs
    tensors: dict[str, Tensor] = {}
    for bid in body_inputs:
        node = graph.nodes.get(bid)
        if node is not None:
            tensors[bid] = node.output
    for out in body_outputs:
        node = graph.nodes.get(out)
        tensors[out] = node.output if node is not None else Tensor(out, tuple(root.output.shape), root.output.dtype)

    # Scalar ConstantOp inputs get embedded as float literals in the kernel
    # body — no kernel parameter, no buffer load.
    literal_constants: dict[str, float] = {}
    for bid in body_inputs:
        node = graph.nodes.get(bid)
        if node is not None and isinstance(node.op, ConstantOp) and node.op.value is not None:
            literal_constants[bid] = float(node.op.value)

    runtime_inputs = tuple(b for b in body_inputs if b not in literal_constants)

    grid, block = _launch_geometry(root.op)
    # TMA descriptors are kernel parameters that come *after* the buffer
    # args (matching the signature emitted by ``render_kernelop``).
    # Collect dedup'd metadata so the backend can encode + bind them.
    desc_stmts = root.op.body.iter_of_type(TmaDescriptor)
    seen: set[str] = set()
    descriptors: list[TmaDescMeta] = []
    desc_names: list[str] = []
    for s in desc_stmts:
        if s.name in seen:
            continue
        seen.add(s.name)
        descriptors.append(
            TmaDescMeta(name=s.name, src_buf=s.src_buf, box_extents=s.box_extents, swizzle=s.swizzle),
        )
        desc_names.append(s.name)
    # Outputs receiving atomic-reduction writes (cross-CTA split-K) must be
    # zero-initialized before each launch so per-CTA partials accumulate
    # cleanly. Anything else can keep its prior contents.
    atomic_outputs: list[str] = []
    seen_atomic: set[str] = set()
    output_set = set(body_outputs)
    for s in root.op.writes:
        if s.reduce_op is not None and s.output in output_set and s.output not in seen_atomic:
            seen_atomic.add(s.output)
            atomic_outputs.append(s.output)
    return CudaOp(
        kernel_source=render_kernelop(root.op, tensors=tensors, literal_constants=literal_constants),
        kernel_name=root.op.name,
        arg_order=(*runtime_inputs, *body_outputs, *desc_names),
        grid=grid,
        block=block,
        smem_bytes=_smem_bytes(root.op),
        tma_descriptors=tuple(descriptors),
        zero_outputs=tuple(atomic_outputs),
    )


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
            total += elements * _nbytes_of(s.dtype)
    return total
