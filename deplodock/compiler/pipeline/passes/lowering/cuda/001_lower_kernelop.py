"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` CUDA source string
and mutates the node's op payload in place. Grid / block geometry is
derived from the ``Tile`` in the body; ``smem_bytes`` is summed
over ``Smem`` decls. ``arg_order`` is ``kernel_op.inputs +
kernel_op.outputs`` keys — matches the kernel signature emitted by
``render_kernelop``.
"""

from __future__ import annotations

from math import prod

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.cuda import CudaOp, TmaDescMeta
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.kernel.ir import TmaDescriptor
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", KernelOp)]

_BLOCK = 256


def rewrite(match: Match, root: Node) -> Graph | None:  # noqa: ARG001 — match required by rule dispatch signature
    # Per-buffer Tensor descriptors (shape + dtype) come straight off
    # ``root.op.inputs`` / ``root.op.outputs`` — the matcher snapped them
    # to the surrounding graph's Tensors via ``populate_io``, including
    # the ``constant`` / ``value`` flags for ConstantOp predecessors.
    tensors: dict[str, Tensor] = {**root.op.inputs, **root.op.outputs}

    # Scalar ConstantOp inputs get embedded as float literals in the kernel
    # body — no kernel parameter, no buffer load.
    literal_constants: dict[str, float] = {n: float(t.value) for n, t in root.op.inputs.items() if t.constant and t.value is not None}

    runtime_inputs = tuple(b for b in root.op.inputs if b not in literal_constants)

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
    output_set = set(root.op.outputs)
    for s in root.op.writes:
        if s.reduce_op is not None and s.output in output_set and s.output not in seen_atomic:
            seen_atomic.add(s.output)
            atomic_outputs.append(s.output)
    return CudaOp(
        kernel_source=render_kernelop(root.op, tensors=tensors, literal_constants=literal_constants),
        kernel_name=root.op.name,
        arg_order=(*runtime_inputs, *root.op.outputs, *desc_names),
        grid=grid,
        block=block,
        smem_bytes=root.op.smem_bytes(),
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
