"""CudaOp — graph-level wrapper around a rendered CUDA kernel.

Produced by ``passes/lowering/cuda`` by rendering each ``KernelOp`` body
to a ``__global__`` source string. The final graph before codegen is
``Graph[CudaOp + InputOp + ConstantOp]``; the CUDA backend walks it in
topological order, emits one ``kernel_name<<<grid, block>>>(args)``
launch per node, and wires buffer pointers by node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op


@dataclass(frozen=True)
class TmaDescMeta:
    """Metadata the CUDA backend needs to encode a TMA descriptor at launch.

    ``name`` matches the kernel signature parameter (added to
    ``arg_order`` after the buffer args). ``src_buf`` names the graph
    buffer whose device pointer + shape feed
    ``cuTensorMapEncodeTiled``. ``box_extents`` and ``swizzle`` are the
    descriptor's per-dim box and swizzle mode."""

    name: str
    src_buf: str
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"
    keep_dims: tuple[int, ...] | None = None


@dataclass
class CudaOp(Op):
    """One CUDA kernel invocation as a graph-op."""

    kernel_source: str = ""  # complete __global__ function
    kernel_name: str = ""
    arg_order: tuple[str, ...] = ()  # kernel-param names in positional order
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    smem_bytes: int = 0
    zero_outputs: tuple[str, ...] = ()
    comment: str = ""
    tma_descriptors: tuple[TmaDescMeta, ...] = field(default_factory=tuple)

    def pretty_body(self) -> str:
        return self.kernel_source
