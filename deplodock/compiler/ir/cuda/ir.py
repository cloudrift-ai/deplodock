"""CudaOp — graph-level wrapper around a rendered CUDA kernel.

Produced by ``passes/lowering/cuda`` by rendering each ``KernelOp.kernel``
to a ``__global__`` source string. The final graph before codegen is
``Graph[CudaOp + InputOp + ConstantOp]``; the CUDA backend walks it in
topological order, emits one ``kernel_name<<<grid, block>>>(args)``
launch per node, and wires buffer pointers by node id.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.base import Op


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
