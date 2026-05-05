"""CUDA IR — device-level graph-op carrying a rendered CUDA kernel.

Submodules:
- :mod:`.ir` — ``CudaOp`` graph-op, produced by
  ``passes/lowering/cuda`` after a ``TileOp`` is rendered to a
  ``__global__`` source string.

The public surface below re-exports the types so callers use
``from deplodock.compiler.ir.cuda import CudaOp``.
"""

from deplodock.compiler.ir.cuda.ir import CudaOp, TmaDescMeta

__all__ = ["CudaOp", "TmaDescMeta"]
