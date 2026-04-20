"""Program-form dialects.

One file per program level, parallel to ``ir/``:

- ``loop`` — ``LoopProgram`` + ``LoopBuffer`` + ``LoopLaunch``. Built by
  ``compile_graph`` after fusion; authoritative shape + launch-order view.
  Pairs with ``ir/loop/ir.py``.
- ``gpu`` — ``GpuProgram`` + ``GpuBuffer`` + ``GpuLaunch``. Produced by
  backend codegen (``backend/cuda/emit``). Pairs with ``ir/kernel_ir.py``.

See ``ARCHITECTURE.md`` for the symmetric pattern (per-IR program form).
"""

from deplodock.compiler.program.gpu import GpuBuffer, GpuLaunch, GpuProgram
from deplodock.compiler.program.loop import LoopBuffer, LoopLaunch, LoopProgram

__all__ = [
    "GpuBuffer",
    "GpuLaunch",
    "GpuProgram",
    "LoopBuffer",
    "LoopLaunch",
    "LoopProgram",
]
