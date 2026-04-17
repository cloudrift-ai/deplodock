"""Loop backend: pure-Python interpreter for ``LoopProgram``.

Evaluates a post-fusion ``LoopProgram`` via numpy whole-tensor ops. Used
as a reference/debugging backend — sits between the ``NumpyBackend``
(evaluates the pre-fusion Graph) and the ``CudaBackend`` (lowers to
``GpuProgram`` and runs via nvcc) so bugs can be attributed to fusion vs
codegen.
"""

from deplodock.compiler.backend.loop.backend import LoopBackend, WrappedLoopProgram

__all__ = ["LoopBackend", "WrappedLoopProgram"]
