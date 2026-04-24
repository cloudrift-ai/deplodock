"""Deplodock compiler IR dialects.

One subpackage per IR level:

- ``base``     — ``Op`` base class + ``InputOp`` / ``ConstantOp`` sentinels.
- ``expr``     — shared expression AST (``Var``, ``BinaryExpr``, ...) +
                 coord-expression helpers for ``IndexMapOp``.
- ``frontend`` — Torch-captured ops (``LinearOp``, ``MatmulOp``, ``SdpaOp``,
                 ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``,
                 ``SliceOp``, ``CatOp``).
- ``tensor``   — minimal post-decomposition IR (``ElementwiseOp``,
                 ``ReduceOp``, ``ScanOp``, ``GatherOp``, ``ScatterOp``,
                 ``IndexMapOp``).
- ``loop``     — Loop IR (``LoopOp`` + ``Axis`` + body-form ``Load`` +
                 SSA ``Assign`` / ``Accum`` / ``Write`` / ``Select``
                 statements). One ``LoopOp`` per GPU kernel.
- ``kernel``   — Kernel IR: C-like AST (``GpuKernel``, ``Stmt`` variants,
                 ``ArrayAccess``, ``CastExpr``, ``VectorLoad``, ...) wrapped in
                 a ``KernelOp`` graph-op.
- ``cuda``     — Device IR (``CudaOp`` carrying rendered ``__global__``
                 source + launch geometry).

The ``Graph`` container and rewrite pipeline live outside ``ir/`` — see
``deplodock.compiler.graph`` and ``deplodock.compiler.pipeline``.

See ``ARCHITECTURE.md`` in this directory for stage-by-stage semantics
and invariants.
"""

from deplodock.compiler.ir.base import ConstantOp, InputOp, Op

__all__ = [
    "ConstantOp",
    "InputOp",
    "Op",
]
