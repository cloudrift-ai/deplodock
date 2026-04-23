"""Deplodock compiler IR dialects.

One file per IR level:

- ``base``     — ``Op`` base class + ``InputOp`` / ``ConstantOp`` sentinels.
- ``graph``    — ``Tensor``, ``Node``, ``Graph`` container + ``Hints``.
- ``expr``     — shared expression AST (``Var``, ``BinOp``, ...) +
                 coord-expression helpers for ``IndexMapOp``.
- ``frontend`` — Torch-captured ops (``LinearOp``, ``MatmulOp``, ``SdpaOp``,
                 ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``,
                 ``SliceOp``, ``CatOp``).
- ``tensor``   — minimal post-decomposition IR (``ElementwiseOp``,
                 ``ReduceOp``, ``ScanOp``, ``GatherOp``, ``ScatterOp``,
                 ``IndexMapOp``).
- ``loop``     — loop IR (``LoopOp`` + ``Axis`` + body-form ``Load`` +
                 SSA ``Assign`` / ``Accum`` / ``Write`` / ``Select``
                 statements). One ``LoopOp`` per GPU kernel.
- ``gpu``      — GPU IR (imperative C-like AST: ``GpuKernel``, ``Stmt``
                 variants, ``ArrayAccess``, ``Cast``, ``VectorLoad``, ...).

See ``ARCHITECTURE.md`` in this directory for stage-by-stage semantics and
invariants.
"""

from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.graph import Graph, Hints, Node, Tensor, resolve_hints

__all__ = [
    "ConstantOp",
    "Graph",
    "Hints",
    "InputOp",
    "Node",
    "Op",
    "Tensor",
    "resolve_hints",
]
