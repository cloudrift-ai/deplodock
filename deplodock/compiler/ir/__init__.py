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
- ``block``    — structural kernel IR (``KernelOp`` + ``Port`` / ``Mux`` /
                 ``Combine`` / ``Assign`` SSA tree).
- ``kernel``   — imperative C-like AST (``KernelDef``, ``Stmt`` variants).

See ``ARCHITECTURE.md`` in this directory for stage-by-stage semantics and
invariants.
"""

from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.ir.graph import Graph, Hints, Node, Tensor, resolve_hints

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
