"""Emmy compiler IR dialects.

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
- ``tile``     — Tile IR (``TileOp`` + ``Tile`` / ``Tile`` / ``Coop`` /
                 ``Sync`` schedule wrappers, reusing Loop IR leaves).
                 Lowers directly to CUDA source.
- ``cuda``     — Device IR (``CudaOp`` carrying rendered ``__global__``
                 source + launch geometry).

The ``Graph`` container and rewrite pipeline live outside ``ir/`` — see
``emmy.compiler.graph`` and ``emmy.compiler.pipeline``.

See ``ARCHITECTURE.md`` in this directory for stage-by-stage semantics
and invariants.
"""

from emmy.compiler.ir.base import ConstantOp, InputOp, Op

__all__ = [
    "ConstantOp",
    "InputOp",
    "Op",
]
