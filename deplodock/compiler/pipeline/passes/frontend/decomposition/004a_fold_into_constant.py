"""Fold ``TransposeOp(ConstantOp)`` into ``ConstantOp.load_ops``.

The fold appends the ``TransposeOp`` to the constant's ``load_ops`` and
produces a fresh ``ConstantOp`` whose ``output.shape`` is the
post-transpose shape. At bind time the loader reads the source tensor
(from safetensors or a live ``nn.Module``) and replays the recorded
``load_ops`` chain through the reference NumPy backend, so downstream
Loads see the post-chain layout.

Why fold here rather than later: a ``TransposeOp`` lowered through
``010_transpose`` becomes an ``IndexMapOp``, which gets fused into
consumer Loads' index expressions. The runtime tensor stays in its
original layout and the access pattern reads the transposed element of
the original storage. That's correct but defeats the smem layout
cuBLAS-style SGEMM kernels rely on (see ``007_stage_inputs``) and
prevents TMA on the asymmetric ``(BN, BM)`` tile shape (see
``010_tma_copy``). Pre-folding the constant solves both without
changing the rest of the graph.

Companion rule ``004b_fold_reshape_into_constant`` does the same for
``ReshapeOp``. The shared body is in ``_fold_constant.py``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import TransposeOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._fold_constant import fold_into_constant

PATTERN = [Pattern("root", TransposeOp)]


def rewrite(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    return fold_into_constant(graph, root, inp_x, out)
