"""Fold ``ReshapeOp(ConstantOp)`` into ``ConstantOp.load_ops`` —
``ReshapeOp`` companion of ``004a_fold_into_constant``."""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import ReshapeOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._fold_constant import fold_into_constant

PATTERN = [Pattern("root", ReshapeOp)]


def rewrite(match: Match, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    return fold_into_constant(graph, root, inp_x, out)
