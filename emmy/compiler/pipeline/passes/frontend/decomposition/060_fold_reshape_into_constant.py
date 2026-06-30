"""Fold ``ReshapeOp(ConstantOp)`` into ``ConstantOp.load_ops`` —
``ReshapeOp`` companion of ``050_fold_into_constant``."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.frontend.ir import ReshapeOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._fold_constant import fold_into_constant

PATTERN = [Pattern("root", ReshapeOp)]


def rewrite(match: Match, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    return fold_into_constant(graph, root, inp_x, out)
