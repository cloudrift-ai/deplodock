"""Minimal tensor IR and structural-kernel compiler."""

from deplodock.compiler.dump import CompilerDump
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.ir.frontend import ReshapeOp, TransposeOp
from deplodock.compiler.ir.graph import Graph, Node, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, GatherOp, ReduceOp, ScanOp, ScatterOp
from deplodock.compiler.matcher import ChainMatch, Group, Production, match_grammar
from deplodock.compiler.rewriter import Pass, PassTrace, Rewriter, Rule, RuleApplication

__all__ = [
    "ChainMatch",
    "CompilerDump",
    "ConstantOp",
    "ElementwiseOp",
    "GatherOp",
    "Graph",
    "Group",
    "InputOp",
    "Node",
    "Op",
    "Pass",
    "PassTrace",
    "Production",
    "ReduceOp",
    "ReshapeOp",
    "Rewriter",
    "Rule",
    "RuleApplication",
    "ScanOp",
    "ScatterOp",
    "Tensor",
    "TransposeOp",
    "match_grammar",
]
