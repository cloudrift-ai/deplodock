"""Minimal tensor IR and graph transformation engine."""

from deplodock.compiler.dump import CompilerDump
from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import Match, match_pattern
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    GatherOp,
    InputOp,
    Op,
    ReduceOp,
    ReshapeOp,
    ScanOp,
    ScatterOp,
    TransposeOp,
)
from deplodock.compiler.pattern import PatternNode, PatternVar, PatternWildcard, parse_pattern
from deplodock.compiler.plan import ExecutionPlan, OpKernel, plan_graph
from deplodock.compiler.rewriter import Pass, PassTrace, Rewriter, Rule, RuleApplication

__all__ = [
    "CompilerDump",
    "ConstantOp",
    "ElementwiseOp",
    "ExecutionPlan",
    "GatherOp",
    "Graph",
    "InputOp",
    "Match",
    "Node",
    "Op",
    "OpKernel",
    "Pass",
    "PassTrace",
    "PatternNode",
    "PatternVar",
    "PatternWildcard",
    "ReduceOp",
    "ReshapeOp",
    "Rewriter",
    "Rule",
    "RuleApplication",
    "ScanOp",
    "ScatterOp",
    "Tensor",
    "TransposeOp",
    "match_pattern",
    "parse_pattern",
    "plan_graph",
]
