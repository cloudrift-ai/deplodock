"""Minimal tensor IR and graph transformation engine."""

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import Match, match_pattern
from deplodock.compiler.ops import (
    ElementwiseOp,
    FusedReduceElementwiseOp,
    GatherOp,
    InputOp,
    Op,
    ReduceOp,
    ScanOp,
    ScatterOp,
)
from deplodock.compiler.pattern import PatternNode, PatternVar, PatternWildcard, parse_pattern
from deplodock.compiler.pipeline import compile_and_run
from deplodock.compiler.rewriter import Pass, Rewriter, Rule
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace, RuleApplication

__all__ = [
    "CompilerTrace",
    "ElementwiseOp",
    "FusedReduceElementwiseOp",
    "GatherOp",
    "Graph",
    "InputOp",
    "Match",
    "Node",
    "Op",
    "Pass",
    "PatternNode",
    "PatternVar",
    "PatternWildcard",
    "ReduceOp",
    "Rewriter",
    "Rule",
    "ScanOp",
    "ScatterOp",
    "Tensor",
    "compile_and_run",
    "ExecutionResult",
    "match_pattern",
    "parse_pattern",
    "PassTrace",
    "RuleApplication",
]
