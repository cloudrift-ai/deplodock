"""Minimal tensor IR and graph transformation engine."""

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import Match, match_pattern
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    FusedAttentionOp,
    FusedRMSNormOp,
    FusedSiLUMulOp,
    FusedSoftmaxOp,
    GatherOp,
    InputOp,
    MatmulOp,
    Op,
    ReduceOp,
    ReshapeOp,
    ScanOp,
    ScatterOp,
    TransposeOp,
)
from deplodock.compiler.pattern import PatternNode, PatternVar, PatternWildcard, parse_pattern
from deplodock.compiler.plan import ExecutionPlan, OpKernel, plan_graph
from deplodock.compiler.rewriter import Pass, Rewriter, Rule
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace, RuleApplication

__all__ = [
    "CompilerTrace",
    "ConstantOp",
    "ElementwiseOp",
    "ExecutionPlan",
    "ExecutionResult",
    "FusedAttentionOp",
    "FusedRMSNormOp",
    "FusedSiLUMulOp",
    "FusedSoftmaxOp",
    "GatherOp",
    "Graph",
    "InputOp",
    "Match",
    "MatmulOp",
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
