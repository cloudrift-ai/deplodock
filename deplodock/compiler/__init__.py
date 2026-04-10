"""Minimal tensor IR and graph transformation engine."""

from deplodock.compiler.benchmark import BenchmarkSuite, run_benchmark_suite
from deplodock.compiler.cuda.lower import MatmulConfig
from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import Match, match_pattern
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    FusedAttentionOp,
    FusedReduceElementwiseOp,
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
from deplodock.compiler.pipeline import compile_and_run
from deplodock.compiler.rewriter import Pass, Rewriter, Rule
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace, RuleApplication

__all__ = [
    "BenchmarkSuite",
    "CompilerTrace",
    "ConstantOp",
    "ElementwiseOp",
    "ExecutionResult",
    "FusedAttentionOp",
    "FusedReduceElementwiseOp",
    "FusedRMSNormOp",
    "FusedSiLUMulOp",
    "FusedSoftmaxOp",
    "GatherOp",
    "Graph",
    "InputOp",
    "Match",
    "MatmulConfig",
    "MatmulOp",
    "Node",
    "Op",
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
    "compile_and_run",
    "match_pattern",
    "parse_pattern",
    "run_benchmark_suite",
]
