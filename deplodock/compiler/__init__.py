"""Minimal tensor IR and structural-kernel compiler."""

from deplodock.compiler.dump import CompilerDump
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from deplodock.compiler.ir.graph import Graph, Node, Tensor
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, ReduceOp, ScanOp, ScatterOp
from deplodock.compiler.matcher import Match, Pattern, match_pattern
from deplodock.compiler.rewriter import run_pass

__all__ = [
    "Match",
    "CompilerDump",
    "ConstantOp",
    "ElementwiseOp",
    "GatherOp",
    "Graph",
    "InputOp",
    "Node",
    "Op",
    "Pattern",
    "ReduceOp",
    "ReshapeOp",
    "ScanOp",
    "ScatterOp",
    "Tensor",
    "TransposeOp",
    "match_pattern",
    "run_pass",
]
