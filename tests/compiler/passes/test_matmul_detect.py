"""Tests for the matmul detector pass (loop/matmul/001_detect_matmul.py).

The pass is annotate-only: it sets ``cuda.matmul.*`` hints on LoopOp nodes
whose body is a canonical contraction, and leaves the graph otherwise
unchanged. Tests verify both positive detection (matmul, linear) and
negative (elementwise-only, softmax-like kernels).
"""

from pathlib import Path

import torch
import torch.nn as nn

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import LOOP_PASSES
from deplodock.compiler.pipeline.engine import run_pass
from deplodock.compiler.trace.torch import trace_module

_PASSES_ROOT = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "pipeline" / "passes"


def _compile_loop(graph: Graph) -> Graph:
    for name in LOOP_PASSES:
        graph = run_pass(graph, _PASSES_ROOT / name)
    return graph


def _loop_nodes(graph: Graph):
    return [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]


def test_detect_linear_matmul():
    # 32×2048 @ 2048×2048 — one of TinyLlama's Q/O shapes; fits default tile.
    m = nn.Linear(2048, 2048, bias=False)
    g = trace_module(m, (torch.randn(32, 2048),))
    g = _compile_loop(g)
    kernels = _loop_nodes(g)
    assert len(kernels) == 1
    h = kernels[0].hints
    assert h.get("cuda.matmul.strategy") == "tma_matmul"
    assert h.get("cuda.matmul.m") == 32
    assert h.get("cuda.matmul.n") == 2048
    assert h.get("cuda.matmul.k") == 2048
    assert h.get("cuda.matmul.tile_m") == 32
    assert h.get("cuda.matmul.tile_n") == 32
    assert h.get("cuda.matmul.block_k") == 64


def test_skip_divisibility():
    # 4×8 @ 8×16 — M=4 doesn't divide BM=32, so we fall back to scalar.
    m = nn.Linear(8, 16, bias=False)
    g = trace_module(m, (torch.randn(4, 8),))
    g = _compile_loop(g)
    kernels = _loop_nodes(g)
    assert len(kernels) == 1
    assert not kernels[0].hints.has("cuda.matmul.strategy")


def test_skip_elementwise_only():
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (16, 32)))
    b = g.add_node(InputOp(), [], Tensor("B", (16, 32)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("C", (16, 32)))
    g.outputs = [ew]

    g = _compile_loop(g)
    for node in _loop_nodes(g):
        assert not node.hints.has("cuda.matmul.strategy")


def test_detect_matmul_shape_mismatch():
    """ElementwiseOp(mul) → ReduceOp over an inner axis doesn't match the
    A[m,k]·B[n,k]→C[m,n] fingerprint (middle-axis reduce instead of trailing),
    so it should not be flagged as tma_matmul."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8, 4)))
    b = g.add_node(InputOp(), [], Tensor("B", (4, 8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(fn="sum", axis=1), [ew], Tensor("C", (4, 1, 4)))
    g.outputs = [c]

    g = _compile_loop(g)
    for node in _loop_nodes(g):
        assert not node.hints.has("cuda.matmul.strategy")
