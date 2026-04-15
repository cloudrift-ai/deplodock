"""Tests for rules/fusion/001_structure_contraction.py.

Verifies that a flat-prologue KernelOp containing sum(mul(A, B)) gets
restructured into a KernelOp with ContractionCore.
"""

import importlib.util
from pathlib import Path

from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    InputOp,
    KernelOp,
    ReduceOp,
)
from deplodock.compiler.pattern import parse_pattern

RULE_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "deplodock"
    / "compiler"
    / "rules"
    / "fusion"
    / "001_structure_contraction.py"
)


def _load_rule():
    spec = importlib.util.spec_from_file_location("structure_contraction", RULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_rule(graph: Graph) -> Graph:
    rule = _load_rule()
    pattern = parse_pattern(rule.PATTERN)
    # Apply until fixed point.
    changed = True
    while changed:
        changed = False
        matches = match_pattern(graph, pattern)
        for match in matches:
            new_graph = rule.rewrite(graph, match)
            if new_graph is not graph:
                graph = new_graph
                changed = True
                break
    return graph


def _make_matmul_graph(m: int = 4, k: int = 8, n: int = 16) -> Graph:
    """Build A(M,K) @ B(K,N) via sum(mul)."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    # Broadcast a(M,K,1) * b(1,K,N) → (M,K,N), reduce k → (M,N). The tracer
    # emits this via explicit unsqueezes in reality; here we mock it.
    mul_out = g.add_node(
        ElementwiseOp("mul"),
        [a, b],
        Tensor("mul", (m, k, n)),
        node_id="mul",
    )
    red = g.add_node(
        ReduceOp("sum", axis=1),
        [mul_out],
        Tensor("C", (m, n)),
        node_id="C",
    )
    g.outputs = [red]
    return g


def test_rule_structures_flat_matmul_kernel():
    """sum(mul(A,B)) → KernelOp(core=ContractionCore(...))."""
    g = _make_matmul_graph()
    fused = auto_fuse(g)
    # auto_fuse produces a flat-prologue KernelOp.
    kernels_before = [n for n in fused.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels_before) == 1
    assert kernels_before[0].op.core is None, "auto_fuse should leave core=None"

    structured = _apply_rule(fused)

    kernels_after = [n for n in structured.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels_after) == 1
    core = kernels_after[0].op.core
    assert isinstance(core, ContractionCore), f"expected ContractionCore, got {type(core).__name__}"
    assert core.a.buffer_id == "A"
    assert core.b.buffer_id == "B"
    # K axis is the last dim of A (1 since A is 2D).
    assert core.k_axis == 1


def test_rule_is_noop_on_non_kernel_ops():
    """Rule returns same graph when applied to non-KernelOp nodes."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4,)), node_id="x")
    g.inputs = [x]
    g.outputs = [x]

    result = _apply_rule(g)
    # No KernelOps to structure; graph returned unchanged.
    assert result is g or len(result.nodes) == len(g.nodes)


def test_rule_is_noop_on_already_structured_kernel():
    """Applying the rule twice doesn't change an already-structured KernelOp."""
    g = _make_matmul_graph()
    fused = auto_fuse(g)
    once = _apply_rule(fused)
    twice = _apply_rule(once)
    # Second application should be idempotent — still one KernelOp with ContractionCore.
    kernels = [n for n in twice.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    assert isinstance(kernels[0].op.core, ContractionCore)


def test_rule_skips_non_matmul_kernels():
    """KernelOps that aren't matmul-shaped stay as flat-prologue (core=None)."""
    # Build a pure elementwise graph: y = x + x (no reduce, no contraction).
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 4)), node_id="x")
    g.inputs = [x]
    y = g.add_node(ElementwiseOp("add"), [x, x], Tensor("y", (4, 4)), node_id="y")
    g.outputs = [y]

    fused = auto_fuse(g)
    result = _apply_rule(fused)
    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    # If fusion produced a kernel (auto_fuse might not for single-op graphs), its core stays None.
    for k in kernels:
        assert k.op.core is None
