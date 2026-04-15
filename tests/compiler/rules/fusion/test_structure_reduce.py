"""Tests for rules/fusion/002_structure_reduce.py."""

import importlib.util
from pathlib import Path

from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.ops import (
    ElementwiseOp,
    InputOp,
    KernelOp,
    ReduceOp,
    ReduceStage,
)
from deplodock.compiler.pattern import parse_pattern

RULES_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "deplodock"
    / "compiler"
    / "rules"
    / "fusion"
)


def _load_rule(name: str):
    spec = importlib.util.spec_from_file_location(name, RULES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_rule(graph: Graph, rule_name: str) -> Graph:
    rule = _load_rule(rule_name)
    pattern = parse_pattern(rule.PATTERN)
    changed = True
    while changed:
        changed = False
        for match in match_pattern(graph, pattern):
            new_graph = rule.rewrite(graph, match)
            if new_graph is not graph:
                graph = new_graph
                changed = True
                break
    return graph


def _apply_pipeline(graph: Graph) -> Graph:
    """Run contraction then reduce rules in order (mimics Pass behavior)."""
    graph = _apply_rule(graph, "001_structure_contraction")
    graph = _apply_rule(graph, "002_structure_reduce")
    return graph


def _make_rmsnorm_graph(n: int = 8, d: int = 16) -> Graph:
    """Build a simplified RMSNorm: mul(x, rsqrt(mean(mul(x,x), axis=-1)))."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (n, d)), node_id="x")
    g.inputs = [x]
    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (n, d)), node_id="sq")
    s = g.add_node(ReduceOp("sum", axis=-1), [sq], Tensor("s", (n,)), node_id="s")
    r = g.add_node(ElementwiseOp("rsqrt"), [s], Tensor("r", (n,)), node_id="r")
    y = g.add_node(ElementwiseOp("mul"), [x, r], Tensor("y", (n, d)), node_id="y")
    g.outputs = [y]
    return g


def _make_softmax_graph(n: int = 4, d: int = 8) -> Graph:
    """Build softmax: exp(x - max) / sum(exp(x - max))."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (n, d)), node_id="x")
    g.inputs = [x]
    mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (n,)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (n, d)), node_id="sub")
    ex = g.add_node(ElementwiseOp("exp"), [sub], Tensor("ex", (n, d)), node_id="ex")
    sm = g.add_node(ReduceOp("sum", axis=-1), [ex], Tensor("sm", (n,)), node_id="sm")
    dv = g.add_node(ElementwiseOp("div"), [ex, sm], Tensor("dv", (n, d)), node_id="dv")
    g.outputs = [dv]
    return g


def test_single_reduce_becomes_one_stage_core():
    """sum(x*x) then mul → core = (ReduceStage(pre_ops=(sq,), reduce=sum),)."""
    g = _make_rmsnorm_graph()
    fused = auto_fuse(g)
    structured = _apply_pipeline(fused)

    kernels = [n for n in structured.nodes.values() if isinstance(n.op, KernelOp)]
    # Rule should fire on at least one kernel; find one with core.
    with_core = [k for k in kernels if isinstance(k.op.core, tuple) and k.op.core]
    assert with_core, f"expected at least one reduce-structured kernel; got cores {[k.op.core for k in kernels]}"

    kernel = with_core[0].op
    assert isinstance(kernel.core, tuple)
    # Stages: one reduce.
    assert len(kernel.core) == 1
    stage = kernel.core[0]
    assert isinstance(stage, ReduceStage)
    assert isinstance(stage.reduce.op, ReduceOp)


def test_softmax_becomes_two_stage_core():
    """div(exp(x - max), sum(exp(x - max))) → core with two ReduceStages."""
    g = _make_softmax_graph()
    fused = auto_fuse(g)
    structured = _apply_pipeline(fused)

    kernels = [n for n in structured.nodes.values() if isinstance(n.op, KernelOp)]
    with_multireduce = [k for k in kernels if isinstance(k.op.core, tuple) and len(k.op.core) >= 2]
    # Softmax may split into multiple kernels by auto_fuse; find any with 2+ stages.
    if with_multireduce:
        kernel = with_multireduce[0].op
        assert len(kernel.core) == 2
        assert isinstance(kernel.core[0].reduce.op, ReduceOp)
        assert kernel.core[0].reduce.op.fn == "max"
        assert kernel.core[1].reduce.op.fn == "sum"
        # Inter-reduce ops: sub and exp.
        assert len(kernel.core[1].pre_ops) == 2
    else:
        # Softmax may be split across multiple single-reduce kernels — verify at least
        # one max and one sum reduce are present in some kernel's core.
        fns = []
        for k in kernels:
            if isinstance(k.op.core, tuple):
                for stage in k.op.core:
                    fns.append(stage.reduce.op.fn)
        assert "max" in fns and "sum" in fns, f"expected max + sum reduces; got {fns}"


def test_reduce_rule_noop_on_pointwise_kernel():
    """Pure elementwise kernel (no reduce) stays with core=None."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 4)), node_id="x")
    g.inputs = [x]
    y = g.add_node(ElementwiseOp("add"), [x, x], Tensor("y", (4, 4)), node_id="y")
    g.outputs = [y]
    fused = auto_fuse(g)
    structured = _apply_rule(fused, "002_structure_reduce")

    for n in structured.nodes.values():
        if isinstance(n.op, KernelOp):
            assert n.op.core is None


def test_rules_ordered_contraction_wins_over_reduce():
    """A sum(mul(A, B)) kernel lands as ContractionCore, not a ReduceCore."""
    from deplodock.compiler.ops import ContractionCore

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    g.inputs = [a, b]
    mul = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("mul", (4, 8, 16)), node_id="mul")
    c = g.add_node(ReduceOp("sum", axis=1), [mul], Tensor("C", (4, 16)), node_id="C")
    g.outputs = [c]

    fused = auto_fuse(g)
    structured = _apply_pipeline(fused)
    kernels = [n for n in structured.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    # Contraction rule should fire first, producing ContractionCore (not a ReduceCore).
    assert isinstance(kernels[0].op.core, ContractionCore)
