"""Tests for the backend-agnostic execution plan layer."""

import json
from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.plan import ExecutionPlan, plan_graph
from deplodock.compiler.rewriter import Rewriter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> Graph:
    with open(FIXTURE_DIR / name) as f:
        return Graph.from_dict(json.load(f))


def _compile(g: Graph) -> Graph:
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    return Rewriter.from_directory(rules_dir).apply(g)


def test_plan_graph_on_fixture():
    """plan_graph produces a valid ExecutionPlan from the TinyLlama fixture."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled, name="tinyllama")

    assert isinstance(plan, ExecutionPlan)
    assert plan.name == "tinyllama"
    assert len(plan.buffers) > 0
    assert len(plan.ops) > 0


def test_plan_graph_buffer_roles():
    """Buffers have correct roles derived from graph inputs/outputs/constants."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    inputs = [b for b in plan.buffers if b.role == "input"]
    outputs = [b for b in plan.buffers if b.role == "output"]
    constants = [b for b in plan.buffers if b.role == "constant"]

    assert len(inputs) == 3
    assert len(outputs) == 1
    assert len(constants) >= 9


def test_plan_graph_op_tags():
    """plan_graph produces expected op tags for a compiled transformer block."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    op_tags = {op.op for op in plan.ops}
    assert "matmul" in op_tags
    assert "attention" in op_tags  # FusedAttentionOp (absorbs softmax + 2 matmuls)
    assert "silu_mul" in op_tags


def test_plan_graph_data_flow():
    """All op inputs reference valid buffer names."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    buf_names = {b.name for b in plan.buffers}
    for op in plan.ops:
        for inp in op.inputs:
            assert inp in buf_names, f"Op {op.op} references unknown input {inp!r}"
        for out in op.outputs:
            assert out in buf_names, f"Op {op.op} references unknown output {out!r}"


def test_plan_graph_matmul_count():
    """Correct number of matmul ops after decomposition + fusion."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    matmul_count = sum(1 for op in plan.ops if op.op == "matmul")
    # 7 projections only. QK + attn@V consumed by FusedAttentionOp.
    assert matmul_count == 7, f"Expected 7 matmul, got {matmul_count}"


def test_plan_graph_is_backend_agnostic():
    """ExecutionPlan contains no CUDA-specific information."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    for op in plan.ops:
        assert not hasattr(op, "kernel_source")
        assert not hasattr(op, "grid")
        assert not hasattr(op, "block")
