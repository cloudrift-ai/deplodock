"""Tests for the backend-agnostic execution plan layer."""

import json
from pathlib import Path

from deplodock.compiler.block_planner import BlockConfig, plan_block
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


# ---- plan_graph tests (graph-driven planning) ----


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
    assert "softmax" in op_tags
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
    assert matmul_count == 9, f"Expected 9 matmul, got {matmul_count}"


def test_plan_graph_is_backend_agnostic():
    """ExecutionPlan contains no CUDA-specific information."""
    compiled = _compile(_load_fixture("tinyllama_layer0.json"))
    plan = plan_graph(compiled)

    for op in plan.ops:
        assert not hasattr(op, "kernel_source")
        assert not hasattr(op, "grid")
        assert not hasattr(op, "block")


# ---- plan_block tests (config-driven, legacy) ----


def test_plan_block_structure():
    """plan_block returns an ExecutionPlan with correct op count and names."""
    cfg = BlockConfig(batch=1, seq_len=32, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_dim=128)
    plan = plan_block(cfg)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.ops) == 10

    op_names = [op.op for op in plan.ops]
    assert op_names == [
        "rmsnorm",
        "triple_matmul",
        "rope",
        "attention_qk",
        "attention_softmax",
        "attention_sv",
        "matmul_residual_add",
        "rmsnorm",
        "dual_matmul_silu_mul",
        "matmul_residual_add",
    ]


def test_plan_block_buffer_roles():
    """Buffers have correct roles."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    inputs = [b for b in plan.buffers if b.role == "input"]
    outputs = [b for b in plan.buffers if b.role == "output"]
    constants = [b for b in plan.buffers if b.role == "constant"]
    assert len(inputs) == 3
    assert len(outputs) == 1
    assert len(constants) == 9
