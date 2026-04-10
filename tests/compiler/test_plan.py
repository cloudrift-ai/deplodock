"""Tests for the backend-agnostic execution plan layer."""

from deplodock.compiler.block_planner import BlockConfig, plan_block
from deplodock.compiler.plan import ExecutionPlan


def test_plan_block_structure():
    """plan_block returns an ExecutionPlan with correct op count and names."""
    cfg = BlockConfig(batch=1, seq_len=32, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_dim=128)
    plan = plan_block(cfg)

    assert isinstance(plan, ExecutionPlan)
    assert plan.name.startswith("llama_block_")
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
    """Buffers have correct roles (input, output, constant, scratch)."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    roles = {b.role for b in plan.buffers}
    assert roles == {"input", "output", "constant", "scratch"}

    inputs = [b for b in plan.buffers if b.role == "input"]
    outputs = [b for b in plan.buffers if b.role == "output"]
    constants = [b for b in plan.buffers if b.role == "constant"]
    assert len(inputs) == 3  # x, cos, sin
    assert len(outputs) == 1  # output
    assert len(constants) == 9  # w_rms1, Wq, Wk, Wv, Wo, w_rms2, Wg, Wu, Wd

    input_names = {b.name for b in inputs}
    assert input_names == {"x", "cos", "sin"}


def test_plan_block_op_params():
    """OpKernel params carry correct dimension info."""
    cfg = BlockConfig(batch=1, seq_len=8, hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, intermediate_dim=64)
    plan = plan_block(cfg)

    rmsnorm_op = plan.ops[0]
    assert rmsnorm_op.op == "rmsnorm"
    assert rmsnorm_op.params["rows"] == 8  # batch * seq_len
    assert rmsnorm_op.params["dim"] == 32
    assert rmsnorm_op.params["eps"] == cfg.eps

    triple_op = plan.ops[1]
    assert triple_op.op == "triple_matmul"
    assert triple_op.params["M"] == 8
    assert triple_op.params["K"] == 32
    assert triple_op.params["Nq"] == 32  # num_heads * head_dim
    assert triple_op.params["Nk"] == 16  # num_kv_heads * head_dim


def test_plan_block_data_flow():
    """Ops reference valid buffer names."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    buf_names = {b.name for b in plan.buffers}

    for op in plan.ops:
        for inp in op.inputs:
            assert inp in buf_names, f"Op {op.op} references unknown input buffer {inp!r}"
        for out in op.outputs:
            assert out in buf_names, f"Op {op.op} references unknown output buffer {out!r}"


def test_plan_is_backend_agnostic():
    """ExecutionPlan contains no CUDA-specific information."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    # No kernel source, no grid/block, no smem_bytes.
    for op in plan.ops:
        assert not hasattr(op, "kernel_source")
        assert not hasattr(op, "grid")
        assert not hasattr(op, "block")

    # BufferSpec has shape tuples, not flat sizes.
    for buf in plan.buffers:
        assert isinstance(buf.shape, tuple)
