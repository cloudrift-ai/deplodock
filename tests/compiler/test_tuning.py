"""Tests for GPU-specific matmul tuning profiles."""

from __future__ import annotations

from deplodock.compiler.backend.cuda.backend import _compile_single
from deplodock.compiler.backend.cuda.tuning import default_matmul_strategy_map
from deplodock.compiler.ops import ElementwiseOp, ReduceOp
from deplodock.compiler.plan import OpKernel


def test_pro6000_dispatch():
    smap, name = default_matmul_strategy_map("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition")
    assert name == "rtx_pro_6000"
    # 4096 should pick TM=26 (Pro 6000 sweet spot from a TM 18-28 sweep, vs
    # 5090's TM=20). The surface from TM=24 to TM=28 is very flat (within 1pp).
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096["thread_m"] == 26
    assert cfg_4096["strategy"] == "tma_db"
    # 1024 needs k_splits=4 on Pro 6000 (188 SMs) to fill the device.
    cfg_1024 = next(c for thr, c in smap if thr >= 1024)
    assert cfg_1024["k_splits"] == 4


def test_5090_dispatch():
    smap, name = default_matmul_strategy_map("NVIDIA GeForce RTX 5090")
    assert name == "rtx_5090"
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096["thread_m"] == 20


def test_unknown_gpu_falls_back_to_5090():
    smap, name = default_matmul_strategy_map("NVIDIA H100 80GB HBM3")
    assert "fallback" in name
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096["thread_m"] == 20


def _make_matmul_op(m, n, k, *, epilogue_ops=None, extra_inputs=None, extra_input_shapes=None):
    """Build an OpKernel for a matmul, optionally with epilogue ops."""
    region_ops = [
        ("ew", ElementwiseOp("mul"), ["A", "B"]),
        ("red", ReduceOp("sum", axis=1), ["ew"]),
    ]
    if epilogue_ops:
        region_ops.extend(epilogue_ops)
    inputs = ["A", "B"] + (extra_inputs or [])
    input_shapes = {"A": (m, k), "B": (k, n)}
    if extra_input_shapes:
        input_shapes.update(extra_input_shapes)
    return OpKernel(
        op="fused_region",
        inputs=inputs,
        outputs=["C"],
        params={
            "M": m,
            "N": n,
            "K": k,
            "shape": (m, n),
            "_region_ops": region_ops,
            "_input_shapes": input_shapes,
        },
    )


def test_m_aware_k_splits():
    """For M=32 (< tile_m=64), k_splits should be > 1 to fill the GPU."""
    op = _make_matmul_op(32, 3584, 3584)
    launch = _compile_single(op)
    # grid.z = k_splits
    assert launch.grid[2] > 1, f"Expected k_splits > 1 for M=32, got grid={launch.grid}"


def test_matmul_epilogue_disables_k_splits():
    """When epilogue is present, k_splits must be 1 even for small M."""
    op = _make_matmul_op(
        32,
        3584,
        3584,
        epilogue_ops=[("ba", ElementwiseOp("add"), ["red", "bias"])],
        extra_inputs=["bias"],
        extra_input_shapes={"bias": (3584,)},
    )
    launch = _compile_single(op)
    assert launch.grid[2] == 1, f"k_splits must be 1 with epilogue, got grid={launch.grid}"
