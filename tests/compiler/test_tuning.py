"""Tests for GPU-specific matmul tuning profiles."""

from __future__ import annotations

from deplodock.compiler.cuda.tuning import default_matmul_strategy_map


def test_pro6000_dispatch():
    smap, name = default_matmul_strategy_map("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition")
    assert name == "rtx_pro_6000"
    # 4096 should pick TM=26 (Pro 6000 sweet spot from a TM 18-28 sweep, vs
    # 5090's TM=20). The surface from TM=24 to TM=28 is very flat (within 1pp).
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096.thread_m == 26
    assert cfg_4096.strategy == "tma_db"
    # 1024 needs k_splits=4 on Pro 6000 (188 SMs) to fill the device.
    cfg_1024 = next(c for thr, c in smap if thr >= 1024)
    assert cfg_1024.k_splits == 4


def test_5090_dispatch():
    smap, name = default_matmul_strategy_map("NVIDIA GeForce RTX 5090")
    assert name == "rtx_5090"
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096.thread_m == 20


def test_unknown_gpu_falls_back_to_5090():
    smap, name = default_matmul_strategy_map("NVIDIA H100 80GB HBM3")
    assert "fallback" in name
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096.thread_m == 20
