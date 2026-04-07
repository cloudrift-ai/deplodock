"""Tests for GPU-specific matmul tuning profiles."""

from __future__ import annotations

from deplodock.compiler.cuda.tuning import default_matmul_strategy_map


def test_pro6000_dispatch():
    smap, name = default_matmul_strategy_map("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition")
    assert name == "rtx_pro_6000"
    # 4096 should pick TM=24 (the Pro 6000 sweet spot, vs 5090's TM=20).
    cfg_4096 = next(c for thr, c in smap if thr >= 4096)
    assert cfg_4096.thread_m == 24
    assert cfg_4096.strategy == "tma_db"


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
