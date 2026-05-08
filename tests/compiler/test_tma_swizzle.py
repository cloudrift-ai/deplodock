"""Accuracy tests for the matmul kernel under each TMA + swizzle config.

Cross-checks four points in the env-flag matrix:

- ``DEPLODOCK_TMA=0`` → cp.async transport with ``+1`` pad_smem.
- ``DEPLODOCK_TMA=1`` (default sm_90+) → ``cp.async.bulk.tensor`` /
  TMA with ``swizzle=NONE``.
- ``DEPLODOCK_TMA_SWIZZLE=1`` (paired with a small ``BN/BM`` so the
  inner box-dim byte size matches a swizzle width) → TMA with
  ``SWIZZLE_128B`` + body-Load XOR decode.

Pinned to a matmul size large enough that the K-outer pipeline + the
swizzle decoder both fire (``N=384`` triggers ``010_double_buffer``,
``011_tma_copy``, and ``015_pipeline_k_outer``). Swizzle-on at the
default ``BN=128`` geometry stays at ``swizzle=NONE`` (inner = 512 B,
no swizzle width matches), so the swizzle row only meaningfully runs
when paired with ``BN=BM=32``.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import requires_cuda

_M = _N = _K = 384
_TOL = 1e-2  # generous fp32 reduction-order tolerance


def _eager() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=1234)
    a = rng.standard_normal((_M, _K), dtype=np.float32)
    b = rng.standard_normal((_K, _N), dtype=np.float32)
    return a, b, a @ b


def _matmul_graph():
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (_M, _K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_K, _N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _run_matmul() -> np.ndarray:
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    a, b, _ = _eager()
    backend = CudaBackend()
    compiled = backend.compile(_matmul_graph())
    out = backend.run(compiled, input_data={"a": a.flatten().tolist(), "b": b.flatten().tolist()})
    return out.outputs["o"].reshape(_M, _N)


def _check_close(actual: np.ndarray, expected: np.ndarray) -> None:
    diff = np.abs(actual - expected)
    peak = float(np.max(np.abs(expected)))
    tol = max(1e-3, _TOL * peak)
    max_diff = float(diff.max())
    assert max_diff < tol, f"max_diff={max_diff:.4f} >= tol={tol:.4f} (peak_eager={peak:.4f})"


@requires_cuda
@pytest.mark.parametrize(
    ("tma", "swizzle", "bn", "bm"),
    [
        ("0", "0", None, None),  # cp.async + pad_smem (default geometry)
        ("1", "0", None, None),  # TMA, swizzle=NONE (default geometry)
        ("1", "0", "32", "32"),  # TMA, swizzle=NONE, small slab (no perf gain, sanity)
        ("1", "1", "32", "32"),  # TMA + SWIZZLE_128B + body-Load XOR decode
    ],
    ids=["cpasync", "tma_no_swizzle", "tma_no_swizzle_bn32", "tma_swizzle_b128"],
)
def test_matmul_accuracy_across_tma_modes(monkeypatch, tma, swizzle, bn, bm):
    monkeypatch.setenv("DEPLODOCK_TMA", tma)
    monkeypatch.setenv("DEPLODOCK_TMA_SWIZZLE", swizzle)
    if bn is not None:
        monkeypatch.setenv("DEPLODOCK_BN", bn)
        monkeypatch.setenv("DEPLODOCK_BM", bm)

    _, _, expected = _eager()
    actual = _run_matmul()
    _check_close(actual, expected)


def test_swizzle_picker_pre_pass(monkeypatch):
    """Picker (in ``012_split_inner_for_swizzle``) maps inner byte size
    to a ``(swizzle_mode, ips_fp32)`` pair. Wider-than-width extents pick
    the largest fitting width. Pure unit test, no CUDA needed."""
    import importlib

    from deplodock.compiler.ir.tile.ir import SwizzleMode

    pick = importlib.import_module(
        "deplodock.compiler.pipeline.passes.lowering.tile.012_split_inner_for_swizzle",
    )._pick_split_swizzle

    monkeypatch.setenv("DEPLODOCK_TMA_SWIZZLE", "1")  # picker doesn't gate; just ensure no env effect
    assert pick(128) == (SwizzleMode.B128, 32)  # exact B128
    assert pick(64) == (SwizzleMode.B64, 16)  # exact B64
    assert pick(32) == (SwizzleMode.B32, 8)  # exact B32
    assert pick(512) == (SwizzleMode.B128, 32)  # 4× B128 → split
    assert pick(256) == (SwizzleMode.B128, 32)  # 2× B128 → split
    assert pick(192) == (SwizzleMode.B64, 16)  # 3× B64 → split
    assert pick(16) is None  # below smallest width
