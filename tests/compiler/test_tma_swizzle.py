"""Accuracy tests for the matmul kernel under each TMA transport config.

Cross-checks the live capability + transport matrix:

- ``--target sm_80`` → cp.async transport with ``+1`` pad_smem.
- ``--target sm_90`` (default sm_90+) → ``cp.async.bulk.tensor`` /
  TMA with ``swizzle=NONE``.

Swizzle support (``012_split_inner_for_swizzle``) was dropped during the
wrap-body refactor; TMA stages always emit with ``swizzle=NONE``.
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
    out, _ = backend.run(compiled, input_data={"a": a.flatten().tolist(), "b": b.flatten().tolist()})
    return out.outputs["o"].reshape(_M, _N)


def _check_close(actual: np.ndarray, expected: np.ndarray) -> None:
    diff = np.abs(actual - expected)
    peak = float(np.max(np.abs(expected)))
    tol = max(1e-3, _TOL * peak)
    max_diff = float(diff.max())
    assert max_diff < tol, f"max_diff={max_diff:.4f} >= tol={tol:.4f} (peak_eager={peak:.4f})"


def _live_capability() -> tuple[int, int]:
    from deplodock.compiler.target import compute_capability  # noqa: PLC0415

    return compute_capability()


@requires_cuda
@pytest.mark.parametrize(
    ("target", "bn", "bm"),
    [
        ((8, 0), None, None),  # cp.async + pad_smem (default geometry)
        ((9, 0), None, None),  # TMA, swizzle=NONE (default geometry)
        ((9, 0), "32", "32"),  # TMA, swizzle=NONE, small slab (no perf gain, sanity)
    ],
    ids=["cpasync", "tma_no_swizzle", "tma_no_swizzle_bn32"],
)
def test_matmul_accuracy_across_tma_modes(monkeypatch, target, bn, bm):
    """Use ``set_target`` to force the codegen path the test wants. Live
    hardware must be ≥ the requested cap so the launch-time smem opt-in
    (derived from ``Context.max_dynamic_smem``) doesn't exceed the actual
    device cap. Tests requesting a cap above the live device are skipped."""
    from deplodock.compiler import target as target_mod

    live = _live_capability()
    if live < target:
        pytest.skip(f"live device sm_{live[0]}{live[1]} < requested sm_{target[0]}{target[1]}")

    target_mod.set_target(target)
    if bn is not None:
        monkeypatch.setenv("DEPLODOCK_BN", bn)
        monkeypatch.setenv("DEPLODOCK_BM", bm)
    try:
        _, _, expected = _eager()
        actual = _run_matmul()
        _check_close(actual, expected)
    finally:
        target_mod.set_target(None)
