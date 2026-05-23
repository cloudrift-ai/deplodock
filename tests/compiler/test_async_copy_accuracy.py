"""CUDA accuracy regression for the wrap-body cp.async path.

050_use_async_copy promotes ``BufferedStage`` to ``AsyncBufferedStage(pipeline_depth=1)``
inside ``SerialTile(serial_outer)``. The materializer emits ``CpAsyncCopy`` per
Source + ``CpAsyncCommit + CpAsyncWait(0) + Sync`` at the wrap boundary instead of
the sync cooperative ``Load+Write``.

Each parametrized config pins planner knobs that force K_o >= 2 (so 010 promotes
to BufferedStage, then 013 swaps to AsyncBufferedStage). The test compiles
through the full CUDA pipeline, asserts an ``AsyncBufferedStage`` actually shows
up AND the rendered kernel source carries ``cp.async`` (a regression that silently
falls back to sync would slip through accuracy alone), and runs the compiled
kernel against the NumpyBackend reference.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import AsyncBufferedStage, TileOp

from .conftest import requires_cuda

_CASES: tuple[tuple[tuple[int, int, int], dict[str, int]], ...] = (
    # K_o = 256/64 = 4 — standard async ring.
    ((128, 256, 128), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
    # K_o = 128/64 = 2 — minimum async ping-pong.
    ((64, 128, 64), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
    # K_o = 512/64 = 8 — deeper ring exercises repeated commit/wait pairs.
    ((64, 512, 64), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
)


def _id(case):
    (m, k, n), knobs = case
    return f"M{m}_K{k}_N{n}_BK{knobs['BK']}"


def _build_matmul(m: int, k: int, n: int) -> tuple[Graph, dict[str, tuple[int, ...]]]:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g, {"a": (m, k), "b": (k, n)}


def _has_async_stage(m: int, k: int, n: int) -> bool:
    """Compile a fresh graph through TILE_PASSES and look for AsyncBufferedStage."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g, _ = _build_matmul(m, k, n)
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    if not isinstance(op, TileOp):
        return False
    return any(isinstance(s, AsyncBufferedStage) for s in op.body.iter())


def _kernel_source(m: int, k: int, n: int) -> str:
    """Compile through CUDA_PASSES and return the rendered kernel source."""
    from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

    g, _ = _build_matmul(m, k, n)
    g2 = Pipeline.build(CUDA_PASSES).run(g)
    return g2.nodes["o"].op.kernel_source


def _random_inputs(input_shapes: dict[str, tuple[int, ...]], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in input_shapes.items()}


def _reference(graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    from deplodock.compiler.backend.numpy import NumpyBackend

    be = NumpyBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs["o"]


def _run_cuda(graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    be = CudaBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs["o"]


@requires_cuda
@pytest.mark.parametrize("case", _CASES, ids=_id)
def test_async_copy_matmul_accuracy(case, monkeypatch):
    (m, k, n), knobs = case
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))

    # Regression gate 1: pass actually fires (AsyncBufferedStage present).
    assert _has_async_stage(m, k, n), (
        f"050_use_async_copy did not produce an AsyncBufferedStage for M={m}, K={k}, N={n} under knobs={knobs}"
    )

    # Regression gate 2: rendered kernel really uses cp.async (no silent
    # fall-back to sync inside the materializer).
    src = _kernel_source(m, k, n)
    assert "cp.async.ca.shared.global" in src, "kernel source missing cp.async issue"
    assert "cp.async.commit_group" in src, "kernel source missing cp.async commit"
    assert "cp.async.wait_group 0" in src, "kernel source missing cp.async wait"

    # Accuracy.
    graph, input_shapes = _build_matmul(m, k, n)
    inputs = _random_inputs(input_shapes)
    ref = _reference(graph, inputs)
    out = _run_cuda(graph, inputs)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out)), "cp.async matmul output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(1e-3, 0.05 * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=0.05)
