"""CUDA accuracy regression for the wrap-body double-buffer path.

040_use_ring_buffers promotes a wrap-body ``Stage`` inside a
``SerialTile(serial_outer)`` to ``BufferedStage(buffer_count=2,
phase=K_o%2)``. The materializer doubles the smem allocation, prepends
the phase to the cooperative-load Write and to every body Load that
reads from staged smem, and drops the leading ``__syncthreads`` because
consecutive iterations write distinct physical slabs.

Each parametrized config pins planner knobs that force K_o >= 2 (so the
double-buffer eligibility check passes). The test compiles the matmul
through the full CUDA pipeline, asserts the resulting TileOp actually
contains a ``BufferedStage`` (so a regression that silently turns off
the promotion would surface), and then runs the compiled kernel and
compares against a NumpyBackend reference.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import BufferedStage, TileOp

from .conftest import requires_cuda

# (M, K, N) with knob pins that force K_o = K / (BK * SPLITK * BR) >= 2
# AND have a coherent BN/BM/FM/FN tile (BN * FN must divide N; BM * FM must
# divide M). Each row pairs the shape with knob values that previously
# crashed or produced wrong output if the wrap-body double-buffer path
# regressed.
_CASES: tuple[tuple[tuple[int, int, int], dict[str, int]], ...] = (
    # K_o = 256/64 = 4 — standard double-buffered ring.
    ((128, 256, 128), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
    # K_o = 128/64 = 2 — minimum ping-pong (one alternation).
    ((64, 128, 64), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
    # K_o = 512/64 = 8 — deep ping-pong, surfaces phase-prepend bugs where
    # the rewritten Load index miscomputes for K_o > 2 iterations.
    ((64, 512, 64), {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}),
)


def _id(case: tuple[tuple[int, int, int], dict[str, int]]) -> str:
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


def _has_buffered_stage(compiled: Graph) -> bool:
    """The CUDA pipeline lowers TileOp → KernelOp → CudaOp; the BufferedStage
    only lives at the TileOp / KernelOp boundary. We re-compile through
    TILE_PASSES to inspect that intermediate IR."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    # Build a fresh graph (same shape/dims) and run only through TILE_PASSES.
    # Re-using the same graph would mutate node ops; cheaper to rebuild from
    # the source graph's input shapes.
    g = Graph()
    for nid in compiled.inputs:
        node = compiled.nodes[nid]
        g.add_node(op=InputOp(), inputs=[], output=node.output, node_id=nid)
    o_node = compiled.nodes["o"]
    # Skip — we just want to know if BufferedStage shows up for the same
    # shape under the same pinned knobs (env vars are already set by the
    # caller's monkeypatch).
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=o_node.output, node_id="o")
    g.inputs = list(compiled.inputs)
    g.outputs = list(compiled.outputs)
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    if not isinstance(op, TileOp):
        return False
    return any(isinstance(s, BufferedStage) for s in op.body.iter())


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
def test_double_buffer_matmul_accuracy(case, monkeypatch):
    """Pin planner knobs that force K_o >= 2 so 040_use_ring_buffers fires,
    then check the CUDA-compiled kernel matches numpy within fp32 tolerance.

    A regression in the wrap-body promotion (wrong phase index, missed Load
    rewrite, dropped trailing sync) surfaces here as a mismatch — the
    pre-refactor 010 had identical eligibility, so any new divergence is a
    bug in this PR's port."""
    (m, k, n), knobs = case
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))

    graph, input_shapes = _build_matmul(m, k, n)
    inputs = _random_inputs(input_shapes)

    # First: confirm the pass actually fires under these knobs. If 010 stopped
    # firing for whatever reason (eligibility tightened, planner shape
    # changed), the accuracy check below would still pass — but on the
    # plain-Stage path, not BufferedStage — which would silently regress
    # double-buffer coverage. Assert explicitly.
    graph_for_inspection, _ = _build_matmul(m, k, n)
    assert _has_buffered_stage(graph_for_inspection), (
        f"040_use_ring_buffers did not produce a BufferedStage for M={m}, K={k}, N={n} under knobs={knobs}"
    )

    ref = _reference(graph, inputs)
    out = _run_cuda(graph, inputs)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out)), "double-buffered matmul output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(1e-3, 0.05 * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=0.05)
