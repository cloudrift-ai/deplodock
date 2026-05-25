"""CUDA accuracy regression for the wrap-body bank-pad path.

070_pad_smem is a BOOL ``PAD_SMEM`` fork. Under PAD_SMEM=True the
materializer reads each ``Source.alloc_extents`` with the ``+1`` pad
folded in, which (1) grows the smem allocation and (2) shifts every
higher-stride row by one float so body Loads' bank distribution flattens.
The kernel must still match the NumpyBackend reference under both
polarities — padding is purely a perf knob, never a correctness one.

Each parametrized config pins planner knobs so the pass actually fires
(at least one source has a fixable conflict). PAD_SMEM is parametrized
explicitly via ``DEPLODOCK_PAD_SMEM`` so both branches of the autotune
fork get measured for accuracy on every run.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import StageBundle, StagePolicy, TileOp

from .conftest import requires_cuda

_SHAPES: tuple[tuple[int, int, int], ...] = (
    (128, 256, 128),
    (64, 128, 64),
)
_PIN_KNOBS = {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}


def _build_matmul(m: int, k: int, n: int) -> tuple[Graph, dict[str, tuple[int, ...]]]:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g, {"a": (m, k), "b": (k, n)}


def _tile_op_pad_summary(m: int, k: int, n: int) -> dict[str, tuple[int, ...]]:
    """Compile through TILE_PASSES and return ``{source_name: pad_tuple}``."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g, _ = _build_matmul(m, k, n)
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    if not isinstance(op, TileOp):
        return {}
    out: dict[str, tuple[int, ...]] = {}
    for s in op.body.iter():
        if isinstance(s, StageBundle) and s.policy == StagePolicy.BUFFERED:
            for src in s.sources:
                out[src.name] = src.pad
    return out


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


def _id(case):
    (m, k, n), polarity = case
    return f"M{m}_K{k}_N{n}_pad{polarity}"


_CASES = [(shape, pol) for shape in _SHAPES for pol in ("true", "false")]


@requires_cuda
@pytest.mark.parametrize("case", _CASES, ids=_id)
def test_pad_smem_matmul_accuracy(case, monkeypatch):
    (m, k, n), polarity = case
    for key, value in _PIN_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))
    monkeypatch.setenv("DEPLODOCK_PAD_SMEM", polarity)

    pads = _tile_op_pad_summary(m, k, n)
    if polarity == "true":
        assert any(p and any(p) for p in pads.values()), f"PAD_SMEM=true produced no padded source for M={m}, K={k}, N={n}: {pads}"
    else:
        assert all(not p or not any(p) for p in pads.values()), f"PAD_SMEM=false leaked pad: {pads}"

    graph, input_shapes = _build_matmul(m, k, n)
    inputs = _random_inputs(input_shapes)
    ref = _reference(graph, inputs)
    out = _run_cuda(graph, inputs)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out)), f"PAD_SMEM={polarity} output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(1e-3, 0.05 * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=0.05)
