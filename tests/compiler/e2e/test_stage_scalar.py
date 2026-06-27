"""Staging — backend-accuracy coverage on the scalar reduce tier.

``stage(read)`` writes ``Schedule.staged[edge] = SYNC`` for a reused gmem read (the
enumeration fork ``120_stage``); ``assemble`` synthesizes the smem slab + cooperative
producer from that annotation. These tests pin the staged path end to end on a scalar
fp32 matmul (the regime that has reuse and lowers today): the full ``TILE_PASSES`` chain
emits a ``StageBundle`` under the all-staged mask, and the staged kernel matches a numpy
reference for every stage subset, including a masked (non-divisor) output axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import StageBundle, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

from ..conftest import requires_cuda


def _matmul_graph(M: int = 64, N: int = 64, K: int = 64) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def test_scalar_matmul_stages_through_pipeline(monkeypatch) -> None:
    """The full ``TILE_PASSES`` chain (greedy) stages a scalar matmul when
    ``DEPLODOCK_STAGE`` pins the all-staged mask (ingested as ``PLACE@<edge>=smem``): the
    emitted ``TileOp`` carries the native ``PLACE@<edge>`` placement and a ``StageBundle``."""
    monkeypatch.setenv("DEPLODOCK_STAGE", "all")
    out = Pipeline.build(TILE_PASSES).run(_matmul_graph(), ctx=Context.from_target((8, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    placed = {k: v for k, v in tile_op.knobs.items() if k.startswith("PLACE@")}
    assert placed and all(v.startswith("smem") for v in placed.values()), placed  # both operands staged
    assert any(isinstance(s, StageBundle) for s in tile_op.body.iter())


@requires_cuda
@pytest.mark.parametrize("stage_mask", ["11", "10", "01", "00"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (64, 47, 64), (128, 128, 128)])
def test_staged_scalar_matmul_matches_reference(monkeypatch, stage_mask, shape) -> None:
    """Every stage subset (both / A-only / B-only / none) lowers to a kernel that
    matches a numpy matmul, including a masked (non-divisor) output axis."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("DEPLODOCK_STAGE", stage_mask)
    # Pin a small in-budget scalar tile: the cold ranker's smart tile pick is retired
    # (greedy cold → emission order), so the deep-BK emission default can overflow the
    # staged smem slab on the larger shape. Legacy env pins route through the ingest mapper.
    for k, v in (("BN", "16"), ("BM", "16"), ("FN", "2"), ("FM", "2"), ("BK", "16")):
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    M, N, K = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(_matmul_graph(M, N, K)), input_data={"a": a, "b": b})[0].outputs
    got = list(out.values())[0].reshape(M, N)
    np.testing.assert_allclose(got, a @ b, atol=1e-3, rtol=1e-3)
