"""Staging — schedule + backend-accuracy coverage on the scalar reduce tier.

The orthogonal ``STAGE`` codec (``d<depth>/sync|cp|tma``) annotates the typed ``Stage``
schedule struct on a Semiring contraction; the materializer assembles the smem slab +
cooperative producer from it. These tests pin the staged path on a scalar fp32 matmul (the
regime that has operand reuse): the ``TILE_PASSES`` chain stamps the ``Stage`` onto the
``SemiringKernel`` schedule, and the staged kernel matches a numpy reference for every stage
subset, including a masked (non-divisor) output axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile import TileOp
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
    """The ``TILE_PASSES`` chain stamps the ``STAGE`` codec onto the scalar matmul's typed
    ``Stage`` schedule struct (the orthogonal codec on the ``SemiringKernel`` arm — no
    resurrected ``StageBundle`` / ``PLACE@<edge>=smem`` placement). ``d1/sync`` is the
    single-buffer plain-``__syncthreads`` staging point."""
    monkeypatch.setenv("DEPLODOCK_STAGE", "d1/sync")
    out = Pipeline.build(TILE_PASSES).run(_matmul_graph(), ctx=Context.from_target((8, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.knobs.get("STAGE") == "d1/sync", tile_op.knobs.get("STAGE")
    stage = tile_op.kernel.schedule.stage
    assert stage is not None and stage.transport == "sync" and stage.depth == 1, stage


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
