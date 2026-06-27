"""Strided-cooperative rows — CUDA backend accuracy.

A pinned 2D cooperative reduce (``BN > 1`` free-axis threads alongside ``BR > 1``
cooperative-K lanes) must compute the same per-row sums as numpy: the cross-thread
combine is a SEGMENTED warp shuffle over each row's ``BR`` lanes, combining each row
independently.

Backend-accuracy port of the deleted ``tests/compiler/passes/test_strided_coop_rows.py``
(the tile/enumeration internal-structure tests are dropped; only the ``CudaBackend``
accuracy assertion survives).
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ReduceOp

from ..conftest import requires_cuda


def _reduce_graph(shape: tuple) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", shape), node_id="x")
    out_shape = (*shape[:-1], 1)
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", out_shape), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return g


@requires_cuda
def test_2d_coop_reduce_accuracy_cuda(monkeypatch):
    """A pinned 2D row (BN=8, BR=16) computes the same per-row sums as numpy —
    the segmented shuffle combines each row independently."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    for key, val in dict(BN=8, BR=16, FN=1, FK=1, BK=2).items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(val))
    g = _reduce_graph((64, 128))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    be = CudaBackend()
    out = be.run(be.compile(g), input_data={"x": x})[0].outputs["o"]
    np.testing.assert_allclose(out, x.sum(-1, keepdims=True), rtol=1e-4, atol=1e-4)
