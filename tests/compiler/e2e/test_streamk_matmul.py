"""Stream-K M3 — end-to-end persistent-CTA matmul correctness.

Pins ``DEPLODOCK_STREAMK=1`` and runs matmuls through the CUDA backend, checking
the result against a numpy reference. Covers the under-occupied case (fewer tiles
than SMs → most CTAs do one tile, the rest idle) and the multi-wave case (more
tiles than SMs → each CTA walks several tiles in its work range). Also asserts
the lowered graph actually took the persistent path (work-range arrays present),
so a silent skip can't make the accuracy check vacuous.

This is the full-tile / one-owner-per-tile variant (no K-split, no atomics yet).
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device")


def _matmul_graph(m: int, k: int, n: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@requires_cuda
@pytest.mark.parametrize(
    "m,k,n",
    [
        (256, 256, 256),  # 128 tiles < 170 SMs → one tile per busy CTA, rest idle
        (1024, 512, 1024),  # many tiles > SMs → each CTA walks several tiles
        (384, 256, 512),  # non-square, non-power-of-two-tile shape
    ],
)
def test_streamk_matmul_matches_numpy(m, k, n, monkeypatch):
    monkeypatch.setenv("DEPLODOCK_STREAMK", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    be = CudaBackend()
    out = be.run(be.compile(_matmul_graph(m, k, n)), input_data={"a": a, "b": b})[0].outputs["o"]

    np.testing.assert_allclose(out.reshape(m, n), a @ b, rtol=2e-3, atol=2e-3)


@requires_cuda
def test_streamk_is_mutually_exclusive_with_splitk(monkeypatch):
    """Stream-K supplies its own K-split (adaptively, in Phase B), so it must not
    compose with fixed split-K. With both pinned, the persistent rewrite
    self-skips and the kernel stays a plain split-K GridTile (no PersistentTile,
    no work-range arrays)."""
    monkeypatch.setenv("DEPLODOCK_STREAMK", "1")
    monkeypatch.setenv("DEPLODOCK_SPLITK", "4")
    lowered = Pipeline.build(CUDA_PASSES).run(_matmul_graph(512, 512, 512))
    cuda_ops = [n.op for n in lowered.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops
    assert not any(op.streamk_work_arrays for op in cuda_ops), "Stream-K must not fire alongside split-K"


@requires_cuda
def test_streamk_lowers_to_persistent_path(monkeypatch):
    """With STREAMK pinned on, the lowered CUDA op carries the work-range
    arrays — proving the accuracy test above ran the persistent kernel, not a
    silently-skipped GridTile fallback."""
    monkeypatch.setenv("DEPLODOCK_STREAMK", "1")
    lowered = Pipeline.build(CUDA_PASSES).run(_matmul_graph(256, 256, 256))
    cuda_ops = [n.op for n in lowered.nodes.values() if isinstance(n.op, CudaOp)]
    matmul_ops = [op for op in cuda_ops if op.streamk_work_arrays]
    assert matmul_ops, "no CudaOp took the Stream-K persistent path"
    op = matmul_ops[0]
    assert op.streamk_work_arrays == ("streamk_work_start", "streamk_work_end")
    assert op.streamk_total_units > 0
    assert op.grid[0] == ("__num_sms__",)
