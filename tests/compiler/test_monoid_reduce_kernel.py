"""Carrier-general cross-partition reduce kernel (Step 3a of
``plans/atomic-free-monoid-combine.md``).

``017``'s ``build_monoid_reduce_tileop`` builds an atomic-free reduce kernel
driven by a ``Combine`` monoid carrier: per-partition state slabs in a
``workspace[S, M, N]``, ``identity``-seeded, folded along ``S`` via the carrier's
``combine_states``. The additive matmul split-K still rides the bit-identical
``Accum`` sum (``_build_reduce_tileop``); this test exercises the NON-additive
``(m, l)`` online-softmax monoid through the new builder directly — a hand-built
scalar monoid split that merges S partition states and matches numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _load_builder():
    import importlib

    mod = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.017_atomic_free_splitk")
    return mod.build_monoid_reduce_tileop


def _ml_carrier():
    from tests.compiler.test_cooperative_combine import _ml_combine

    return _ml_combine("m", "l", "s")


def _reduce_graph(s: int, m: int, n: int) -> Graph:
    build = _load_builder()
    carrier = _ml_carrier()
    reduce_op = build(
        carrier=carrier,
        init_ops=(ElementwiseImpl("maximum"), ElementwiseImpl("add")),
        workspaces=("ws_m", "ws_l"),
        out_name="out",
        s_extent=s,
        m_extent=m,
        n_extent=n,
        dtype=F32,
        out_value="l",  # the merged denominator l_global
        name="monoid_ml__reduce",
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("ws_m", (Dim(s), Dim(m), Dim(n))), node_id="ws_m")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("ws_l", (Dim(s), Dim(m), Dim(n))), node_id="ws_l")
    g.add_node(op=reduce_op, inputs=["ws_m", "ws_l"], output=Tensor("out", (Dim(m), Dim(n))), node_id="out")
    g.inputs = ["ws_m", "ws_l"]
    g.outputs = ["out"]
    return g


@pytest.mark.skipif(not _has_cuda(), reason="monoid reduce kernel runs on CUDA")
@pytest.mark.parametrize("s,m,n", [(4, 16, 16), (8, 16, 32), (3, 20, 20)])
def test_monoid_reduce_merges_partition_states(s: int, m: int, n: int) -> None:
    """Merging S per-partition (m_s, l_s) online-softmax states reproduces the
    global denominator l = Σ_s l_s · exp(m_s − max_s m_s)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(3)
    ws_m = (rng.standard_normal((s, m, n)) * 3.0).astype(np.float32)
    ws_l = (rng.random((s, m, n)).astype(np.float32) + 0.1) * 5.0  # positive denominators
    be = CudaBackend()
    compiled = be.compile(_reduce_graph(s, m, n))
    out = np.asarray(be.run(compiled, input_data={"ws_m": ws_m, "ws_l": ws_l})[0].outputs["out"]).reshape(m, n)
    m_g = ws_m.max(axis=0)
    l_g = (ws_l * np.exp(ws_m - m_g)).sum(axis=0)
    np.testing.assert_allclose(out, l_g, rtol=1e-5, atol=1e-5)
