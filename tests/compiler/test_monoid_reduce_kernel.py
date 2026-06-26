"""Carrier-general cross-partition reduce kernel.

The atomic-free split-K combine block, rebuilt against the block-DAG Tile IR:
``enumeration/_partition.monoid_reduce_tilegraph`` builds a combine kernel driven by a
``Monoid`` carrier — per-partition state slabs in a ``workspace[S, M, N]``,
``identity``-seeded, folded along ``S`` via the carrier's ``combine_states`` (the new-IR
successor of the deleted ``017``'s ``build_monoid_reduce_tileop``). The additive matmul
split-K rides the bit-identical ``Accum`` sum (:func:`additive_reduce_tilegraph`); this
test exercises the NON-additive ``(m, l)`` online-softmax monoid through the builder
directly — a hand-built scalar monoid split that merges S partition states and matches
numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._partition import monoid_reduce_tilegraph, reduce_tilegraphop


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _ml_carrier():
    from tests.compiler.test_cooperative_combine import _ml_combine

    return _ml_combine("m", "l", "s")


def _reduce_graph(s: int, m: int, n: int) -> Graph:
    carrier = _ml_carrier()
    tg = monoid_reduce_tilegraph(
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
    reduce_op = reduce_tilegraphop(tg)
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


# --- The carrier-generic deferred finalize: ONE entry point, dispatched on the carrier ---
# (the blog's thesis — sum, online softmax (m,d), flash (m,d,o) are the SAME cross-partition
# reduce, differing only in carrier state).


def _deferred_graph(carrier, *, workspaces, s, m, n, **kw) -> Graph:
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._partition import deferred_combine_tilegraph

    tg = deferred_combine_tilegraph(
        carrier, workspaces=tuple(workspaces), out_name="out", s_extent=s, m_extent=m, n_extent=n, dtype=F32, name="deferred__reduce", **kw
    )
    g = Graph()
    for w in workspaces:
        g.add_node(op=InputOp(), inputs=[], output=Tensor(w, (Dim(s), Dim(m), Dim(n))), node_id=w)
    g.add_node(op=reduce_tilegraphop(tg), inputs=list(workspaces), output=Tensor("out", (Dim(m), Dim(n))), node_id="out")
    g.inputs, g.outputs = list(workspaces), ["out"]
    return g


@pytest.mark.skipif(not _has_cuda(), reason="deferred finalize runs on CUDA")
def test_deferred_finalize_additive_carrier_sums_partitions() -> None:
    """The generic selector routes a 1-component additive ``Accum`` to the plain ``Σ_s`` fold —
    the trivial carrier (matmul split-K), same entry point as the twisted monoid below."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.stmt import Accum

    s, m, n = 8, 16, 32
    rng = np.random.default_rng(7)
    ws = rng.standard_normal((s, m, n)).astype(np.float32)
    g = _deferred_graph(Accum(name="acc", value="p"), workspaces=("ws",), s=s, m=m, n=n)
    be = CudaBackend()
    out = np.asarray(be.run(be.compile(g), input_data={"ws": ws})[0].outputs["out"]).reshape(m, n)
    np.testing.assert_allclose(out, ws.sum(axis=0), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_cuda(), reason="deferred finalize runs on CUDA")
def test_deferred_finalize_flash_attention_carrier_merges_states() -> None:
    """The flash attention ``(m, l, O)`` twisted monoid — the carrier a split-KV / flash-decoding
    producer would emit per CTA — merges through the SAME generic selector: the cross-partition
    ``O = Σ_s O_s·exp(m_s − max_s m_s)`` (the e^{Δm} rescale, NOT a plain sum), so the deferred
    KERNEL finalize is correct for the attention carrier (atomic would be illegal — non-additive).
    The kernel-level proof of "attention's cross-CTA combine"; the producer that feeds it is the
    remaining flash split-KV work."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline.passes.loop.recognize._flash import flash_combine

    s, m, n = 8, 16, 32
    rng = np.random.default_rng(13)
    ws_m = (rng.standard_normal((s, m, n)) * 3.0).astype(np.float32)
    ws_l = (rng.random((s, m, n)).astype(np.float32) + 0.1) * 5.0
    ws_o = (rng.standard_normal((s, m, n)) * 2.0).astype(np.float32)
    g = _deferred_graph(
        flash_combine("m", "l", "o", "s", "v"),
        workspaces=("ws_m", "ws_l", "ws_o"),
        s=s,
        m=m,
        n=n,
        init_ops=(ElementwiseImpl("maximum"), ElementwiseImpl("add"), ElementwiseImpl("add")),
        out_value="o",  # the merged unnormalized output accumulator O
    )
    be = CudaBackend()
    out = np.asarray(
        be.run(be.compile(g), input_data={"ws_m": ws_m, "ws_l": ws_l, "ws_o": ws_o})[0].outputs["out"]
    ).reshape(m, n)
    o_g = (ws_o * np.exp(ws_m - ws_m.max(axis=0))).sum(axis=0)
    np.testing.assert_allclose(out, o_g, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_cuda(), reason="deferred finalize runs on CUDA")
def test_deferred_finalize_twisted_monoid_carrier_merges_states() -> None:
    """The SAME selector routes the twisted ``(m, l)`` online-softmax ``Monoid`` to the rescaled
    cross-partition merge — carrier-generic dispatch, not a flash special case."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    s, m, n = 8, 16, 32
    rng = np.random.default_rng(11)
    ws_m = (rng.standard_normal((s, m, n)) * 3.0).astype(np.float32)
    ws_l = (rng.random((s, m, n)).astype(np.float32) + 0.1) * 5.0
    g = _deferred_graph(
        _ml_carrier(),
        workspaces=("ws_m", "ws_l"),
        s=s,
        m=m,
        n=n,
        init_ops=(ElementwiseImpl("maximum"), ElementwiseImpl("add")),
        out_value="l",
    )
    be = CudaBackend()
    out = np.asarray(be.run(be.compile(g), input_data={"ws_m": ws_m, "ws_l": ws_l})[0].outputs["out"]).reshape(m, n)
    l_g = (ws_l * np.exp(ws_m - ws_m.max(axis=0))).sum(axis=0)
    np.testing.assert_allclose(out, l_g, rtol=1e-5, atol=1e-5)
