"""Carrier-general cross-partition reduce kernel (BACKEND-ACCURACY).

The atomic-free split-K combine block, run on ``CudaBackend`` and asserted against numpy:
``enumeration/_partition.deferred_combine_tilegraph`` builds a combine kernel for any carrier
(the additive ``Accum`` lowered as a degenerate ``Monoid`` via ``Accum.as_monoid``, folded by
``deferred_combine_tilegraph``) — for a ``Monoid`` carrier, per-partition state slabs in a
``workspace[S, M, N]``, ``identity``-seeded, folded along ``S`` via the carrier's ``combine_states``.
Exercises the NON-additive ``(m, l)`` online-softmax monoid, the additive ``Accum`` (matmul
split-K), the flash ``(m, l, O)`` twisted monoid, and the twisted ``(m, l)`` monoid through the
builder directly.

NOTE: every test here builds its reference graph through the tile-internal
``deferred_combine_tilegraph`` / ``reduce_tilegraphop`` helpers, so they are tile-entangled — they
will break when the tile IR is rebuilt and are expected to be xfailed at that point.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.stmt import Assign, Monoid
# tile IR demolished — pending rebuild (see plans/tile-ir-rebuild.md); guarded so the
# module collects and its tests register as xfail rather than a collection error.
try:
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._partition import deferred_combine_tilegraph, reduce_tilegraphop
except ModuleNotFoundError:
    deferred_combine_tilegraph = reduce_tilegraphop = None


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _ml_combine(m: str, ll: str, s: str) -> Monoid:
    """The online-softmax ``(m, l)`` monoid: state max + denominator, folding one
    score ``s``. Authors both ``merge`` (fold a partial) and ``combine_states``
    (merge two partition states) — the asymmetric monoid can't auto-derive."""
    merge = (
        Assign(f"{m}_mx", "maximum", (m, s)),
        Assign(f"{m}_dm", "subtract", (m, f"{m}_mx")),
        Assign(f"{m}_al", "exp", (f"{m}_dm",)),
        Assign(f"{m}_ds", "subtract", (s, f"{m}_mx")),
        Assign(f"{m}_p", "exp", (f"{m}_ds",)),
        Assign(f"{m}_lm", "multiply", (ll, f"{m}_al")),
        Assign(ll, "add", (f"{m}_lm", f"{m}_p")),
        Assign(m, "copy", (f"{m}_mx",)),
    )
    mb, lb = f"{m}__o", f"{ll}__o"
    combine_states = (
        Assign(f"{m}_cmx", "maximum", (m, mb)),
        Assign(f"{m}_cda", "subtract", (m, f"{m}_cmx")),
        Assign(f"{m}_ca", "exp", (f"{m}_cda",)),
        Assign(f"{m}_cdb", "subtract", (mb, f"{m}_cmx")),
        Assign(f"{m}_cb", "exp", (f"{m}_cdb",)),
        Assign(f"{m}_cla", "multiply", (ll, f"{m}_ca")),
        Assign(f"{m}_clb", "multiply", (lb, f"{m}_cb")),
        Assign(ll, "add", (f"{m}_cla", f"{m}_clb")),
        Assign(m, "copy", (f"{m}_cmx",)),
    )
    return Monoid(
        state=(m, ll),
        partial=(s,),
        merge=merge,
        identity=(Literal(-1e30), Literal(0.0)),
        commutative=True,
        axes=("k",),
        state_b=(mb, lb),
        combine_states=combine_states,
    )


def _ml_carrier():
    return _ml_combine("m", "l", "s")


def _reduce_graph(s: int, m: int, n: int) -> Graph:
    carrier = _ml_carrier()
    tg = deferred_combine_tilegraph(
        carrier,
        init_ops=(ElementwiseImpl("maximum"), ElementwiseImpl("add")),
        workspaces=("ws_m", "ws_l"),
        out_name="out",
        s_extent=s,
        out_shape=(m, n),
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
    tg = deferred_combine_tilegraph(
        carrier, workspaces=tuple(workspaces), out_name="out", s_extent=s, out_shape=(m, n), dtype=F32, name="deferred__reduce", **kw
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
    out = np.asarray(be.run(be.compile(g), input_data={"ws_m": ws_m, "ws_l": ws_l, "ws_o": ws_o})[0].outputs["out"]).reshape(m, n)
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
