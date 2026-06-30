"""Strided-cooperative rows — cooperative-K (``BR > 1``) alongside free-axis
THREAD tiles (``BN·BM > 1``).

The whole-CTA cooperative form ties ``BR > 1`` to ``BN = BM = 1`` (the combine
spans the CTA over a single cooperative THREAD axis). The strided form lifts
that: the cross-thread combine becomes a SEGMENTED warp shuffle over each row's
``BR`` lanes, valid only when the cooperative axis is the innermost THREAD axis
(its lanes form a contiguous BR-aligned intra-warp group) and ``BR`` is a power
of two ≤ ``warp_size``.

Ports the still-valid invariants of the deleted ``test_strided_coop_rows.py`` to
the rebuilt move-composer enumeration (``enumeration/_moves.coop_reduce_offers``)
and the live ``kernel/_combine.cooperative_combine_geometry``. The old test's
lowered-structure assertions (``ThreadTile`` axis order / ``WarpShuffle`` length)
targeted the legacy materializer's IR; here we pin the geometry query the
materializer drives instead, plus CUDA accuracy of a pinned 2D row.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._combine import cooperative_combine_geometry
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, iter_dag
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import coop_reduce_offers

from ..conftest import requires_cuda

_CC = (12, 0)
_WARP = 32


def _reduce_graph(shape: tuple) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", shape), node_id="x")
    out_shape = (*shape[:-1], 1)
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", out_shape), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return g


def _reduce_dag(shape: tuple) -> IterDag:
    out = Pipeline.build(LOOP_PASSES).run(_reduce_graph(shape), ctx=Context.from_target(_CC))
    lo = next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    return iter_dag(lo)


# --- enumeration -----------------------------------------------------


def test_whole_cta_form_keeps_wide_br():
    """Without free-axis THREAD tiles (``BN=BM=1``) the cooperative reduce keeps
    CTA-wide ``BR > warp_size`` candidates — the combine spans the whole CTA."""
    offers = coop_reduce_offers(_reduce_dag((64, 256)), warp_size=_WARP)
    assert offers, "whole-CTA cooperative reduce produced no offers"
    assert max(br for _, _, br, _ in offers) > _WARP, "BN=BM=1 form must keep CTA-wide BR > warp_size"


def test_strided_rows_clip_br_to_segmented_shuffle(monkeypatch):
    """A pinned ``BN > 1`` (free-axis threads alongside) clips every cooperative
    ``BR`` to a power of two ≤ ``warp_size`` — the segmented warp-shuffle combine's
    validity window."""
    monkeypatch.setenv("EMMY_BN", "8")
    offers = coop_reduce_offers(_reduce_dag((64, 256)), warp_size=_WARP)
    assert offers, "strided-cooperative reduce produced no offers"
    for _, _, br, _ in offers:
        assert br <= _WARP, f"strided row with BR > warp_size: {br}"
        assert br & (br - 1) == 0, f"strided row with non-pow2 BR: {br}"


def test_strided_rows_respect_thread_cap(monkeypatch):
    """``bn·br`` stays within the CTA thread budget (1024) on strided rows."""
    monkeypatch.setenv("EMMY_BN", "8")
    for _, _, br, _ in coop_reduce_offers(_reduce_dag((64, 256)), warp_size=_WARP):
        assert 8 * br <= 1024, f"bn·br exceeds CTA thread budget: 8·{br}"


# --- cooperative_combine_geometry ------------------------------------


def test_combine_geometry_whole_cta():
    """Single (all-cooperative) THREAD axis: any size — the combine spans the
    CTA and ``emit_combine`` picks warp / hierarchical / tree-halve."""
    k_c = Axis("k_c", 64)
    tid, n = cooperative_combine_geometry((k_c,), frozenset({"k_c"}), warp_size=_WARP)
    assert (tid, n) == ("k_c", 64)


def test_combine_geometry_segmented_rows():
    """Free-axis threads alongside: returns the SEGMENT size (BR), not the CTA
    size; the cooperative axis must be the innermost THREAD axis."""
    n_t = Axis("n_t", 16)
    k_c = Axis("k_c", 32)
    tid, n = cooperative_combine_geometry((n_t, k_c), frozenset({"k_c"}), warp_size=_WARP)
    assert (tid, n) == ("k_c", 32)


def test_combine_geometry_rejects_non_innermost_coop_axis():
    """The cooperative axis must be the innermost (fastest) THREAD axis —
    otherwise a row's lanes aren't a contiguous aligned segment."""
    n_t = Axis("n_t", 16)
    k_c = Axis("k_c", 32)
    with pytest.raises(ValueError, match="innermost"):
        cooperative_combine_geometry((k_c, n_t), frozenset({"k_c"}), warp_size=_WARP)


def test_combine_geometry_rejects_unshufflable_segment():
    """With free-axis threads present, BR must be a power of two ≤ ``warp_size``."""
    with pytest.raises(ValueError, match="power-of-two"):
        cooperative_combine_geometry((Axis("n_t", 4), Axis("k_c", 64)), frozenset({"k_c"}), warp_size=_WARP)
    with pytest.raises(ValueError, match="power-of-two"):
        cooperative_combine_geometry((Axis("n_t", 4), Axis("k_c", 24)), frozenset({"k_c"}), warp_size=_WARP)


def test_combine_geometry_requires_single_coop_axis():
    """Exactly one cooperative THREAD axis (planner construction); zero or two
    raise."""
    with pytest.raises(ValueError, match="exactly one"):
        cooperative_combine_geometry((Axis("n_t", 16),), frozenset({"k_c"}), warp_size=_WARP)


# --- symbolic-row strided coop ---------------------------------------


def test_symbolic_row_strided_offers(monkeypatch):
    """A symbolic leading axis (``seq_len``, 8, 128) with a pinned ``BN=8`` still
    offers strided-cooperative ``BR`` clipped to the segmented-shuffle window —
    the form the dynamic q/k-norm kernels deploy."""
    monkeypatch.setenv("EMMY_BN", "8")
    offers = coop_reduce_offers(_reduce_dag((Dim("seq_len"), 8, 128)), warp_size=_WARP)
    assert offers, "symbolic-row strided cooperative reduce produced no offers"
    for _, _, br, _ in offers:
        assert br <= _WARP and br & (br - 1) == 0, f"symbolic-row strided BR not pow2<=warp: {br}"


# --- CUDA accuracy ---------------------------------------------------


@requires_cuda
def test_2d_coop_reduce_accuracy_cuda(monkeypatch):
    """A pinned 2D row (BN=8, BR=16) computes the same per-row sums as numpy —
    the segmented shuffle combines each row independently."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    for key, val in dict(BN=8, BR=16, FN=1, FK=1, BK=2).items():
        monkeypatch.setenv(f"EMMY_{key}", str(val))
    g = _reduce_graph((64, 128))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    be = CudaBackend()
    out = be.run(be.compile(g), input_data={"x": x})[0].outputs["o"]
    np.testing.assert_allclose(out, x.sum(-1, keepdims=True), rtol=1e-4, atol=1e-4)
