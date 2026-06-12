"""Strided-cooperative rows — cooperative-K (``BR > 1``) alongside free-axis
THREAD tiles (``BN·BM > 1``).

v1 tied ``BR > 1`` to ``BN = BM = 1`` (the materializer's combine spanned the
whole CTA over a single THREAD axis). v2 lifts that: the combine becomes a
SEGMENTED warp shuffle over each row's BR lanes, valid when the cooperative
axis is the innermost THREAD axis (its lanes form a contiguous BR-aligned
intra-warp group) and BR is a power of two ≤ warp_size. These tests pin:

- the enumeration rule (2D rows admitted; their BR clipped to pow2 ≤ warp;
  CTA-wide BR > warp still exclusive to the BN=BM=1 form),
- ``cooperative_combine_geometry`` (the materializer's combine query),
- the lowered structure (K_c innermost THREAD axis; ``WarpShuffle`` with the
  SEGMENT length, no ``TreeHalve``) on static and symbolic-row graphs,
- CUDA accuracy of a pinned 2D row.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.kernel.ir import TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import cooperative_combine_geometry
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import enumerate_cartesian

from ..conftest import requires_cuda

_CTX = Context(compute_capability=(12, 0))


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _reduce_graph(shape: tuple) -> Graph:
    g = Graph()
    _input(g, "x", shape)
    out_shape = (*shape[:-1], 1)
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", out_shape), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    return g


# --- enumeration -----------------------------------------------------


def test_reduce_enumeration_admits_br_with_free_threads():
    """The reduce mode emits 2D rows (BN·BM > 1 AND BR > 1) — the v1
    constraint is lifted."""
    combos = enumerate_cartesian(E_M=1, E_N=64, E_K=128, ctx=_CTX, priority_mode="reduce")
    two_d = [p for p in combos if p["BN"] * p["BM"] > 1 and p["BR"] > 1]
    assert two_d, "reduce enumeration produced no strided-cooperative (BN·BM>1, BR>1) row"


def test_2d_rows_clip_br_to_segmented_shuffle():
    """Every 2D row's BR is a power of two ≤ warp_size (the segmented
    warp-shuffle combine's validity window); CTA-wide BR > warp_size rows
    survive only in the BN=BM=1 form."""
    combos = enumerate_cartesian(E_M=1, E_N=64, E_K=256, ctx=_CTX, priority_mode="reduce")
    saw_wide_1d = False
    for p in combos:
        if p["BN"] * p["BM"] > 1 and p["BR"] > 1:
            assert p["BR"] <= _CTX.warp_size, f"2D row with BR > warp_size: {p}"
            assert p["BR"] & (p["BR"] - 1) == 0, f"2D row with non-pow2 BR: {p}"
        if p["BN"] == 1 and p["BM"] == 1 and p["BR"] > _CTX.warp_size:
            saw_wide_1d = True
    assert saw_wide_1d, "BN=BM=1 rows must keep CTA-wide BR > warp_size candidates"


def test_2d_rows_respect_thread_cap():
    """BN·BM·BR stays within max_threads_per_cta on 2D rows."""
    combos = enumerate_cartesian(E_M=1, E_N=256, E_K=256, ctx=_CTX, priority_mode="reduce")
    for p in combos:
        assert p["BN"] * p["BM"] * p["BR"] <= _CTX.max_threads_per_cta, p


# --- cooperative_combine_geometry ------------------------------------


def test_combine_geometry_whole_cta():
    """Single (all-cooperative) THREAD axis: any size — the combine spans
    the CTA and emit_combine picks warp/hierarchical/tree-halve."""
    k_c = Axis("k_c", 64)
    tid, n = cooperative_combine_geometry((k_c,), frozenset({"k_c"}), warp_size=32)
    assert (tid, n) == ("k_c", 64)


def test_combine_geometry_segmented_rows():
    """Free-axis threads alongside: returns the SEGMENT size (BR), not the
    CTA size."""
    n_t = Axis("n_t", 16)
    k_c = Axis("k_c", 32)
    tid, n = cooperative_combine_geometry((n_t, k_c), frozenset({"k_c"}), warp_size=32)
    assert (tid, n) == ("k_c", 32)


def test_combine_geometry_rejects_non_innermost_coop_axis():
    """The cooperative axis must be the innermost (fastest) THREAD axis —
    otherwise a row's lanes aren't a contiguous aligned segment."""
    n_t = Axis("n_t", 16)
    k_c = Axis("k_c", 32)
    with pytest.raises(ValueError, match="innermost"):
        cooperative_combine_geometry((k_c, n_t), frozenset({"k_c"}), warp_size=32)


def test_combine_geometry_rejects_unshufflable_segment():
    """With free-axis threads present, BR must be a power of two ≤ warp_size."""
    n_t = Axis("n_t", 4)
    with pytest.raises(ValueError, match="power-of-two"):
        cooperative_combine_geometry((n_t, Axis("k_c", 64)), frozenset({"k_c"}), warp_size=32)
    with pytest.raises(ValueError, match="power-of-two"):
        cooperative_combine_geometry((n_t, Axis("k_c", 24)), frozenset({"k_c"}), warp_size=32)


# --- lowered structure ------------------------------------------------


def _pin(monkeypatch, **knobs) -> None:
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))


def _lowered_tiles(g: Graph):
    out = Pipeline.build(KERNEL_PASSES).run(g)
    stmts = [s for node in out.nodes.values() if getattr(node.op, "body", None) is not None for s in node.op.body.iter()]
    thread_tiles = [s for s in stmts if isinstance(s, ThreadTile)]
    return stmts, thread_tiles


def _assert_2d_coop_tile(stmts, thread_tiles, *, bn: int, br: int) -> None:
    """One ThreadTile with the K_c axis INNERMOST (= last, fastest threadIdx
    bits), a segmented WarpShuffle of length BR, and no TreeHalve."""
    (tt,) = thread_tiles
    assert [ax.extent.as_static() for ax in tt.axes] == [bn, br], [ax.name for ax in tt.axes]
    coop_names = {name for s in tt.body.iter() if isinstance(s, Accum) for name in s.axes}
    assert tt.axes[-1].name in coop_names, "cooperative K_c axis must be the innermost THREAD axis"
    assert tt.axes[0].name not in coop_names, "free row axis must not be cooperative"
    shuffles = [s for s in stmts if isinstance(s, WarpShuffle)]
    assert shuffles and all(s.length == br for s in shuffles), [s.length for s in shuffles]
    assert not any(isinstance(s, TreeHalve) for s in stmts), "segmented combine must not fall back to TreeHalve"


def test_lowering_2d_coop_static_rows(monkeypatch):
    """Pinned BN=8, BR=16 on a static (64, 128) sum: 128-thread CTA with the
    row axis as threads and a 16-lane segmented shuffle per row."""
    _pin(monkeypatch, BN=8, BR=16, FN=1, FK=1, BK=2)
    stmts, thread_tiles = _lowered_tiles(_reduce_graph((64, 128)))
    _assert_2d_coop_tile(stmts, thread_tiles, bn=8, br=16)


def test_lowering_2d_coop_symbolic_rows(monkeypatch):
    """Symbolic leading axis (seq_len, 8, 128): the symbolic axis stays
    whole-to-grid (exact symbolic grid, no mask) while the static row axis
    thread-binds alongside the BR lanes — the strided-cooperative form the
    dynamic q/k-norm kernels deploy."""
    _pin(monkeypatch, BN=8, BR=16, FN=1, FK=1, BK=2)
    stmts, thread_tiles = _lowered_tiles(_reduce_graph((Dim("seq_len"), 8, 128)))
    _assert_2d_coop_tile(stmts, thread_tiles, bn=8, br=16)
    grids = [s for s in stmts if isinstance(s, GridTile)]
    assert grids, "expected a GridTile"
    sym_axes = [ax for g_ in grids for ax in g_.axes if not ax.extent.is_static]
    assert sym_axes, "symbolic seq axis must bind to the grid"
    assert all("seq_len" in ax.extent.expr.free_vars() for ax in sym_axes)


# --- CUDA accuracy -----------------------------------------------------


@requires_cuda
def test_2d_coop_reduce_accuracy_cuda(monkeypatch):
    """A pinned 2D row computes the same sums as numpy — the segmented
    shuffle combines each row independently."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    _pin(monkeypatch, BN=8, BR=16, FN=1, FK=1, BK=2)
    g = _reduce_graph((64, 128))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(g), input_data={"x": x})[0].outputs["o"]
    np.testing.assert_allclose(out, x.sum(-1, keepdims=True), rtol=1e-4, atol=1e-4)
