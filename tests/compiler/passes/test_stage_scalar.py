"""R1 (Staging) — the ``stage`` move on the scalar reduce tier.

``plans/tile-ir-block-dag.md`` R1: ``stage(read)`` writes ``Schedule.staged[edge] =
SYNC`` for a reused gmem read (the enumeration fork ``120_stage``); ``assemble``
synthesizes the smem slab + cooperative producer from that annotation
(``assembly/_slab``). These tests pin the new architecture end to end on a scalar
fp32 matmul (the regime that has reuse and lowers today):

- ``stage_candidates`` finds the reused operands off the DERIVED ``Block.reads``;
- ``assemble`` emits a ``StageBundle`` whose ``Source`` geometry reconstructs the
  original σ-rewritten gmem index (cache axes = THREAD/REGISTER tile + K stage; GRID
  and the serial-outer K fold into the slab origin);
- the staged kernel matches a numpy reference for every stage mask, including a
  masked (non-divisor) output axis.

The warp-tier MMA staging probe (``test_stage_inputs_mma_probe.py``) needs the
``atomize`` tier (R4) and stays quarantined until then.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import Buffer, Space, StageBundle, StagePolicy, TileOp, Transport
from deplodock.compiler.pipeline import LOOP_PASSES, TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import build_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._stage import stage_candidates

from ..conftest import requires_cuda

_MM_KNOBS = {"BN": 8, "FN": 2, "BM": 8, "FM": 2, "BK": 16, "FK": 1, "SPLITK": 1}


def _matmul_graph(M: int = 64, N: int = 64, K: int = 64) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _staged_tilegraph(graph: Graph, knobs: dict):
    """Build the matmul ``TileGraph`` with ``knobs`` and stage every candidate."""
    from dataclasses import replace

    loop = next(n.op for n in Pipeline.build(LOOP_PASSES).run(graph).nodes.values() if type(n.op).__name__ == "LoopOp")
    dag = iter_dag(loop)
    regime = classify(dag)
    buffers = {name: Buffer(name=name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM) for name, t in loop.inputs.items()}
    tg = build_dag(dag, knobs, kernel_name="k_matmul", target_names=regime.target_names, buffers=buffers)
    cands = stage_candidates(tg)
    staged = {e: Transport.SYNC for e in cands}
    return replace(tg, schedule=replace(tg.schedule, staged=staged)), cands


def test_stage_candidates_finds_reused_matmul_inputs() -> None:
    """Both matmul operands are stageable (each is reused across the other's free
    axis — fan-in); the candidate set is the derived input-source ``Edge``s."""
    tg, cands = _staged_tilegraph(_matmul_graph(), _MM_KNOBS)
    assert {e.buffer for e in cands} == {"a", "b"}
    assert all(e.dst == "k_matmul" for e in cands)


def test_stage_candidates_empty_for_pointwise() -> None:
    """A pointwise (no K-tower) nest has no per-stage slab — nothing to stage."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (64, 64)), node_id="x")
    g.add_node(op=ElementwiseOp(op=ElementwiseImpl("relu")), inputs=["x"], output=Tensor("y", (64, 64)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    loop = next(n.op for n in Pipeline.build(LOOP_PASSES).run(g).nodes.values() if type(n.op).__name__ == "LoopOp")
    dag = iter_dag(loop)
    tg = build_dag(dag, {"BN": 8, "FN": 1, "BM": 8, "FM": 1}, kernel_name="k_relu", target_names=classify(dag).target_names)
    assert stage_candidates(tg) == []


def test_assemble_synthesizes_smem_slabs() -> None:
    """``assemble`` emits one SYNC ``StageBundle`` with a ``Source`` per operand; the
    affine ``source_index`` reconstructs the operand's original σ-rewritten gmem
    index, so the cooperative producer reads exactly what the un-staged Load did."""
    tg, _ = _staged_tilegraph(_matmul_graph(), _MM_KNOBS)
    tile_op = assemble_block(tg, knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_matmul")
    bundles = [s for s in tile_op.body.iter() if isinstance(s, StageBundle)]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.policy is StagePolicy.SYNC
    srcs = {s.buf: s for s in bundle.sources}
    assert set(srcs) == {"a", "b"}
    # A's slab caches the M tile (BM·FM = 16) × the K stage (BK = 16) = 256 elems;
    # the GRID + serial-outer-K terms are not cache axes (they fold into origin).
    a = srcs["a"]
    assert int(np.prod([ax.extent.as_static() for ax in a.cache_axes])) == 16 * 16
    assert a.dtype is not None and a.dtype.nbytes == 4
    # Every cache-axis Var is bound by a tile layer in the tower (no dangling symbol).
    from deplodock.compiler.ir.axis import Axis

    cache_vars = {ax.name for s in bundle.sources for ax in s.cache_axes}
    bound: set[str] = set()
    for stmt in tile_op.body.iter():
        bound |= {ax.name for ax in getattr(stmt, "axes", ()) if isinstance(ax, Axis)}
        ax = getattr(stmt, "axis", None)
        if isinstance(ax, Axis):
            bound.add(ax.name)
    assert cache_vars <= bound, f"dangling cache vars: {cache_vars - bound}"


def test_scalar_matmul_stages_through_pipeline(monkeypatch) -> None:
    """The full ``TILE_PASSES`` chain (greedy) stages a scalar matmul when
    ``DEPLODOCK_STAGE`` pins the all-staged mask: the emitted ``TileOp`` carries the
    ``STAGE`` knob and a ``StageBundle``."""
    monkeypatch.setenv("DEPLODOCK_STAGE", "all")
    out = Pipeline.build(TILE_PASSES).run(_matmul_graph(), ctx=Context.from_target((8, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.knobs.get("STAGE") == "11"
    assert any(isinstance(s, StageBundle) for s in tile_op.body.iter())


@requires_cuda
@pytest.mark.parametrize("stage_mask", ["11", "10", "01", "00"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (64, 47, 64), (128, 128, 128)])
def test_staged_scalar_matmul_matches_reference(monkeypatch, stage_mask, shape) -> None:
    """Every stage subset (both / A-only / B-only / none) lowers to a kernel that
    matches a numpy matmul, including a masked (non-divisor) output axis."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("DEPLODOCK_STAGE", stage_mask)
    M, N, K = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(_matmul_graph(M, N, K)), input_data={"a": a, "b": b})[0].outputs
    got = list(out.values())[0].reshape(M, N)
    np.testing.assert_allclose(got, a @ b, atol=1e-3, rtol=1e-3)
