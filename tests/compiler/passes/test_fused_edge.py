"""The SMEM fused-edge assemble (``plans/dag-edge-placement-split-as-enumeration.md``).

The fused realization of an ``SMEM``-placed edge: a MAP producer ``--xn-->`` SEMIRING
matmul consumer kept in **one kernel**, the ``xn`` intermediate riding an smem slab the
producer fills (`relu(x) @ w`). ``assemble_fused`` builds it by reusing the existing
``StageBundle.compute`` phase ("sibling-smem → own-smem"): stage ``x`` into a slab, the
producer becomes the compute phase writing ``xn_smem``, the consumer matmul reads it.

The structural test pins the fused kernel shape; the CUDA test pins it runs correctly —
one kernel computing ``relu(x) @ w``, the matmul reading ``xn`` from smem (no gmem
round-trip).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import Buffer, Edge, Space, StageBundle, TileGraph, TileOp, Transport
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._fused import assemble_fused, is_fused_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from tests.compiler.conftest import requires_cuda
from tests.compiler.passes.test_tile_ir_invariants import _oracle_tilegraph

_KN = {"BN": 16, "FN": 2, "BM": 16, "FM": 2, "BK": 16, "FK": 1, "SPLITK": 1}


def _mm_xn_graph(M=64, K=64, N=64) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("xn", (M, K), F16), node_id="xn")
    g.add_node(InputOp(), [], Tensor("w", (K, N), F16), node_id="w")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("o", (M, N), F16), node_id="o")
    g.inputs, g.outputs = ["xn", "w"], ["o"]
    return g


def _relu_producer(M, K) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), F16), node_id="x")
    g.add_node(ElementwiseOp("relu"), ["x"], Tensor("xn", (M, K), F16), node_id="xn")
    g.inputs, g.outputs = ["x"], ["xn"]
    return g


def _mul_producer(M, K) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (M, K), F16), node_id="y")
    g.add_node(ElementwiseOp("multiply"), ["x", "y"], Tensor("xn", (M, K), F16), node_id="xn")
    g.inputs, g.outputs = ["x", "y"], ["xn"]
    return g


def _producer_block(producer_graph: Graph):
    out = Pipeline.build(LOOP_PASSES).run(producer_graph, ctx=Context.from_target((12, 0)))
    lo = next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    return replace(seed_graph(iter_dag(lo), kernel_name="prod").blocks[0], name="prod")


def _fused_graph(producer_graph: Graph, M=64, K=64, N=64) -> TileGraph:
    """``f(x, …) @ w`` as a fused 2-block ``TileGraph``: a (logical) MAP producer + a
    tiled matmul consumer, the ``xn`` edge SMEM-placed (one launch group + staged)."""
    cons_tg = _oracle_tilegraph(_mm_xn_graph(M, K, N), _KN)
    cons = cons_tg.blocks[0]
    prod = _producer_block(producer_graph)
    buffers = {n: Buffer(n, (M, K), F16, space=Space.GMEM) for n in (*producer_graph.inputs, "xn")}
    buffers["w"] = Buffer("w", (K, N), F16, space=Space.GMEM)
    buffers["o"] = Buffer("o", (M, N), F16, space=Space.GMEM)
    xn_edge = Edge(src="prod", dst=cons.name, buffer="xn")
    sched = replace(cons_tg.schedule, launch={"prod": 0, cons.name: 0}, staged={xn_edge: Transport.SYNC})
    return TileGraph(name="fused", buffers=buffers, blocks=(prod, cons), schedule=sched)


def test_fused_graph_detected():
    tg = _fused_graph(_relu_producer(64, 64))
    assert is_fused_graph(tg)  # two blocks, one launch group


def test_assemble_fused_builds_compute_phase():
    """The fused ``TileOp`` is one kernel whose ``xn`` slab is filled by a
    ``StageBundle.compute`` phase (the producer relu), staging ``x`` (not ``xn``) from
    gmem — the producer rides the consumer's slab, no gmem round-trip of ``xn``."""
    top = assemble_fused(_fused_graph(_relu_producer(64, 64)), knobs=_KN, base_knobs={}, kernel_name="k_fused")
    assert isinstance(top, TileOp)
    bundles = [s for s in top.body.iter() if isinstance(s, StageBundle)]
    assert bundles, "fused kernel must carry a StageBundle"
    fused = [b for b in bundles if b.compute is not None]
    assert fused, "the xn slab must be filled by a compute phase (the producer)"
    # the bundle stages x (the producer input), not xn (which is the compute output)
    staged_bufs = {src.buf for b in fused for src in b.sources}
    assert "x" in staged_bufs and "xn" not in staged_bufs
    # the compute phase applies the producer transform (relu) and writes the xn slab
    from deplodock.compiler.ir.stmt import Assign, Write

    assigns = [s for b in fused for s in b.compute.iter() if isinstance(s, Assign)]
    assert any(getattr(a.op, "name", "") == "relu" for a in assigns)
    writes = [s for b in fused for s in b.compute.iter() if isinstance(s, Write)]
    assert any(w.output.startswith("xn") for w in writes)


@requires_cuda
@pytest.mark.parametrize(
    "producer, np_ref",
    [
        (_relu_producer, lambda ins: np.maximum(ins["x"], 0)),  # single-input MAP
        (_mul_producer, lambda ins: ins["x"] * ins["y"]),  # multi-input MAP (same-shape operands)
    ],
    ids=["relu", "multiply"],
)
def test_fused_map_matmul_runs_correctly(producer, np_ref):
    """End-to-end: ``f(x, …) @ w`` computes in **one** launch, matching a numpy
    reference — the matmul reads ``xn`` from smem (the fused edge), no separate producer
    kernel. Covers a single-input (relu) and a multi-input (multiply) MAP producer."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    M = K = N = 64
    pg = producer(M, K)
    top = assemble_fused(_fused_graph(pg, M, K, N), knobs=_KN, base_knobs={}, kernel_name="k_fused")
    fg = Graph()
    for name in (*pg.inputs, "w"):
        shape = (K, N) if name == "w" else (M, K)
        fg.add_node(InputOp(), [], Tensor(name, shape, F16), node_id=name)
    fg.add_node(top, [*pg.inputs, "w"], Tensor("o", (M, N), F16), node_id="o")
    fg.inputs, fg.outputs = [*pg.inputs, "w"], ["o"]

    compiled = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(fg, ctx=Context.from_target((12, 0)))
    rng = np.random.default_rng(0)
    ins = {name: rng.standard_normal((M, K) if name != "w" else (K, N)).astype(np.float16) for name in (*pg.inputs, "w")}
    res = CudaBackend().run(compiled, input_data=ins)[0].outputs
    got = list(res.values())[0].reshape(M, N).astype(np.float32)
    ref = np_ref({k: v.astype(np.float32) for k, v in ins.items()}) @ ins["w"].astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=3e-1, rtol=3e-1)
