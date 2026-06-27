"""The SMEM fused-edge assemble — backend-accuracy coverage.

The fused realization of an ``SMEM``-placed edge: a MAP producer ``--xn-->`` SEMIRING
matmul consumer kept in **one kernel**, the ``xn`` intermediate riding an smem slab the
producer fills (`relu(x) @ w`). These are the end-to-end accuracy tests: compile a fused
graph, run it on ``CudaBackend``, and assert the result matches a numpy / torch reference
— one kernel computing ``f(x, …) @ w``, the matmul reading ``xn`` from smem (no gmem
round-trip).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import Buffer, Edge, Placement, Space, TileGraph, TileGraphOp, TileOp, Transport
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import build_dag, seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from tests.compiler.conftest import requires_cuda

_KN = {
    fam.split_key("a1"): fam.enc_split(16, 2),
    fam.split_key("a0"): fam.enc_split(16, 2),
    fam.reduce_key("a2"): fam.enc_reduce(serial=16, fold=1, cta=1),
}


# ── ``_oracle_tilegraph`` (copied from the deleted ``test_tile_ir_invariants``) ──────────
def _loop_dag_buffers(graph: Graph):
    """The fused ``LoopOp`` + its ``iter_dag`` / regime / logical buffers — the seed the
    move composer tiles. Mirrors ``010_build`` so the oracle path matches the pipeline."""
    loop = next(n.op for n in Pipeline.build(LOOP_PASSES).run(graph).nodes.values() if type(n.op).__name__ == "LoopOp")
    dag = iter_dag(loop)
    buffers = {name: Buffer(name=name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM) for name, t in loop.inputs.items()}
    return loop, dag, classify(dag), buffers


def _oracle_tilegraph(graph: Graph, knobs: dict):
    """``build_dag`` (the composition oracle) for ``graph`` at ``knobs``."""
    _loop, dag, regime, buffers = _loop_dag_buffers(graph)
    return build_dag(dag, knobs, kernel_name="k_matmul", target_names=regime.target_names, buffers=buffers)


# ── ``_norm_linear_graph`` (copied from the deleted ``test_cut_offers``) ─────────────────
_S, _H, _I = 32, 1024, 3072


def _norm_linear_graph() -> Graph:
    """RMSNorm → Linear (f16): fusion yields the prologue-demoted matmul that
    005 offers the structural split on."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, _S, _H), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (_H,), f16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (_I, _H), f16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, _S, _H), f16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, _S, _I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg"]
    g.outputs = ["o"]
    return g


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


def _broadcast_producer(M, K) -> Graph:
    """``xn[m,k] = x[m,k] · rs[m] · cs[k]`` — a row broadcast (``rs[m]`` over k) and a col
    broadcast (``cs[k]`` over m), the scale-application shape of rmsnorm."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("rs", (M, 1), F16), node_id="rs")
    g.add_node(InputOp(), [], Tensor("cs", (1, K), F16), node_id="cs")
    g.add_node(ElementwiseOp("multiply"), ["x", "rs"], Tensor("t", (M, K), F16), node_id="t")
    g.add_node(ElementwiseOp("multiply"), ["t", "cs"], Tensor("xn", (M, K), F16), node_id="xn")
    g.inputs, g.outputs = ["x", "rs", "cs"], ["xn"]
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
    buffers = {n: Buffer(n, tuple(producer_graph.nodes[n].output.shape), F16, space=Space.GMEM) for n in producer_graph.inputs}
    buffers["xn"] = Buffer("xn", (M, K), F16, space=Space.GMEM)
    buffers["w"] = Buffer("w", (K, N), F16, space=Space.GMEM)
    buffers["o"] = Buffer("o", (M, N), F16, space=Space.GMEM)
    xn_edge = Edge(src="prod", dst=cons.name, buffer="xn")
    sched = replace(cons_tg.schedule, launch={"prod": 0, cons.name: 0}, staged={xn_edge: Transport.SYNC})
    return TileGraph(name="fused", buffers=buffers, blocks=(prod, cons), schedule=sched)


@requires_cuda
@pytest.mark.parametrize(
    "producer, np_ref",
    [
        (_relu_producer, lambda ins: np.maximum(ins["x"], 0)),  # single-input MAP
        (_mul_producer, lambda ins: ins["x"] * ins["y"]),  # multi-input MAP (same-shape operands)
        (_broadcast_producer, lambda ins: ins["x"] * ins["rs"] * ins["cs"]),  # row + col broadcasts
    ],
    ids=["relu", "multiply", "broadcast"],
)
def test_fused_map_matmul_runs_correctly(producer, np_ref):
    """End-to-end: ``f(x, …) @ w`` computes in **one** launch, matching a numpy
    reference — the matmul reads ``xn`` from smem (the fused edge), no separate producer
    kernel. Covers single-input (relu), multi-input (multiply), and **broadcast-operand**
    (``x·rs[m]·cs[k]`` — the rmsnorm scale-application shape) MAP producers."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    M, K, N = 32, 64, 32  # M != K so the row / col broadcasts are unambiguous
    pg = producer(M, K)
    shape_of = {name: tuple(d.as_static() for d in pg.nodes[name].output.shape) for name in pg.inputs}
    shape_of["w"] = (K, N)
    top = assemble_block(_fused_graph(pg, M, K, N), knobs=_KN, base_knobs={}, kernel_name="k_fused")
    fg = Graph()
    for name in (*pg.inputs, "w"):
        fg.add_node(InputOp(), [], Tensor(name, shape_of[name], F16), node_id=name)
    fg.add_node(top, [*pg.inputs, "w"], Tensor("o", (M, N), F16), node_id="o")
    fg.inputs, fg.outputs = [*pg.inputs, "w"], ["o"]

    compiled = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(fg, ctx=Context.from_target((12, 0)))
    rng = np.random.default_rng(0)
    ins = {name: rng.standard_normal(shape_of[name]).astype(np.float16) for name in (*pg.inputs, "w")}
    res = CudaBackend().run(compiled, input_data=ins)[0].outputs
    got = list(res.values())[0].reshape(M, N).astype(np.float32)
    ref = np_ref({k: v.astype(np.float32) for k, v in ins.items()}) @ ins["w"].astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=0.1, rtol=2e-2)


def _loopop(g: Graph):
    out = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((12, 0)))
    return next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")


def _renamed_producer(producer_graph: Graph, M, K) -> Graph:
    """``producer_graph`` rewired to write the ``o__xn`` intermediate (the consumer's
    input name) instead of ``xn``."""
    pg = Graph()
    for nid, node in producer_graph.nodes.items():
        out = node.output
        name = "o__xn" if nid == "xn" else nid
        shape = tuple(d.as_static() for d in out.shape)
        pg.add_node(node.op, [("o__xn" if i == "xn" else i) for i in node.inputs], Tensor(name, shape, F16), node_id=name)
    pg.inputs, pg.outputs = list(producer_graph.inputs), ["o__xn"]
    return pg


def _fused_seed_op(producer_graph: Graph, M, K, N) -> TileGraphOp:
    """A fused 2-block ``TileGraphOp`` seed: a LOGICAL matmul consumer (``o = xn @ w``,
    un-tiled — the enumeration tiles it) + a logical MAP producer (``producer_graph``,
    rewired to write ``o__xn``), the ``xn`` edge SMEM-placed. Carries the consumer's
    dag/regime so the enumeration forks tile ``blocks[0]`` (the consumer) while
    preserving ``blocks[1]`` (the producer)."""
    cg = Graph()
    cg.add_node(InputOp(), [], Tensor("o__xn", (M, K), F16), node_id="o__xn")
    cg.add_node(InputOp(), [], Tensor("w", (K, N), F16), node_id="w")
    cg.add_node(MatmulOp(), ["o__xn", "w"], Tensor("o", (M, N), F16), node_id="o")
    cg.inputs, cg.outputs = ["o__xn", "w"], ["o"]
    clo = _loopop(cg)
    cdag = iter_dag(clo)
    creg = classify(cdag)
    cons = seed_graph(cdag, kernel_name="k_cons").blocks[0]

    plo = _loopop(_renamed_producer(producer_graph, M, K))
    prod = replace(seed_graph(iter_dag(plo), kernel_name="prod").blocks[0], name="prod")

    buffers: dict = {}
    for lo in (clo, plo):
        for name, t in {**lo.inputs, **lo.outputs}.items():
            buffers[name] = Buffer(name, tuple(t.shape), t.dtype, space=Space.GMEM)
    tg = TileGraph(
        name="fused",
        buffers=buffers,
        blocks=(cons, prod),
        schedule=replace(_oracle_tilegraph(_mm_xn_graph(M, K, N), _KN).schedule, binding={}, launch={}, staged={}),
    )
    tg = tg.place_edge(Edge(src="prod", dst="k_cons", buffer="o__xn"), Placement.SMEM)
    return TileGraphOp(name="k_fused", tilegraph=tg, dag=cdag, algebra=creg.algebra, target_names=creg.target_names, buffers=buffers)


@requires_cuda
@pytest.mark.parametrize("tier", ["scalar", "warp"])
@pytest.mark.parametrize(
    "producer, np_ref",
    [(_mul_producer, lambda i: i["x"] * i["y"]), (_broadcast_producer, lambda i: i["x"] * i["rs"] * i["cs"])],
    ids=["mul", "broadcast"],
)
def test_fused_edge_lowers_through_enumeration_and_assembly(monkeypatch, tier, producer, np_ref):
    """The full tile pipeline (enumeration tiles the consumer + assembly dispatches to
    ``assemble_fused``) lowers a fused 2-block seed to one correct kernel — proving the
    body-move auxiliary-block preservation + the assembly fused dispatch. Covers the
    scalar tier and the **warp (mma.sync) tier** (the cut-beating form — the matmul reads
    the cooperatively-computed ``xn`` from smem via ``ldmatrix``), over a same-shape and a
    **broadcast-operand** (``x·rs[m]·cs[k]`` — the rmsnorm scale shape) producer."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    if tier == "scalar":
        monkeypatch.setenv("DEPLODOCK_BN", "16")
        monkeypatch.setenv("DEPLODOCK_BM", "16")
        monkeypatch.setenv("DEPLODOCK_MMA", "0")
    M, K, N = 32, 64, 32
    pg = producer(M, K)
    shape_of = {name: tuple(d.as_static() for d in pg.nodes[name].output.shape) for name in pg.inputs}
    shape_of["w"] = (K, N)
    inputs = [*pg.inputs, "w"]
    g = Graph()
    for name in inputs:
        g.add_node(InputOp(), [], Tensor(name, shape_of[name], F16), node_id=name)
    g.add_node(_fused_seed_op(pg, M, K, N), inputs, Tensor("o", (M, N), F16), node_id="o")
    g.inputs, g.outputs = inputs, ["o"]

    out = Pipeline.build(["lowering/tile/enumeration", "lowering/tile/assembly"]).run(g, ctx=Context.from_target((12, 0)))
    assert any(isinstance(n.op, TileOp) for n in out.nodes.values())  # one fused kernel, not two
    assert sum(1 for n in out.nodes.values() if isinstance(n.op, TileOp)) == 1
    compiled = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(out, ctx=Context.from_target((12, 0)))
    rng = np.random.default_rng(0)
    ins = {name: rng.standard_normal(shape_of[name]).astype(np.float16) for name in inputs}
    res = CudaBackend().run(compiled, input_data=ins)[0].outputs
    got = list(res.values())[0].reshape(M, N).astype(np.float32)
    ref = np_ref({k: v.astype(np.float32) for k, v in ins.items()}) @ ins["w"].astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=0.1, rtol=2e-2)


@requires_cuda
def test_fused_rmsnorm_matmul_runs_correctly(monkeypatch):
    """The **MONOID** producer (rmsnorm) fuses: ``rmsnorm(x)·nw @ w`` computes in one
    kernel — a cooperative reduce PROLOGUE (``CoopReduce``) computes the per-row scale
    into smem, and the scale-application compute phase reads it as a broadcast operand
    feeding the matmul. The reduce over the full row precedes the matmul K-loop. Scalar
    tier here; the live warp-tier path is covered by
    ``test_offering_fork_fused_edge_runs_correctly[warp]``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.algebra import AlgebraKind
    from deplodock.compiler.ir.stmt import Write
    from deplodock.compiler.ir.tile.ir import Placement, TileGraphOp
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract import _fission, seed_demoted

    monkeypatch.setenv("DEPLODOCK_BN", "16")
    monkeypatch.setenv("DEPLODOCK_BM", "16")
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    M, K, N = 32, 1024, 3072
    g = _norm_linear_graph()
    lo = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((12, 0)))
    node = next(n for n in lo.nodes.values() if type(n.op).__name__ == "LoopOp")
    f = _fission(node.op, graph=lo, node_id=node.id, out_tensor=node.output)
    cdag = iter_dag(f.consumer)
    creg = classify(cdag)
    tg = seed_demoted(node.op, graph=lo, node_id=node.id, out_tensor=node.output)
    prod = next(b for b in tg.blocks if b.carrier and b.carrier.kind is AlgebraKind.MONOID)
    cons = next(b for b in tg.blocks if b.carrier and b.carrier.kind is AlgebraKind.SEMIRING)
    xn = next(e for e in tg.edges if e.src == prod.name and e.dst == cons.name)
    tg = replace(tg, blocks=(cons, prod)).place_edge(xn, Placement.SMEM)
    op = TileGraphOp(name="k_fused", tilegraph=tg, dag=cdag, algebra=creg.algebra, target_names=creg.target_names, buffers=tg.buffers)

    ext = {nid: n.output for nid, n in lo.nodes.items()}
    out_name = next(w.output for w in cons.compute.iter_of_type(Write))
    inputs = [n for n in tg.buffers if n not in {"o__xn", "o", out_name}]
    fg = Graph()
    for nid in inputs:
        t = ext[nid]
        fg.add_node(InputOp(), [], Tensor(nid, tuple(t.shape), t.dtype), node_id=nid)
    fg.add_node(op, inputs, Tensor(out_name, tuple(node.output.shape), node.output.dtype), node_id=out_name)
    fg.inputs, fg.outputs = inputs, [out_name]

    res = Pipeline.build(["lowering/tile/enumeration", "lowering/tile/assembly"]).run(fg, ctx=Context.from_target((12, 0)))
    assert sum(1 for n in res.nodes.values() if isinstance(n.op, TileOp)) == 1  # ONE fused kernel
    compiled = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(res, ctx=Context.from_target((12, 0)))
    rng = np.random.default_rng(0)
    data = {}
    for nid in inputs:
        shp, npdt = tuple(d.as_static() for d in ext[nid].shape), ext[nid].dtype.np
        if nid == "xn_mean_count":
            data[nid] = np.full(shp, float(K), dtype=npdt)  # the rmsnorm reduction size
        elif nid == "xn_eps":
            data[nid] = np.full(shp, 1e-6, dtype=npdt)
        else:
            data[nid] = rng.standard_normal(shp).astype(npdt)
    got = list(CudaBackend().run(compiled, input_data=data)[0].outputs.values())[0].reshape(M, N).astype(np.float32)
    x, nw, wg = data["x"].astype(np.float32), data["nw"].astype(np.float32), data["wg"].astype(np.float32)
    xn_ref = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    # The MONOID fused-prologue path (cooperative rms-scale reduce + fp16 matmul accumulate
    # at K=1024) carries ~7% relative error on the large elements — a known-fragile path.
    # rtol=0.1 reflects that real
    # precision (tighter than the old blanket rtol=0.5) so a regression is still caught;
    # atol=0.5 absorbs the near-zero elements where relative error is meaningless.
    np.testing.assert_allclose(got, xn_ref @ wg.T, atol=0.5, rtol=0.1)


_TILE_PASSES = ["lowering/tile/split", "lowering/tile/enumeration", "lowering/tile/assembly"]


@requires_cuda
@pytest.mark.parametrize("tier", ["warp", "scalar"])
def test_offering_fork_fused_edge_runs_correctly(monkeypatch, tier):
    """End-to-end through the LIVE pass chain (no hand-built seed): the offering fork's
    kept SMEM edge lowers ``RMSNorm → linear`` to one fused kernel that matches the torch
    reference — proving ``seed_fused`` wires the demoted matmul into the fused assemble.
    Covers the **warp tier** (the natural greedy pick — the matmul reads the
    cooperatively-reduced per-row scale from smem via ``ldmatrix``) and the scalar tier."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    if tier == "scalar":
        monkeypatch.setenv("DEPLODOCK_BN", "16")
        monkeypatch.setenv("DEPLODOCK_BM", "16")
        monkeypatch.setenv("DEPLODOCK_MMA", "0")
    else:
        # Pin a small in-budget warp tile: the cold-ranker's smart tile pick is being
        # retired (greedy cold → emission order), so the warp tier must be pinned rather
        # than relying on the prior to avoid the largest-BK smem-overflow emission default.
        # Legacy env pins route through the ingest mapper (DEPLODOCK_BK → REDUCE@<axis>).
        monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
        monkeypatch.setenv("DEPLODOCK_WM", "2")
        monkeypatch.setenv("DEPLODOCK_WN", "2")
        monkeypatch.setenv("DEPLODOCK_FM", "2")
        monkeypatch.setenv("DEPLODOCK_FN", "2")
        monkeypatch.setenv("DEPLODOCK_BK", "2")
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "0")  # the kept fused edge
    ctx = Context.from_target((12, 0))
    s, h, i = 32, 1024, 3072
    lo = Pipeline.build(LOOP_PASSES).run(_norm_linear_graph(), ctx=ctx)
    tiled = Pipeline.build(_TILE_PASSES).run(lo, ctx=ctx)
    assert sum(1 for n in tiled.nodes.values() if isinstance(n.op, TileOp)) == 1  # one fused kernel
    out = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(tiled, ctx=ctx)

    rng = np.random.default_rng(0)
    ins = {
        "x": rng.standard_normal((1, s, h)).astype(np.float16),
        "nw": rng.standard_normal((h,)).astype(np.float16),
        "wg": rng.standard_normal((i, h)).astype(np.float16),
    }
    got = list(CudaBackend().run(out, input_data=ins)[0].outputs.values())[0].reshape(s, i).astype(np.float32)
    x, nw, wg = ins["x"][0].astype(np.float32), ins["nw"].astype(np.float32), ins["wg"].astype(np.float32)
    rms = x * (1.0 / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    # [warp] is accurate (~0.06 abs); [scalar] takes the fragile fp16 fused-prologue path
    # (~7% on large elements). rtol=0.1 covers both — far tighter than the old rtol=0.5.
    np.testing.assert_allclose(got, rms @ wg.T, atol=0.1, rtol=0.1)


@requires_cuda
@pytest.mark.parametrize(
    "expr",
    ["(0.5 * x) @ w", "torch.sigmoid(x) @ w", "torch.relu(x) @ w"],
    ids=["scale_const", "sigmoid", "relu"],
)
def test_fused_map_producer_warp_tier_live(monkeypatch, expr):
    """MAP-producer demoted matmuls fuse correctly at the **warp tier** through the live
    pass chain, matching torch. Covers a **scalar-constant** operand (``0.5·x`` — the
    fully-broadcast operand read straight from gmem, no slab) alongside a unary activation,
    the producer shapes a real model emits before a linear."""

    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.base import InputOp

    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "0")  # the kept SMEM fused edge (warp tier)
    code = (
        "import torch\ntorch.manual_seed(0)\n"
        "x = torch.randn(64, 256, dtype=torch.float16) * 0.3\n"
        "w = torch.randn(256, 128, dtype=torch.float16) * 0.3\n" + expr
    )
    g, _slug, (mod, args, kwargs) = graph_from_code(code)
    out = Pipeline.build([*LOOP_PASSES, *_TILE_PASSES, "lowering/kernel", "lowering/cuda"]).run(g, ctx=Context.from_target((12, 0)))
    ref = mod(*args, **kwargs).detach().cpu().numpy().astype(np.float32)
    argv = [a.detach().cpu().numpy() for a in args]
    ins = {}
    for nid, node in out.nodes.items():
        if isinstance(node.op, InputOp):
            shp = tuple(d.as_static() for d in node.output.shape)
            match = next((a for a in argv if a.shape == shp), None)
            ins[nid] = match.astype(node.output.dtype.np)
    got = list(CudaBackend().run(out, input_data=ins)[0].outputs.values())[0].reshape(ref.shape).astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=0.1, rtol=2e-2)
