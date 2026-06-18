"""Tests for the demoted-matmul split (``lowering/tile/005_split_demoted`` + the
``_split_demoted`` cut builder).

A fused computed-operand cone (gated-MLP norm prologue, elementwise scale) keeps a matmul
off the warp tier; rule 005 offers a structural fork splitting the kernel into ``xn``
producer(s) + the clean gemm before partition tiles anything — one producer per computed
multiply operand (rotary QK^T gets two; a weight-side scale's N cone gets a [K, N]
buffer). The offer is gated only on the
cut's WELL-FORMEDNESS (signal-driven design: profitability is the tuner's question, never a
predicate's). Covers: offered / not-offered cuttability gates, the fragment's structure,
the rule's option list (fused first) + the offered-marker idempotence, ``DEPLODOCK_SPLIT_CONE``
pinning, greedy structural pricing (cold compile keeps today's kernel sets; a trained prior
deploys the split; a failed structural pick falls back to keep-fused),
tune exploring both branches, no-GPU numpy accuracy of the spliced graph, and CUDA accuracy
of both the scalar-tier and mma.sync split paths.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, ReshapeOp, RmsNormOp, TransposeOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, TileOp
from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline, RuleSkipped, TuningSearch
from deplodock.compiler.pipeline.fork import OptionFork
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile._split_demoted import try_split_demoted
from deplodock.compiler.pipeline.search.db import SearchDB
from tests.compiler.conftest import drain_tune

from ..conftest import requires_cuda

split_rule = importlib.import_module(
    "deplodock.compiler.pipeline.passes.lowering.tile.005_split_demoted",
)

_S, _H, _I = 32, 1024, 3072


@pytest.fixture(autouse=True)
def _force_sm120(monkeypatch, tmp_path):
    """sm_120 target (auto-mma enumerates for the split gemm) + an isolated,
    untrained prior so greedy picks are deterministic regardless of the host's
    checkpoint."""
    monkeypatch.setenv("DEPLODOCK_PRIOR_FILE", str(tmp_path / "prior.json"))
    target_mod.set_target((12, 0))
    yield
    target_mod.set_target(None)


def _norm_linear_graph(dtype_name: str = "f16") -> Graph:
    """RMSNorm → Linear: fusion produces the prologue-demoted matmul kernel
    (the gated-MLP plan's shape, single matmul until dual-Mma lands)."""
    dt = _dt.get(dtype_name)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, _S, _H), dt), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (_H,), dt), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (_I, _H), dt), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, _S, _H), dt), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, _S, _I), dt), node_id="o")
    g.inputs = ["x", "nw", "wg"]
    g.outputs = ["o"]
    return g


def _scale_matmul_graph(M: int = 128, K: int = 128, N: int = 128) -> Graph:
    """Elementwise scale → MatmulOp: fusion produces an in-cell-cone demoted
    matmul with the B operand in [K, N] layout (the mma-lowerable form)."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("s", (M, K), f16), node_id="s")
    g.add_node(InputOp(), [], Tensor("w", (K, N), f16), node_id="w")
    g.add_node(ElementwiseOp("multiply"), ["x", "s"], Tensor("xs", (M, K), f16), node_id="xs")
    g.add_node(MatmulOp(), ["xs", "w"], Tensor("o", (M, N), f16), node_id="o")
    g.inputs = ["x", "s", "w"]
    g.outputs = ["o"]
    return g


def _double_scale_linear_graph(M: int = 128, K: int = 128, N: int = 128) -> Graph:
    """Elementwise scale on BOTH operands → Linear ([N, K] weight): fusion
    produces a doubly-computed matmul cell (the rotary-QK^T shape). The
    two-producer cut materializes the N-indexed cone at [K, N], so the
    consumer gemm gets the canonical B layout — warp-tier eligible even
    though the original Linear access was transposed [n, k]."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("sx", (M, K), f16), node_id="sx")
    g.add_node(InputOp(), [], Tensor("w", (N, K), f16), node_id="w")
    g.add_node(InputOp(), [], Tensor("sw", (N, K), f16), node_id="sw")
    g.add_node(ElementwiseOp("multiply"), ["x", "sx"], Tensor("xs", (M, K), f16), node_id="xs")
    g.add_node(ElementwiseOp("multiply"), ["w", "sw"], Tensor("ws", (N, K), f16), node_id="ws")
    g.add_node(LinearOp(), ["xs", "ws"], Tensor("o", (M, N), f16), node_id="o")
    g.inputs = ["x", "sx", "w", "sw"]
    g.outputs = ["o"]
    return g


def _gated_mlp_graph(S: int = 32, H: int = 256, I: int = 512) -> Graph:  # noqa: E741
    """RMSNorm → (gate Matmul, up Matmul) → multiply: fusion inlines the norm
    chain once per matmul (two SSA-duplicated cones sharing the leaf Loads)
    into ONE dual-accum K loop — the Qwen3 gated-MLP kernel of
    plans/qwen3-embedding-layer0-tune-findings.md finding 2. [K, N] weights
    (the trace pre-transposes Linear weights), so the split gemms are
    warp-tier eligible."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), f16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (H, I), f16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (H, I), f16), node_id="wu")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), f16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "wg"], Tensor("gate", (1, S, I), f16), node_id="gate")
    g.add_node(MatmulOp(), ["xn", "wu"], Tensor("up", (1, S, I), f16), node_id="up")
    g.add_node(ElementwiseOp("multiply"), ["gate", "up"], Tensor("o", (1, S, I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg", "wu"]
    g.outputs = ["o"]
    return g


def _pure_matmul_graph(M: int = 128, K: int = 128, N: int = 128) -> Graph:
    """Plain Matmul, both operands stageable Loads — no cut applies."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (M, K), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (K, N), f16), node_id="w")
    g.add_node(MatmulOp(), ["x", "w"], Tensor("o", (M, N), f16), node_id="o")
    g.inputs = ["x", "w"]
    g.outputs = ["o"]
    return g


def _collapsed_matmul_graph(HD: int = 4, S: int = 32, D: int = 64, N: int = 64) -> Graph:
    """Transpose+reshape → Matmul (+ residual): fusion collapses the layout ops
    into the matmul's A load, whose K then folds across TWO index dims
    (`attn[0, ((m*K + k)/D) % HD, m, (m*K + k) % D]`) — the o_proj attn-out
    shape of plans/qwen3-embedding-layer0-tune-findings.md finding 3. Both
    operands are plain Loads, so only the layout-materializing cut applies."""
    f16 = _dt.get("f16")
    K = HD * D
    g = Graph()
    g.add_node(InputOp(), [], Tensor("attn", (1, HD, S, D), f16), node_id="attn")
    g.add_node(InputOp(), [], Tensor("w", (K, N), f16), node_id="w")
    g.add_node(InputOp(), [], Tensor("res", (1, S, N), f16), node_id="res")
    g.add_node(TransposeOp(axes=(1, 2)), ["attn"], Tensor("at", (1, S, HD, D), f16), node_id="at")
    g.add_node(ReshapeOp(shape=(1, S, K)), ["at"], Tensor("ar", (1, S, K), f16), node_id="ar")
    g.add_node(MatmulOp(), ["ar", "w"], Tensor("mm", (1, S, N), f16), node_id="mm")
    g.add_node(ElementwiseOp("add"), ["mm", "res"], Tensor("o", (1, S, N), f16), node_id="o")
    g.inputs = ["attn", "w", "res"]
    g.outputs = ["o"]
    return g


def _fuse(graph: Graph, cc=(12, 0)) -> Graph:
    return Pipeline.build(LOOP_PASSES).run(graph, ctx=Context.from_target(cc), db=SearchDB())


def _fused_loop_node(graph: Graph):
    nodes = [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]
    assert len(nodes) == 1, "fusion must collapse the chain into one kernel"
    return nodes[0]


def _split(graph_or_fused: Graph, cc=(12, 0)) -> Graph | None:
    fused = graph_or_fused
    node = _fused_loop_node(fused)
    return try_split_demoted(node.op, Context.from_target(cc), graph=fused, node_id=node.id, out_tensor=node.output)


# ---------------------------------------------------------------------------
# The cut builder: well-formedness gates (no profitability prediction)
# ---------------------------------------------------------------------------


def test_split_offered_for_prologue_demotion() -> None:
    """The fused norm→linear kernel splits into an xn producer + a clean gemm."""
    fused = _fuse(_norm_linear_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    loops = {nid: n for nid, n in frag.nodes.items() if isinstance(n.op, LoopOp)}
    assert set(loops) == {f"{node.id}__xn", f"{node.id}__mm"}
    xn = frag.nodes[f"{node.id}__xn"]
    mm = frag.nodes[f"{node.id}__mm"]
    # Producer: the norm cone materialized at (rows, K) in the operand dtype.
    assert tuple(d.as_static() for d in xn.output.shape) == (_S, _H)
    assert xn.output.dtype == _dt.get("f16")
    assert xn.op.name.endswith("_xn")
    # Consumer: loads xn + the weight. The Linear-derived gemm has a
    # transposed-B operand (wg is [N, K] — K in the last index dim of BOTH
    # loads); ``011_lower_atom_cell._classify_ab`` now recovers A/B from the
    # output coordinates, so the eligibility gate admits it for the tensor-core
    # tier (B lowers gmem-direct via the trans helper — same Q@K^T class).
    assert xn.id in mm.inputs
    ctx = Context.from_target((12, 0))
    assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())
    assert frag.outputs == [mm.id]
    # Structural features restamped per body — the split kernels must not
    # featurize as the fused kernel for the learned prior.
    assert xn.op.knobs.get("S_n_accum") == 1.0  # the norm's row-stat reduce
    assert mm.op.knobs.get("S_n_accum") == 1.0 and mm.op.knobs.get("S_n_load") == 2.0


def test_split_offered_for_in_cell_cone() -> None:
    frag = _split(_fuse(_scale_matmul_graph()))
    assert frag is not None
    assert sum(1 for n in frag.nodes.values() if isinstance(n.op, LoopOp)) == 2
    # The MatmulOp-derived gemm keeps its canonical [K, N] B operand — the
    # rebuilt consumer is genuinely warp-tier eligible.
    mm = next(n for nid, n in frag.nodes.items() if nid.endswith("__mm"))
    ctx = Context.from_target((12, 0))
    assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())


def test_split_offered_for_f32_too() -> None:
    """Signal-driven: the builder checks cuttability only — no dtype / device /
    tier prediction. Whether an f32 split pays is the tuner's question."""
    assert _split(_fuse(_norm_linear_graph("f32"))) is not None


def test_split_offered_for_weight_side_cone() -> None:
    """An N-reading cone beside a plain Load (a weight-side scale — the
    dequant shape): one producer materializing at [K, N], consumer warp-tier
    eligible. Falls out of the unified per-operand rule; the old A-side-only
    dispatch bailed here."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (128, 128), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (128, 128), f16), node_id="w")
    g.add_node(InputOp(), [], Tensor("sw", (128, 128), f16), node_id="sw")
    g.add_node(ElementwiseOp("multiply"), ["w", "sw"], Tensor("ws", (128, 128), f16), node_id="ws")
    g.add_node(MatmulOp(), ["x", "ws"], Tensor("o", (128, 128), f16), node_id="o")
    g.inputs = ["x", "w", "sw"]
    g.outputs = ["o"]
    fused = _fuse(g)
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    loops = {nid for nid, n in frag.nodes.items() if isinstance(n.op, LoopOp)}
    assert loops == {f"{node.id}__xn", f"{node.id}__mm"}
    xn = frag.nodes[f"{node.id}__xn"]
    # The cone reads (k, n) and no rows — a 2-D [K, N] buffer, K NOT in the
    # last dim, so the consumer's B load keeps the canonical layout.
    assert tuple(d.as_static() for d in xn.output.shape) == (128, 128)
    mm = frag.nodes[f"{node.id}__mm"]
    ctx = Context.from_target((12, 0))
    assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())


def test_two_sided_split_offered_for_double_cone() -> None:
    """Both multiply operands computed (rotary QK^T's shape): the cut builds
    TWO producers. The N-indexed cone lands at [K, N] — K out of the last
    index dim — so the consumer gemm is genuinely warp-tier eligible, which
    the one-sided Linear split can never be (transposed-B)."""
    fused = _fuse(_double_scale_linear_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    loops = {nid: n for nid, n in frag.nodes.items() if isinstance(n.op, LoopOp)}
    assert set(loops) == {f"{node.id}__xna", f"{node.id}__xnb", f"{node.id}__mm"}
    xna = frag.nodes[f"{node.id}__xna"]
    xnb = frag.nodes[f"{node.id}__xnb"]
    mm = frag.nodes[f"{node.id}__mm"]
    # A at (rows, K); B at (K, N) — the row-free scale cone stays 2-D.
    assert tuple(d.as_static() for d in xna.output.shape) == (128, 128)
    assert tuple(d.as_static() for d in xnb.output.shape) == (128, 128)
    assert xna.op.name.endswith("_xna") and xnb.op.name.endswith("_xnb")
    assert xna.id in mm.inputs and xnb.id in mm.inputs
    ctx = Context.from_target((12, 0))
    assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())
    assert frag.outputs == [mm.id]
    # Structural features restamped per body.
    assert mm.op.knobs.get("S_n_load") == 2.0


def test_two_sided_split_matches_fused_on_numpy() -> None:
    from deplodock.compiler.backend.numpy import NumpyBackend

    fused = _fuse(_double_scale_linear_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    split_graph = fused.copy()
    split_graph.splice(frag, consumed={node.id}, output=node.id)
    assert sum(1 for n in split_graph.nodes.values() if isinstance(n.op, LoopOp)) == 3

    rng = np.random.default_rng(0)
    npf16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
        "sx": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
        "w": (rng.standard_normal((128, 128), dtype=np.float32) * 0.05).astype(npf16),
        "sw": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
    }
    be = NumpyBackend()
    ref = be.run(be.compile(_double_scale_linear_graph()), input_data=dict(inputs))[0].outputs["o"]
    out = be.run(be.compile(split_graph), input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


def test_gated_mlp_split_offered_with_duplicated_cones() -> None:
    """The dual-accum gate+up kernel splits into ONE shared xn producer (the
    two SSA-duplicated norm chains value-number to a single class), one clean
    gemm per accum, and the pointwise combine consumer — finding 2's fix: the
    fused dual-matmul cell can never pass the one-matmul-per-K-loop mma gate,
    but each extracted gemm is genuinely warp-tier eligible."""
    S, H, I = 32, 256, 512  # noqa: E741
    fused = _fuse(_gated_mlp_graph(S, H, I))
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    loops = {nid: n for nid, n in frag.nodes.items() if isinstance(n.op, LoopOp)}
    assert set(loops) == {f"{node.id}__xn", f"{node.id}__mm0", f"{node.id}__mm1", f"{node.id}__mm"}
    # ONE xn (the duplicated chains share it), materialized at the cell-leaf dtype.
    xn = frag.nodes[f"{node.id}__xn"]
    assert tuple(d.as_static() for d in xn.output.shape) == (S, H)
    assert xn.output.dtype == _dt.get("f16")
    # Per-accum gemms: (rows, N) f32 — the accumulator's own precision.
    ctx = Context.from_target((12, 0))
    for i in (0, 1):
        mm = frag.nodes[f"{node.id}__mm{i}"]
        assert tuple(d.as_static() for d in mm.output.shape) == (S, I)
        assert mm.output.dtype == _dt.get("f32")
        assert xn.id in mm.inputs
        assert mm.op.name.endswith(f"_mm{i}")
        assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())
    # The consumer is the pointwise combine: reads both mm buffers, no reduce.
    cons = frag.nodes[f"{node.id}__mm"]
    assert {f"{node.id}__mm0", f"{node.id}__mm1"} <= set(cons.inputs)
    assert cons.op.knobs.get("S_n_accum") == 0.0
    assert frag.outputs == [cons.id]
    # Structural features restamped per body — each gemm featurizes as a
    # clean 2-load matmul, not as the fused kernel.
    for i in (0, 1):
        mm = frag.nodes[f"{node.id}__mm{i}"]
        assert mm.op.knobs.get("S_n_accum") == 1.0 and mm.op.knobs.get("S_n_load") == 2.0


def test_gated_mlp_split_matches_fused_on_numpy() -> None:
    from deplodock.compiler.backend.numpy import NumpyBackend

    fused = _fuse(_gated_mlp_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    split_graph = fused.copy()
    split_graph.splice(frag, consumed={node.id}, output=node.id)
    assert sum(1 for n in split_graph.nodes.values() if isinstance(n.op, LoopOp)) == 4

    rng = np.random.default_rng(0)
    inputs = _gated_mlp_inputs(rng)
    be = NumpyBackend()
    ref = be.run(be.compile(_gated_mlp_graph()), input_data=dict(inputs))[0].outputs["o"]
    out = be.run(be.compile(split_graph), input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


def test_layout_split_offered_for_folded_k_load() -> None:
    """A plain-Load matmul operand whose K folds across index dims (the
    collapsed reshape/transpose o_proj read) is a DEGENERATE cone — the split
    materializes it through a contiguizing copy producer, and the consumer
    gemm (canonical [rows, K] A load, residual epilogue intact) is genuinely
    warp-tier eligible. Finding 3's fix: without the offer, no structural
    escape existed (both operands plain Loads = 'pure cell')."""
    HD, S, D, N = 4, 32, 64, 64
    fused = _fuse(_collapsed_matmul_graph(HD, S, D, N))
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    loops = {nid: n for nid, n in frag.nodes.items() if isinstance(n.op, LoopOp)}
    assert set(loops) == {f"{node.id}__xn", f"{node.id}__mm"}
    xn = frag.nodes[f"{node.id}__xn"]
    # The producer is the pure contiguizing copy: [rows, K] at the operand
    # dtype, no compute (one load, no assigns).
    assert tuple(d.as_static() for d in xn.output.shape) == (S, HD * D)
    assert xn.output.dtype == _dt.get("f16")
    assert xn.op.knobs.get("S_n_load") == 1.0 and xn.op.knobs.get("S_n_assign") == 0.0
    mm = frag.nodes[f"{node.id}__mm"]
    assert xn.id in mm.inputs
    ctx = Context.from_target((12, 0))
    assert any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())
    assert frag.outputs == [mm.id]


def test_no_layout_split_for_single_k_dim_load() -> None:
    """A pure matmul whose A load keeps K in one index dim is already
    stageable — no cut applies (materializing it would be pure overhead)."""
    fused = _fuse(_pure_matmul_graph())
    node = _fused_loop_node(fused)
    assert try_split_demoted(node.op, Context.from_target((12, 0)), graph=fused, node_id=node.id, out_tensor=node.output) is None


def test_layout_split_matches_fused_on_numpy() -> None:
    from deplodock.compiler.backend.numpy import NumpyBackend

    fused = _fuse(_collapsed_matmul_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    split_graph = fused.copy()
    split_graph.splice(frag, consumed={node.id}, output=node.id)
    assert sum(1 for n in split_graph.nodes.values() if isinstance(n.op, LoopOp)) == 2

    rng = np.random.default_rng(0)
    inputs = _collapsed_matmul_inputs(rng)
    be = NumpyBackend()
    ref = be.run(be.compile(_collapsed_matmul_graph()), input_data=dict(inputs))[0].outputs["o"]
    out = be.run(be.compile(split_graph), input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


def test_no_split_when_multi_accum_cell_has_stray_stmt() -> None:
    """Multi-accum extraction must claim every cell stmt into some gemm; a
    stray Assign in the K loop (no gemm home) bails conservatively."""
    fused = _fuse(_gated_mlp_graph())
    node = _fused_loop_node(fused)
    from deplodock.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt

    def add_stray(stmts) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Loop):
                if s.is_reduce and any(isinstance(c, Load) and c.input == "wg" for c in s.body):
                    ld = next(c for c in s.body if isinstance(c, Load) and c.input == "wg")
                    stray = Assign(name="stray", op="negative", args=(ld.names[0],))
                    out.append(Loop(axis=s.axis, body=Body((*s.body, stray)), unroll=s.unroll))
                else:
                    out.append(Loop(axis=s.axis, body=Body(add_stray(s.body)), unroll=s.unroll))
            else:
                out.append(s)
        return tuple(out)

    strayed = LoopOp(body=Body(add_stray(node.op.body)))
    assert try_split_demoted(strayed, Context.from_target((12, 0)), graph=fused, node_id=node.id, out_tensor=node.output) is None


def test_no_split_when_cone_escapes() -> None:
    """A cone value consumed past the multiply can't be cut out — bail.

    Modifies the real fused body: a second Write inside the K loop stores the
    multiply's cone arg, so the moved value escapes the slice."""
    fused = _fuse(_norm_linear_graph())
    node = _fused_loop_node(fused)
    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Write

    k_loop = None

    def find_k(stmts):
        nonlocal k_loop
        for s in stmts:
            if isinstance(s, Loop):
                if s.is_reduce and any(isinstance(c, Load) and c.input == "wg" for c in s.body):
                    k_loop = s
                else:
                    find_k(s.body)

    find_k(node.op.body)
    assert k_loop is not None
    mul = next(c for c in k_loop.body if getattr(c, "name", None) == k_loop.body[-1].value)
    cone_arg = next(a for a in mul.args if not any(isinstance(c, Load) and a in c.names and c.input == "wg" for c in k_loop.body))

    def leak_cone(stmts) -> tuple[Stmt, ...]:
        out = []
        for s in stmts:
            if s is k_loop:
                leak = Write(output="leak", index=(Var(s.axis.name),), values=(cone_arg,))
                out.append(Loop(axis=s.axis, body=Body((*s.body, leak)), unroll=s.unroll))
            elif isinstance(s, Loop):
                out.append(Loop(axis=s.axis, body=Body(leak_cone(s.body)), unroll=s.unroll))
            else:
                out.append(s)
        return tuple(out)

    leaky = LoopOp(body=Body(leak_cone(node.op.body)))
    assert try_split_demoted(leaky, Context.from_target((12, 0)), graph=fused, node_id=node.id, out_tensor=node.output) is None


# ---------------------------------------------------------------------------
# The 005 rule: option list, marker idempotence, pins
# ---------------------------------------------------------------------------


class _StubMatch:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph


def _rule_rewrite(fused: Graph, cc=(12, 0)):
    node = _fused_loop_node(fused)
    return split_rule.rewrite(Context.from_target(cc), _StubMatch(fused), node)


def test_rule_offers_fused_first_then_split() -> None:
    fused = _fuse(_norm_linear_graph())
    node = _fused_loop_node(fused)
    options = _rule_rewrite(fused)
    assert isinstance(options, list) and len(options) == 2
    keep, split = options
    # Fused FIRST — the greedy cold pick keeps today's kernel sets. The
    # keep-fused option is the same body with the decision knob stamped:
    # SPLIT_CONE=False ("considered, declined") vs absent ("never offered") is
    # the learned prior's training signal AND the rule's idempotence guard.
    assert isinstance(keep, LoopOp)
    assert keep.knobs[split_rule.SPLIT_CONE.name] is False
    assert keep.body.structural_key() == node.op.body.structural_key()
    assert isinstance(split, OptionFork) and isinstance(split.option, Graph)
    # Ranking knobs carry the offer site's full knob base under the decision
    # delta — feature-identical to the keep side's lifted fork (and to the
    # composed Σ rows the two-level tuner trains the prior on).
    assert split.knobs == {**node.op.knobs, split_rule.SPLIT_CONE.name: True}
    # Both split kernels carry the decision on op.knobs too — their perf rows
    # train the prior with SPLIT_CONE=1.
    for n in split.option.nodes.values():
        if isinstance(n.op, LoopOp):
            assert n.op.knobs[split_rule.SPLIT_CONE.name] is True


def test_rule_knob_guard_skips_reconsider() -> None:
    """Once a branch's decision is applied (knob on op.knobs), batch
    re-enumeration in fork children skips the kernel instead of re-offering
    (multi-site graphs compound combinatorially otherwise)."""
    fused = _fuse(_norm_linear_graph())
    node = _fused_loop_node(fused)
    keep = _rule_rewrite(fused)[0]
    node.op = keep  # what Candidate.apply does when the keep-fused option resolves
    with pytest.raises(RuleSkipped, match="already considered"):
        _rule_rewrite(fused)


def test_rule_skips_uncuttable() -> None:
    # A pure matmul (no cone) is not cuttable — the rule skips, no marker.
    g = Graph()
    f16 = _dt.get("f16")
    g.add_node(InputOp(), [], Tensor("x", (128, 128), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (128, 128), f16), node_id="w")
    g.add_node(MatmulOp(), ["x", "w"], Tensor("o", (128, 128), f16), node_id="o")
    g.inputs = ["x", "w"]
    g.outputs = ["o"]
    pure = _fuse(g)
    with pytest.raises(RuleSkipped, match="not a cuttable"):
        _rule_rewrite(pure)
    # Never-offered kernels stay knob-free — the prior's "not considered" state.
    assert split_rule.SPLIT_CONE.name not in _fused_loop_node(pure).op.knobs


def test_pin_split_cone_forces_each_branch(monkeypatch) -> None:
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    assert isinstance(_rule_rewrite(_fuse(_norm_linear_graph())), Graph)
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "0")
    declined = _rule_rewrite(_fuse(_norm_linear_graph()))
    assert isinstance(declined, LoopOp)
    assert declined.knobs[split_rule.SPLIT_CONE.name] is False


# ---------------------------------------------------------------------------
# Greedy compile: the structural option is never the cold pick
# ---------------------------------------------------------------------------


def _lowered_kernel_ids(graph: Graph) -> list[str]:
    from deplodock.compiler.ir.cuda.ir import CudaOp

    return [nid for nid, n in graph.nodes.items() if isinstance(n.op, CudaOp)]


def test_greedy_compile_keeps_fused_kernel() -> None:
    out = Pipeline.build(CUDA_PASSES).run(_norm_linear_graph(), ctx=Context.from_target((12, 0)), db=SearchDB())
    assert len(_lowered_kernel_ids(out)) == 1, "greedy must never pick the structural split cold (untrained prior)"


class _SplitFavoringPrior:
    """Stub 'trained' prior: every knob row carrying ``SPLIT_CONE=True`` (the
    split fragment's kernels) prices cheap, everything else expensive — so the
    structural pricing (`policy/greedy._pick_structural`) predicts the split's
    Σ (2 kernels × 1.0) below the fused side's 100.0 and deploys it. Constant
    within a side, so ordinary tile picks tie and fall to emission order."""

    fitted = True

    def mean_scores(self, rows: list[dict]) -> list[float]:
        return [1.0 if r.get(split_rule.SPLIT_CONE.name) is True else 100.0 for r in rows]


class _ColdPrior(_SplitFavoringPrior):
    fitted = False


def test_greedy_trained_prior_deploys_split() -> None:
    """With the trained prior predicting the split kernels cheaper, unpinned
    greedy prices the structural option and deploys the two-kernel split
    (plans/structural-forks-in-two-level.md step 3 — the deploy path). The
    structural pick is a trace fact: a ``Decision`` with ``chosen_kind ==
    "graph"``."""
    from deplodock.compiler.pipeline import TILE_PASSES
    from deplodock.compiler.pipeline.pipeline import Run
    from deplodock.compiler.pipeline.search.policy import greedy_decide

    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0)))
    terminal, trace = run.resolve(_norm_linear_graph(), greedy_decide(prior=_SplitFavoringPrior()))
    tiles = [nid for nid, n in terminal.nodes.items() if isinstance(n.op, TileOp)]
    assert any(d.chosen_kind == "graph" for d in trace), "the structural pick must appear in the trace"
    assert len(tiles) == 2, f"the trained-prior pick must deploy the split, got {tiles}"
    assert any(nid.endswith("__xn") for nid in tiles)


def test_greedy_cold_stub_prior_keeps_fused() -> None:
    """The same prediction surface but unfitted: pricing is gated on the
    trained model, so the structural leaf stays filtered."""
    from deplodock.compiler.pipeline import TILE_PASSES
    from deplodock.compiler.pipeline.pipeline import Run
    from deplodock.compiler.pipeline.search.policy import greedy_decide

    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0)))
    terminal, trace = run.resolve(_norm_linear_graph(), greedy_decide(prior=_ColdPrior()))
    assert not any(d.chosen_kind == "graph" for d in trace)
    assert sum(1 for n in terminal.nodes.values() if isinstance(n.op, TileOp)) == 1


def test_greedy_structural_pick_falls_back_on_lowering_failure(monkeypatch) -> None:
    """A structural pick whose fragment kernel fails to lower must re-drive
    down the keep-fused branch (the structural choice is blockable as a whole):
    the xn producer's every KernelOp variant fails validate, so the first drive
    leaves it un-lowered and the retry retires structural picks — the compile
    succeeds with the fused kernel instead of raising LoweringError."""
    from deplodock.compiler.ir.kernel.ir import KernelOp
    from deplodock.compiler.pipeline.search import prior as prior_pkg

    monkeypatch.setattr(prior_pkg, "load_prior", lambda *a, **kw: _SplitFavoringPrior())
    real_validate = KernelOp.validate
    monkeypatch.setattr(KernelOp, "validate", lambda self, ctx: False if self.name.endswith("_xn") else real_validate(self, ctx))

    out = Pipeline.build(CUDA_PASSES).run(_norm_linear_graph(), ctx=Context.from_target((12, 0)), db=SearchDB())
    assert len(_lowered_kernel_ids(out)) == 1, "the failed structural pick must fall back to the fused kernel"


def test_pinned_split_lowers_two_kernels(monkeypatch) -> None:
    # The Linear-derived transposed-B gemm is atom-ineligible (mirrors the
    # cell tagger), so no warp rows enumerate — the split lowers on the
    # scalar tier with no knob pins needed.
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    out = Pipeline.build(CUDA_PASSES).run(_norm_linear_graph(), ctx=Context.from_target((12, 0)), db=SearchDB())
    ids = _lowered_kernel_ids(out)
    assert len(ids) == 2
    assert any(nid.endswith("__xn") for nid in ids)


# ---------------------------------------------------------------------------
# Tune: both branches are real terminals
# ---------------------------------------------------------------------------


def test_tune_explores_fused_and_split_terminals(monkeypatch) -> None:
    """Through TILE_PASSES (no render — keeps the test off the kernel/cuda
    stages) the tuning search reaches terminals on both sides of the fork:
    one-kernel (fused) and two-kernel (split) tile graphs. Knob pins shrink
    the variant space; the loop early-exits once both kinds are seen."""
    for k, v in {"WM": "2", "WN": "2", "FM": "1", "FN": "8", "BK": "2", "BM": "8", "BN": "64", "BR": "1", "SPLITK": "1", "FK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    seen: set[int] = set()

    def _saw(t) -> bool:
        seen.add(sum(1 for n in t.graph.nodes.values() if isinstance(n.op, TileOp)))
        return {1, 2} <= seen  # early-exit once both kinds are seen

    drain_tune(
        Pipeline.build(TILE_PASSES),
        _norm_linear_graph(),
        search=TuningSearch(patience=10**6),
        ctx=Context.from_target((12, 0)),
        db=SearchDB(),
        on=_saw,
    )
    assert {1, 2} <= seen, f"both fused (1-kernel) and split (2-kernel) terminals must appear, saw {seen}"


# ---------------------------------------------------------------------------
# Accuracy — numpy (no GPU) and CUDA
# ---------------------------------------------------------------------------


def _norm_linear_inputs(rng) -> dict:
    npf16 = np.dtype(np.float16)
    return {
        "x": rng.standard_normal((1, _S, _H), dtype=np.float32).astype(npf16),
        "nw": rng.standard_normal((_H,), dtype=np.float32).astype(npf16),
        "wg": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(npf16),
    }


def _collapsed_matmul_inputs(rng, HD: int = 4, S: int = 32, D: int = 64, N: int = 64) -> dict:
    npf16 = np.dtype(np.float16)
    return {
        "attn": rng.standard_normal((1, HD, S, D), dtype=np.float32).astype(npf16),
        "w": (rng.standard_normal((HD * D, N), dtype=np.float32) * 0.05).astype(npf16),
        "res": rng.standard_normal((1, S, N), dtype=np.float32).astype(npf16),
    }


def _gated_mlp_inputs(rng, S: int = 32, H: int = 256, I: int = 512) -> dict:  # noqa: E741
    npf16 = np.dtype(np.float16)
    return {
        "x": rng.standard_normal((1, S, H), dtype=np.float32).astype(npf16),
        "nw": rng.standard_normal((H,), dtype=np.float32).astype(npf16),
        "wg": (rng.standard_normal((H, I), dtype=np.float32) * 0.05).astype(npf16),
        "wu": (rng.standard_normal((H, I), dtype=np.float32) * 0.05).astype(npf16),
    }


def _assert_close(out, ref) -> None:
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out.astype(np.float32)))
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), atol=max(1e-2, 0.05 * peak), rtol=0.05)


def test_split_matches_fused_on_numpy() -> None:
    """The spliced two-kernel graph computes the same function as the fused one."""
    from deplodock.compiler.backend.numpy import NumpyBackend

    fused = _fuse(_norm_linear_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    split_graph = fused.copy()
    split_graph.splice(frag, consumed={node.id}, output=node.id)
    assert sum(1 for n in split_graph.nodes.values() if isinstance(n.op, LoopOp)) == 2

    rng = np.random.default_rng(0)
    inputs = _norm_linear_inputs(rng)
    be = NumpyBackend()
    ref = be.run(be.compile(_norm_linear_graph()), input_data=dict(inputs))[0].outputs["o"]
    out = be.run(be.compile(split_graph), input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_split_scalar_accuracy_cuda(monkeypatch) -> None:
    """Pinned split, gemm on the scalar tier: two kernels, matches numpy."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    target_mod.set_target(None)  # live device
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    rng = np.random.default_rng(0)
    inputs = _norm_linear_inputs(rng)
    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_norm_linear_graph()), input_data=dict(inputs))[0].outputs["o"]
    be = CudaBackend()
    compiled = be.compile(_norm_linear_graph())
    assert len(_lowered_kernel_ids(compiled)) == 2
    out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_split_mma_accuracy_cuda(monkeypatch) -> None:
    """Pinned split on a MatmulOp-derived demotion: the clean gemm runs on
    mma.sync and matches numpy. (Linear-derived gemms can't mma-lower — the
    transposed-B operand is unclassifiable by the cell tagger, and the
    eligibility gate now mirrors that, keeping them off the warp tier.)"""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    target_mod.set_target(None)
    for k, v in {"SPLIT_CONE": "1", "MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "4", "FN": "8", "BK": "2"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    M, K, N = 128, 128, 128
    rng = np.random.default_rng(0)
    npf16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((M, K), dtype=np.float32).astype(npf16),
        "s": rng.standard_normal((M, K), dtype=np.float32).astype(npf16),
        "w": (rng.standard_normal((K, N), dtype=np.float32) * 0.05).astype(npf16),
    }
    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_scale_matmul_graph(M, K, N)), input_data=dict(inputs))[0].outputs["o"]
    be = CudaBackend()
    compiled = be.compile(_scale_matmul_graph(M, K, N))
    mma_kernels = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp) and "mma.sync" in n.op.kernel_source]
    assert len(mma_kernels) == 1, "the split gemm must lower on the warp tier"
    out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_two_sided_split_mma_accuracy_cuda(monkeypatch) -> None:
    """Pinned two-sided split: three kernels, the consumer gemm runs on
    mma.sync (the [K, N] xn_b materialization fixes the Linear-transposed
    B layout), and the output matches numpy."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    target_mod.set_target(None)
    for k, v in {"SPLIT_CONE": "1", "MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "4", "FN": "8", "BK": "2"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    rng = np.random.default_rng(0)
    npf16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
        "sx": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
        "w": (rng.standard_normal((128, 128), dtype=np.float32) * 0.05).astype(npf16),
        "sw": rng.standard_normal((128, 128), dtype=np.float32).astype(npf16),
    }
    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_double_scale_linear_graph()), input_data=dict(inputs))[0].outputs["o"]
    be = CudaBackend()
    compiled = be.compile(_double_scale_linear_graph())
    ids = _lowered_kernel_ids(compiled)
    assert len(ids) == 3
    mma_kernels = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp) and "mma.sync" in n.op.kernel_source]
    assert len(mma_kernels) == 1, "the two-sided split gemm must lower on the warp tier"
    out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_gated_mlp_split_mma_accuracy_cuda(monkeypatch) -> None:
    """Pinned split on the gated-MLP dual-accum kernel: three kernels (the
    ``006_merge_split_glue`` re-fusion folds the pointwise combine into one
    gemm's epilogue), BOTH extracted gemms run on mma.sync, and the output
    matches numpy (finding 2's end-to-end shape — the real Qwen3 kernel goes
    51 µs scalar-fused → ~13 µs split on an RTX 5090)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    target_mod.set_target(None)
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    rng = np.random.default_rng(0)
    inputs = _gated_mlp_inputs(rng)
    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_gated_mlp_graph()), input_data=dict(inputs))[0].outputs["o"]
    be = CudaBackend()
    compiled = be.compile(_gated_mlp_graph())
    ids = _lowered_kernel_ids(compiled)
    assert len(ids) == 3, "xn + gemm + (gemm with the combine folded as its epilogue)"
    mma_kernels = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp) and "mma.sync" in n.op.kernel_source]
    assert len(mma_kernels) == 2, "both extracted gemms must lower on the warp tier"
    out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_layout_split_mma_accuracy_cuda(monkeypatch) -> None:
    """Pinned split on the collapsed-layout matmul (finding 3's o_proj shape):
    the contiguizing copy producer + the clean gemm on mma.sync, output
    matches numpy (the real o_proj kernel goes 25 µs scalar-fused → ~6 µs
    split on an RTX 5090)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    target_mod.set_target(None)
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    rng = np.random.default_rng(0)
    inputs = _collapsed_matmul_inputs(rng)
    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_collapsed_matmul_graph()), input_data=dict(inputs))[0].outputs["o"]
    be = CudaBackend()
    compiled = be.compile(_collapsed_matmul_graph())
    ids = _lowered_kernel_ids(compiled)
    assert len(ids) == 2
    mma_kernels = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp) and "mma.sync" in n.op.kernel_source]
    assert len(mma_kernels) == 1, "the layout-split gemm must lower on the warp tier"
    out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


# ---------------------------------------------------------------------------
# Symbolic row axes (dynamic shapes): the cut admits Dim('seq_len') rows
# ---------------------------------------------------------------------------


def _sym_norm_linear_graph() -> Graph:
    """``_norm_linear_graph`` with the seq axis symbolic (the dynamic-trace
    shape every Qwen3 layer-0 kernel carries)."""
    from deplodock.compiler.dim import Dim

    dt = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Dim("seq_len"), _H), dt), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (_H,), dt), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (_I, _H), dt), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, Dim("seq_len"), _H), dt), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, Dim("seq_len"), _I), dt), node_id="o")
    g.inputs = ["x", "nw", "wg"]
    g.outputs = ["o"]
    return g


def test_split_offered_for_symbolic_rows() -> None:
    """Symbolic ROW extents don't bail the cut: the producers / consumer
    re-emit the row Loops verbatim and the ``xn`` buffer carries the symbolic
    Dim (allocated from the runtime extent like any symbolic intermediate)."""
    fused = _fuse(_sym_norm_linear_graph())
    frag = _split(fused)
    assert frag is not None, "the cut must offer on symbolic-row graphs"
    xn_nodes = [n for nid, n in frag.nodes.items() if "__xn" in nid]
    assert xn_nodes, "expected an xn producer in the fragment"
    seq_dims = [d for d in xn_nodes[0].output.shape if not d.is_static]
    assert seq_dims and "seq_len" in seq_dims[0].expr.free_vars(), (
        f"xn buffer must carry the symbolic seq dim, got {xn_nodes[0].output.shape}"
    )


def test_gated_mlp_split_offered_for_symbolic_rows() -> None:
    """The multi-accum (gated-MLP) cut also admits symbolic rows: each
    extracted gemm's ``mm_i`` buffer carries the symbolic leading dim."""
    from deplodock.compiler.dim import Dim

    f16 = _dt.get("f16")
    S, H, I = Dim("seq_len"), 256, 512  # noqa: E741
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), f16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (H, I), f16), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (H, I), f16), node_id="wu")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), f16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "wg"], Tensor("gate", (1, S, I), f16), node_id="gate")
    g.add_node(MatmulOp(), ["xn", "wu"], Tensor("up", (1, S, I), f16), node_id="up")
    g.add_node(ElementwiseOp("multiply"), ["gate", "up"], Tensor("o", (1, S, I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg", "wu"]
    g.outputs = ["o"]

    fused = _fuse(g)
    frag = _split(fused)
    assert frag is not None, "the multi-accum cut must offer on symbolic-row graphs"
    mm_nodes = [n for nid, n in frag.nodes.items() if "__mm" in nid and nid.endswith(("0", "1"))]
    assert len(mm_nodes) == 2, "expected two extracted gemm producers"
    for n in mm_nodes:
        assert any(not d.is_static for d in n.output.shape), f"mm buffer must carry the symbolic leading dim, got {n.output.shape}"


def test_collapsed_attn_out_split_offered_for_symbolic_rows() -> None:
    """The o_proj collapsed attn-out cut admits symbolic rows even though its
    K-folded operand index references the symbolic dim in its strides (the head
    stride is ``seq_len * D``). That ``seq_len`` read is a legitimate runtime
    quantity, not an unmodeled scope — the regression that kept the dynamic
    o_proj fused-scalar at seq_len=512 (~461 us) instead of warp-tier."""
    from deplodock.compiler.dim import Dim

    f16 = _dt.get("f16")
    HD, S, D, N = 4, Dim("seq_len"), 64, 64
    K = HD * D
    g = Graph()
    g.add_node(InputOp(), [], Tensor("attn", (1, HD, S, D), f16), node_id="attn")
    g.add_node(InputOp(), [], Tensor("w", (K, N), f16), node_id="w")
    g.add_node(InputOp(), [], Tensor("res", (1, S, N), f16), node_id="res")
    g.add_node(TransposeOp(axes=(1, 2)), ["attn"], Tensor("at", (1, S, HD, D), f16), node_id="at")
    g.add_node(ReshapeOp(shape=(1, S, K)), ["at"], Tensor("ar", (1, S, K), f16), node_id="ar")
    g.add_node(MatmulOp(), ["ar", "w"], Tensor("mm", (1, S, N), f16), node_id="mm")
    g.add_node(ElementwiseOp("add"), ["mm", "res"], Tensor("o", (1, S, N), f16), node_id="o")
    g.inputs = ["attn", "w", "res"]
    g.outputs = ["o"]

    frag = _split(_fuse(g))
    assert frag is not None, "the collapsed attn-out cut must offer on symbolic-row graphs"
    xn_nodes = [n for nid, n in frag.nodes.items() if "__xn" in nid]
    assert xn_nodes, "expected a contiguizing xn producer in the fragment"
    assert any(not d.is_static for d in xn_nodes[0].output.shape), (
        f"xn buffer must carry the symbolic row dim, got {xn_nodes[0].output.shape}"
    )


def test_split_offered_for_symbolic_k() -> None:
    """A symbolic reduce extent IS admitted now that the warp tier accepts a
    MASKED K (hint-tiled, the partial final slab zero-filled in smem — see
    ``010_partition_loops`` / ``_stage_expand``). The split hands the cell tagger
    a clean symbolic-K gemm that reaches the tensor-core tier, exactly the
    static-K case; before masked-K MMA this was bailed (no tier upgrade).
    (Symbolic N and symbolic ROW axes were already admitted — tests above.)"""
    from deplodock.compiler.dim import Dim

    f16 = _dt.get("f16")
    K = Dim("k_len")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (32, K), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("s", (32, K), f16), node_id="s")
    g.add_node(InputOp(), [], Tensor("w", (K, 64), f16), node_id="w")
    g.add_node(ElementwiseOp("multiply"), ["x", "s"], Tensor("xs", (32, K), f16), node_id="xs")
    g.add_node(MatmulOp(), ["xs", "w"], Tensor("o", (32, 64), f16), node_id="o")
    g.inputs = ["x", "s", "w"]
    g.outputs = ["o"]
    frag = _split(_fuse(g))
    assert frag is not None, "symbolic-K demoted matmul should now split (masked-K warp tier)"
    loops = {n.id for n in frag.nodes.values() if isinstance(n.op, LoopOp)}
    assert loops == {"o__xn", "o__mm"}


def test_symbolic_split_matches_fused_on_numpy() -> None:
    """The spliced symbolic-row graph computes the same function as the fused
    one at a runtime seq != the 512 hint."""
    from deplodock.compiler.backend.numpy import NumpyBackend

    fused = _fuse(_sym_norm_linear_graph())
    node = _fused_loop_node(fused)
    frag = _split(fused)
    assert frag is not None
    split_graph = fused.copy()
    split_graph.splice(frag, consumed={node.id}, output=node.id)

    rng = np.random.default_rng(0)
    seq = 48
    npf16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((1, seq, _H), dtype=np.float32).astype(npf16),
        "nw": rng.standard_normal((_H,), dtype=np.float32).astype(npf16),
        "wg": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(npf16),
    }
    be = NumpyBackend()
    ref = be.run(be.compile(_sym_norm_linear_graph()), input_data=dict(inputs))[0].outputs["o"]
    out = be.run(be.compile(split_graph), input_data=dict(inputs))[0].outputs["o"]
    _assert_close(out, ref)


@requires_cuda
def test_symbolic_split_accuracy_cuda(monkeypatch) -> None:
    """Pinned symbolic-row split on the live GPU: one compiled two-kernel
    program serves two runtime seqs (33 and 80 — neither the hint), matching
    numpy at both."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    target_mod.set_target(None)  # live device
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    rng = np.random.default_rng(0)
    npf16 = np.dtype(np.float16)
    shared = {
        "nw": rng.standard_normal((_H,), dtype=np.float32).astype(npf16),
        "wg": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(npf16),
    }
    be = CudaBackend()
    compiled = be.compile(_sym_norm_linear_graph())
    assert len(_lowered_kernel_ids(compiled)) == 2
    ref_be = NumpyBackend()
    ref_compiled = ref_be.compile(_sym_norm_linear_graph())
    for seq in (33, 80):
        inputs = {"x": rng.standard_normal((1, seq, _H), dtype=np.float32).astype(npf16), **shared}
        ref = ref_be.run(ref_compiled, input_data=dict(inputs))[0].outputs["o"]
        out = be.run(compiled, input_data=dict(inputs))[0].outputs["o"]
        _assert_close(out, ref)
