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
pinning, greedy never picking the structural option (compile keeps today's kernel sets),
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
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, TileOp
from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline, RuleSkipped, TuningSearch
from deplodock.compiler.pipeline.fork import OptionFork
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile._split_demoted import try_split_demoted
from deplodock.compiler.pipeline.search.db import SearchDB

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
    # Consumer: loads xn + the weight. The Linear-derived gemm keeps its
    # transposed-B operand (wg is [N, K] — K in the last index dim of BOTH
    # loads), which ``011_lower_atom_cell._classify_ab`` cannot tag — the
    # eligibility gate mirrors that honestly, so this split lands on the
    # scalar register-tile tier (informational here, NOT a gate the builder
    # applies; the MatmulOp-derived split below reaches the warp tier).
    assert xn.id in mm.inputs
    ctx = Context.from_target((12, 0))
    assert not any(is_atom_eligible(atom, mm.op, ctx, graph=frag) for atom in ATOM_REGISTRY.values())
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
    assert split.knobs == {split_rule.SPLIT_CONE.name: True}
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
    assert len(_lowered_kernel_ids(out)) == 1, "greedy must never pick the structural split"


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
    for t in Pipeline.build(TILE_PASSES).tune(
        _norm_linear_graph(),
        search=TuningSearch(patience=10**6),
        ctx=Context.from_target((12, 0)),
        db=SearchDB(),
    ):
        seen.add(sum(1 for n in t.graph.nodes.values() if isinstance(n.op, TileOp)))
        if {1, 2} <= seen:
            break
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
