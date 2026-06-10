"""Tests for the demoted-matmul split (``lowering/tile/_split_demoted`` + the
``UNFUSE`` fork option in ``010_partition_loops``).

A fused computed-operand cone (gated-MLP norm prologue, elementwise scale) keeps a matmul
off the warp tier; the planner offers a structural fork splitting the kernel into an ``xn``
producer + the clean gemm. Covers: the split builder's offered / not-offered gates, the
fragment's structure, the partition rewrite's option list (fused first), ``DEPLODOCK_UNFUSE``
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
from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline, TuningSearch
from deplodock.compiler.pipeline.fork import Fork, OptionFork
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile._split_demoted import try_split_demoted
from deplodock.compiler.pipeline.search.db import SearchDB

from ..conftest import requires_cuda

partition = importlib.import_module(
    "deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops",
)

_S, _H, _I = 32, 1024, 3072


@pytest.fixture(autouse=True)
def _force_sm120(monkeypatch, tmp_path):
    """sm_120 target (auto-mma enumerates) + an isolated, untrained prior so
    greedy picks are deterministic regardless of the host's checkpoint."""
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
# The split builder: offered / not offered
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
    # Consumer: loads xn + the weight, and is genuinely warp-tier eligible.
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


def test_no_split_for_f32() -> None:
    """The atom registry is f16/bf16-only — an f32 twin never reaches the warp
    tier, so splitting buys nothing."""
    assert _split(_fuse(_norm_linear_graph("f32"))) is None


def test_no_split_below_sm90_without_pin(monkeypatch) -> None:
    """Auto-enumerated mma.sync is Hopper+; on sm_80 the split is offered only
    under an explicit DEPLODOCK_MMA pin (mirrors the planner's warp-row gate)."""
    target_mod.set_target((8, 0))
    fused = _fuse(_norm_linear_graph(), cc=(8, 0))
    assert _split(fused, cc=(8, 0)) is None
    monkeypatch.setenv("DEPLODOCK_MMA", next(iter(ATOM_REGISTRY)))
    assert _split(fused, cc=(8, 0)) is not None


def test_no_split_with_mma_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    assert _split(_fuse(_norm_linear_graph())) is None


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
# The partition rewrite: option list + pins
# ---------------------------------------------------------------------------


class _StubMatch:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph


def _partition_rewrite(fused: Graph, cc=(12, 0)):
    node = _fused_loop_node(fused)
    return partition.rewrite(Context.from_target(cc), node, _StubMatch(fused))


def test_partition_offers_fused_first_then_split() -> None:
    options = _partition_rewrite(_fuse(_norm_linear_graph()))
    assert isinstance(options, list) and len(options) == 2
    fused_opt, split_opt = options
    # Fused FIRST — the greedy cold pick keeps today's kernel sets.
    assert isinstance(fused_opt, (Fork, TileOp)) and not isinstance(fused_opt, OptionFork)
    assert isinstance(split_opt, OptionFork)
    assert split_opt.knobs == {partition.UNFUSE.name: True}
    assert isinstance(split_opt.option, Graph)


def test_partition_no_split_for_f32() -> None:
    result = _partition_rewrite(_fuse(_norm_linear_graph("f32")))
    assert not isinstance(result, list), "f32 keeps today's single fused outcome"


def test_pin_unfuse_forces_each_branch(monkeypatch) -> None:
    monkeypatch.setenv("DEPLODOCK_UNFUSE", "1")
    assert isinstance(_partition_rewrite(_fuse(_norm_linear_graph())), Graph)
    monkeypatch.setenv("DEPLODOCK_UNFUSE", "0")
    assert not isinstance(_partition_rewrite(_fuse(_norm_linear_graph())), (list, Graph))


# ---------------------------------------------------------------------------
# Greedy compile: the structural option is never the cold pick
# ---------------------------------------------------------------------------


def _lowered_kernel_ids(graph: Graph) -> list[str]:
    from deplodock.compiler.ir.cuda.ir import CudaOp

    return [nid for nid, n in graph.nodes.items() if isinstance(n.op, CudaOp)]


def test_greedy_compile_keeps_fused_kernel() -> None:
    out = Pipeline.build(CUDA_PASSES).run(_norm_linear_graph(), ctx=Context.from_target((12, 0)), db=SearchDB())
    assert len(_lowered_kernel_ids(out)) == 1, "greedy must never pick the structural split"


def test_pinned_unfuse_lowers_two_kernels(monkeypatch) -> None:
    # WM=WN=1 prunes the warp rows (single-warp mma is unservable), keeping the
    # gemm on the scalar tier — the mma materializer can't lower the
    # Linear-derived transposed-B operand yet (pre-existing; see PR notes).
    monkeypatch.setenv("DEPLODOCK_UNFUSE", "1")
    monkeypatch.setenv("DEPLODOCK_WM", "1")
    monkeypatch.setenv("DEPLODOCK_WN", "1")
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
    the variant space so the full drain stays fast."""
    for k, v in {"WM": "2", "WN": "2", "FM": "1", "FN": "8", "BK": "2", "BM": "8", "BN": "64", "BR": "1", "SPLITK": "1", "FK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    terminals = list(
        Pipeline.build(TILE_PASSES).tune(
            _norm_linear_graph(),
            search=TuningSearch(patience=10**6),
            ctx=Context.from_target((12, 0)),
            db=SearchDB(),
        )
    )
    sizes = sorted(sum(1 for n in t.graph.nodes.values() if isinstance(n.op, TileOp)) for t in terminals)
    assert 1 in sizes, "the fused branch must reach a terminal"
    assert 2 in sizes, "the split branch must reach a terminal"


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
    monkeypatch.setenv("DEPLODOCK_UNFUSE", "1")
    monkeypatch.setenv("DEPLODOCK_WM", "1")
    monkeypatch.setenv("DEPLODOCK_WN", "1")
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
    mma.sync and matches numpy. (Linear-derived gemms can't mma-lower yet —
    the transposed-B operand is rejected by the cell tagger; pre-existing.)"""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    target_mod.set_target(None)
    for k, v in {"UNFUSE": "1", "MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "4", "FN": "8", "BK": "2"}.items():
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
