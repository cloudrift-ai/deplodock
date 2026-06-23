"""The cut-offer policy — the derived tier-monotonicity predicate (R7 edge placement).

``plans/dag-edge-placement-split-as-enumeration.md`` relocates the demoted-matmul cut's
*offer* decision out of the legacy ``005_split_demoted`` monolith into a single auditable
derived-view query, ``enumeration/_cut.py``. These tests pin the predicate's **tightness**
(the plan's stated key risk): it must offer exactly on a demoted matmul (a fused operand
cone keeps it below any buildable tier) and decline a clean matmul / pointwise body.
"""

from __future__ import annotations

from deplodock.compiler import dtype as _dt
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._cut import Tier, cut_offers, tier
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag

_CC = (12, 0)


def _fused_loop(graph: Graph):
    """The single fused ``LoopOp`` the LOOP passes produce for ``graph`` — the seed the
    cut offer reads (still un-tiled; the tile phase's split runs on exactly this)."""
    out = Pipeline.build(LOOP_PASSES).run(graph, ctx=Context.from_target(_CC))
    return next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")


def _dtype_of(loop_op):
    def f(buf):
        if buf in loop_op.inputs:
            return loop_op.inputs[buf].dtype
        if buf in loop_op.outputs:
            return loop_op.outputs[buf].dtype
        return None

    return f


def _norm_linear_graph() -> Graph:
    """RMSNorm → Linear (f16): fusion folds the norm into the matmul reduce, demoting
    the matmul (a computed operand cone) below the warp tier."""
    f16 = _dt.get("f16")
    s, h, i = 32, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, s, h), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (h,), f16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (i, h), f16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, s, h), f16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, s, i), f16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]
    return g


def _f16_matmul_graph(M=128, K=128, N=128) -> Graph:
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), f16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), f16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), f16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _f32_matmul_graph(M=128, K=128, N=128) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _pointwise_graph(N=256) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (N,)), node_id="x")
    g.add_node(ElementwiseOp("relu"), ["x"], Tensor("o", (N,)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return g


def _tier(graph: Graph) -> Tier:
    lo = _fused_loop(graph)
    return tier(iter_dag(lo), compute_capability=_CC, dtype_of=_dtype_of(lo))


def _offer(graph: Graph):
    lo = _fused_loop(graph)
    return cut_offers(lo, compute_capability=_CC, dtype_of=_dtype_of(lo))


# --- tier lattice ------------------------------------------------------------


def test_tier_warp_for_clean_f16_matmul():
    # a clean f16 gemm reaches the tensor-core tier inline — already maximal, no cut
    assert _tier(_f16_matmul_graph()) is Tier.WARP_MMA


def test_tier_scalar_reduce_for_f32_matmul():
    # f32 has no tensor-core atom — the clean matmul tops out at the scalar reduce tier
    assert _tier(_f32_matmul_graph()) is Tier.SCALAR_REDUCE


def test_tier_map_for_pointwise():
    assert _tier(_pointwise_graph()) is Tier.MAP


def test_tier_unbuildable_for_demoted_matmul():
    # the fused norm cone defeats every regime — the move composer can't lower it
    assert _tier(_norm_linear_graph()) is Tier.UNBUILDABLE


def test_tier_lattice_is_ordered():
    assert Tier.UNBUILDABLE < Tier.MAP < Tier.SCALAR_REDUCE < Tier.COOP_REDUCE < Tier.WARP_MMA


# --- the offer predicate -----------------------------------------------------


def test_cut_offered_and_forced_on_demoted_matmul():
    d = _offer(_norm_linear_graph())
    assert d.offered and d.force  # the fused form is UNBUILDABLE → the cut must be taken
    assert d.tier_inline is Tier.UNBUILDABLE


def test_no_cut_on_clean_matmul():
    for g in (_f16_matmul_graph(), _f32_matmul_graph()):
        d = _offer(g)
        assert not d.offered and not d.force  # already at its maximal tier inline


def test_no_cut_on_pointwise():
    d = _offer(_pointwise_graph())
    assert not d.offered and not d.force
