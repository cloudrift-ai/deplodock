"""MMA staging by default — M6 of ``plans/mma-smem-staging.md``.

After M6 the ``ATOM_KIND`` skip in ``020_stage_inputs`` is gone and
MMA matmuls stage through smem unconditionally. This test pins that
invariant: a small MMA matmul produces at least one ``StageBundle``
post-tile-lowering. End-to-end correctness is covered by
``test_matmul_mma.py``.

The file name dates from M1 (when the gate was probe-toggled). The
probe env var was removed in M6; the test name is preserved so anyone
chasing the milestone history finds it here.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import StageBundle
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _mma_matmul_graph(*, M: int = 64, N: int = 64, K: int = 64) -> Graph:
    """Minimal F16×F16 matmul graph the MMA planner picks up."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    i = Axis("i", M)
    j = Axis("j", N)
    k = Axis("k", K)
    g.add_node(
        op=LoopOp(
            body=(
                Loop(
                    axis=i,
                    body=(
                        Loop(
                            axis=j,
                            body=(
                                Loop(
                                    axis=k,
                                    body=(
                                        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                        Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                        Accum(name="acc", value="p"),
                                    ),
                                ),
                                Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        inputs=["a", "b"],
        output=Tensor("c", (M, N), dtype=F16),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def _has_stage_bundle(graph: Graph) -> bool:
    for node in graph.nodes.values():
        body = getattr(node.op, "body", None)
        if body is None:
            continue
        for stmt in body.iter():
            if isinstance(stmt, StageBundle):
                return True
    return False


def test_mma_matmul_stages_through_smem(monkeypatch):
    """An MMA-eligible F16×F16 matmul produces at least one StageBundle
    in the post-tile body. Pins the M6 invariant: the ATOM_KIND skip
    block in 020_stage_inputs is gone, so the staging admission path
    runs for every warp-tier MMA TileOp."""
    # Pin the s16816 atom + a multi-warp tile: single-warp mma.sync is pruned
    # (ldmatrix is smem→register only), and an ``MMA=<kind>`` pin enumerates
    # the kind at any arch. 64²: WM=2 FM=2 → M-tile 64, WN=2 FN=4 → N-tile
    # 2·4·atom_n(8)=64, BK=2 → K-stage 32.
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "2")
    monkeypatch.setenv("DEPLODOCK_FN", "4")
    monkeypatch.setenv("DEPLODOCK_BK", "2")
    g = _mma_matmul_graph()
    out = Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((8, 0)))
    kop = out.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    assert _has_stage_bundle(out)
