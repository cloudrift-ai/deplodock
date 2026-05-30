"""M1 of ``plans/mma-smem-staging.md`` — probe-gate plumbing.

Verifies the ``DEPLODOCK_MMA_STAGE_PROBE`` env var flips the
``ATOM_KIND`` skip in ``020_stage_inputs.rewrite``. The probe-on
behavior of actually emitting a ``StageBundle`` is M3-dependent
(``_classify`` must recognize the atom σ first); these tests only
pin the gate.
"""

from __future__ import annotations

import pytest

from deplodock import config
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


def test_mma_stage_probe_reads_env(monkeypatch):
    """``config.mma_stage_probe`` reflects the env var, default OFF."""
    monkeypatch.delenv("DEPLODOCK_MMA_STAGE_PROBE", raising=False)
    assert config.mma_stage_probe() is False
    monkeypatch.setenv("DEPLODOCK_MMA_STAGE_PROBE", "1")
    assert config.mma_stage_probe() is True
    monkeypatch.setenv("DEPLODOCK_MMA_STAGE_PROBE", "0")
    assert config.mma_stage_probe() is False


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


@pytest.mark.parametrize("probe_on", [False, True])
def test_pipeline_runs_to_tile_under_probe(monkeypatch, probe_on):
    """Pipeline runs cleanly through ``lowering/tile`` with the probe in
    either state. Probe-OFF skips staging on ATOM_KIND (no
    ``StageBundle``). Probe-ON exercises the M3 / M5 path: 020's
    ``_classify`` stamps ``AffineAddressing.block`` on the atom-strided
    σ and admits the eligible operand(s) — at least one ``StageBundle``
    appears in the post-tile body.
    """
    if probe_on:
        monkeypatch.setenv("DEPLODOCK_MMA_STAGE_PROBE", "1")
    else:
        monkeypatch.delenv("DEPLODOCK_MMA_STAGE_PROBE", raising=False)
    monkeypatch.setenv("DEPLODOCK_MMA", "1")

    g = _mma_matmul_graph()
    out = Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((8, 0)))
    kop = out.nodes["c"].op
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16"
    if probe_on:
        assert _has_stage_bundle(out), "expected at least one StageBundle when probe is ON"
    else:
        assert not _has_stage_bundle(out)
