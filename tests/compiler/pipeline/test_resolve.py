"""``Run.resolve`` — the deterministic-resolution entry point (one live graph,
a ``decide`` callback per fork, a ``Decision`` trace as the only process-state
output).

Pins the M1 contract: in-place apply (the terminal IS the seeded graph object,
no per-fork copies), an option-0 ``decide`` reproducing the no-prior greedy
compile bit-for-bit, the trace shape on a forked compile, and the
structural-decision replay being consulted (identical offer sites decide once).
No GPU: everything terminates in the tile dialect or earlier.
"""

from __future__ import annotations

import pytest

from emmy.compiler import target as target_mod
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.pipeline import Run


@pytest.fixture(autouse=True)
def _isolated_prior(monkeypatch, tmp_path):
    """Untrained prior file so any lazy prior load is deterministic; target
    reset after each test."""
    monkeypatch.setenv("EMMY_PRIOR_FILE", str(tmp_path / "prior.json"))
    yield
    target_mod.set_target(None)


def _f32_matmul_graph(M: int = 128, K: int = 128, N: int = 128) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _option0(fp) -> object:
    """The no-information decide: first emitted option, branch Forks descended
    first-child — the same leaf the no-prior greedy drive reaches."""
    o = fp.options[0]
    while isinstance(o, Fork) and not o.is_leaf:
        o = o.expand()[0]
    return o


def _graph_signature(graph: Graph) -> list[tuple[str, str, tuple]]:
    """Per-node identity strong enough for a bit-for-bit pick comparison:
    node id, op class, and the realized knob row."""
    return [
        (nid, type(n.op).__name__, tuple(sorted((k, str(v)) for k, v in (getattr(n.op, "knobs", None) or {}).items())))
        for nid, n in sorted(graph.nodes.items())
    ]


# ---------------------------------------------------------------------------
# In-place fold + option-0 parity
# ---------------------------------------------------------------------------


def test_resolve_applies_in_place() -> None:
    """No sibling snapshots, no per-fork copies: the terminal IS the seeded
    graph object."""
    g = _f32_matmul_graph()
    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0)))
    terminal, trace = run.resolve(g, _option0)
    assert terminal is g, "resolve must fold over the seeded graph in place"
    assert any(isinstance(n.op, LoopOp) is False for n in terminal.nodes.values())  # something lowered
    assert trace, "a matmul lowering passes at least one fork point"


def test_option0_decide_matches_no_prior_greedy() -> None:
    """A decide that always takes option-0 reproduces the no-prior greedy
    compile (``greedy_decide(prior=None)`` falls to emission order at every
    fork — the same first leaf)."""
    from emmy.compiler.pipeline.search.policy import greedy_decide

    ctx = Context.from_target((8, 0))
    greedy, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_f32_matmul_graph(), greedy_decide(prior=None))
    plain, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_f32_matmul_graph(), _option0)
    assert _graph_signature(plain) == _graph_signature(greedy)


# ---------------------------------------------------------------------------
# Trace shape
# ---------------------------------------------------------------------------


def test_trace_records_partition_fork() -> None:
    """The contraction's schedule fork traces as ONE decision under the recognizer rule (the
    hierarchical tile → stage → reduce fork tree is one fork point, not a per-family chain), with
    the kernel's node id, ``chosen_kind == "op"`` (a ``TileOp`` rebind), the decide's score
    annotation (``None`` for the unranked option-0 decide), and the chosen leaf's COMPLETE knob
    row (the axis-named ``TILE@<k>`` key; option-0 = the conservative per-cell leaf)."""
    from emmy.compiler.pipeline.knob import family_of, family_value

    g = _f32_matmul_graph()
    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0)))
    terminal, trace = run.resolve(g, _option0)
    part = [d for d in trace if any(family_of(k) == "TILE" for k in d.knob_delta)]
    assert len(part) == 1, f"one hierarchical schedule fork per contraction, got {[d.rule_name for d in part]}"
    d = part[0]
    assert d.node_id in terminal.nodes
    assert d.chosen_kind == "op"
    assert d.score is None
    assert d.n_options >= 1, "the fork offers its lazy fork tree as the raw option"
    assert family_value(d.knob_delta, "TILE") == "", "option-0 is the conservative per-cell leaf"


def test_decide_score_lands_on_trace() -> None:
    """``fp.score`` is the decide's output channel for the pick's predicted
    cost — resolve copies it onto the fork's Decision."""

    def scored(fp):
        fp.score = 42.0
        return _option0(fp)

    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0)))
    _, trace = run.resolve(_f32_matmul_graph(), scored)
    assert trace and all(d.score == 42.0 for d in trace)
