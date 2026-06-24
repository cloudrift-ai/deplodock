"""``Run.resolve`` — the deterministic-resolution entry point (one live graph,
a ``decide`` callback per fork, a ``Decision`` trace as the only process-state
output; see ``plans/resolve-trace-driver.md`` M1).

Pins the M1 contract: in-place apply (the terminal IS the seeded graph object,
no per-fork copies), an option-0 ``decide`` reproducing the no-prior greedy
compile bit-for-bit, the trace shape on a forked compile, and the
structural-decision replay being consulted (identical offer sites decide once).
No GPU: everything terminates in the tile dialect or earlier.
"""

from __future__ import annotations

import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.pipeline import Run, _is_structural_option


@pytest.fixture(autouse=True)
def _isolated_prior(monkeypatch, tmp_path):
    """Untrained prior file so any lazy prior load is deterministic; target
    reset after each test."""
    monkeypatch.setenv("DEPLODOCK_PRIOR_FILE", str(tmp_path / "prior.json"))
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


def _norm_linear(prefix: str, g: Graph | None = None) -> Graph:
    """RMSNorm → Linear (f16): fusion yields the prologue-demoted matmul whose
    keep-vs-split offer (``tile/005_split_demoted``) is a structural fork."""
    f16 = _dt.get("f16")
    g = g if g is not None else Graph()
    x, nw, wg, xn, o = (f"{prefix}{n}" for n in ("x", "nw", "wg", "xn", "o"))
    g.add_node(InputOp(), [], Tensor(x, (1, 32, 1024), f16), node_id=x)
    g.add_node(InputOp(), [], Tensor(nw, (1024,), f16), node_id=nw)
    g.add_node(InputOp(), [], Tensor(wg, (3072, 1024), f16), node_id=wg)
    g.add_node(RmsNormOp(eps=1e-6), [x, nw], Tensor(xn, (1, 32, 1024), f16), node_id=xn)
    g.add_node(LinearOp(), [xn, wg], Tensor(o, (1, 32, 3072), f16), node_id=o)
    g.inputs += [x, nw, wg]
    g.outputs += [o]
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
    from deplodock.compiler.pipeline.search.policy import greedy_decide

    ctx = Context.from_target((8, 0))
    greedy, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_f32_matmul_graph(), greedy_decide(prior=None))
    plain, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_f32_matmul_graph(), _option0)
    assert _graph_signature(plain) == _graph_signature(greedy)


# ---------------------------------------------------------------------------
# Trace shape
# ---------------------------------------------------------------------------


def test_trace_records_partition_fork() -> None:
    """The inner partition forks trace under their per-family rule names with the
    kernel's node id and the decide's score annotation (None for the unranked
    option-0 decide); ``chosen_kind`` is ``"op"`` for tile rebinds. The block-DAG
    Tile IR splits the old monolithic ``010_enumerate`` into the per-family
    chain (``010_reduce_tile`` → ``020_thread_tile`` → ``030_register_tile`` →
    ``050_stage``), so one kernel records that chain rather than one fork, and
    the cumulative knob row carries the complete tile (``{BM, BN}``) by the
    thread-tile fork."""
    g = _f32_matmul_graph()
    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0)))
    terminal, trace = run.resolve(g, _option0)
    part = [d for d in trace if d.rule_name in {"010_reduce_tile", "020_thread_tile", "030_register_tile"}]
    assert [d.rule_name for d in part] == ["010_reduce_tile", "020_thread_tile", "030_register_tile"], (
        f"the inner tile-fork chain for one kernel, got {[d.rule_name for d in trace]}"
    )
    for d in part:
        assert d.node_id in terminal.nodes
        assert d.chosen_kind == "op"
        assert d.score is None
        assert d.n_options >= 1, "each family fork emits its lazy fork tree as the raw option"
    thread = next(d for d in part if d.rule_name == "020_thread_tile")
    assert {"BM", "BN"} <= set(thread.knob_delta), f"the thread-tile decision carries the complete free tile row, got {thread.knob_delta}"


def test_decide_score_lands_on_trace() -> None:
    """``fp.score`` is the decide's output channel for the pick's predicted
    cost — resolve copies it onto the fork's Decision."""

    def scored(fp):
        fp.score = 42.0
        return _option0(fp)

    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0)))
    _, trace = run.resolve(_f32_matmul_graph(), scored)
    assert trace and all(d.score == 42.0 for d in trace)


# ---------------------------------------------------------------------------
# Structural forks: decide sees them, replay is consulted
# ---------------------------------------------------------------------------


def test_structural_replay_consulted() -> None:
    """Two structurally identical offer sites decide ONCE: the second 005 offer
    replays the first decision read off the graph (``Op.source`` + stamped
    decision knobs), so the decide callback sees one structural fork and the
    terminal carries both splits. Replays are not decisions — no trace entry."""
    from deplodock.compiler.pipeline.search.two_level import outer_pipeline

    g = _norm_linear("b", _norm_linear("a"))
    seen: list[str] = []

    def split_first(fp):
        seen.append(fp.match.rule.name)
        structural = [o for o in fp.options if _is_structural_option(o)]
        if structural:
            assert fp.structural
            return structural[0]
        return _option0(fp)

    terminal, trace = Run(pipeline=outer_pipeline(), ctx=Context.from_target((12, 0))).resolve(g, split_first)
    assert seen.count("005_split_demoted") == 1, f"the second offer site must replay, decide saw {seen}"
    assert sum(1 for d in trace if d.rule_name == "005_split_demoted") == 1
    assert sum(1 for n in terminal.nodes.values() if isinstance(n.op, LoopOp)) == 4, "both sites must take the split side"
    split = next(d for d in trace if d.rule_name == "005_split_demoted")
    assert split.chosen_kind == "graph"
    assert split.knob_delta.get("CUT") == "1"
