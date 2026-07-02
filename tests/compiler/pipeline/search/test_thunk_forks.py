"""Thunk-bearing fork support: ``LazyCandidate.pending`` carries a ``Fork``,
the search loop expands it before resolving, and chains of Forks produce a
leaf op exactly once.

Tests build a small custom Pipeline with a single rule whose ``rewrite``
returns ``Fork`` options. Each Fork's ``expand`` either returns more Forks
(branch level) or a concrete ``Op`` (leaf level). The harness then drives
the pipeline through both single-shot `Pipeline.run` and the `TuningSearch` sweep
and asserts the cursor advances exactly once and the resolved leaf carries
the expected knob delta.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp, Op
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.pipeline import Pass, Pattern, Pipeline, Rule
from tests.compiler.conftest import drain_tune


@dataclass(frozen=True)
class ThunkFork(Fork):
    """Minimal branch-``Fork`` test stand-in (the production ``ThunkFork`` was removed with its
    last rule): ``expand_fn(knobs)`` produces the next level, so these engine tests can build
    arbitrary lazy fork shapes without the tree builder."""

    knobs: dict = field(default_factory=dict)
    expand_fn: Callable = lambda knobs: []
    is_leaf: bool = False

    def expand(self):
        return self.expand_fn(self.knobs)


# A tiny stub Op for testing. Carries an arbitrary ``knobs`` dict that the
# rule's rewrite stamps on emitted variants — that's all the test needs
# to check fork dispatch outcomes.
@dataclass(eq=False)
class _StubOp(Op):
    tag: str = "stub"
    knobs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def __class_getitem__(cls, _):  # for any generic usage in patterns
        return cls

    def __init__(self, tag: str = "stub", knobs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.tag = tag
        self.knobs = knobs or {}


def _make_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,), "f32"), node_id="x")
    g.add_node(op=_StubOp(tag="root"), inputs=["x"], output=Tensor("y", (4,), "f32"), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _build_rule(rewrite_fn) -> Rule:
    """Wrap ``rewrite_fn`` in a Rule matching the stub op at root."""
    rule = Rule(
        name="__test_thunk__",
        pattern=[Pattern(name="root", op_type=_StubOp)],
        rewrite=rewrite_fn,
        param_names=tuple(inspect.signature(rewrite_fn).parameters.keys()),
    )
    return rule


def _build_pipeline(rewrite_fn) -> Pipeline:
    rule = _build_rule(rewrite_fn)
    pass_ = Pass(name="__test__", rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


def _final_op(graph: Graph) -> _StubOp:
    """Pull the (single) StubOp at the 'y' node."""
    op = graph.nodes["y"].op
    assert isinstance(op, _StubOp), f"expected _StubOp, got {type(op).__name__}"
    return op


# ---------------------------------------------------------------------------


def test_single_level_fork_resolves_leaf() -> None:
    """One Fork → its expand returns one concrete Op leaf. Pipeline must
    dispatch on is_expandable, expand the Fork, then resolve the leaf."""

    def rewrite(root):  # noqa: ARG001
        leaf = _StubOp(tag="leaf", knobs={"L": 1})
        # Single-level fork: expand returns [leaf]
        return [ThunkFork(knobs={"L": 1}, expand_fn=lambda knobs: [leaf])]

    pipeline = _build_pipeline(rewrite)
    out = pipeline.run(_make_graph())
    op = _final_op(out)
    assert op.tag == "leaf"
    assert op.knobs.get("L") == 1


def test_two_level_fork_chain_resolves_leaf() -> None:
    """Outer Fork expands to inner Forks; each inner expands to a leaf.
    Verifies the search loop re-dispatches on is_expandable after each
    expansion until a concrete Op is reached."""

    def rewrite(root):  # noqa: ARG001
        def make_inner(outer_v: int):
            def leaf(knobs: dict, ov: int = outer_v) -> list[_StubOp]:
                return [_StubOp(tag=f"leaf-{ov}-{knobs['B']}", knobs={"A": ov, "B": knobs["B"]})]

            return [ThunkFork(knobs={"B": b}, expand_fn=leaf) for b in (10, 20)]

        return [ThunkFork(knobs={"A": a}, expand_fn=lambda knobs: make_inner(knobs["A"])) for a in (1, 2)]

    pipeline = _build_pipeline(rewrite)
    out = pipeline.run(_make_graph())
    op = _final_op(out)
    # Single-shot (uniform PUCT) picks option 0 at every fork → A=1, B=10.
    assert op.knobs.get("A") == 1
    assert op.knobs.get("B") == 10
    assert op.tag == "leaf-1-10"


def test_tuning_enumerates_thunk_leaves() -> None:
    """Under TuningSearch, every leaf of a 2-level thunk chain must
    materialize. With ``patience=10**6`` the search exhausts the tree."""
    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.search.db import SearchDB

    def rewrite(root):  # noqa: ARG001
        def make_inner(outer_v: int):
            def leaf(knobs: dict, ov: int = outer_v) -> list[_StubOp]:
                return [_StubOp(tag=f"leaf-{ov}-{knobs['B']}", knobs={"A": ov, "B": knobs["B"]})]

            return [ThunkFork(knobs={"B": b}, expand_fn=leaf) for b in (10, 20)]

        return [ThunkFork(knobs={"A": a}, expand_fn=lambda knobs: make_inner(knobs["A"])) for a in (1, 2)]

    pipeline = _build_pipeline(rewrite)
    search = TuningSearch(patience=10**6)
    leaves: set[tuple[int, int]] = set()
    for cand in drain_tune(pipeline, _make_graph(), search=search, db=SearchDB()):
        op = _final_op(cand.graph)
        leaves.add((op.knobs["A"], op.knobs["B"]))
    # 2 outer × 2 inner = 4 leaves; expect all to materialize.
    assert leaves == {(1, 10), (1, 20), (2, 10), (2, 20)}


def test_mixed_fork_and_concrete_op_siblings() -> None:
    """One rule batch returns a mix of Forks and concrete Ops. Single-shot
    picks option 0 (a Fork that expands to a leaf), but both branches
    must be valid for tune to enumerate."""
    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.search.db import SearchDB

    direct_leaf = _StubOp(tag="direct", knobs={"K": 99})

    def rewrite(root):  # noqa: ARG001
        # Option 0: Fork that expands to a leaf with K=1.
        # Option 1: concrete Op already (K=99).
        return [
            ThunkFork(knobs={"K": 1}, expand_fn=lambda knobs: [_StubOp(tag="via-fork", knobs=dict(knobs))]),
            direct_leaf,
        ]

    pipeline = _build_pipeline(rewrite)
    # Single-shot: option 0 → Fork → leaf K=1.
    out_greedy = pipeline.run(_make_graph())
    assert _final_op(out_greedy).knobs.get("K") == 1

    # Tuning: both K=1 (via-fork) and K=99 (direct) materialize.
    search = TuningSearch(patience=10**6)
    leaves: set[int] = set()
    for cand in drain_tune(pipeline, _make_graph(), search=search, db=SearchDB()):
        leaves.add(_final_op(cand.graph).knobs["K"])
    assert leaves == {1, 99}


def test_is_expandable_discriminates_fork_vs_op() -> None:
    """Unit check on LazyCandidate.is_expandable: True only when pending's
    option is a Fork."""
    from deplodock.compiler.pipeline.search.candidate import Candidate, LazyCandidate

    # Build a real Candidate and reach into LazyCandidate construction
    # to verify is_expandable() against three pending shapes.
    graph = _make_graph()
    pipeline = _build_pipeline(lambda root: None)  # noqa: ARG005
    # We don't need to actually run; just construct a Match-shaped tuple
    # via Pipeline.match to get a real Match.
    cur = pipeline.passes[0].rules[0]
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.pipeline import Run
    from deplodock.compiler.pipeline.search.db import SearchDB

    run = Run(pipeline=pipeline, ctx=Context.probe(), search=TuningSearch(), db=SearchDB())
    cand = Candidate(run=run, graph=graph, cursor=None)  # cursor unused for these calls

    matches = pipeline.match(graph, cur)
    assert matches, "pattern must match the stub node"
    match = matches[0]

    # Branch Fork → True
    branch = ThunkFork(knobs={"X": 1}, expand_fn=lambda knobs: [_StubOp()])
    lc_fork = LazyCandidate.from_option(inner=cand, cursor=cand.cursor, match=match, option=branch)
    assert lc_fork.is_expandable() is True

    # Op option → lifted into a leaf OptionFork → False (resolves directly)
    lc_op = LazyCandidate.from_option(inner=cand, cursor=cand.cursor, match=match, option=_StubOp(tag="leaf"))
    assert lc_op.is_expandable() is False

    # None pending → False
    lc_none = LazyCandidate(inner=cand, cursor=cand.cursor, pending=None)
    assert lc_none.is_expandable() is False
