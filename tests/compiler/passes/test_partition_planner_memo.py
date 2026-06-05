"""Tests for the partition planner's lazy Fork-tree build + structural
features (``010_partition_loops`` / ``fork.build_fork_tree`` /
``992_stamp_structural_features``).

The lazy tree builds branch Forks only along the explored path; the structural
``S_*`` features are the variant identity the score cache keys on. All
assertions are call-count / identity based — no wall-time flakiness.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import fork as fork_mod
from deplodock.compiler.pipeline.fork import Fork, Level, build_fork_tree
from deplodock.compiler.pipeline.knob import STRUCT_PREFIX
from deplodock.compiler.tensor import Tensor

_planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")
BR, BM, BN, FM, FN = (_planner.BR, _planner.BM, _planner.BN, _planner.FM, _planner.FN)
# ``structure_features`` now lives in the stamp pass (loaded under a bare stem).
_stamp = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.992_stamp_structural_features")
structure_features = _stamp.structure_features


@pytest.fixture(autouse=True)
def _no_stray_pins(monkeypatch):
    """Plan cache keys and enumeration counts are pin-sensitive — clear every
    ``DEPLODOCK_<KNOB>`` (alias spellings included) a previous test in the
    same worker may have leaked."""
    from deplodock.compiler.pipeline.passes.lowering.tile import _enumeration  # noqa: PLC0415

    for knob in _enumeration._PLANNER_KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
        for alias in knob.aliases:
            monkeypatch.delenv(f"DEPLODOCK_{alias}", raising=False)


def _ctx() -> Context:
    return Context(compute_capability="sm_80")


def _loop_op_matmul(*, a: str = "a", b: str = "b", o: str = "o", i: str = "i", j: str = "j", k: str = "k") -> LoopOp:
    """Plain matmul LoopOp with parameterized buffer / axis names."""
    i_ax, j_ax, k_ax = Axis(i, 128), Axis(j, 128), Axis(k, 64)
    return LoopOp(
        body=(
            Loop(
                axis=i_ax,
                body=(
                    Loop(
                        axis=j_ax,
                        body=(
                            Loop(
                                axis=k_ax,
                                body=(
                                    Load(name=f"{a}_v", input=a, index=(Var(i), Var(k))),
                                    Load(name=f"{b}_v", input=b, index=(Var(k), Var(j))),
                                    Assign(name="prod", op=ElementwiseImpl("multiply"), args=(f"{a}_v", f"{b}_v")),
                                    Accum(name="acc", value="prod", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Write(output=o, index=(Var(i), Var(j)), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _graph_with_dtypes(dtype_a: str, dtype_b: str, *, a: str = "a", b: str = "b") -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor(a, (128, 64), dtype_a), node_id=a)
    g.add_node(InputOp(), [], Tensor(b, (64, 128), dtype_b), node_id=b)
    return g


def _stamped(loop_op: LoopOp, graph: Graph | None = None) -> LoopOp:
    """Stamp the ``S_*`` structural features onto a hand-built LoopOp — the job
    ``992_stamp_structural_features`` does in the real pipeline. Ad-hoc callers
    of ``_plan_kernel`` must do this themselves (there is no fallback)."""
    from dataclasses import replace  # noqa: PLC0415

    return replace(loop_op, knobs={**loop_op.knobs, **structure_features(loop_op.body, graph)})


def _build_tree(plan, ctx: Context) -> Fork:
    """The planner's canonical scalar level layout (mirrors ``rewrite()``)."""
    return build_fork_tree(
        params=plan.params,
        levels=[
            Level((_planner.MMA.name,), lambda p: (p["MMA"],) if "MMA" in p else ()),
            Level((BR.name,), lambda p: (p.get("BR", 1),)),
            Level((BM.name, BN.name), lambda p: (p.get("BM", 1), p.get("BN", 1))),
            Level((_planner.WM.name, _planner.WN.name), lambda p: (p.get("WM", 1), p.get("WN", 1))),
            Level((FM.name, FN.name), lambda p: (p["FM"], p["FN"])),
        ],
        materialize=lambda p: _planner._materialize(plan, p),
        score=lambda p, cache: _planner._score_variant(plan, p, ctx, cache),
    )


def test_dtype_signature_separates_structural_features():
    """Same body structure, different operand dtypes → different structural
    features (the ``S_dtype_*`` multiset gates the fp16 half2 window)."""
    feats_f16 = structure_features(_loop_op_matmul().body, _graph_with_dtypes("f16", "f16"))
    feats_f32 = structure_features(_loop_op_matmul().body, _graph_with_dtypes("f32", "f32"))
    feats_none = structure_features(_loop_op_matmul().body, None)
    assert feats_f16 != feats_f32
    assert feats_f16 != feats_none and feats_f32 != feats_none
    # Same structure + same dtypes → same features even across distinct op
    # objects with different SSA / buffer / axis names (counts are name-invariant).
    other = _loop_op_matmul(a="w", b="x", o="y", i="i2", j="j2", k="k2")
    assert structure_features(other.body, _graph_with_dtypes("f16", "f16", a="w", b="x")) == feats_f16


def test_lazy_tree_builds_only_expanded_path(monkeypatch):
    """Walking only one root→leaf path must instantiate O(path · level
    fanout) Forks, not one per param — and each branch's lazily computed
    score must equal the max over its expanded children (exactness vs the
    eager build)."""
    created = {"n": 0}

    def _counting(cls):
        class Counting(cls):
            def __init__(self, *args, **kwargs):
                created["n"] += 1
                super().__init__(*args, **kwargs)

        return Counting

    monkeypatch.setattr(fork_mod, "_Branch", _counting(fork_mod._Branch))
    monkeypatch.setattr(fork_mod, "_Leaf", _counting(fork_mod._Leaf))
    ctx = _ctx()
    plan = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert plan is not None and len(plan.params) > 8
    tree = _build_tree(plan, ctx)

    node = tree
    path: list[Fork] = [node]
    while not node.is_leaf:
        children = node.expand()
        assert node.score() == pytest.approx(max(c.score() for c in children)), "lazy branch score != max over expanded children"
        node = children[0]
        path.append(node)
    assert created["n"] < len(plan.params), f"lazy tree created {created['n']} Forks for a single-path walk over {len(plan.params)} params"


def test_score_variant_matches_lazy_score():
    """``_score_variant`` is a thin wrapper over ``TileOp.lazy_score`` — the
    two must agree for every variant when fed the same stamped knob dict."""
    ctx = _ctx()
    plan = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert plan is not None
    for p in plan.params:
        lazy = TileOp.lazy_score(ctx, knobs={**plan.base_knobs, **p}, shapes=plan.shape)
        assert _planner._score_variant(plan, p, ctx) == pytest.approx(lazy)


def test_structural_features_stamped_by_last_loop_pass():
    """``loop/fusion/992_stamp_structural_features`` runs last in the loop
    dialect: every fused LoopOp leaves the loop passes carrying its ``S_*``
    structural features equal to :func:`structure_features` — identity settles
    with the final fused body, before any tune-DB keying or tile-stage scoring
    sees the op."""
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
    from deplodock.compiler.pipeline.search.db import SearchDB

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (64, 128)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (128, 32)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (64, 32)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    fused = Pipeline.build(LOOP_PASSES).run(g, ctx=_ctx(), db=SearchDB())
    loops = [n.op for n in fused.nodes.values() if isinstance(n.op, LoopOp)]
    assert loops, "fusion should leave at least one LoopOp"
    for op in loops:
        struct = {k: v for k, v in op.knobs.items() if k.startswith(STRUCT_PREFIX)}
        assert struct == structure_features(op.body, fused)


def test_scores_shared_across_identical_loop_ops(monkeypatch):
    """The search-owned value-keyed score cache (the planner keys each
    variant by ``(ctx, merged knobs)`` — complete thanks to the ``S_*``
    structural-feature knobs) makes structurally identical LoopOps — different
    SSA / buffer / axis names — share every score: scoring the second kernel's tree
    through the SAME search computes ZERO new scores. (Expansion alone
    never scores — ranking is search policy, so the read happens via
    ``Search.score_of``.)"""
    from deplodock.compiler.pipeline.search.policy import GreedySearch

    calls = {"n": 0}
    real = TileOp.lazy_score

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(TileOp, "lazy_score", staticmethod(counting))
    ctx = _ctx()
    plan_1 = _planner._plan_kernel(_stamped(_loop_op_matmul()), ctx, kernel_name="k_l0")
    plan_2 = _planner._plan_kernel(_stamped(_loop_op_matmul(a="w", b="x", o="y", i="i2", j="j2", k="k2")), ctx, kernel_name="k_l1")
    assert plan_1 is not None and plan_2 is not None
    struct_1 = {k: v for k, v in plan_1.base_knobs.items() if k.startswith("S_")}
    struct_2 = {k: v for k, v in plan_2.base_knobs.items() if k.startswith("S_")}
    assert struct_1 and struct_1 == struct_2, "structurally identical ops must share structural features"
    search = GreedySearch()
    tree_1 = _build_tree(plan_1, ctx)
    assert calls["n"] == 0, "building a tree must not score"
    tree_1.expand()
    assert calls["n"] == 0, "expansion must not score — ranking is the search's job"
    search.score_of(tree_1)  # root prior = max over every param
    n_first = calls["n"]
    assert n_first == len(plan_1.params)
    search.score_of(_build_tree(plan_2, ctx))
    assert calls["n"] == n_first, "second kernel's tree re-scored rows instead of hitting the search's value-keyed cache"
