"""Tests for the partition planner's structural enumeration memo + the lazy
Fork-tree build (``010_partition_loops._ENUM_MEMO`` / ``fork_tree.build_fork_tree``).

A whole-model graph repeats the same kernel shapes once per transformer layer;
the memo shares the enumerated ``TileParams`` and their scores across
structurally identical LoopOps, and the lazy tree builds branch Forks only
along the explored path. All assertions are call-count / identity based —
no wall-time flakiness.
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
from deplodock.compiler.pipeline import fork_tree
from deplodock.compiler.pipeline.fork_tree import Level, build_fork_tree
from deplodock.compiler.pipeline.pipeline import Fork
from deplodock.compiler.tensor import Tensor

_planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")
BR, BM, BN, FM, FN, BK, SPLITK = (_planner.BR, _planner.BM, _planner.BN, _planner.FM, _planner.FN, _planner.BK, _planner.SPLITK)


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    """Fresh memo + no stray planner pins: enumeration counts and memo keys
    are pin-sensitive, so clear every ``DEPLODOCK_<KNOB>`` a previous test in
    the same worker may have leaked."""
    from deplodock.compiler.pipeline.passes.lowering.tile import _enumeration  # noqa: PLC0415

    for knob in _enumeration._PLANNER_KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
        for alias in knob.aliases:
            monkeypatch.delenv(f"DEPLODOCK_{alias}", raising=False)
    _planner._reset_enum_memo()
    yield
    _planner._reset_enum_memo()


def _ctx() -> Context:
    return Context(compute_capability="sm_80")


def _loop_op_matmul(*, a: str = "a", b: str = "b", o: str = "o", i: str = "i", j: str = "j", k: str = "k") -> LoopOp:
    """Plain matmul LoopOp with parameterized buffer / axis names — two calls
    with different names are structurally identical (``Body.structural_key``
    canonicalizes SSA, buffer, and free-axis names away)."""
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


def _build_tree(plan, ctx: Context) -> Fork | list[Fork]:
    """The planner's canonical scalar level layout (mirrors ``rewrite()``)."""
    return build_fork_tree(
        params=plan.params,
        levels=[
            Level((BR.name,), lambda p: (p.br,)),
            Level((BM.name, BN.name), lambda p: (p.bm, p.bn)),
            Level((FM.name, FN.name), lambda p: (p.fm, p.fn)),
            Level((BK.name, SPLITK.name), lambda p: (p.bk, p.splitk)),
        ],
        materialize=lambda p: _planner._materialize(plan, p),
        score=lambda p: _planner._score_variant(plan, p, ctx),
    )


def test_enumeration_memoized_across_identical_loop_ops(monkeypatch):
    """Two structurally identical LoopOps (different SSA / buffer / axis
    names) must share one ``enumerate_cartesian`` run — the second
    ``_plan_kernel`` is a memo hit."""
    calls = {"n": 0}
    real = _planner.enumerate_cartesian

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_planner, "enumerate_cartesian", counting)
    ctx = _ctx()
    plan_1 = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    n_after_first = calls["n"]
    plan_2 = _planner._plan_kernel(_loop_op_matmul(a="w", b="x", o="y", i="i2", j="j2", k="k2"), ctx, kernel_name="k_l1")
    assert plan_1 is not None and plan_2 is not None
    assert n_after_first >= 1
    assert calls["n"] == n_after_first, "second structurally identical LoopOp re-ran enumerate_cartesian"
    # The SAME params tuple object is shared — the id()-keyed score cache
    # relies on this (see _Plan docstring).
    assert plan_1.params is plan_2.params


def test_scores_shared_across_identical_loop_ops(monkeypatch):
    """Building the Fork tree for two structurally identical plans must
    compute each param's lazy score exactly once in total: the second
    tree's ``_score_variant`` calls hit the shared ``score_cache``."""
    calls = {"n": 0}
    real = TileOp.lazy_score

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(TileOp, "lazy_score", staticmethod(counting))
    ctx = _ctx()
    plan_1 = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    plan_2 = _planner._plan_kernel(_loop_op_matmul(a="w", b="x", o="y", i="i2", j="j2", k="k2"), ctx, kernel_name="k_l1")
    _build_tree(plan_1, ctx)
    assert calls["n"] == len(plan_1.params)
    _build_tree(plan_2, ctx)
    assert calls["n"] == len(plan_1.params), "second tree re-scored params instead of hitting the shared score_cache"


def test_dtype_signature_separates_memo_entries():
    """Same body structure, different operand dtypes → different memo keys
    (dtypes gate the fp16 half2 window enumeration)."""
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    key_f16 = _planner._enum_cache_key(loop_op, ctx, _graph_with_dtypes("f16", "f16"))
    key_f32 = _planner._enum_cache_key(loop_op, ctx, _graph_with_dtypes("f32", "f32"))
    key_none = _planner._enum_cache_key(loop_op, ctx, None)
    assert key_f16 != key_f32
    assert key_f16 != key_none and key_f32 != key_none


def test_plan_rides_op_metadata(monkeypatch):
    """Re-planning the SAME LoopOp object is an op-metadata hit — the
    stamped ``_Plan`` comes back without re-running classification or
    enumeration — and a pin flip invalidates the stamp (the stored memo
    key no longer matches)."""
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    plan_1 = _planner._plan_kernel(loop_op, ctx, kernel_name="k_l0")
    assert plan_1 is not None
    assert loop_op.meta["plan"][1] is plan_1
    plan_2 = _planner._plan_kernel(loop_op, ctx, kernel_name="k_l0")
    assert plan_2 is plan_1, "same op object must return the stamped plan"
    monkeypatch.setenv("DEPLODOCK_BK", "16")
    plan_3 = _planner._plan_kernel(loop_op, ctx, kernel_name="k_l0")
    assert plan_3 is not plan_1, "a pin flip must invalidate the op-metadata stamp"
    assert all(p.bk == 16 for p in plan_3.params)


def test_mma_pin_lands_in_memo_key(monkeypatch):
    """``MMA`` is a planner knob, so the pin snapshot covers it: flipping
    ``DEPLODOCK_MMA`` must produce a fresh memo key (it gates the warp-tier
    enumeration)."""
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    key_default = _planner._enum_cache_key(loop_op, ctx, None)
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    key_off = _planner._enum_cache_key(loop_op, ctx, None)
    assert key_default != key_off


def test_pin_snapshot_invalidates_memo(monkeypatch):
    """A ``DEPLODOCK_<KNOB>`` pin flipped mid-process must land on a fresh
    memo key — ``enumerate_cartesian`` folds pins live via ``Knob.narrow``."""
    calls = {"n": 0}
    real = _planner.enumerate_cartesian

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_planner, "enumerate_cartesian", counting)
    ctx = _ctx()
    _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    n_unpinned = calls["n"]
    monkeypatch.setenv("DEPLODOCK_BK", "16")
    plan_pinned = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert calls["n"] > n_unpinned, "pinned re-plan replayed the unpinned cached enumeration"
    assert plan_pinned is not None
    assert all(p.bk == 16 for p in plan_pinned.params)


def test_lazy_tree_builds_only_expanded_path(monkeypatch):
    """Walking only the option-0 path must instantiate O(path) Forks, not one
    per param — and each branch's lazily computed score must equal the max
    over its expanded children (exactness vs the eager build)."""
    created = {"n": 0}

    class CountingFork(Fork):
        def __init__(self, *args, **kwargs):
            created["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fork_tree, "Fork", CountingFork)
    ctx = _ctx()
    plan = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert plan is not None and len(plan.params) > 8
    tree = _build_tree(plan, ctx)

    node = tree if isinstance(tree, Fork) else tree[0]
    path: list[Fork] = [node]
    while not node.is_leaf:
        children = node.expand()
        assert node.score == pytest.approx(max(c.score for c in children)), "lazy branch score != max over expanded children"
        node = children[0]
        path.append(node)
    assert created["n"] < len(plan.params), f"lazy tree created {created['n']} Forks for a single-path walk over {len(plan.params)} params"


def test_score_variant_parity_with_uncached_lazy_score():
    """The precomputed score inputs (``score_n_staged`` /
    ``score_write_inner_fv``) must not change the score: ``_score_variant``
    equals a from-scratch ``TileOp.lazy_score`` for every variant."""
    ctx = _ctx()
    plan = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert plan is not None
    for p in plan.params:
        assert _planner._score_variant(plan, p, ctx) == pytest.approx(TileOp.lazy_score(ctx, shapes=plan.shape, params=p))
