"""Tests for the partition planner's op-metadata plan cache + the lazy
Fork-tree build (``010_partition_loops`` / ``fork_tree.build_fork_tree``).

The finished ``_Plan`` rides its LoopOp as op metadata, keyed by
``_plan_cache_key`` so pin / ctx / dtype changes invalidate the stamp; the
lazy tree builds branch Forks only along the explored path. All assertions
are call-count / identity based — no wall-time flakiness.
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


def test_plan_rides_op_metadata(monkeypatch):
    """Re-planning the SAME LoopOp object is an op-metadata hit — the
    stamped ``_Plan`` comes back without re-running classification or
    enumeration — and a pin flip invalidates the stamp (the stored cache
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


def test_op_metadata_hit_skips_enumeration(monkeypatch):
    """The stamp hit must not re-run ``enumerate_cartesian``."""
    calls = {"n": 0}
    real = _planner.enumerate_cartesian

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_planner, "enumerate_cartesian", counting)
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    _planner._plan_kernel(loop_op, ctx, kernel_name="k_l0")
    n_first = calls["n"]
    assert n_first >= 1
    _planner._plan_kernel(loop_op, ctx, kernel_name="k_l0")
    assert calls["n"] == n_first, "op-metadata hit re-ran enumerate_cartesian"


def test_dtype_signature_separates_cache_keys():
    """Same body structure, different operand dtypes → different plan cache
    keys (dtypes gate the fp16 half2 window enumeration)."""
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    key_f16 = _planner._plan_cache_key(loop_op, ctx, _graph_with_dtypes("f16", "f16"))
    key_f32 = _planner._plan_cache_key(loop_op, ctx, _graph_with_dtypes("f32", "f32"))
    key_none = _planner._plan_cache_key(loop_op, ctx, None)
    assert key_f16 != key_f32
    assert key_f16 != key_none and key_f32 != key_none


def test_mma_pin_lands_in_cache_key(monkeypatch):
    """``MMA`` is a planner knob, so the pin snapshot covers it: flipping
    ``DEPLODOCK_MMA`` must produce a fresh plan cache key (it gates the
    warp-tier enumeration)."""
    ctx = _ctx()
    loop_op = _loop_op_matmul()
    key_default = _planner._plan_cache_key(loop_op, ctx, None)
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    key_off = _planner._plan_cache_key(loop_op, ctx, None)
    assert key_default != key_off


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
    assert created["n"] < len(plan.params), (
        f"lazy tree created {created['n']} Forks for a single-path walk over {len(plan.params)} params"
    )


def test_score_variant_matches_lazy_score():
    """``_score_variant`` is a thin wrapper over ``TileOp.lazy_score`` — the
    two must agree for every variant."""
    ctx = _ctx()
    plan = _planner._plan_kernel(_loop_op_matmul(), ctx, kernel_name="k_l0")
    assert plan is not None
    for p in plan.params:
        assert _planner._score_variant(plan, p, ctx) == pytest.approx(TileOp.lazy_score(ctx, shapes=plan.shape, params=p))
