"""Tests for the hierarchical Fork tree emitted by ``010_partition_loops``.

The planner emits one ``MMA → BR → (BM,BN) → (WM,WN) → (FM,FN) →
(BK,SPLITK) → TileOp leaf`` Fork tree from a ``_Plan`` (cheap-to-build
pre-materialization state); tier-foreign levels collapse (single value).
Levels with a single value collapse (no Fork wrapper). Branch scores
propagate ``max`` from leaves so the search picks high-Q subtrees first.
Leaf ``expand`` thunks materialize the chosen TileOp on demand —
:func:`_materialize` runs ``_build_split_body`` + body normalization
only when the leaf is actually resolved.

We exercise the builder directly via importlib (the module name has a
``000_`` numeric prefix and isn't importable with normal syntax) on
real planner-emitted plans rather than synthetic stubs — that way we
catch knob-name drift and score-formula shape changes.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork, Level, build_fork_tree

_planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")
_plan_kernel = _planner._plan_kernel
_materialize = _planner._materialize
_score_variant = _planner._score_variant
BR, BM, BN, FM, FN, BK, SPLITK = (_planner.BR, _planner.BM, _planner.BN, _planner.FM, _planner.FN, _planner.BK, _planner.SPLITK)


def _build_fork_tree_lazy(plan, ctx: Context) -> Fork | list[Fork]:
    """Mirror the planner's own call site in ``rewrite()`` — builds the same
    Fork tree from a ``_Plan`` + ``ctx`` via the shared
    ``pipeline/fork.py`` builder, with the planner's canonical level
    layout. Single source of truth for the test assertions below."""
    return build_fork_tree(
        params=plan.params,
        levels=[
            Level((_planner.MMA.name,), lambda p: (p["MMA"],) if "MMA" in p else ()),
            Level((BR.name,), lambda p: (p.get("BR", 1),)),
            Level((BM.name, BN.name), lambda p: (p.get("BM", 1), p.get("BN", 1))),
            Level((_planner.WM.name, _planner.WN.name), lambda p: (p.get("WM", 1), p.get("WN", 1))),
            Level((FM.name, FN.name), lambda p: (p["FM"], p["FN"])),
            Level((BK.name, SPLITK.name), lambda p: (p["BK"], p["SPLITK"])),
        ],
        materialize=lambda p: _materialize(plan, p),
        score=lambda p, cache: _score_variant(plan, p, ctx, cache),
    )


def _ctx() -> Context:
    """Pinned sm_80 context — keeps variant counts stable across CI hosts."""
    return Context(compute_capability="sm_80")


def _loop_op_matmul(*, m: int = 128, n: int = 128, k: int = 64) -> LoopOp:
    """Synthetic plain-matmul LoopOp meeting ``is_matmul_reduce``:
    reduce body has ≥2 K-indexed Loads from distinct buffers and at
    least one Accum. Exercises the planner's matmul branch (where
    SPLITK > 1 variants are emitted)."""
    i_ax, j_ax, k_ax = Axis("i", m), Axis("j", n), Axis("k", k)
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
                                    # ``is_reduce`` is a derived property: True iff the body
                                    # contains an Accum — so just including Accum is enough.
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="prod", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="prod", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Write(output="o", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _loop_op_pointwise(*, m: int = 4, n: int = 8) -> LoopOp:
    """Simple pointwise LoopOp (matches the pattern used in the existing
    planner-rules test)."""
    i_ax, j_ax = Axis("i", m), Axis("j", n)
    return LoopOp(
        body=(
            Loop(
                axis=i_ax,
                body=(
                    Loop(
                        axis=j_ax,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _matmul_plan():
    """Return the planner's ``_Plan`` for a plain matmul. Multi-variant by
    construction (matmul gets the full BN×BM×FM×FN×BK×SPLITK grid)."""
    plan = _plan_kernel(_loop_op_matmul(), _ctx(), kernel_name="k_matmul")
    assert plan is not None and len(plan.params) > 1, "matmul should yield multiple variants"
    return plan


def _pointwise_plan():
    """Planner's ``_Plan`` for a pointwise kernel — all BR=1."""
    plan = _plan_kernel(_loop_op_pointwise(), _ctx(), kernel_name="k_pointwise")
    assert plan is not None and len(plan.params) > 1, "pointwise should yield multiple variants"
    return plan


def _walk_leaves(tree: Fork | list[Fork]) -> list[Fork]:
    roots = [tree] if isinstance(tree, Fork) else list(tree)
    leaves: list[Fork] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.is_leaf:
            leaves.append(node)
            continue
        stack.extend(node.expand())
    return leaves


def _walk_branches(tree: Fork | list[Fork]) -> list[Fork]:
    roots = [tree] if isinstance(tree, Fork) else list(tree)
    branches: list[Fork] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.is_leaf:
            continue
        branches.append(node)
        stack.extend(node.expand())
    return branches


def test_single_variant_collapses_to_single_leaf_on_expand():
    """With one variant in, the lazy root expands straight to one leaf
    Fork — every level above the leaf collapses. (The planner's
    ``rewrite()`` short-circuits a 1-param plan to a bare ``_materialize``
    before ever building a tree; this pins the builder's own behavior.)"""
    plan = _matmul_plan()
    one_plan = _planner._Plan(
        shape=plan.shape,
        leading=plan.leading,
        base_knobs=plan.base_knobs,
        kernel_name=plan.kernel_name,
        params=plan.params[:1],
    )
    tree = _build_fork_tree_lazy(one_plan, _ctx())
    assert isinstance(tree, Fork)
    assert not tree.is_leaf and tree.knobs == {}
    (leaf,) = tree.expand()
    assert leaf.is_leaf


def test_multi_variant_matmul_emits_branch_forks():
    plan = _matmul_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    branches = _walk_branches(tree)
    assert branches, "expected branch Forks for a multi-variant matmul"


def test_leaves_preserve_every_variant():
    """Tree's leaf set must equal the planner's enumerated params (no dup/drop)."""
    plan = _matmul_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    leaves = _walk_leaves(tree)
    assert len(leaves) == len(plan.params), f"leaf count {len(leaves)} doesn't match variant count {len(plan.params)}"


def test_leaf_knobs_are_bk_splitk_only():
    """Leaf Forks pin only (BK, SPLITK) — the rest is committed by ancestor
    branch Forks. The DB lookup at fork-replay time matches deltas
    per-level, so the partition must be tight."""
    plan = _matmul_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    for leaf in _walk_leaves(tree):
        assert set(leaf.knobs.keys()) <= {"BK", "SPLITK"}, f"leaf knobs leak into non-(BK,SPLITK): {leaf.knobs}"


def test_branch_score_is_max_of_children():
    """Max-propagated scores: each branch Fork's score equals the max
    score among its direct children (branches or leaves)."""
    plan = _matmul_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    for branch in _walk_branches(tree):
        children = branch.expand()
        assert children
        expected = max(c.score() for c in children)
        assert branch.score() == pytest.approx(expected), f"branch.score {branch.score()} != max(child scores) {expected}"


def test_max_score_descent_reaches_best_variant():
    """Siblings are unranked (the search picks the max prior via
    ``Search.score_of``); descending by max score at every level must
    land on the leaf whose prior equals the global best over the whole
    enumeration — i.e. greedy's pick = argmax ``TileOp.lazy_score``.
    ``lazy_score`` is the ONLY scorer (there is no post-materialization
    ``Op.score``), so the picked leaf's fork prior is compared against
    a from-scratch argmax over every enumerated variant."""
    plan = _matmul_plan()
    ctx = _ctx()
    tree = _build_fork_tree_lazy(plan, ctx)

    node = tree if isinstance(tree, Fork) else max(tree, key=lambda f: f.score())
    while not node.is_leaf:
        node = max(node.expand(), key=lambda f: f.score())

    def _lazy(p):
        return TileOp.lazy_score(ctx, knobs={**plan.base_knobs, **p}, shapes=plan.shape)

    best = max(_lazy(p) for p in plan.params)
    assert node.score() == pytest.approx(best), f"max-descent leaf prior {node.score()} != best lazy_score {best}"


def test_pointwise_collapses_br_layer():
    """Pointwise kernels have BR=1 fixed, so the BR Fork layer must
    collapse — no branch Fork should ever pin BR for them."""
    plan = _pointwise_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    for branch in _walk_branches(tree):
        assert "BR" not in branch.knobs, f"BR leaked into branch for pointwise: {branch.knobs}"


def test_branch_knobs_partition_cleanly():
    """The union of (root-fork knobs, walked branch knobs, leaf knobs)
    along any root→leaf path must cover the planner's level knobs
    ``{MMA, BR, BM, BN, WM, WN, FM, FN, BK, SPLITK}`` at most once each —
    no knob duplicated across levels (tier-foreign levels collapse)."""
    plan = _matmul_plan()
    tree = _build_fork_tree_lazy(plan, _ctx())
    expected = {"MMA", "BR", "BM", "BN", "WM", "WN", "FM", "FN", "BK", "SPLITK"}

    # Walk one root→leaf path and check.
    def _first_path(node: Fork) -> list[Fork]:
        path = [node]
        while not node.is_leaf:
            node = node.expand()[0]
            path.append(node)
        return path

    roots = [tree] if isinstance(tree, Fork) else list(tree)
    for root in roots:
        path = _first_path(root)
        seen: dict[str, int] = {}
        for fork in path:
            for k in fork.knobs:
                seen[k] = seen.get(k, 0) + 1
        # Collapsed knobs (single-value across all variants) are absent from
        # the path's Forks but pinned by the leaf TileOp — accept missing
        # knobs that don't vary across variants. The check that matters:
        # no knob appears twice (would mean overlapping branch deltas).
        duplicates = {k: c for k, c in seen.items() if c > 1}
        assert not duplicates, f"knobs appear at multiple levels: {duplicates}"
        # Every pinned knob is one of the planner's expected set.
        assert set(seen) <= expected, f"unexpected knob keys in path: {set(seen) - expected}"
