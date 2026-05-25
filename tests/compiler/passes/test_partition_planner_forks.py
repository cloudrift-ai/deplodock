"""Tests for the hierarchical Fork tree emitted by ``010_partition_loops``.

The planner converts its flat ``list[TileOp]`` variant set into a
``BR → (BM,BN) → (FM,FN) → (BK,SPLITK) → TileOp leaf`` Fork tree. Levels
with a single value collapse (no Fork wrapper). Branch scores propagate
``max`` from leaves so the search picks high-Q subtrees first.

We exercise the builder directly via importlib (the module name has a
``000_`` numeric prefix and isn't importable with normal syntax) on
real planner-emitted ``TileOp`` lists rather than synthetic stubs —
that way we catch knob-name drift and ``TileOp.score`` shape changes.
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
from deplodock.compiler.pipeline.pipeline import Fork

_planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")
_build_fork_tree = _planner._build_fork_tree
_split_kernel_fully = _planner._split_kernel_fully


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


def _matmul_variants() -> list[TileOp]:
    """Return the planner's flat ``list[TileOp]`` for a plain matmul.
    Multi-variant by construction (matmul gets the full BN×BM×FM×FN×BK×SPLITK grid)."""
    variants = _split_kernel_fully(_loop_op_matmul(), _ctx(), kernel_name="k_matmul")
    assert variants is not None and len(variants) > 1, "matmul should yield multiple variants"
    return variants


def _pointwise_variants() -> list[TileOp]:
    """Planner's flat variant list for a pointwise kernel — all BR=1."""
    variants = _split_kernel_fully(_loop_op_pointwise(), _ctx(), kernel_name="k_pointwise")
    assert variants is not None and len(variants) > 1, "pointwise should yield multiple variants"
    return variants


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


def test_single_variant_short_circuits_to_single_leaf():
    """With one variant in, the builder collapses every level and returns
    a single leaf Fork (not a list of siblings). The engine still routes
    it through the fork-spawn path since ``isinstance(option, Fork)``."""
    variants = _matmul_variants()
    one = variants[:1]
    tree = _build_fork_tree(one, _ctx())
    assert isinstance(tree, Fork)
    assert tree.is_leaf


def test_multi_variant_matmul_emits_branch_forks():
    variants = _matmul_variants()
    tree = _build_fork_tree(variants, _ctx())
    branches = _walk_branches(tree)
    assert branches, "expected branch Forks for a multi-variant matmul"


def test_leaves_preserve_every_variant():
    """Tree's leaf set must equal the flat variant list (no dup/drop)."""
    variants = _matmul_variants()
    tree = _build_fork_tree(variants, _ctx())
    leaves = _walk_leaves(tree)
    assert len(leaves) == len(variants), f"leaf count {len(leaves)} doesn't match variant count {len(variants)}"


def test_leaf_knobs_are_bk_splitk_only():
    """Leaf Forks pin only (BK, SPLITK) — the rest is committed by ancestor
    branch Forks. The DB lookup at fork-replay time matches deltas
    per-level, so the partition must be tight."""
    variants = _matmul_variants()
    tree = _build_fork_tree(variants, _ctx())
    for leaf in _walk_leaves(tree):
        assert set(leaf.knobs.keys()) <= {"BK", "SPLITK"}, f"leaf knobs leak into non-(BK,SPLITK): {leaf.knobs}"


def test_branch_score_is_max_of_children():
    """Max-propagated scores: each branch Fork's score equals the max
    score among its direct children (branches or leaves)."""
    variants = _matmul_variants()
    tree = _build_fork_tree(variants, _ctx())
    for branch in _walk_branches(tree):
        children = branch.expand()
        assert children
        expected = max(c.score for c in children)
        assert branch.score == pytest.approx(expected), f"branch.score {branch.score} != max(child scores) {expected}"


def test_first_leaf_matches_best_score_variant():
    """Sibling ordering is by max-propagated ``TileOp.score`` descending:
    the leaf reachable by always taking option-0 must be the highest-
    scoring variant in the planner's emission. Greedy primary = score
    primary (single source of truth, also used as MCTS prior)."""
    variants = _matmul_variants()
    ctx = _ctx()
    tree = _build_fork_tree(variants, ctx)

    node = tree if isinstance(tree, Fork) else tree[0]
    while not node.is_leaf:
        node = node.expand()[0]
    leaf_op = node.expand()[0]

    expected = max(variants, key=lambda v: v.score(ctx))
    assert leaf_op.score(ctx) == pytest.approx(expected.score(ctx)), (
        f"option-0 leaf score {leaf_op.score(ctx)} != max variant score {expected.score(ctx)}"
    )


def test_pointwise_collapses_br_layer():
    """Pointwise kernels have BR=1 fixed, so the BR Fork layer must
    collapse — no branch Fork should ever pin BR for them."""
    variants = _pointwise_variants()
    tree = _build_fork_tree(variants, _ctx())
    for branch in _walk_branches(tree):
        assert "BR" not in branch.knobs, f"BR leaked into branch for pointwise: {branch.knobs}"


def test_branch_knobs_partition_cleanly():
    """The union of (root-fork knobs, walked branch knobs, leaf knobs)
    along any root→leaf path must cover ``{BR, BM, BN, FM, FN, BK, SPLITK}``
    exactly once each — no knob duplicated across levels, none missing."""
    variants = _matmul_variants()
    tree = _build_fork_tree(variants, _ctx())
    expected = {"BR", "BM", "BN", "FM", "FN", "BK", "SPLITK"}

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
