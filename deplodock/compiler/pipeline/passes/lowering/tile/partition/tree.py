"""Generative schedule tree for the pointwise regime.

Unlike the legacy ``build_fork_tree`` (which groups a pre-enumerated flat
cartesian post-hoc), this tree generates children move-by-move: the root
offers legal thread tiles, each thread branch offers the register tiles legal
*given that thread tile* (incremental budget pruning), and each leaf
materializes. The two-level MCTS branches on these genuine move decisions —
coarse (thread fan-out) near the root, fine (register cells) at the leaves —
reusing the ``Fork`` / ``LazyCandidate`` engine unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_pointwise_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import (
    reg_knobs,
    reg_offers,
    thread_knobs,
    thread_offers,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import PointwiseSkeleton


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree."""

    skel: PointwiseSkeleton
    budget: Budget
    base_knobs: dict
    kernel_name: str


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete move stack (thread + register tiles); ``expand`` materializes
    the ``TileOp``."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [build_pointwise_tile(self.ctx.skel, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


@dataclass(frozen=True)
class _ThreadChosen(Fork):
    """Thread tile pinned; ``expand`` offers the register tiles legal under the
    cell budget."""

    ctx: _Ctx
    thread: tuple[int, int]
    knobs: dict

    def expand(self) -> list:
        return [
            _Leaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.skel, reg)}) for reg in reg_offers(self.ctx.skel, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _ChooseThread(Fork):
    """Root: ``expand`` offers the legal thread tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _ThreadChosen(ctx=self.ctx, thread=t, knobs=thread_knobs(self.ctx.skel, t))
            for t in thread_offers(self.ctx.skel, self.ctx.budget)
        ]


def build_pointwise_tree(skel: PointwiseSkeleton, *, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Return the root ``Fork`` of the generative tree, a bare ``TileOp`` when
    only one variant is legal (mirrors the legacy single-variant short-circuit),
    or ``None`` when nothing is legal (the dispatcher falls through to legacy)."""
    budget = Budget()
    threads = thread_offers(skel, budget)
    regs = reg_offers(skel, budget)
    if not threads or not regs:
        return None
    ctx = _Ctx(skel=skel, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name)
    if len(threads) == 1 and len(regs) == 1:
        full = {**thread_knobs(skel, threads[0]), **reg_knobs(skel, regs[0])}
        return build_pointwise_tile(skel, full, kernel_name=kernel_name, base_knobs=base_knobs)
    return _ChooseThread(ctx=ctx, knobs={})
