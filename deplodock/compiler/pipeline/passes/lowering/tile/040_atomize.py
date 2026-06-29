"""Atomize — resolve the algebra→hardware-atom binding structurally, on the schedule.

The warp matmul materializer used to ``lower()`` the ``Semiring`` to flat loop-IR and then
re-recognize which operand is the mma ``a`` vs ``b`` (by axis-in-index), whether ``b`` is
transposed, the fold accumulator, and the projection epilogue. Every one of those facts is
already first-class on the ``Semiring`` node (``operands`` / ``fold`` / ``reduce_axis`` /
``out``) and the grid. This pass reads them **structurally** — off each operand's own leaf
``Load`` index, never a flattened loop — and stamps an :class:`AtomBinding` onto the
``SemiringSchedule`` (a sibling of the ``WarpTile`` geometry decision).

The binding rides the **schedule**, not the op tree: ``op_cache_key`` digests
``lower(op.op)`` (not the schedule), so the perf / prior cache stays byte-identical, and the
``Semiring`` combine remains the single source of truth. Runs at ``040`` — after
``030_split`` rewrites operand indices for cross-CTA slices, and after ``020_schedule`` has
chosen the ``WarpTile`` (split partials drop ``warp_tile``, so they fall through here).

The pass dispatches on kernel **kind** (the typed ``*Kernel`` seam, mirroring ``lower``):

- a warp-tiled ``SemiringKernel`` → :func:`bind_contraction` resolves the operand→role
  :class:`AtomBinding`;
- a cooperative / ILP ``MonoidKernel`` → :func:`_atomize_monoid` resolves the
  :class:`ReduceBinding` (the ``MonoidAtom`` dtype + partition widths; the shuffle/tree fold
  sequence stays derived).

**Recursion seam (deferred — warp-flash).** Flash is a ``Monoid`` (online-softmax) over a
nested contraction, so a kind-recursive atomize would bind the inner QK^T / PV with the same
:func:`bind_contraction` the root uses — that function is node-addressable for exactly this
reuse. It is **not wired yet** because flash's inner contractions are not structural
``Semiring`` nodes today: ``_flash._flash_op`` ``lower()``-s the QK score (a degenerate
``Monoid``) straight into loop-IR inside the score ``Map`` body, and carries no per-node
``WarpTile``. Wiring the recursion requires warp-flash to first (1) keep the inner
contractions as ``Semiring`` nodes and (2) attach inner warp geometry — at which point the
``MonoidKernel`` arm walks ``carrier.partial`` for nested ``Semiring``\\ s and calls
:func:`bind_contraction` per inner contraction. Until then a tree-walk would bind nothing, so
it is intentionally absent."""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile import (
    AtomBinding,
    MonoidAtom,
    MonoidKernel,
    Operand,
    ReduceBinding,
    SemiringKernel,
    TileOp,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", TileOp)]


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs (the materializer's helper)."""
    return {v for e in index for v in e.free_vars()}


def _operand_leaf(operand) -> Load:
    """The buffer ``Load`` of a one-``Load`` operand ``Map`` — Phase-1's gmem-direct
    contraction operand. A non-``Load`` leaf (a nested reduction / staged-fill prologue) is
    out of scope here (it bails, matching the materializer's gmem-direct guard)."""
    leaf = operand.body[-1] if isinstance(operand, Map) and operand.body else None
    if not isinstance(leaf, Load):
        raise LoweringError("warp tier: a contraction compute prologue isn't supported (gmem-direct, no staging)")
    return leaf


def bind_contraction(semi, m_name: str, n_name: str, epilogue: Body) -> AtomBinding:
    """Resolve the operand→role :class:`AtomBinding` for a contraction ``semi`` whose output is
    indexed by grid axes ``m_name`` / ``n_name``, with projection ``epilogue``.

    **Node-addressable** — it binds any ``Semiring`` node, not just a kernel root — so warp-flash
    can reuse it on flash's nested QK^T / PV contractions (the recursion seam in the module
    docstring). A/B are bound by which output axis each operand's OWN leaf ``Load`` index carries
    (Phase 1: each operand is a one-``Load`` ``Map``); ``b_trans`` from B's last index component."""
    k_name = semi.reduce_axis.name
    leaves = [_operand_leaf(o) for o in semi.operands]
    a_leaf = next((ld for ld in leaves if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in leaves if n_name in _idx_vars(ld.index)), None)
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    b_trans = k_name in b_leaf.index[-1].free_vars()  # B[n,k] (K last) vs canonical B[k,n]
    return AtomBinding(a=Operand(a_leaf, "a"), b=Operand(b_leaf, "b"), b_trans=b_trans, acc=semi.out, epilogue=epilogue)


def _atomize_semiring(tile: TileOp) -> AtomBinding:
    """The root contraction: extract the ``Semiring`` + output grid + projection epilogue (the
    ``Map`` body, or empty for a bare contraction) and delegate to :func:`bind_contraction`."""
    node = tile.op  # a Semiring, or a Map(source=Semiring) projection
    grid = tile.kernel.schedule.place.grid
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    epilogue = node.body if isinstance(node, Map) else Body(())
    return bind_contraction(node.reduce_node, grid[-2].name, grid[-1].name, epilogue)


def _atomize_monoid(tile: TileOp) -> ReduceBinding:
    """Resolve the cooperative-combine binding off the ``Monoid`` carrier + its ``ReducePlan``.
    The accumulator dtype is read off the carrier's combine program (the same fact
    ``emit_combine`` derives at kernel time); the fold mechanism stays derived on the binding."""
    kernel = tile.kernel
    plan = kernel.schedule.reduce
    carrier = tile.op.reduce_node
    dtype = next((a.dtype for a in carrier.combine_states if a.dtype is not None), None) or F32
    return ReduceBinding(atom=MonoidAtom(dtype=dtype), coop=plan.coop, reg=plan.reg)


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    kernel = tile.kernel
    sched = kernel.schedule if kernel is not None else None
    # A warp-tiled Semiring contraction → the operand→role AtomBinding (split partials drop
    # warp_tile, so they fall through here).
    if isinstance(kernel, SemiringKernel) and getattr(sched, "warp_tile", None) is not None:
        if sched.bind is not None:
            raise RuleSkipped("already atomized")  # idempotent / fixpoint-safe
        return replace(tile, kernel=replace(kernel, schedule=replace(sched, bind=_atomize_semiring(tile))))
    # A cooperative / ILP Monoid reduce → the cooperative-combine ReduceBinding (the scalar
    # tier — coop == reg == 1 — has no combine to bind and falls through).
    plan = getattr(sched, "reduce", None) if isinstance(kernel, MonoidKernel) else None
    if plan is not None and (plan.coop > 1 or plan.reg > 1):
        if sched.bind is not None:
            raise RuleSkipped("already atomized")
        return replace(tile, kernel=replace(kernel, schedule=replace(sched, bind=_atomize_monoid(tile))))
    raise RuleSkipped("not a warp contraction or cooperative reduce — nothing to atomize")
