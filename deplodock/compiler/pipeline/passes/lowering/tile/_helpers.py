"""Shared utilities for ``lowering/tile`` rules.

- :func:`single_tile` — extract the unique outer ``GridTile`` /
  ``ThreadTile`` / ``WarpTile`` from a ``TileOp.body``, raising
  ``RuleSkipped`` if there isn't exactly one. Eliminates the
  identical 5-line preamble at the top of nearly every rule in this
  directory.
- :func:`parallel_tile_of` — return the per-binding-tier inner scope
  (ThreadTile or WarpTile) for an outer ``GridTile`` / ``ThreadTile`` /
  ``WarpTile``. ``thread_tile_of`` is kept as a deprecated alias.
- :func:`is_matmul_reduce` — predicate on a reduce ``Loop``: body has
  ≥2 distinct K-indexed buffer Loads + at least one ``Accum``. The
  multiply between the two K-indexed Loads is implicit (the only way
  two distinct K-indexed buffer Loads can contribute to an Accum
  in this IR is through a fused multiply-accumulate). Used by
  ``010_partition_loops`` to locate the K reduce inside an output
  body; downstream K-outer passes (``010``, ``015``) key off the
  planner-stamped ``Role.SERIAL_OUTER`` / ``Role.STAGE_INNER`` tags
  instead of re-deriving the matmul shape structurally.
- :func:`compute_capability` — re-exported from
  :mod:`deplodock.compiler.target` so passes can import it locally.
  Honors the ``--target sm_NN`` CLI override.
- :func:`loads_reading` — collect every body Load reading a Stage by
  name. Bank-conflict analysis itself lives in
  :mod:`deplodock.compiler.diagnostics.bank_conflicts` and is shared
  with the visualizer.

The file is prefixed ``_`` so the engine's rule loader skips it
(``engine._load_rules`` filters ``startswith("_")``).
"""

from __future__ import annotations

import logging

from deplodock.compiler.ir.algebra import matmul_reduce
from deplodock.compiler.ir.stmt import Accum, Body, Load, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import GridTile, ParallelTile, SerialTile, StridedTile, ThreadTile, WarpTile
from deplodock.compiler.pipeline import RuleSkipped

_logger = logging.getLogger(__name__)


def accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another Accum's
    running value. Permits multiple independent Accums; rejects online
    algorithms (online softmax, Welford). Used by reduction-rule
    tests as a structural predicate."""
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


from deplodock.compiler.target import compute_capability  # noqa: E402,F401


def parallel_tile_of(outer: ParallelTile) -> ThreadTile | WarpTile:
    """Return the per-binding-tier inner scope for an outer
    ``GridTile`` / ``ThreadTile`` / ``WarpTile``.

    The cooperative form (``GridTile`` wrapping ``ThreadTile`` /
    ``WarpTile``) puts the per-tier scope one level deeper; the
    pointwise form (standalone ``ThreadTile``) IS the per-thread scope.
    Downstream passes that operate on the per-binding-tier scope
    (``020_stage_inputs``, ``010_split_register_axes``, ...) call this
    helper instead of branching on the outer flavor.

    Renamed from ``thread_tile_of`` to reflect the addition of
    ``WarpTile`` (which is also a valid per-tier scope under the same
    outer ``GridTile`` shape). ``thread_tile_of`` is kept as a
    deprecation-warning alias.
    """
    if isinstance(outer, (ThreadTile, WarpTile)):
        return outer
    if isinstance(outer, GridTile):
        for child in outer.body:
            if isinstance(child, (ThreadTile, WarpTile)):
                return child
        raise RuleSkipped("GridTile without an inner ThreadTile/WarpTile")
    raise TypeError(f"parallel_tile_of: expected GridTile/ThreadTile/WarpTile, got {type(outer).__name__}")


def thread_tile_of(outer: ParallelTile) -> ThreadTile | WarpTile:
    """Deprecated alias for :func:`parallel_tile_of`. Kept for the
    transition until in-flight MMA / WS-refactor consumers drop their
    legacy import. Emits a ``DeprecationWarning`` via the module logger
    so call sites surface in test logs.
    """
    _logger.warning("thread_tile_of is deprecated; use parallel_tile_of", stacklevel=2)
    return parallel_tile_of(outer)


def replace_parallel_tile_body(outer: ParallelTile, new_body) -> ParallelTile:
    """Rebuild ``outer`` with the per-binding-tier scope's body replaced.

    Preserves the GridTile wrapper (if any). Cooperativity is recovered
    from ``Accum.axes`` at materialize / render time so no per-tile tag
    needs propagating here.

    Renamed from ``replace_thread_tile_body`` to reflect ``WarpTile``
    support. ``replace_thread_tile_body`` is kept as a deprecation-
    warning alias.
    """
    new_body = Body.coerce(new_body) if not isinstance(new_body, Body) else new_body
    if isinstance(outer, ThreadTile):
        return ThreadTile(axes=outer.axes, body=new_body)
    if isinstance(outer, WarpTile):
        return WarpTile(axes=outer.axes, body=new_body)
    if isinstance(outer, GridTile):
        # Locate the inner per-tier scope (ThreadTile or WarpTile) and rebuild it.
        new_outer_body: list = []
        for child in outer.body:
            if isinstance(child, ThreadTile):
                new_outer_body.append(ThreadTile(axes=child.axes, body=new_body))
            elif isinstance(child, WarpTile):
                new_outer_body.append(WarpTile(axes=child.axes, body=new_body))
            else:
                new_outer_body.append(child)
        return GridTile(axes=outer.axes, body=Body(new_outer_body), swizzle_group_m=outer.swizzle_group_m)
    raise TypeError(f"replace_parallel_tile_body: expected GridTile/ThreadTile/WarpTile, got {type(outer).__name__}")


def replace_thread_tile_body(outer: ParallelTile, new_body) -> ParallelTile:
    """Deprecated alias for :func:`replace_parallel_tile_body`."""
    _logger.warning("replace_thread_tile_body is deprecated; use replace_parallel_tile_body", stacklevel=2)
    return replace_parallel_tile_body(outer, new_body)


def single_tile(body: Body) -> tuple[int, ParallelTile]:
    """Locate the (sole) outermost ``GridTile`` / ``ThreadTile`` /
    ``WarpTile`` in a TileOp body.

    ``TileOp.__post_init__`` enforces *at most* one outer tile, so this
    only needs to handle the zero-tile case — which happens for the
    degenerate single-thread serial body ``001_launch_geometry`` produces
    when a LoopOp has no outer free-Loop chain to strip. Raises
    ``RuleSkipped`` in that case so the rule cleanly bails.

    Returns the outermost tile flavor regardless of cooperative shape
    (``GridTile`` wrapping a ``ThreadTile`` / ``WarpTile``, or a standalone
    ``ThreadTile``). Callers that need the per-binding-tier scope navigate
    via :func:`parallel_tile_of` (or ``tile.body``).
    """
    for i, s in enumerate(body):
        if isinstance(s, (GridTile, ThreadTile, WarpTile)):
            return (i, s)
    raise RuleSkipped("TileOp has no outer ParallelTile (degenerate single-thread body)")


def is_matmul_reduce(loop) -> bool:
    """True iff ``loop`` is a reduce-loop whose body matches the matmul
    signature: ≥2 distinct buffers with K-indexed Loads (where K is
    ``loop.axis.name``) plus at least one ``Accum`` (or its tensor-core fused
    form ``Mma``, after ``tile/011_lower_atom_cell`` has rewritten the cell).

    Accepts both Loop-IR ``Loop`` / ``StridedLoop`` (the pre-launch_geometry
    shape seen by ``010_partition_loops``) and Tile-IR ``SerialTile`` /
    ``StridedTile`` (the post-launch_geometry shape). Used by
    ``010_partition_loops`` to locate the matmul K reduce inside a LoopOp
    body, and by downstream tile passes to confirm a matmul-shaped reduce
    survived.

    The structural core lives in ``ir/algebra.matmul_reduce`` (the single
    source the bottom-up `AlgebraKind` classifier shares); this wrapper adds
    the tile-layer type guard.
    """
    return isinstance(loop, (Loop, StridedLoop, SerialTile, StridedTile)) and matmul_reduce(loop)


def segmentable_k_extent(load: Load, k_name: str) -> int | None:
    """For a folded-K matmul operand (the reduce axis spread across >1 index dim
    — the collapsed-reshape / transpose ``o_proj`` attn-out read), return the
    inner contiguous extent ``C`` when the read is **segmentable**; else ``None``.

    Segmentable signature: the innermost (stride-1) index dim is ``expr % C`` with
    a literal ``C`` carrying ``k_name`` — i.e. K's contiguous run has extent ``C``
    and the remaining K-dims are higher-order delinearization of the same flat
    offset. A C-aligned K split (``k_per_block == C`` ⇒ ``K_o`` = strided-outer
    segment, ``K_i·atom_k`` = the contiguous inner run) folds the delinearization
    to a clean ``[outer, …, inner]`` read (the range-aware ``Expr.simplify``), so
    the matmul reaches the mma tier reading gmem directly — no transpose producer.

    Returns ``None`` for a single-K-dim load (already stageable) or a fold whose
    inner dim isn't a literal-modulus contiguous run (not safely C-alignable)."""
    from deplodock.compiler.ir.expr import BinaryExpr, Literal  # noqa: PLC0415

    if not load.index:
        return None
    k_dims = [d for d, e in enumerate(load.index) if k_name in e.free_vars()]
    if len(k_dims) <= 1:
        return None
    inner = load.index[-1]
    if (
        isinstance(inner, BinaryExpr)
        and inner.op == "%"
        and isinstance(inner.right, Literal)
        and inner.right.dtype == "int"
        and isinstance(inner.right.value, int)
        and k_name in inner.left.free_vars()
    ):
        return inner.right.value
    return None


def collect_invariant_names(stmt: Stmt) -> set[str]:
    """SSA names that ``stmt`` defines and exposes to its enclosing scope.

    For leaf stmts (Load, Assign, Select, Accum, Stage, etc.)
    that's just ``stmt.defines()``. For wrapper stmts (Loop, Tile,
    Cond, StridedLoop) it recursively collects every Accum name in
    every nested body — those are the values the wrapper exposes
    upward once the loop / scope closes.

    Used by passes that need to know "what cross-loop SSA names are
    safe to read" — names defined in a *prior sibling stmt* at the
    same scope are loop-invariant w.r.t. any subsequent K-outer Loop
    here, so cross-loop reads of them don't compound fp32 drift in a
    pipelined rewrite the way reads of a *current* Accum's running
    value would.
    """
    out = set(stmt.defines())
    for body in stmt.nested():
        for s in body.iter():
            out.update(s.defines())
    return out


# ---------------------------------------------------------------------------
# Stage / Load helpers
# ---------------------------------------------------------------------------


def loads_reading(body: Body, stage_name: str) -> list[Load]:
    """Collect every Load anywhere in ``body`` reading from ``stage_name``."""
    return [s for s in body.iter() if isinstance(s, Load) and s.input == stage_name]
