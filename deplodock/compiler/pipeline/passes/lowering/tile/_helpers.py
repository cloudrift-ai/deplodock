"""Shared utilities for ``lowering/tile`` rules.

- :func:`single_tile` — extract the unique ``Tile`` from a ``TileOp.body``,
  raising ``RuleSkipped`` if there isn't exactly one. Eliminates the
  identical 5-line preamble at the top of nearly every rule in this
  directory.
- :func:`is_matmul_reduce` — predicate on a reduce ``Loop``: body has
  ≥2 distinct K-indexed buffer Loads + at least one ``Accum``. The
  multiply between the two K-indexed Loads is implicit (the only way
  two distinct K-indexed buffer Loads can contribute to an Accum
  in this IR is through a fused multiply-accumulate).
- :func:`is_matmul_k_outer` — predicate for a top-level free ``Loop``
  wrapping a single reduce Loop with a pure-compute body
  (Load / Assign / Accum + at least one Accum). Rule-specific gates
  layer on top via the ``extra_gate`` callback.
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
from collections.abc import Callable

from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage
from deplodock.compiler.pipeline import RuleSkipped

_logger = logging.getLogger(__name__)


def accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another Accum's
    running value. Permits multiple independent Accums; rejects online
    algorithms (online softmax, Welford). Salvaged from the deleted
    ``004_launch_geometry``; retained because the kernel-rule tests
    use it as a structural predicate."""
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


from deplodock.compiler.target import compute_capability  # noqa: E402,F401


def single_tile(body: Body) -> tuple[int, Tile]:
    """Locate the (sole) ``Tile`` in a TileOp body.

    ``TileOp.__post_init__`` enforces *at most* one Tile, so this only
    needs to handle the zero-Tile case — which happens for the
    degenerate single-thread serial body ``001_tileify`` produces when
    a LoopOp has no outer free-Loop chain to strip. Raises
    ``RuleSkipped`` in that case so the rule cleanly bails.
    """
    for i, s in enumerate(body):
        if isinstance(s, Tile):
            return (i, s)
    raise RuleSkipped("TileOp has no Tile (degenerate single-thread body)")


def is_matmul_reduce(loop: Loop) -> bool:
    """True iff ``loop`` is a reduce ``Loop`` whose body matches the
    matmul signature: ≥2 distinct buffers with K-indexed Loads (where
    K is ``loop.axis.name``) plus at least one ``Accum``.

    Doesn't check body purity — that lives in :func:`is_matmul_k_outer`.
    Used directly by ``002_chunk_matmul_k`` (which needs to fire on
    matmul-shaped reduces wherever they sit, not only at the top level
    under a K-outer wrapper).
    """
    if not (isinstance(loop, Loop) and loop.is_reduce):
        return False
    K_name = loop.axis.name
    bufs = {ld.input for ld in loop.body.of_type(Load) if K_name in {v for e in ld.index for v in e.free_vars()}}
    if len(bufs) < 2:
        return False
    return any(isinstance(s, Accum) for s in loop.body)


def collect_invariant_names(stmt: Stmt) -> set[str]:
    """SSA names that ``stmt`` defines and exposes to its enclosing scope.

    For leaf stmts (Load, Assign, Select, Accum, Stage, Combine, etc.)
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


def is_matmul_k_outer(
    loop: Stmt,
    *,
    extra_gate: Callable[[Loop, Loop], bool] = lambda k_outer, k_inner: True,
) -> bool:
    """True iff ``loop`` is a non-reduce free ``Loop`` wrapping exactly
    one reduce Loop (the K-inner) whose body is pure compute
    (``Load`` / ``Assign`` / ``Accum`` only, with at least one Accum).

    ``extra_gate(k_outer, k_inner)`` runs as a final check after the
    structural gates pass; rules layer their own constraints (e.g.
    ``≥2 K-indexed buffers`` for register_tile, ``≥1 Stage in
    k_outer.body`` for double_buffer, ``no cross-loop SSA reads`` for
    pipeline_k_outer) via this hook so the structural part stays in one
    place. Idempotence-style markers go in extra_gate too.
    """
    if not (isinstance(loop, Loop) and not loop.is_reduce):
        return False
    reduces = [c for c in loop.body if isinstance(c, Loop) and c.is_reduce]
    if len(reduces) != 1:
        return False
    k_inner = reduces[0]
    if not all(isinstance(c, (Load, Assign, Accum)) for c in k_inner.body):
        return False
    if not any(isinstance(c, Accum) for c in k_inner.body):
        return False
    return extra_gate(loop, k_inner)


# ---------------------------------------------------------------------------
# Stage / Load helpers
# ---------------------------------------------------------------------------


def loads_reading(body: Body, stage_name: str) -> list[Load]:
    """Collect every Load anywhere in ``body`` reading from ``stage_name``."""
    return [s for s in body.iter() if isinstance(s, Load) and s.input == stage_name]


# ---------------------------------------------------------------------------
# Shared knob choice constants (single source of truth)
# ---------------------------------------------------------------------------

# Per-nest-level register-tile factor choices. Used by
# ``008_register_tile`` (legacy fork) and ``000_partition_planner``
# (planner-driven matmul register-tile fork).
TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# Cap on total per-thread replication (∏ factors). Mirrors
# ``008_register_tile._MAX_CELLS_PER_THREAD``: NVRTC compile time
# explodes on more-unrolled bodies.
MAX_CELLS_PER_THREAD: int = 128


# ---------------------------------------------------------------------------
# Axis replication (factored out of legacy 008_register_tile so the planner-
# driven 006a_register_tile_planned can keep working after 008 is deleted)
# ---------------------------------------------------------------------------


def replicate_along_axis(body: Body, axis: str, factor: int, sigma_for: Callable[[int], Sigma]) -> Body:
    """F× replicate every stmt whose value transitively depends on
    ``axis``. Each such stmt is emitted ``factor`` times with σ given
    by ``sigma_for(i)`` and SSA names suffixed ``_<i>``. Stmts that
    don't depend on ``axis`` pass through. Block stmts recurse into
    their bodies and rebuild via :meth:`Stmt.with_bodies`; a wrapper
    that shadows ``axis`` isn't itself replicated (the fold's bound-
    axis filter keeps shadowed references local).

    Dependency analysis is one :meth:`Body.fold` over the def-use DAG
    with bound-axis filtering. ``keep[name]`` records which SSA names
    must carry the suffix vs. pass through unchanged (Tile-input
    buffers, constants, axis-free producers)."""

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        # Stage cache-axis Vars are smem-local — they don't vary per replica.
        # Mark them bound here so Stages aren't tagged for replication; only
        # the consumer Loads (which σ-rewrite cache-axis Vars) multiply.
        local_bound = bound | frozenset(ax.name for ax in s.axes) if isinstance(s, Stage) else bound
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in local_bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in s.defines()}

    def rename_for(i: int):
        def _rename(name: str) -> str:
            return f"{name}_{i}" if keep.get(name, False) else name

        return _rename

    def go(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            nested = s.nested()
            if nested:
                out.append(s.with_bodies(tuple(go(child) for child in nested)))
            elif axis in deps.get(id(s), frozenset()):
                for i in range(factor):
                    out.append(s.rewrite(rename_for(i), sigma_for(i)))
            else:
                out.append(s)
        return Body(out)

    return go(body)
