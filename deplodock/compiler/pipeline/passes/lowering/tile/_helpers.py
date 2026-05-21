"""Shared utilities for ``lowering/tile`` rules.

- :func:`single_tile` — extract the unique ``Tile`` from a ``TileOp.body``,
  raising ``RuleSkipped`` if there isn't exactly one. Eliminates the
  identical 5-line preamble at the top of nearly every rule in this
  directory.
- :func:`is_matmul_reduce` — predicate on a reduce ``Loop``: body has
  ≥2 distinct K-indexed buffer Loads + at least one ``Accum``. The
  multiply between the two K-indexed Loads is implicit (the only way
  two distinct K-indexed buffer Loads can contribute to an Accum
  in this IR is through a fused multiply-accumulate). Used by
  ``000_partition_planner`` to locate the K reduce inside an output
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

from deplodock.compiler.ir.stmt import Accum, Body, Load, Loop, Stmt, Tile
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


def single_tile(body: Body) -> tuple[int, Tile]:
    """Locate the (sole) ``Tile`` in a TileOp body.

    ``TileOp.__post_init__`` enforces *at most* one Tile, so this only
    needs to handle the zero-Tile case — which happens for the
    degenerate single-thread serial body ``001_launch_geometry`` produces when
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


# ---------------------------------------------------------------------------
# Stage / Load helpers
# ---------------------------------------------------------------------------


def loads_reading(body: Body, stage_name: str) -> list[Load]:
    """Collect every Load anywhere in ``body`` reading from ``stage_name``."""
    return [s for s in body.iter() if isinstance(s, Load) and s.input == stage_name]
