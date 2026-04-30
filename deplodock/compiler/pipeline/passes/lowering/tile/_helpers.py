"""Shared utilities for ``lowering/tile`` rules.

Three helpers:

- :func:`single_tile` — extract the unique ``Tile`` from a ``TileOp.body``,
  raising ``RuleSkipped`` if there isn't exactly one. Eliminates the
  identical 5-line preamble at the top of nearly every rule in this
  directory.
- :func:`is_matmul_reduce` — predicate on a reduce ``Loop``: body has
  ≥2 distinct K-indexed buffer Loads + at least one ``Accum``. The
  multiply between the two K-indexed Loads is implicit (the only way
  two distinct K-indexed buffer Loads can contribute to an Accum
  in this IR is through a fused multiply-accumulate).
- :func:`is_matmul_k_outer` / :func:`find_matmul_k_outer` — predicate
  / locator for a top-level free ``Loop`` wrapping a single reduce
  Loop with a pure-compute body (Load / Assign / Accum + at least
  one Accum). Rule-specific gates layer on top via the ``extra_gate``
  callback.

The file is prefixed ``_`` so the engine's rule loader skips it
(``engine._load_rules`` filters ``startswith("_")``).
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Stmt, Tile
from deplodock.compiler.pipeline.engine import RuleSkipped


def single_tile(body: tuple[Stmt, ...]) -> tuple[int, Tile]:
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
    Used directly by ``002_split_matmul_k`` (which needs to fire on
    matmul-shaped reduces wherever they sit, not only at the top level
    under a K-outer wrapper).
    """
    if not (isinstance(loop, Loop) and loop.is_reduce):
        return False
    K_name = loop.axis.name
    bufs = {ld.input for ld in loop.loads if K_name in {v for e in ld.index for v in e.free_vars()}}
    if len(bufs) < 2:
        return False
    return any(isinstance(s, Accum) for s in loop.body)


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
    pipeline_async) via this hook so the structural part stays in one
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


def find_matmul_k_outer(
    body: tuple[Stmt, ...],
    *,
    extra_gate: Callable[[Loop, Loop], bool] = lambda k_outer, k_inner: True,
) -> int | None:
    """Return the index of the first top-level stmt in ``body`` that
    satisfies :func:`is_matmul_k_outer` (with the supplied extra gate),
    or ``None`` if no top-level Loop matches.

    Used by ``008_register_tile`` to locate the K-outer it splices the
    rewritten Loop back in around. Other rules (``013``, ``015``) walk
    every top-level Loop applying :func:`is_matmul_k_outer` directly,
    since they may fire on more than one match per Tile.
    """
    for i, s in enumerate(body):
        if is_matmul_k_outer(s, extra_gate=extra_gate):
            return i
    return None
