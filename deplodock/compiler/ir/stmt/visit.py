"""Body-tree walkers and small visitor utilities.

``iter_body`` yields every Stmt in pre-order via ``Stmt.nested``;
``map_body`` is a flat 1:N transformer; ``_recurse_through_block`` is
the shared helper that the normalization passes use to recurse through
non-Loop block stmts (StridedLoop / Tile / Cond).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.blocks import Cond, StridedLoop, Tile


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    """Yield every ``Stmt`` in ``body`` in pre-order, recursing into each
    stmt's ``nested()`` bodies.

    Works across all IRs (Loop, Tile, Kernel) without type-switching:
    every block-structured Stmt subclass overrides ``Stmt.nested`` to
    return its child body tuples, and this walker drives off that
    method. Callers that want only leaves can filter with ``isinstance``.
    """
    for s in body:
        yield s
        for child_body in s.nested():
            yield from iter_body(child_body)


def map_body(
    body: tuple[Stmt, ...],
    fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]],
) -> tuple[Stmt, ...]:
    """Flat body transformer: apply ``fn`` to each stmt, splice its result
    into the output. ``fn`` may return:

    - a single ``Stmt`` (kept in place of the input),
    - ``None`` (drop the input), or
    - an iterable of ``Stmt`` (inline all of them — useful for 1:N
      expansions like loop unrolling or size-1 Loop inlining).

    ``fn`` is called on *every* stmt including ``Loop`` wrappers; recursion
    into a Loop's body is the caller's responsibility (typically by writing
    a self-recursive ``fn`` that returns ``Loop(axis=..., body=map_body(s.body, fn))``
    for Loop cases). Lets callers pick their own policy for axis renames,
    Loop skipping, or selective recursion.
    """
    out: list[Stmt] = []
    for s in body:
        r = fn(s)
        if r is None:
            continue
        if isinstance(r, Stmt):
            out.append(r)
        else:
            out.extend(r)
    return tuple(out)


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def _recurse_through_block(s: Stmt, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Stmt | None:
    """If ``s`` is a non-Loop block stmt, return a copy with its body / bodies
    re-walked through ``fn`` via ``map_body``. Else return ``None``."""
    if isinstance(s, StridedLoop):
        return StridedLoop(axis=s.axis, start=s.start, step=s.step, body=map_body(s.body, fn))
    if isinstance(s, Tile):
        return Tile(axes=s.axes, body=map_body(s.body, fn))
    if isinstance(s, Cond):
        return Cond(cond=s.cond, body=map_body(s.body, fn), else_body=map_body(s.else_body, fn))
    return None
