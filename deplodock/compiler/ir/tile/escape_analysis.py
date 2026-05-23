"""Coordination summary for a Tile-IR / Kernel-IR body.

A single body walk yields three derived sets the materializer + render
consume to pick atomic-add, broadcast-guard, and cooperative-combine
emission:

- ``atomic_axes(write)`` — enclosing ``GridTile.axes`` NOT in the
  Write's index. Non-empty ⇒ multiple CTAs race ⇒ ``atomicAdd``.
- ``accum_cooperative_axes[name]`` — for each Accum, the subset of
  ``Accum.axes`` that's also an enclosing ``ThreadTile`` axis. Non-empty
  ⇒ per-thread partials need cross-thread combining at the Accum's
  escape point.
- ``broadcast_axes(write)`` — cooperative thread axes (any axis in any
  ``accum_cooperative_axes`` set) NOT in the Write's index. Non-empty ⇒
  ``Cond(axis == 0)`` guard required so only one thread of the
  cooperative group performs the store.

A ``TileOp`` body has at most one outer ``GridTile`` and one outer
``ThreadTile`` (enforced by ``TileOp.__post_init__``), and axis names
are unique within the body after ``normalize_body``, so axis sets are
*global* — no per-stmt scope walk needed. Staging-buffer Writes (smem
stores to ``Smem`` decls or ``Stage.sources``) are excluded from both
atomic and broadcast classification: those targets are per-thread slab
slots, not racing global stores, and the warp-shuffle / smem
tree-halve code in ``_emit_combine`` already carries its own per-lane
guards.

See ``plans/derive-coordination-from-body.md`` for the refactor history
(coordination pass deletion + Accum.axes follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt import Accum, Body
from deplodock.compiler.ir.stmt.leaves import Write
from deplodock.compiler.ir.tile.ir import GridTile, Stage, ThreadTile, TileOp


@dataclass(frozen=True)
class EscapeAnalysis:
    """Coordination metadata for a ``TileOp`` body.

    ``cooperative_thread_axes`` is the union of every cooperative axis
    set across the body's Accums. ``accum_cooperative_axes`` is the
    per-Accum set so callers can match a Combine emission point to the
    Accum that drives it. ``Write`` keys use ``id(...)`` internally
    because ``Write.index`` may hold ``BinaryExpr`` nodes that aren't
    hashable — use the :meth:`atomic_axes` / :meth:`broadcast_axes`
    accessors. ``writes`` is the analyzed Writes in body-walk order so
    callers can iterate deterministically.
    """

    cooperative_thread_axes: frozenset[str] = frozenset()
    accum_cooperative_axes: dict[str, frozenset[str]] = field(default_factory=dict)
    writes: tuple[Write, ...] = field(default_factory=tuple)
    _write_atomic_axes: dict[int, frozenset[str]] = field(default_factory=dict)
    _write_broadcast_axes: dict[int, frozenset[str]] = field(default_factory=dict)

    def atomic_axes(self, w: Write) -> frozenset[str]:
        """Block axes NOT in ``w.index`` — non-empty ⇒ ``atomicAdd``."""
        return self._write_atomic_axes.get(id(w), frozenset())

    def broadcast_axes(self, w: Write) -> frozenset[str]:
        """Cooperative thread axes NOT in ``w.index`` — non-empty ⇒
        ``Cond(axis == 0)`` guard around the Write."""
        return self._write_broadcast_axes.get(id(w), frozenset())


def analyze(tile_op_or_body: TileOp | Body) -> EscapeAnalysis:
    """Compute :class:`EscapeAnalysis` for a ``TileOp`` (or its body
    directly — useful when the render path wants to analyze a fragment
    or when test fixtures want to skip ``TileOp`` normalization)."""
    body = tile_op_or_body.body if isinstance(tile_op_or_body, TileOp) else tile_op_or_body

    try:
        from deplodock.compiler.ir.kernel.ir import Smem  # noqa: PLC0415
    except ImportError:
        Smem = None  # type: ignore[assignment]

    block_axes: set[str] = set()
    thread_axes: set[str] = set()
    staging_buffers: set[str] = set()
    accums: list[Accum] = []
    writes: list[Write] = []

    for s in body.iter():
        if isinstance(s, GridTile):
            block_axes.update(ax.name for ax in s.axes)
        elif isinstance(s, ThreadTile):
            thread_axes.update(ax.name for ax in s.axes)
        elif isinstance(s, Stage):
            staging_buffers.update(src.name for src in s.sources)
        elif Smem is not None and isinstance(s, Smem):
            staging_buffers.add(s.name)
        elif isinstance(s, Accum):
            accums.append(s)
        elif isinstance(s, Write):
            writes.append(s)

    block_axes_fz = frozenset(block_axes)
    thread_axes_fz = frozenset(thread_axes)
    staging_buffers_fz = frozenset(staging_buffers)

    accum_cooperative_axes: dict[str, frozenset[str]] = {acc.name: frozenset(acc.axes) & thread_axes_fz for acc in accums}
    cooperative_thread_axes = frozenset().union(*accum_cooperative_axes.values()) if accum_cooperative_axes else frozenset()

    write_atomic_axes: dict[int, frozenset[str]] = {}
    write_broadcast_axes: dict[int, frozenset[str]] = {}
    for w in writes:
        if w.output in staging_buffers_fz:
            write_atomic_axes[id(w)] = frozenset()
            write_broadcast_axes[id(w)] = frozenset()
            continue
        idx_vars = _free_vars_in_index(w)
        write_atomic_axes[id(w)] = block_axes_fz - idx_vars
        write_broadcast_axes[id(w)] = cooperative_thread_axes - idx_vars

    return EscapeAnalysis(
        cooperative_thread_axes=cooperative_thread_axes,
        accum_cooperative_axes=accum_cooperative_axes,
        writes=tuple(writes),
        _write_atomic_axes=write_atomic_axes,
        _write_broadcast_axes=write_broadcast_axes,
    )


def _free_vars_in_index(w: Write) -> frozenset[str]:
    """Union of all axis Vars referenced by ``w.index``."""
    out: set[str] = set()
    for e in w.index:
        out |= e.free_vars()
    return frozenset(out)


__all__ = ["EscapeAnalysis", "analyze"]
