"""Escape analysis on Tile-IR bodies — derives coordination decisions.

Computes three queries from a ``TileOp`` body:

- ``atomic_axes(write)`` — enclosing ``GridTile`` axis names NOT in the
  Write's index. Non-empty ⇒ multiple CTAs race ⇒ ``atomicAdd``. Purely
  structural — derivable from Write index vs. enclosing block axes.
- ``broadcast_axes(write)`` — cooperative thread axes (derived from
  ``Accum.axes`` ∩ enclosing ``ThreadTile.axes``) NOT in the Write's
  index. Non-empty ⇒ ``Cond(axis == 0)`` guard required.
- ``accum_cooperative_axes[name]`` — for each ``Accum.name``, the
  intersection of its declared reduction axes with the enclosing
  ``ThreadTile.axes``. Non-empty ⇒ a cross-thread combine is required
  at the Accum's escape point.

**Cooperativity is derived from ``Accum.axes``.** The planner populates
``Accum.axes`` with every loop / thread axis the Accum reduces over, and
σ-split propagates renames through (a single K axis becoming K_o × K_i ×
K_c expands into three sub-axes on the Accum). When any of those axes is
also bound by an enclosing ``ThreadTile``, the per-thread partials need
combining cross-thread — that's the cooperative-K case. Plain-matmul
Accums have only serial-loop axes, so the intersection is empty and no
combine fires.

The materializer / Kernel-IR render consume the analysis directly: there
is no separate coordination pass. See
``plans/derive-coordination-from-body.md`` for the refactor history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt import Accum, Body
from deplodock.compiler.ir.stmt.leaves import Write
from deplodock.compiler.ir.tile.ir import GridTile, Stmt, ThreadTile, TileOp


@dataclass(frozen=True)
class EscapeAnalysis:
    """Per-stmt coordination metadata derived from a ``TileOp`` body.

    ``cooperative_thread_axes`` is the union of every enclosing
    ``ThreadTile.cooperative_axes`` (read from the planner-emitted tag,
    not derived). ``accum_cooperative_axes`` maps each Accum to the
    subset that actually encloses it. ``Write`` keys use ``id(...)``
    internally because ``Write.index`` may hold ``BinaryExpr`` nodes
    that aren't hashable — use the :meth:`atomic_axes` /
    :meth:`broadcast_axes` accessors rather than touching the private
    dicts. ``writes`` holds the analyzed Writes in body-walk order so
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


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def analyze(tile_op_or_body: TileOp | Body) -> EscapeAnalysis:
    """Compute :class:`EscapeAnalysis` for a ``TileOp`` (or its body
    directly — useful when the materializer wants to analyze a fragment
    or when test fixtures want to skip ``TileOp`` normalization that
    renames axes)."""
    body = tile_op_or_body.body if isinstance(tile_op_or_body, TileOp) else tile_op_or_body
    # Single body walk: collect each Accum / Write with its scope chain
    # (sequence of enclosing block stmts root-first). The chain lets us
    # answer "which enclosing tile axes are above this stmt" without a
    # second traversal.
    accums: list[tuple[Accum, tuple[Stmt, ...]]] = []
    writes: list[tuple[Write, tuple[Stmt, ...]]] = []
    _collect(body, (), accums, writes)
    # Per-thread Writes to smem staging buffers must NOT be classified
    # as atomic — each thread owns its own slab slot, no racing. The
    # materializer's cooperative-load nest produces Writes whose index
    # references the cooperative-load loop var (not the block axes), so
    # block-axis-missing would otherwise flag them.
    staging_buffers = _staging_buffer_names(body)

    # --- Atomic-Write classification ---
    # Pure index analysis: any enclosing GridTile axis missing from a
    # Write's index means CTAs race ⇒ atomic. Skipped for staging-buffer
    # targets (smem).
    write_atomic_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        if w.output in staging_buffers:
            write_atomic_axes[id(w)] = frozenset()
            continue
        block_axes = _block_axis_names_in_chain(chain)
        missing = block_axes - _free_vars_in_index(w)
        write_atomic_axes[id(w)] = frozenset(missing)

    # --- Cooperative axes per Accum ---
    # Intersection of the Accum's declared reduction axes with the
    # enclosing ThreadTile's axes. Anything in both sets is a
    # cooperative thread axis — per-thread partials need combining
    # cross-thread before the Accum value escapes.
    accum_cooperative_axes: dict[str, frozenset[str]] = {}
    for acc, chain in accums:
        thread_axes = _enclosing_thread_axes(chain)
        accum_cooperative_axes[acc.name] = frozenset(acc.axes) & thread_axes

    cooperative_thread_axes = frozenset().union(*accum_cooperative_axes.values()) if accum_cooperative_axes else frozenset()

    # --- Broadcast-Write classification ---
    # A Write needs a Cond-guard for axis t iff t is a cooperative thread
    # axis derived from any Accum in the body AND t is not referenced by
    # the Write's index. Staging-buffer writes are skipped: the warp-
    # shuffle / smem tree-halve code in ``_emit_combine`` carries its own
    # per-lane / per-warp guards, and a second guard would collapse the
    # broadcast write to thread 0 only (leaving the rest of the smem
    # slab uninitialized).
    write_broadcast_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        if w.output in staging_buffers:
            write_broadcast_axes[id(w)] = frozenset()
            continue
        thread_axes = _enclosing_thread_axes(chain)
        coop_axes = cooperative_thread_axes & thread_axes
        w_idx_vars = _free_vars_in_index(w)
        write_broadcast_axes[id(w)] = frozenset(coop_axes - w_idx_vars)

    return EscapeAnalysis(
        cooperative_thread_axes=cooperative_thread_axes,
        accum_cooperative_axes=accum_cooperative_axes,
        writes=tuple(w for w, _ in writes),
        _write_atomic_axes=write_atomic_axes,
        _write_broadcast_axes=write_broadcast_axes,
    )


# ---------------------------------------------------------------------------
# Body traversal — collect leaf stmts with scope chains
# ---------------------------------------------------------------------------


def _collect(
    body: Body,
    chain: tuple[Stmt, ...],
    accums: list[tuple[Accum, tuple[Stmt, ...]]],
    writes: list[tuple[Write, tuple[Stmt, ...]]],
) -> None:
    """Walk ``body`` recursively, recording each ``Accum`` / ``Write``
    paired with its enclosing-stmt chain (root-first)."""
    for s in body:
        if isinstance(s, Accum):
            accums.append((s, chain))
        elif isinstance(s, Write):
            writes.append((s, chain))
        for sub in s.nested():
            _collect(sub, chain + (s,), accums, writes)


# ---------------------------------------------------------------------------
# Scope queries
# ---------------------------------------------------------------------------


def _block_axis_names_in_chain(chain: tuple[Stmt, ...]) -> frozenset[str]:
    """Set of axis names bound by every ``GridTile`` enclosing the leaf."""
    out: set[str] = set()
    for s in chain:
        if isinstance(s, GridTile):
            out.update(ax.name for ax in s.axes)
    return frozenset(out)


def _staging_buffer_names(body: Body) -> frozenset[str]:
    """Names declared as ``Smem`` (or ``Stage.sources``) anywhere in the
    body — staging buffers, not global outputs. Writes to these names
    are per-thread slab stores, not racing global stores."""
    # Local imports to avoid a Tile-IR cycle and to keep Kernel-IR
    # optional for callers analyzing pre-materialize bodies.
    out: set[str] = set()
    try:
        from deplodock.compiler.ir.kernel.ir import Smem  # noqa: PLC0415
    except ImportError:
        Smem = None  # type: ignore[assignment]
    from deplodock.compiler.ir.tile.ir import Stage  # noqa: PLC0415

    for s in body.iter():
        if Smem is not None and isinstance(s, Smem):
            out.add(s.name)
        if isinstance(s, Stage):
            for src in s.sources:
                out.add(src.name)
    return frozenset(out)


def _enclosing_thread_axes(chain: tuple[Stmt, ...]) -> frozenset[str]:
    """Union of ``ThreadTile.axes`` (names) across every ``ThreadTile``
    enclosing the leaf."""
    out: set[str] = set()
    for s in chain:
        if isinstance(s, ThreadTile):
            out.update(ax.name for ax in s.axes)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Index / value queries
# ---------------------------------------------------------------------------


def _free_vars_in_index(w: Write) -> frozenset[str]:
    """Union of all axis Vars referenced by the Write's index."""
    out: set[str] = set()
    for e in w.index:
        out |= e.free_vars()
    return frozenset(out)


__all__ = ["EscapeAnalysis", "analyze"]
