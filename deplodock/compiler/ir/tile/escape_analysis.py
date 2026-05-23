"""Escape analysis on Tile-IR bodies — derives coordination decisions.

Computes three queries from a ``TileOp`` body:

- ``atomic_axes(write)`` — enclosing ``GridTile`` axis names NOT in the
  Write's index. Non-empty ⇒ multiple CTAs race ⇒ ``atomicAdd``. Purely
  structural — derivable from Write index vs. enclosing block axes.
- ``broadcast_axes(write)`` — enclosing cooperative thread axes NOT in
  the Write's index. Non-empty ⇒ ``Cond(axis == 0)`` guard required.
- ``accum_cooperative_axes[name]`` — for each ``Accum.name``, the set of
  cooperative thread axes whose loop the Accum is updated inside and
  whose escape requires a cross-thread ``Combine``.

**Cooperativity is taken from ``ThreadTile.cooperative_axes`` directly**
— not derived from data flow. After the partition planner emits a body,
the cooperative-stride pattern lives in the load *index expression*
(``x[..., a2*512 + a3*256 + a1]``) rather than in any loop kind, so a
structural rule like "StridedLoop bound to thread axis ⇒ cooperative"
misses BR=K degenerate kernels and post-staging shapes. The planner's
tag is the structural source of truth; the helper treats it as input
and derives only the operational consequences (Combine emission,
atomic stamping, Cond-guard placement).

This frames ``001_coordination``'s remaining responsibility as
*emitting* the operational markers (Combine / Write.reduce_op / Cond
wrappers) so the materializer can stay mechanical. The intent of the
larger refactor (``plans/derive-coordination-from-body.md``) is to
delete those markers in favor of materializer-side queries against this
helper — see the plan for the milestone breakdown.
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

    # --- Atomic-Write classification ---
    # Pure index analysis: any enclosing GridTile axis missing from a
    # Write's index means CTAs race ⇒ atomic.
    write_atomic_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        block_axes = _block_axis_names_in_chain(chain)
        missing = block_axes - _free_vars_in_index(w)
        write_atomic_axes[id(w)] = frozenset(missing)

    # --- Cooperative axes per Accum ---
    # Read from the enclosing ThreadTile's planner-emitted tag rather
    # than try to derive. See module docstring for why derivation is
    # unreliable post-staging.
    accum_cooperative_axes: dict[str, frozenset[str]] = {}
    for acc, chain in accums:
        accum_cooperative_axes[acc.name] = _enclosing_cooperative_axes(chain)

    cooperative_thread_axes = frozenset().union(*accum_cooperative_axes.values()) if accum_cooperative_axes else frozenset()

    # --- Broadcast-Write classification ---
    # A Write needs a Cond-guard for axis t iff t is in the enclosing
    # ThreadTile.cooperative_axes AND t is not referenced by the Write's
    # index. This conservatively guards any write that sits at
    # cooperative scope without using the cooperative thread var — same
    # rule ``001_coordination``'s ``_guard_scalar_write`` applies.
    write_broadcast_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        coop_axes = _enclosing_cooperative_axes(chain)
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


def _enclosing_cooperative_axes(chain: tuple[Stmt, ...]) -> frozenset[str]:
    """Union of ``ThreadTile.cooperative_axes`` across every ``ThreadTile``
    enclosing the leaf. This is the planner's structural source of truth
    for cooperativity — see module docstring."""
    out: set[str] = set()
    for s in chain:
        if isinstance(s, ThreadTile):
            out.update(s.cooperative_axes)
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


def cross_check_against_tags(tile_op: TileOp) -> None:
    """Soundness check used during the M2 migration: assert that the
    helper's predictions agree with the planner-emitted coordination
    tags (``GridTile.splitk_axes``) and the coordination-pass-emitted
    leaf markers (``Write.reduce_op``) on a single post-coordination
    ``TileOp``.

    Raises ``AssertionError`` on disagreement — used by the materializer
    to fail loudly if any kernel in the test zoo would diverge. Will be
    deleted in M3 when the materializer switches to the helper as the
    sole source of truth and the markers are removed.

    Cooperativity isn't cross-checked here: the helper reads
    ``ThreadTile.cooperative_axes`` directly (see module docstring), so
    they agree by construction.

    Checked invariants:

    1. Every ``GridTile.splitk_axes`` axis appears in the atomic-axis
       set of at least one Write inside that GridTile.
    2. Every ``Write`` with ``reduce_op != None`` is classified atomic
       by the helper (``atomic_axes`` non-empty).
    """
    result = analyze(tile_op)
    body = tile_op.body if isinstance(tile_op, TileOp) else tile_op

    # Invariant 1: every splitk axis is missing from at least one Write's index
    # somewhere in its scope.
    splitk_seen: dict[str, bool] = {}
    for s in body.iter():
        if isinstance(s, GridTile) and s.splitk_axes:
            for ax in s.splitk_axes:
                splitk_seen.setdefault(ax, False)
    for w in result.writes:
        for ax in result.atomic_axes(w):
            if ax in splitk_seen:
                splitk_seen[ax] = True
    for ax, seen in splitk_seen.items():
        assert seen, (
            f"GridTile.splitk_axes contains {ax!r} but no enclosed Write classifies it as atomic. "
            f"Helper missed a Write or the tag is stale."
        )

    # Invariant 2: every reduce_op Write is classified atomic.
    for w in result.writes:
        if w.reduce_op is not None:
            assert result.atomic_axes(w), (
                f"Write(output={w.output!r}) has reduce_op={w.reduce_op.name!r} but escape analysis "
                f"found no enclosing block axis missing from its index. Disagreement on atomicity."
            )


__all__ = ["EscapeAnalysis", "analyze", "cross_check_against_tags"]
