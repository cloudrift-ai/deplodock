"""Escape analysis on Tile-IR bodies.

Recovers coordination decisions (cooperative reduce, atomic write,
broadcast-write guard) from the structural shape of a ``TileOp`` body,
without depending on the planner-emitted tags
(``ThreadTile.cooperative_axes`` / ``GridTile.splitk_axes``) or the
marker stmts (``Combine``, ``Write.reduce_op``) that today's
``001_coordination`` pass produces.

Why: those tags and stmts are derivable from data flow. Keeping them in
the IR introduces a class of bugs (out-of-sync tag vs. body) and forces
a separate coordination pass to interpret them. This helper computes the
same answers by walking the body, so the materializer can consume the
result directly and the tags / stmts / pass can be deleted.

See ``plans/derive-coordination-from-body.md`` for the full refactor
plan.

Three queries are exposed via :class:`EscapeAnalysis`:

- ``accum_cooperative_axes[name]`` — for each ``Accum.name``, the set of
  enclosing ``ThreadTile`` axis names through which the accumulated
  value escapes its reduce loop. Non-empty ⇒ a cross-thread combine
  over those axes is required at the escape point.
- ``write_atomic_axes[write]`` — for each ``Write``, the set of
  enclosing ``GridTile`` axis names NOT referenced by the Write's
  index. Non-empty ⇒ multiple CTAs race ⇒ ``atomicAdd`` required.
- ``write_broadcast_axes[write]`` — for each ``Write``, the set of
  enclosing thread axes that are functionally cooperative (some Accum
  feeding the Write escapes through them) AND NOT referenced by the
  Write's index. Non-empty ⇒ ``Cond(axis == 0)`` guard required so only
  one thread of the cooperative group performs the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt import Accum, Assign, Body
from deplodock.compiler.ir.stmt.leaves import Write
from deplodock.compiler.ir.tile.ir import (
    GridTile,
    SerialTileBase,
    Stmt,
    ThreadTile,
    TileOp,
)


@dataclass(frozen=True)
class EscapeAnalysis:
    """Per-stmt coordination metadata derived from data flow on a ``TileOp``.

    ``accum_cooperative_axes`` is keyed by ``Accum.name`` (an SSA name,
    hashable). ``Write`` keys use ``id(...)`` internally because the IR's
    ``Write.index`` may hold ``BinaryExpr`` nodes that aren't hashable —
    use the :meth:`atomic_axes` / :meth:`broadcast_axes` accessors
    rather than touching ``_write_atomic_axes`` directly. ``writes``
    holds the analyzed Writes in body-walk order so callers can iterate
    deterministically.
    """

    accum_cooperative_axes: dict[str, frozenset[str]] = field(default_factory=dict)
    writes: tuple[Write, ...] = field(default_factory=tuple)
    _write_atomic_axes: dict[int, frozenset[str]] = field(default_factory=dict)
    _write_broadcast_axes: dict[int, frozenset[str]] = field(default_factory=dict)

    @property
    def cooperative_thread_axes(self) -> frozenset[str]:
        """Union of every thread axis any Accum escapes through. These
        are the functionally-cooperative axes of the analyzed TileOp."""
        out: set[str] = set()
        for axes in self.accum_cooperative_axes.values():
            out.update(axes)
        return frozenset(out)

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
    assigns: list[tuple[Assign, tuple[Stmt, ...]]] = []
    _collect(body, (), accums, writes, assigns)

    # Build def-use over SSA Assigns so we can ask "does this Write's
    # value transitively depend on Accum X?" — needed for the broadcast-
    # guard check (Rule 4) and the bias-decomposition heuristic.
    assign_deps = {a.name: tuple(a.args) for a, _ in assigns}
    accum_consumers = {acc.name: _names_reaching_into(acc.name, assign_deps) for acc, _ in accums}
    # Reverse: for each SSA name, which Accums it transitively depends on.
    accum_seeds_by_name: dict[str, set[str]] = {}
    for acc_name, reach in accum_consumers.items():
        for n in reach:
            accum_seeds_by_name.setdefault(n, set()).add(acc_name)
        accum_seeds_by_name.setdefault(acc_name, set()).add(acc_name)

    # --- Atomic-Write classification (Rule 2) ---
    write_atomic_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        block_axes = _block_axis_names_in_chain(chain)
        missing = block_axes - _free_vars_in_index(w)
        write_atomic_axes[id(w)] = frozenset(missing)

    # --- Accum cooperativity (Rule 1) ---
    # For each Accum, find the innermost enclosing reduce loop (the
    # SerialTile/StridedTile whose body directly holds the Accum). Then
    # find the enclosing ThreadTile axes above that loop. The Accum
    # escapes through axis t iff some consumer of acc.name (recursively
    # via Assigns and via Writes whose values depend on it) sits at a
    # position outside the reduce loop but still inside the ThreadTile
    # binding t. Cooperativity for t requires additionally that the
    # consumer Write (when applicable) doesn't reference t — if it does,
    # threads write distinct cells and the value is per-thread, not
    # cooperative.
    accum_cooperative_axes: dict[str, frozenset[str]] = {}
    for acc, acc_chain in accums:
        reduce_loop = _innermost_reduce_loop(acc_chain)
        enclosing_thread_axes = _thread_axes_above(acc_chain, reduce_loop)
        if not enclosing_thread_axes:
            accum_cooperative_axes[acc.name] = frozenset()
            continue
        coop: set[str] = set()
        # Inspect every Write whose value transitively depends on this
        # Accum, plus uses-via-Assign that escape the reduce loop. A
        # thread axis t is cooperative for this Accum iff some such
        # consumer escapes the reduce loop AND doesn't reference t in
        # its index (Writes) or its enclosing scope (Assigns escape
        # implicitly when they feed an unguarded Write).
        for w, w_chain in writes:
            if not _value_depends_on_accum(w, acc.name, accum_seeds_by_name):
                continue
            if reduce_loop is not None and reduce_loop in w_chain:
                continue  # consumer is still inside the reduce loop
            w_idx_vars = _free_vars_in_index(w)
            for t in enclosing_thread_axes:
                if t not in w_idx_vars:
                    coop.add(t)
        accum_cooperative_axes[acc.name] = frozenset(coop)

    coop_thread_axes: set[str] = set()
    for axes in accum_cooperative_axes.values():
        coop_thread_axes.update(axes)

    # --- Broadcast-Write classification (Rule 4) ---
    write_broadcast_axes: dict[int, frozenset[str]] = {}
    for w, chain in writes:
        thread_axes = _thread_axis_names_in_chain(chain)
        # Only cooperative thread axes drive a guard. If the axis is
        # output-partition (no Accum escapes through it), every thread
        # writes a distinct cell anyway.
        candidate_coop_axes = thread_axes & coop_thread_axes
        w_idx_vars = _free_vars_in_index(w)
        # Restrict further: only the cooperative axes shared with the
        # Accum chain that actually feeds this Write contribute. A Write
        # that doesn't depend on any cooperative Accum doesn't need a
        # guard even if its index happens to omit a coop axis.
        feeding_coops: set[str] = set()
        for acc_name, coop_for_acc in accum_cooperative_axes.items():
            if _value_depends_on_accum(w, acc_name, accum_seeds_by_name):
                feeding_coops.update(coop_for_acc)
        guard_axes = (candidate_coop_axes & feeding_coops) - w_idx_vars
        write_broadcast_axes[id(w)] = frozenset(guard_axes)

    return EscapeAnalysis(
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
    assigns: list[tuple[Assign, tuple[Stmt, ...]]],
) -> None:
    """Walk ``body`` recursively, recording each ``Accum`` / ``Write`` /
    ``Assign`` paired with its enclosing-stmt chain (root-first)."""
    for s in body:
        if isinstance(s, Accum):
            accums.append((s, chain))
        elif isinstance(s, Write):
            writes.append((s, chain))
        elif isinstance(s, Assign):
            assigns.append((s, chain))
        for sub in s.nested():
            _collect(sub, chain + (s,), accums, writes, assigns)


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


def _thread_axis_names_in_chain(chain: tuple[Stmt, ...]) -> frozenset[str]:
    """Set of axis names bound by every ``ThreadTile`` enclosing the leaf."""
    out: set[str] = set()
    for s in chain:
        if isinstance(s, ThreadTile):
            out.update(ax.name for ax in s.axes)
    return frozenset(out)


def _innermost_reduce_loop(chain: tuple[Stmt, ...]) -> Stmt | None:
    """Return the innermost ``SerialTileBase`` in ``chain`` whose body is
    a reduce (``is_reduce`` is True — i.e. contains an ``Accum``).
    ``None`` if no reduce loop encloses the leaf."""
    for s in reversed(chain):
        if isinstance(s, SerialTileBase) and s.is_reduce:
            return s
    return None


def _thread_axes_above(chain: tuple[Stmt, ...], anchor: Stmt | None) -> frozenset[str]:
    """Set of thread-axis names bound by ``ThreadTile`` ancestors at or
    above ``anchor`` in ``chain``. If ``anchor`` is ``None``, all
    ``ThreadTile`` axes in ``chain`` count."""
    out: set[str] = set()
    for s in chain:
        if isinstance(s, ThreadTile):
            out.update(ax.name for ax in s.axes)
        if anchor is not None and s is anchor:
            break
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


# ---------------------------------------------------------------------------
# Def-use over SSA Assigns
# ---------------------------------------------------------------------------


def _names_reaching_into(seed: str, assign_deps: dict[str, tuple[str, ...]]) -> frozenset[str]:
    """Forward def-use closure: every SSA name whose definition transitively
    depends on ``seed`` (via Assign-dependency edges)."""
    out: set[str] = set()
    stack = [seed]
    while stack:
        n = stack.pop()
        # Walk all assigns: if any uses n in its args, add the defined name.
        for defined, args in assign_deps.items():
            if defined in out:
                continue
            if n in args:
                out.add(defined)
                stack.append(defined)
    return frozenset(out)


def _value_depends_on_accum(
    w: Write,
    accum_name: str,
    accum_seeds_by_name: dict[str, set[str]],
) -> bool:
    """True iff any ``Write.values`` SSA name has ``accum_name`` among
    its transitive Accum sources."""
    for v in w.values:
        if accum_name in accum_seeds_by_name.get(v, set()):
            return True
    return False


__all__ = ["EscapeAnalysis", "analyze"]
