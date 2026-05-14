"""Search-tree data classes — one ``Candidate`` per point in the
autotune space, plus its cursor / per-step result records.

A ``Candidate`` is what the engine pops, advances by one rule
application, and pushes back. Forks at multi-option rewrite points
share a single deep-copied snapshot of the parent's graph and
materialize lazily on first ``.graph`` access — see :class:`ForkOrigin`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph

if TYPE_CHECKING:
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.pattern import Match


@dataclass
class Cursor:
    """Pipeline resume state for a ``Candidate``.

    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass.

    The advance/wrap/pass-step logic lives in :meth:`Candidate.apply`
    (driven by ``match.is_last``) so eager and lazy candidates
    transition through the cursor states the same way."""

    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

@dataclass(frozen=True)
class ForkOrigin:
    """Lazy-fork recipe. Materializing a Candidate from this means:
    deep-copy ``parent_snapshot``, build a fresh :class:`Match` against
    the copy, apply ``option`` to it, return the resulting graph.

    Multiple sibling forks share the same ``parent_snapshot`` via Python
    references — refcount frees it when the last lazy sibling has been
    materialized or dropped from the search queue. ``match`` carries
    everything the rule application needs: the rewrite site (read by
    ``TuningSearch`` to derive its tip key) and the pipeline location
    (``pass_idx`` / ``pass_name``, replayed back to ``on_apply`` when
    the fork materializes so the diff/dump rendering reports the
    *producer's* pass, not whichever pass the consumer is on by the
    time it pops the fork)."""

    parent_snapshot: Graph
    rule_name: str
    match: Match
    option: Op | Graph


ApplyCallback = Callable[["Graph", str, "Match", "Op | Graph"], None]
PassFinishCallback = Callable[[int, str, Graph], None]


@dataclass
class Candidate:
    """A single point in the search space. The engine pops a candidate,
    advances it by one rule application attempt, pushes the resulting
    successor(s) back onto the search queue, and yields the candidate
    when ``cursor.pass_idx`` reaches the end of the pipeline.

    ``graph`` is owned by this candidate. The active rollout's
    Candidate holds a concrete ``_graph``; forks at multi-option
    rewrite points are *lazy* — they hold an ``_origin`` (see
    :class:`ForkOrigin`) and materialize on first ``.graph`` access.

    ``ctx`` is shared by reference. ``cursor`` is the pipeline cursor.

    ``on_apply`` is a per-batch hook fired from inside :meth:`apply`
    just before the rewrite mutates ``graph``. The engine sets it once
    per rule batch (shared by ``cand`` and its sibling forks) so the
    debug/dump rendering of "this rewrite happened" lives next to the
    apply itself — the search loop doesn't have to log a fork's apply
    that hasn't materialized yet."""

    ctx: Context
    cursor: Cursor = field(default_factory=Cursor)
    on_apply: ApplyCallback | None = None
    on_pass_finish: PassFinishCallback | None = None
    _graph: Graph | None = field(default=None, repr=False)
    _origin: ForkOrigin | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        assert (self._graph is None) ^ (self._origin is None), "Candidate must have exactly one of _graph (concrete) or _origin (lazy)"

    @property
    def graph(self) -> Graph:
        """Materialize the graph on demand. Lazy forks copy the shared
        ``parent_snapshot`` once and apply their stored option in
        place; subsequent accesses return the cached result."""
        if self._graph is None:
            origin = self._origin
            assert origin is not None  # invariant enforced by __post_init__
            self._graph = origin.parent_snapshot.copy()
            self._origin = None  # drop the shared snapshot reference; refcount-GC kicks in
            self.apply(origin.rule_name, origin.match.remap(self._graph), origin.option)
        return self._graph

    @graph.setter
    def graph(self, g: Graph) -> None:
        self._graph = g
        self._origin = None

    def apply(self, rule_name: str, match: Match, option: Op | Graph | None) -> None:
        """Apply one rewrite outcome to ``self.graph`` in place, then
        update ``cursor`` for the rewrite kind and (when
        ``match.is_last``) for the rule-batch boundary.

        Per-rewrite cursor accounting: functional ``Graph`` splices
        bump ``n_applied`` so the end-of-pass logic knows the scan must
        restart; in-place ``Op`` rebinds bump nothing (the rule's
        idempotence guard handles re-firing).

        Per-batch cursor accounting (only when ``match.is_last``):
        advance ``rule_idx`` by 1, and when it reaches
        ``match.n_rules`` either restart the scan (if any functional
        rewrites accumulated) or fire ``on_pass_finish`` and step to
        the next pass. Owning the advance here keeps eager driver
        applies and lazy fork-materialization applies on the same code
        path — a fork's cursor moves forward identically to the
        cand's, regardless of when the materialization happens.

        ``Op`` rebinds ``root.op`` (id, inputs, hints kept); ``Graph``
        is a fragment spliced via ``Graph.splice``. On the ``Op`` path
        the chain ``Op.source`` is stamped with the op being replaced
        (unless the rule already set it) and the predecessor's
        ``knobs`` are merged forward — so the rewrite chain threads
        through every in-place rebind for free.

        Fires ``on_apply(graph_before, rule_name, match, option)``
        first when set, so the engine's debug/dump rendering sees the
        pre-rewrite graph. ``match.pass_idx`` / ``match.pass_name``
        carry the producer's pass location (which may differ from
        ``self.cursor.pass_idx`` for a lazily-materialized fork)."""
        # Local imports to break the cycle with ``compiler.ir.base``.
        from deplodock.compiler.ir.base import Op as _Op  # noqa: PLC0415

        if option is not None:
            graph = self.graph
            if self.on_apply is not None:
                self.on_apply(graph, rule_name, match, option)
            if isinstance(option, _Op):
                old_op = graph.nodes[match.root_node_id].op
                if option is not old_op and option.source is None:
                    option.source = old_op
                    option.knobs = {**old_op.knobs, **option.knobs}
                graph.nodes[match.root_node_id].op = option
            else:
                assert isinstance(option, Graph), f"expected Graph, Op, list, or RuleSkipped; got {type(option).__name__}"
                graph.splice(option, consumed=match.consumed, output=match.output or match.root_node_id)
                self.cursor.n_applied += 1
        if match.is_last:
            self._advance_batch(match.n_rules, match.pass_idx, match.pass_name)

    def _advance_batch(self, n_rules: int, pass_idx: int | None, pass_name: str | None) -> None:
        """Move ``cursor.rule_idx`` past the just-finished rule batch.
        Wraps to the next pass (firing ``on_pass_finish``) when the
        scan completes with no functional rewrites; otherwise restarts
        the scan from rule 0 to apply newly-spawned matches."""
        cur = self.cursor
        cur.rule_idx += 1
        if cur.rule_idx < n_rules:
            return
        finished = cur.n_applied == 0
        cur.rule_idx = 0
        cur.n_applied = 0
        if finished:
            if self.on_pass_finish is not None and pass_idx is not None and pass_name is not None:
                self.on_pass_finish(pass_idx, pass_name, self.graph)
            cur.pass_idx += 1
