"""Search-tree data classes — :class:`Candidate` (concrete graph state)
and :class:`LazyCandidate` (a parent + a chain of pending applications
that materializes via :meth:`resolve`).

Sibling forks at multi-option rewrite points share a single ``inner``
``Candidate`` (the parent's snapshot) by reference; each fork holds its
own one-element chain. ``resolve()`` is the single entry point that
turns a lazy candidate into a concrete one — copy the inner's graph
once, replay the chain of ``(match, option)`` pairs through
``Candidate.apply``, drop the chain.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph

# Use the engine logger so the existing debug-emit toggles (rule-
# skipped lines under ``compile -vv``) keep working without callers
# having to also bump this module's level.
_logger = logging.getLogger("deplodock.compiler.pipeline.engine")

if TYPE_CHECKING:
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.engine import Match, Pass


@dataclass
class Cursor:
    """Pipeline resume state for a candidate.

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


ApplyCallback = Callable[["Graph", "Match", "Op | Graph"], None]
PassFinishCallback = Callable[["Pass", Graph], None]


@dataclass
class Candidate:
    """A concrete point in the search space — owns a real ``graph``.

    ``ctx`` is shared by reference across siblings. ``cursor`` tracks
    pipeline resume state. ``on_apply`` is fired from inside
    :meth:`apply` just before each rewrite mutates ``graph`` (the
    engine installs it on the root candidate; forks inherit).
    ``on_pass_finish`` fires from inside :meth:`apply` when the rule
    batch ending the pass is applied.

    :class:`LazyCandidate` is the deferred-apply counterpart used for
    autotune fork siblings; both expose :meth:`resolve` so the search
    loop can treat them uniformly."""

    ctx: Context
    graph: Graph
    cursor: Cursor = field(default_factory=Cursor)
    on_apply: ApplyCallback | None = None
    on_pass_finish: PassFinishCallback | None = None

    def resolve(self) -> Candidate:
        """Identity — already concrete. Provided so callers can resolve
        any candidate uniformly."""
        return self

    def lazy(self) -> LazyCandidate:
        """Wrap in a zero-chain :class:`LazyCandidate`. The search
        layer always handles ``LazyCandidate``; this helper lifts a
        concrete cand back into that interface (e.g. before pushing
        the rollout's current cand back to ``Search.push``)."""
        return LazyCandidate(inner=self, cursor=self.cursor, chain=[])

    def try_rewrite(self, match: Match) -> list[Op | Graph] | None:
        """Eager mode (called by the search loop): invoke
        ``match.rule.rewrite`` against this candidate's graph,
        validate the result, and either apply the single chosen
        option or — for a multi-option fork — return the option list
        for the caller to spawn ``LazyCandidate`` siblings from.
        Returns ``None`` when no rewrite was applied (``RuleSkipped``,
        empty options after validation, or single option applied
        successfully).

        Cursor advance is unconditional on ``match.is_last`` — even
        when the rewrite skipped or produced no valid option — so the
        search loop terminates on quiescent batches where every match
        is skipped by the rule's own idempotence guard. The
        multi-option return path is the one exception: the cursor
        advance is left to the eventual fork's apply on resolve."""
        from deplodock.compiler.ir.base import Op as _Op  # noqa: PLC0415
        from deplodock.compiler.pipeline.engine.driver import _build_rewrite_kwargs  # noqa: PLC0415
        from deplodock.compiler.pipeline.engine.pipeline import RuleSkipped  # noqa: PLC0415
        from deplodock.compiler.pipeline.rule_diff import display_name, emit, format_skipped  # noqa: PLC0415

        if not match.is_alive():
            # Earlier applies in this batch invalidated the match's
            # consumed nodes. Skip the rewrite, but still advance the
            # cursor when this was the last match — otherwise the
            # search loop would re-pop the same rule batch forever.
            self._advance_if_last(match)
            return None
        rule = match.rule
        assert rule is not None, "Match.rule must be set before try_rewrite"
        try:
            result = rule.rewrite(**_build_rewrite_kwargs(rule, match, self.ctx))
        except RuleSkipped as exc:
            if _logger.isEnabledFor(logging.DEBUG):
                emit(format_skipped(display_name(match.pass_name, match.rule_name or ""), match.root_node_id, exc.reason))
            self._advance_if_last(match)
            return None
        options = list(result) if isinstance(result, (list, tuple)) else [result]
        options = [o for o in options if not isinstance(o, _Op) or o.validate(self.ctx)]
        if not options:
            self._advance_if_last(match)
            return None
        if len(options) > 1:
            # Defer to a fork — caller spawns ``LazyCandidate``
            # siblings. Cursor advance happens via the fork's apply on
            # resolve, so we leave it alone here.
            return options
        self.apply(match, options[0])
        return None

    def apply(self, match: Match, option: Op | Graph) -> None:
        """Lazy mode (called by ``LazyCandidate.resolve`` and
        internally by :meth:`try_rewrite` for single-option matches):
        apply the specific ``option`` to this candidate's graph.
        Mutates the graph, fires ``on_apply``, bumps cursor
        ``n_applied`` for functional splices, and advances the rule-
        batch cursor when ``match.is_last``.

        ``Op`` rebinds ``root.op`` (id / inputs / hints kept);
        ``Graph`` is a fragment spliced via ``Graph.splice``. On the
        ``Op`` path the chain ``Op.source`` is stamped with the op
        being replaced (unless the rule already set it) and the
        predecessor's ``knobs`` are merged forward — so the rewrite
        chain threads through every in-place rebind for free."""
        from deplodock.compiler.ir.base import Op as _Op  # noqa: PLC0415

        if self.on_apply is not None:
            self.on_apply(self.graph, match, option)
        if isinstance(option, _Op):
            old_op = self.graph.nodes[match.root_node_id].op
            if option is not old_op and option.source is None:
                option.source = old_op
                option.knobs = {**old_op.knobs, **option.knobs}
            self.graph.nodes[match.root_node_id].op = option
        else:
            assert isinstance(option, Graph), f"expected Graph or Op; got {type(option).__name__}"
            self.graph.splice(option, consumed=match.consumed, output=match.output or match.root_node_id)
            self.cursor.n_applied += 1
        self._advance_if_last(match)

    def _advance_if_last(self, match: Match) -> None:
        if match.is_last:
            self._advance_batch(match.pass_)

    def _advance_batch(self, pass_: Pass | None) -> None:
        """Move ``cursor.rule_idx`` past the just-finished rule batch.
        Wraps to the next pass (firing ``on_pass_finish``) when the
        scan completes with no functional rewrites; otherwise restarts
        the scan from rule 0 to apply newly-spawned matches."""
        cur = self.cursor
        cur.rule_idx += 1
        n_rules = len(pass_.rules) if pass_ is not None else 0
        if cur.rule_idx < n_rules:
            return
        finished = cur.n_applied == 0
        cur.rule_idx = 0
        cur.n_applied = 0
        if finished:
            if self.on_pass_finish is not None and pass_ is not None and pass_.name:
                self.on_pass_finish(pass_, self.graph)
            cur.pass_idx += 1


@dataclass
class LazyCandidate:
    """Deferred-apply counterpart of :class:`Candidate`. Holds a parent
    ``inner`` Candidate (whose ``graph`` is the snapshot to clone from
    and whose ``ctx`` / ``on_apply`` / ``on_pass_finish`` propagate
    onto the resolved Candidate) and a ``chain`` of ``(match, option)``
    pairs to replay on resolve.

    Sibling forks at the same rewrite point share ``inner`` by
    reference — only one snapshot is ever held in memory per fork
    point. Each fork's chain is its own short list (typically a single
    pair carrying the alt option for that fork's match site).

    ``cursor`` is the lazy candidate's own pipeline cursor (typically a
    copy of the parent's cursor at fork-creation time)."""

    inner: Candidate
    cursor: Cursor
    chain: list[tuple[Match, Op | Graph | None]]

    def resolve(self) -> Candidate:
        """Materialize: copy ``inner.graph``, build a fresh Candidate
        carrying our cursor and the inner's callbacks, replay the
        chain through its ``apply``, drop the chain so a second resolve
        is a no-op (returns the cached resolved Candidate). Multiple
        sibling ``LazyCandidate`` instances pointing at the same
        ``inner`` each get their own copy — the snapshot is shared
        only across siblings, not across resolve calls."""
        if not self.chain:
            return self.inner
        resolved = Candidate(
            ctx=self.inner.ctx,
            graph=self.inner.graph.copy(),
            cursor=self.cursor,
            on_apply=self.inner.on_apply,
            on_pass_finish=self.inner.on_pass_finish,
        )
        for match, option in self.chain:
            resolved.apply(match.remap(resolved.graph), option)
        self.chain = []
        self.inner = resolved
        return resolved
