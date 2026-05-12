"""Search-tree data classes — one ``Candidate`` per point in the
autotune space, plus its cursor / trace / per-step result records.

A ``Candidate`` is what the engine pops, advances by one rule
application, and pushes back. Forks at multi-option rewrite points
clone the candidate (deep-copy of ``graph``, shared ``ctx``) and place
the alternative on the search queue.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph


@dataclass(frozen=True)
class TraceEntry:
    """One rule application in a candidate's history. ``choice_idx`` is
    the option picked at fork points (0 for deterministic single-option
    rules)."""

    rule_name: str
    choice_idx: int = 0


@dataclass
class Cursor:
    """Pipeline resume state for a ``Candidate``.

    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass."""

    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    def advance(
        self,
        result: RuleResult,
        n_rules: int,
        on_pass_finish: Callable[[int], None] | None = None,
    ) -> None:
        """Drive the cursor forward by one rule attempt.

        First, update ``n_applied`` (functional fires drive end-of-scan
        restart logic) and ``rule_idx`` (only advanced when no
        functional fire happened — in-place rebinds and zero-fire
        batches stay on the same graph state, so re-scanning the rule
        would loop or no-op). Then, when ``rule_idx`` reaches
        ``n_rules``, transition: restart from rule 0 if functional
        rewrites accumulated this scan, otherwise invoke ``on_pass_end``
        with the just-finished ``pass_idx`` and advance to the next
        pass."""
        self.n_applied += result.n_functional
        if result.n_functional == 0:
            self.rule_idx += 1
        if self.rule_idx < n_rules:
            return
        finished = self.n_applied == 0
        self.rule_idx = 0
        self.n_applied = 0
        if finished:
            if on_pass_finish is not None:
                on_pass_finish(self.pass_idx)
            self.pass_idx += 1

    def fork(self, n_applied_delta: int) -> Cursor:
        """A copy at the same ``(pass_idx, rule_idx)`` with ``n_applied``
        shifted by ``n_applied_delta`` — used when spawning autotune
        alternatives mid-batch."""
        return Cursor(self.pass_idx, self.rule_idx, self.n_applied + n_applied_delta)


@dataclass
class RuleResult:
    """Outcome of one rule-application iteration.

    * ``forks`` — alternative candidates spawned at autotune fork points
      (empty for deterministic rules).
    * ``n_functional`` — count of ``Graph`` (functional) rewrites applied
      to the candidate's own graph in this batch.
    * ``n_inplace`` — count of ``Op`` (in-place rebind) rewrites applied
      to the candidate's own graph in this batch."""

    forks: list[Candidate] = field(default_factory=list)
    n_functional: int = 0
    n_inplace: int = 0

    @property
    def fired(self) -> bool:
        return (self.n_functional + self.n_inplace) > 0


@dataclass
class Candidate:
    """A single point in the search space. The engine pops a candidate,
    advances it by one rule application attempt, pushes the resulting
    successor(s) back onto the search queue, and yields the candidate
    when ``cursor.pass_idx`` reaches the end of the pipeline.

    ``graph`` is owned by this candidate (deep-copied on multi-option
    forks; mutated in place on single-option steps). ``ctx`` is shared
    by reference. ``trace`` is the immutable history of rule
    applications on this branch. ``cursor`` is the pipeline cursor."""

    graph: Graph
    ctx: Context
    trace: tuple[TraceEntry, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
