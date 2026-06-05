"""``Search`` ABC shared by the concrete policies.

The engine ↔ policy protocol is TOKEN-THREADED: every ``pop`` hands back
an opaque token alongside the candidate, and the engine passes that token
back explicitly — ``push(parent=token)`` to attribute spawned children to
their fork point, ``observe(token, ...)`` to attribute a terminal's
measurement. Lineage never depends on call order (no "most recently
popped" hidden state), so the policy's bookkeeping stays correct even if
the engine interleaves pops, pushes, and observes.

- ``pop`` returns ``(token, candidate)`` (``None`` ends the run);
- ``push`` enqueues spawned candidates (unranked fork siblings) under
  ``parent`` — the token of the pop that produced them, or ``None`` for
  the seed candidate that starts a run;
- ``score_of`` is the value-keyed scorer the policies rank with — the
  search owns the cache, the ``Fork`` carries the compute;
- ``observe`` is called by :meth:`Pipeline.tune` once per yielded
  terminal, with the terminal's token and the aggregated ``reward``
  (``1/total_us``) + ``status`` from the bench. Default is a no-op;
  :class:`TuningSearch` overrides it to backprop on its MCTS tree.

Tokens are minted by the policy and opaque to the engine:
:class:`TuningSearch` (the only policy) uses its ``SearchNode`` as the token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats


class Search(ABC):
    def __init__(self, *, score_cache: dict[Hashable, float] | None = None) -> None:
        # Value-keyed variant-score cache, handed to every ``Fork.score``
        # compute. Owned by the search because scoring is ranking policy:
        # one cache per search run shares scores across every fork point
        # it ranks. The KEYING is the scorer's own (it knows its value
        # identity — the partition planner keys ``(ctx, merged knobs)``,
        # complete because the ``SID`` structural-identity knob rides the
        # dict), so structurally identical kernels (the same layer
        # repeated through a whole model) hit the same entries. Pass a
        # shared dict to pool entries across several searches (see
        # ``two_level.inner_reward``, which tunes per-op slices with one
        # fresh search each).
        self.score_cache: dict[Hashable, float] = score_cache if score_cache is not None else {}

    def score_of(self, fork: Fork | None) -> float:
        """The fork's planner prior — ``fork.score(self.score_cache)``;
        ``None`` (a no-pending wrapper) scores ``0.0``.

        NOTE: the policies no longer rank with this — the static prior was
        nuked from selection in favor of the learned
        :class:`~deplodock.compiler.pipeline.search.prior.Prior`. This
        accessor is retained as the lazy-planner score *compute* (exercised
        directly by the partition-planner tests and available as latent
        infra); it never fires ``fork.expand()``, so reading it stays cheap."""
        return fork.score(self.score_cache) if fork is not None else 0.0

    @abstractmethod
    def push(self, *cands: LazyCandidate, parent: object | None = None) -> None:
        """Enqueue the spawned candidates — all siblings of one fork
        point, in rule-emission order.

        ``parent`` is the token of the :meth:`pop` whose candidate
        spawned these siblings, or ``None`` for the seed candidate that
        starts a run. Selection is the policy's job: :class:`TuningSearch`
        ranks the frontier by PUCT over its learned prior (a single-shot
        compile, prior absent, descends emission-order)."""

    @abstractmethod
    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        """Return ``(token, candidate)`` for the next candidate to
        explore, or ``None`` to end the run. The engine threads the
        token back verbatim into :meth:`push` (as ``parent``) and
        :meth:`observe`."""

    def observe(self, token: object | None, stats: PerfStats, status: str, candidate: object | None = None) -> None:  # noqa: B027
        """Hook for the policy to consume a terminal's measurement.
        ``token`` is the one the terminal was popped with; ``stats``
        aggregates its per-iter latencies in microseconds (median /
        mean / variance / etc.); ``status`` is ``"ok"`` or
        ``"bench_fail"``. ``candidate`` is the *resolved* terminal Candidate —
        its ``graph`` carries the realized op ``knobs`` (the full config,
        including knobs stamped at deterministic lowering steps that never
        appear as forks). Default no-op; :class:`TuningSearch` overrides it."""
