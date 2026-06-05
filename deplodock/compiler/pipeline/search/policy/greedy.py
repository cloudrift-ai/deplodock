"""Single-shot compile driver — stops at the first terminal candidate.

This is NOT an exploration policy (the learned prior / PUCT in
:class:`TuningSearch` is the only one of those). It is the deterministic
``Pipeline.run`` driver for ``compile`` / ``run`` and the assembled-graph
lowering: it emits a single option-0 lowering with **O(1) work per step**.

It deliberately keeps *no* MCTS tree. Routing a whole-model compile through
``TuningSearch`` instead is O(N²) in the number of rule applications — its
``pop`` re-descends from the root each call — so a transformer-block compile
effectively hangs. A single pending slot avoids that entirely."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.policy.base import Search


class GreedySearch(Search):
    """Stop at the first terminal candidate.

    ``push`` keeps exactly one candidate per fork point and drops the
    rest, so only a single pending slot is ever needed. When the engine
    yields a terminal without pushing it back, the next ``pop`` finds
    the slot empty and returns ``None``, ending the search.

    Keeps the **first** emitted sibling (the rule's option-0 default) — there
    is no prior in single-shot compile (no benches to fit one). Wiring a
    warm-started prior into compile is a deferred follow-up."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None

    def push(self, *cands: LazyCandidate, parent: object | None = None) -> None:
        del parent  # greedy keeps no lineage — one pending slot, no tree
        self._pending = cands[0] if cands else None

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None
