"""``FallbackPrior`` — composes the learned + analytic priors into the single
ranking surface the search consumes.

``score`` / ``mean_score`` use the learned :class:`CatBoostPrior` once it's
``fitted`` and fall back to the :class:`AnalyticPrior` (cold-start heuristic)
otherwise — so the policies always get a usable ranking and no longer special-
case "cold → emission order". Everything else (training: ``add_rows`` /
``maybe_refit`` / ``checkpoint`` / ``fit`` / ``to_json``, and inspection:
``_dataset`` / ``trajectory`` / ``summary`` for diagnostics) delegates to the
learned half, so ``tune`` trains and checkpoints CatBoost exactly as before and
``fitted`` reflects whether a *trained* model exists.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.prior.analytic import AnalyticPrior
from deplodock.compiler.pipeline.search.prior.base import Prior
from deplodock.compiler.pipeline.search.prior.catboost import CatBoostPrior


class FallbackPrior(Prior):
    """Learned prior with an analytic cold-start fallback. Not a dataset owner —
    its training / inspection surface is the ``learned`` prior's; only ``score`` /
    ``mean_score`` blend in the analytic fallback."""

    def __init__(self, learned: Prior, analytic: Prior | None = None) -> None:
        # Deliberately NOT calling super().__init__() — this prior holds no
        # dataset of its own; every stateful attribute delegates to ``learned``
        # (see __getattr__), so there's no second reservoir to diverge.
        self.learned = learned
        self.analytic = analytic if analytic is not None else AnalyticPrior()

    @property
    def fitted(self) -> bool:
        # Reflects the LEARNED model (so diagnostics / `eval prior` report whether
        # real tuning data exists). The policies no longer gate on this — they
        # always call score()/mean_score(), which fall back to analytic when cold.
        return self.learned.fitted

    def score(self, knobs: dict) -> float:
        return self.learned.score(knobs) if self.learned.fitted else self.analytic.score(knobs)

    def mean_score(self, knobs: dict) -> float:
        return self.learned.mean_score(knobs) if self.learned.fitted else self.analytic.mean_score(knobs)

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        return self.learned.mean_scores(knobs_list) if self.learned.fitted else self.analytic.mean_scores(knobs_list)

    # --- training + inspection: delegate to the learned half ------------------
    def fit(self) -> None:
        self.learned.fit()

    def add_rows(self, rows) -> None:
        self.learned.add_rows(rows)

    def maybe_refit(self, *, force: bool = False) -> bool:
        return self.learned.maybe_refit(force=force)

    def checkpoint(self) -> None:
        self.learned.checkpoint()

    def to_json(self) -> dict | None:
        return self.learned.to_json()

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        self.learned.record_bench(knobs, median, status)

    def summary(self, label: str) -> str:
        return self.learned.summary(label)

    def __getattr__(self, name: str):
        # Read-through for anything not defined here (``_dataset`` / ``trajectory``
        # / ``_path`` / counters that diagnostics + offline refit inspect).
        # ``__getattr__`` only fires for genuinely-missing attributes, so the
        # explicit overrides above always win.
        return getattr(self.learned, name)


def load_prior(*, seed: int = 0, path=None) -> FallbackPrior:
    """The one global prior the search loads: the learned ``CatBoostPrior``
    (warm-started from its checkpoint if present) wrapped with an
    ``AnalyticPrior`` cold-start fallback."""
    return FallbackPrior(CatBoostPrior.load(seed=seed, path=path), AnalyticPrior())
