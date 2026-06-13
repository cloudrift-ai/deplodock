"""``FallbackPrior`` — composes the learned + analytic priors into the single
ranking surface the search consumes.

``mean_score`` / ``mean_scores`` / ``pick`` (the deploy + eval + diagnostics
surface) use the learned :class:`CatBoostPrior` once it's ``fitted`` and fall
back to the :class:`AnalyticPrior` (cold-start heuristic) otherwise — so the
policies always get a usable ranking and no longer special-case "cold → emission
order". :meth:`score` (the MCTS *selection* signal — see :mod:`policy.mcts`) is
the one surface that BLENDS the two even when fitted, so PUCT explores the region
the heuristic prices well but the data-poor learned model buries (the golden-sweep
fp16 finding). Everything else (training: ``add_rows`` / ``maybe_refit`` /
``checkpoint`` / ``fit`` / ``to_json``, and inspection: ``_dataset`` /
``trajectory`` / ``summary`` for diagnostics) delegates to the learned half, so
``tune`` trains and checkpoints CatBoost exactly as before and ``fitted`` reflects
whether a *trained* model exists.

The two priors are on **different scales**: CatBoost regresses ``log(latency µs)``
so its ``score`` is calibrated µs, whereas the analytic prior is fit by
learning-to-rank (``scripts/golden_knob_heuristics.py``) so its ``score`` —
``exp(-0.1·quality)`` — is an *ordinal* proxy with arbitrary magnitude (only its
order is meaningful; its neutral "no opinion" value is exactly ``1.0``). So
:meth:`score` keeps the learned µs as the scale and folds the analytic in as a
dimensionless **multiplier** ``analytic**W`` centered at that neutral 1.0 — the
learned half owns the per-shape µs anchor, the analytic half contributes only its
ranking. ``W`` (``config.analytic_tilt``) sizes the nudge: small enough to
perturb the learned order, not replace it.
"""

from __future__ import annotations

from deplodock import config
from deplodock.compiler.pipeline.search.prior.analytic import AnalyticPrior
from deplodock.compiler.pipeline.search.prior.base import Prior
from deplodock.compiler.pipeline.search.prior.catboost import CatBoostPrior


class FallbackPrior(Prior):
    """Learned prior with an analytic cold-start fallback. Not a dataset owner —
    its training / inspection surface is the ``learned`` prior's. ``mean_score`` /
    ``mean_scores`` fall back to analytic only while cold; ``score`` (the MCTS
    selection signal) blends the analytic multiplier in even when fitted."""

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
        # MCTS-selection signal ONLY (deploy/eval go through mean_score/pick).
        # Cold: the analytic prior IS the ranking. Fitted: keep the learned µs as
        # the scale and tilt it by the analytic's dimensionless ranking multiplier
        # (``analytic**W``, neutral 1.0) so PUCT still explores a region the
        # heuristic prices well but the learned model — having never measured it —
        # buries (the fp16 small-BK warp tiles at large squares). ``W=0`` recovers
        # pure-learned selection. Scale-correct because the analytic factor is
        # centered at its no-opinion 1.0, so a config the heuristic has no view on
        # leaves the learned prediction untouched.
        if not self.learned.fitted:
            return self.analytic.score(knobs)
        w = config.analytic_tilt()
        learned = self.learned.score(knobs)
        if w == 0.0 or learned <= 0.0:
            return learned
        return learned * self.analytic.score(knobs) ** w

    def mean_score(self, knobs: dict) -> float:
        return self.learned.mean_score(knobs) if self.learned.fitted else self.analytic.mean_score(knobs)

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        return self.learned.mean_scores(knobs_list) if self.learned.fitted else self.analytic.mean_scores(knobs_list)

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        # Measured -O3 evidence lives in the LEARNED half's reservoir (the
        # analytic prior has no dataset), and applies even while the model is
        # cold — a freshly-seeded reservoir below ``min_rows`` still holds real
        # measurements worth deploying. The score fallback below then ranks the
        # evidence-less case through whichever half is live.
        ev = self.learned.evidence_pick(rows)
        if ev is not None:
            return ev
        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=scores.__getitem__)
        return best_i, scores[best_i]

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
