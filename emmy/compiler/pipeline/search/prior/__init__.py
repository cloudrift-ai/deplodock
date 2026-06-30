"""Learned tuning prior: a :class:`Prior` base (bounded reservoir dataset, batched
refit, end-of-run sanity stats) and the concrete :class:`CatBoostPrior`."""

from __future__ import annotations

from emmy.compiler.pipeline.search.prior.analytic import AnalyticPrior
from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.catboost import CatBoostPrior
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior, load_prior


def prior_from_json(obj: dict) -> Prior:
    """Reconstruct a fitted prior from a persisted :meth:`Prior.to_json` dict
    (used to load a tuned prior into a later ``compile`` / ``run``)."""
    return CatBoostPrior.from_json(obj)


__all__ = ["AnalyticPrior", "CatBoostPrior", "FallbackPrior", "Prior", "load_prior", "prior_from_json"]
