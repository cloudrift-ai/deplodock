"""Learned tuning prior: a :class:`Prior` base (bounded reservoir dataset, batched
refit, end-of-run sanity stats) and the concrete :class:`CatBoostPrior`."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.prior.base import Prior
from deplodock.compiler.pipeline.search.prior.catboost_prior import CatBoostPrior


def prior_from_json(obj: dict) -> Prior:
    """Reconstruct a fitted prior from a persisted :meth:`Prior.to_json` dict
    (used to load a tuned prior into a later ``compile`` / ``run``)."""
    return CatBoostPrior.from_json(obj)


__all__ = ["CatBoostPrior", "Prior", "prior_from_json"]
