"""Inner-``tune`` learned prior: a :class:`Prior` base (model interface + the
end-of-run sanity stats) and the concrete :class:`BayesianRidgePrior`."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.prior.base import Prior
from deplodock.compiler.pipeline.search.prior.bayesian_ridge import BayesianRidgePrior

__all__ = ["BayesianRidgePrior", "Prior"]
