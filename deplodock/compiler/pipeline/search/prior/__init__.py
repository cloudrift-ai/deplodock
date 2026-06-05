"""Inner-``tune`` learned prior: a :class:`Prior` base (model interface + the
end-of-run sanity stats) and the concrete :class:`BayesianRidgePrior`."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.prior.base import Prior
from deplodock.compiler.pipeline.search.prior.bayesian_ridge import BayesianRidgePrior


def prior_from_bytes(blob: bytes) -> Prior:
    """Reconstruct a fitted prior from a persisted :meth:`Prior.to_bytes` blob
    (the inverse of :meth:`Prior.to_bytes`; used to load a tuned prior into a
    later ``compile`` / ``run``)."""
    return BayesianRidgePrior.from_bytes(blob)


__all__ = ["BayesianRidgePrior", "Prior", "prior_from_bytes"]
