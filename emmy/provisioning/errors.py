"""Exceptions used by the cloud VM provisioning layer.

The orchestrator in :mod:`deplodock.provisioning.cloud` distinguishes three
outcomes from a provider's ``create_instance``:

* **success** — returns a ``VMConnectionInfo``.
* **capacity-class failure** — the current candidate (provider + instance
  type + zone) has no usable capacity; raise :class:`CapacityExhausted`
  and the orchestrator advances to the next candidate.
* **terminal failure** — auth, malformed request, or any error that is
  guaranteed to recur regardless of candidate; raise
  :class:`TerminalProvisionError` and the orchestrator aborts.

Any other exception is treated as transient and the orchestrator retries
the same candidate a small number of times before moving on.
"""


class CapacityExhausted(Exception):
    """The current candidate has no capacity available.

    Providers raise this for HTTP 503 / 429 on rent, GCP
    ``ZONE_RESOURCE_POOL_EXHAUSTED`` / ``STOCKOUT`` / ``QUOTA_EXCEEDED``,
    or when a status poll reports a terminal "no capacity" state
    (e.g. CloudRift ``Inactive``). The orchestrator catches this and
    advances to the next candidate without further same-candidate retries.
    """


class TerminalProvisionError(Exception):
    """A non-retryable failure (bad credentials, malformed request, etc.).

    The orchestrator does not retry and does not fall back to other
    candidates — it surfaces the error to the caller immediately.
    """
