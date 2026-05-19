"""Knob-formatting helpers for tune logs and the summary table.

Knobs are forwarded along the rewrite chain via every Op-rebind and
end up on the final ``CudaOp``. Some are real tuning parameters
(``BM``, ``BN``, ``FM``, ``FN``, ...); others are pass-marker
booleans (``blockify``, ``register_tile``, ``stage:*``,
``stage_inputs``) that just record "this pass ran". The marker
booleans are noise in tune output, so we drop them when rendering.
"""

from __future__ import annotations


def format_tuning_knobs(knobs: dict) -> str:
    """Render ``knobs`` as a compact ``key=value`` string, dropping
    boolean-valued entries (pass markers). Empty after filtering → ``-``."""
    filtered = {k: v for k, v in knobs.items() if not isinstance(v, bool)}
    if not filtered:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(filtered.items()))
