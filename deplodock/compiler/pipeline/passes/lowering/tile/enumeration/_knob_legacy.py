"""Legacy knob ingest (read-only) — the deprecation ramp for the rank-2 GEMM-letter
schema (``plans/algebra-knob-naming-schema.md`` Step 2).

The native ``MOVE@element`` schema (``_families``) is the source of truth everywhere
the pipeline faces inward — storage, ``eval`` display, new goldens, new tests all speak
native. The legacy names (``BN``/``BM``/``BK``/``WM``/``MMA``/…) survive only as an
**ingest-only** translation at the read boundary, so an existing env pin
(``DEPLODOCK_BK=16``) or a legacy-recorded golden YAML still resolves. There is
deliberately **no** reverse (``project``): nothing ever maps native → legacy.

The bridge is the **element-resolution rule** — the canonical DAG ranking the move
offers already use:

- free axes ranked innermost-first (rank-0 = "N" = ``dag.inner_n``, rank-1 = "M" =
  ``dag.outer_m``);
- reduce axes ranked primary-first (rank-0 = "K" = ``dag.k_node``);
- edges ranked by the staging candidate order.

A legacy pin can only reach the rank-0 slots, by design — a streaming flash's 2nd reduce
axis (``REDUCE@dd``), a 3rd free axis, or an INLINE non-score edge have **no** legacy name
and aren't reachable through the ramp. That is the intended nudge toward the native pin.
This module is the ONE place the canonical ranking is named for the legacy bridge.
"""

from __future__ import annotations

from deplodock import config


def _int_pin(name: str) -> int | None:
    """A legacy ``DEPLODOCK_<NAME>`` env pin as an int, or ``None`` when unset /
    unparseable."""
    raw = config.knob_raw(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def reduce_fields(dag, axis_name: str) -> tuple[int | None, int | None, int | None, int | None]:
    """Legacy ``DEPLODOCK_BK`` / ``FK`` / ``SPLITK`` / ``BR`` → the native
    ``(serial, fold, cta, coop)`` REDUCE factors, applied to the **primary** reduce
    axis only (rank-0 = ``dag.k_node``). Each field is the pinned int or ``None``
    (unpinned → the offer keeps its full menu). A non-primary reduce axis (a streaming
    flash's nested QK^T ``dd``) has no legacy name → all ``None``."""
    if not dag.reduce or axis_name != dag.k_node.loop.axis.name:
        return (None, None, None, None)
    return (_int_pin("BK"), _int_pin("FK"), _int_pin("SPLITK"), _int_pin("BR"))
