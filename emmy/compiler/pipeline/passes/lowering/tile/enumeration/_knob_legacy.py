"""Legacy knob ingest (read-only) — the deprecation ramp for the rank-2 GEMM-letter
schema.

The native ``MOVE@element`` schema (``_families``) is the source of truth everywhere
the pipeline faces inward — storage, ``eval`` display, new goldens, new tests all speak
native. The legacy names (``BN``/``BM``/``BK``/``WM``/``MMA``/…) survive only as an
**ingest-only** translation at the read boundary, so an existing env pin
(``EMMY_BK=16``) or a legacy-recorded golden YAML still resolves. There is
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

from emmy import config


def _int_pin(name: str) -> int | None:
    """A legacy ``EMMY_<NAME>`` env pin as an int, or ``None`` when unset /
    unparseable."""
    raw = config.knob_raw(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def _free_rank(dag, axis_name: str) -> int | None:
    """The canonical free-axis rank of ``axis_name`` — ``0`` for the innermost free
    axis (``dag.inner_n``, legacy "N"), ``1`` for the next-out (``dag.outer_m``,
    legacy "M"), else ``None`` (a 3rd+ free axis has no legacy name)."""
    if axis_name == dag.inner_n.axis.name:
        return 0
    if dag.outer_m is not None and axis_name == dag.outer_m.axis.name:
        return 1
    return None


def split_par(dag, axis_name: str) -> int | None:
    """Legacy parallel-binding pin for a free axis: rank-0 (innermost N) takes
    ``BN`` (thread) or ``WN`` (warp), rank-1 (outer M) takes ``BM`` / ``WM`` — the
    tier is mutually exclusive (a kernel is scalar XOR warp), so whichever is set
    wins. A 3rd+ free axis has no legacy name → ``None``."""
    rank = _free_rank(dag, axis_name)
    if rank == 0:
        return _int_pin("BN") or _int_pin("WN")
    if rank == 1:
        return _int_pin("BM") or _int_pin("WM")
    return None


def split_reg(dag, axis_name: str) -> int | None:
    """Legacy register-cell pin for a free axis: rank-0 (innermost N) takes ``FN``,
    rank-1 (outer M) takes ``FM``."""
    rank = _free_rank(dag, axis_name)
    if rank == 0:
        return _int_pin("FN")
    if rank == 1:
        return _int_pin("FM")
    return None


def stage_mask(n: int) -> int | None:
    """Legacy ``EMMY_STAGE`` bitmask over ``n`` ranked staged read-sites
    (``"11"`` / ``"all"`` / ``"none"`` / int), ingested as the placement pin. ``None``
    when unset (auto-enumerate)."""
    raw = config.knob_raw("STAGE")
    if raw is None:
        return None
    s = raw.strip()
    if s == "all":
        return (1 << n) - 1
    if s == "none" or s == "":
        return 0
    if len(s) == n and all(c in "01" for c in s):
        return sum(int(c) << i for i, c in enumerate(s))
    try:
        return int(s, 0) & ((1 << n) - 1)
    except ValueError:
        return None


def tma_pin() -> bool | None:
    """Legacy ``EMMY_TMA`` transport pin (ingested as the ``:tma`` vs ``:sync``
    xport for staged ``PLACE@<edge>``). ``None`` when unset."""
    raw = config.knob_raw("TMA")
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cut_pin() -> bool | None:
    """Legacy ``EMMY_CUT`` / ``EMMY_SPLIT_CONE`` (a width-1 bitmask; any non-zero
    bit = cut), ingested as the demoted-cone keep-vs-cut decision. ``None`` when unset."""
    raw = config.knob_raw("CUT")
    if raw is None:
        raw = config.knob_raw("SPLIT_CONE")
    if raw is None or raw == "":
        return None
    s = raw.strip().lower()
    if s == "all":
        return True
    if s == "none":
        return False
    if s and all(c in "01" for c in s):
        return "1" in s
    try:
        return int(s, 0) != 0
    except ValueError:
        return None


def chain_pin() -> bool:
    """Legacy ``EMMY_CHAIN`` opt-in (ingested as ``PLACE@<score>=inline`` — the
    FA-2 shared-score warp-chain). ``False`` when unset."""
    raw = config.knob_raw("CHAIN")
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def atom_raw() -> str | None:
    """The legacy ``EMMY_MMA`` env pin (``0``/``scalar`` / a kind / auto), ingested
    as the atom control for the matmul cell. ``None`` when unset."""
    return config.knob_raw("MMA")


def reduce_fields(dag, axis_name: str) -> tuple[int | None, int | None, int | None, int | None]:
    """Legacy ``EMMY_BK`` / ``FK`` / ``SPLITK`` / ``BR`` → the native
    ``(serial, fold, cta, coop)`` REDUCE factors, applied to the **primary** reduce
    axis only (rank-0 = ``dag.k_node``). Each field is the pinned int or ``None``
    (unpinned → the offer keeps its full menu). A non-primary reduce axis (a streaming
    flash's nested QK^T ``dd``) has no legacy name → all ``None``."""
    if not dag.reduce or axis_name != dag.k_node.loop.axis.name:
        return (None, None, None, None)
    return (_int_pin("BK"), _int_pin("FK"), _int_pin("SPLITK"), _int_pin("BR"))
