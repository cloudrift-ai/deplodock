"""Knob descriptor — the canonical schema for a tunable / forkable rule parameter.

Each ``Knob`` declares one tuning dimension: its name (also the env-var
key), its type (driving parse + pretty), the autotune candidate hints,
and a short help string. Rules emit knob values into ``TileOp.knobs``
dicts; the env layer reads ``DEPLODOCK_<NAME>`` to pin a knob across
the whole run; pretty-printing routes through the descriptor so
display matches storage.

Knobs are declared as plain module-level constants inside the rule
that owns them (e.g. ``BN`` / ``BM`` in ``005_blockify_launch``). The
:func:`registry` introspects every loaded rule module under
``deplodock.compiler.pipeline.passes.`` and collects every ``Knob``
instance — no ``register(...)`` wrapper, no manual bookkeeping.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from deplodock import config

# The ``deplodock/`` package dir (knob.py → pipeline → compiler → deplodock).
_PKG_ROOT = Path(__file__).resolve().parents[2]


class KnobType(Enum):
    """Knob value type — drives ``Knob.parse`` and ``Knob.pretty``."""

    INT = "int"
    BOOL = "bool"
    BINMASK = "binmask"


@dataclass(frozen=True)
class Knob:
    """Schema for one tunable parameter.

    ``name`` is used as the key in ``TileOp.knobs`` dicts and to derive
    the env override ``DEPLODOCK_<NAME>``. ``hints`` is the autotune
    candidate list (a guideline, not a constraint — rules apply their
    own structural validity gates). ``help`` is a short docstring shown
    by future tooling (``deplodock knobs``)."""

    name: str
    type: KnobType
    hints: tuple = ()
    help: str = ""

    @property
    def env(self) -> str:
        return config.knob_var(self.name)

    def parse(self, raw: str, *, width: int | None = None) -> Any:
        """Decode an env-string value per ``type``. ``width`` is required
        for ``BINMASK`` (number of candidate buffers in the current tile).

        BINMASK accepts a binary string ``"101"`` (char ``i`` selects
        ranked-buffer ``i``; length must match ``width``), the keywords
        ``"all"`` / ``"none"``, or a decimal / ``0x``-hex int (clamped
        to ``width`` bits)."""
        s = raw.strip()
        if self.type is KnobType.INT:
            return int(s, 0)
        if self.type is KnobType.BOOL:
            return s.lower() in {"1", "true", "yes", "on"}
        if self.type is KnobType.BINMASK:
            if width is None:
                raise ValueError(f"BINMASK knob {self.name!r} parse needs width")
            if s == "all":
                return (1 << width) - 1
            if s == "none":
                return 0
            if len(s) == width and all(c in "01" for c in s):
                return sum(int(c) << i for i, c in enumerate(s))
            return int(s, 0) & ((1 << width) - 1)
        raise ValueError(f"unhandled knob type: {self.type}")

    def pretty(self, value: Any, *, width: int | None = None) -> str:
        """Render ``value`` for display / storage. ``BINMASK`` produces
        a fixed-width binary string (char ``i`` = bit ``i``); other types
        round-trip through ``str()``."""
        if self.type is KnobType.BINMASK:
            if width is None:
                raise ValueError(f"BINMASK knob {self.name!r} pretty needs width")
            return "".join("1" if (value >> i) & 1 else "0" for i in range(width))
        return str(value)

    def narrow(self, candidates: Iterable[Any]) -> tuple:
        """Intersect ``candidates`` with the env pin ``DEPLODOCK_<NAME>``.

        Folds env-driven pinning into the same iteration that produces
        hint-driven candidates, so callers don't enumerate-then-filter:

            bn_choices = BN.narrow(_TUNE_AXIS_CHOICES)  # 1-tuple if pinned

        Returns ``tuple(candidates)`` unchanged when the env var is
        absent; a 1-tuple ``(pinned,)`` when the pin matches a candidate;
        ``()`` when the pin doesn't match anything in ``candidates``.
        Callers that need a peer-kernel fallback (empty after structural
        gates → invalid pin for *this* shape → re-enumerate without
        pins) implement that policy rule-side.

        ``BINMASK`` isn't supported — it would need ``width``, and no
        rule enumerates BINMASK candidates today."""
        raw = config.knob_raw(self.name)
        if raw is None:
            return tuple(candidates)
        if self.type is KnobType.BINMASK:
            raise ValueError(f"Knob.narrow not supported for BINMASK ({self.name!r})")
        pinned = self.parse(raw)
        return tuple(c for c in candidates if c == pinned)


# --- Registry --------------------------------------------------------------

_REGISTRY: dict[str, Knob] | None = None


def _walk_modules() -> list[ModuleType]:
    # Every ``Knob`` is declared in a rule module under ``deplodock/`` — but rule
    # modules are loaded via ``importlib.util.spec_from_file_location`` and registered
    # under their bare file stem (e.g. ``005_blockify_launch``), not the full
    # ``deplodock.compiler.pipeline.passes.…`` dotted path, so we can't filter on
    # ``__name__``. Filter on ``__file__`` living under the package instead: this keeps
    # every Knob-bearing module while skipping the thousands of stdlib / third-party
    # modules — which both slows the walk and, for e.g. ``torch.distributed``, emits a
    # spurious ``FutureWarning`` when ``vars(mod)`` materializes its deprecated members.
    #
    # Snapshot ``sys.modules`` with ``tuple(...)`` (one GIL-held C call) before walking:
    # the per-module body releases the GIL, and a concurrent kernel compile / bench
    # worker importing a module would otherwise mutate ``sys.modules`` mid-iteration
    # ("dictionary changed size during iteration").
    mods: list[ModuleType] = []
    for m in tuple(sys.modules.values()):
        f = getattr(m, "__file__", None) if m is not None else None
        if f and Path(f).is_relative_to(_PKG_ROOT):
            mods.append(m)
    return mods


def registry() -> dict[str, Knob]:
    """``{name: Knob}`` table, built lazily on first access by walking
    every loaded module for module-level ``Knob`` attributes.

    Rule modules are imported at pipeline-startup (to collect their
    ``PATTERN`` / ``rewrite``), so by the time anyone asks for knobs
    they're all present in ``sys.modules``. Duplicate names across
    rules (e.g. multiple rules declaring ``BN``) collapse to the
    first-seen ``Knob`` — declarations should agree on type / hints."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        for mod in _walk_modules():
            try:
                attrs = vars(mod).values()
            except TypeError:
                continue  # some built-in modules don't expose __dict__
            for attr in attrs:
                if isinstance(attr, Knob) and attr.name not in _REGISTRY:
                    _REGISTRY[attr.name] = attr
    return _REGISTRY


def get(name: str) -> Knob | None:
    """Lookup a registered ``Knob`` by name. ``None`` if no rule has
    declared it."""
    return registry().get(name)


def reset_registry() -> None:
    """Clear the lazy registry cache. Test-only — pytest fixtures that
    monkeypatch ``sys.modules`` use this to force a rebuild."""
    global _REGISTRY
    _REGISTRY = None


# --- Aggregate env var ------------------------------------------------------


def apply_knobs_env(raw: str | None = None) -> dict[str, str]:
    """Splat ``DEPLODOCK_KNOBS="K1=V1,K2=V2,..."`` into individual
    ``DEPLODOCK_<K>=V`` env vars.

    Convenience for setting many knobs from one place (CLI invocation,
    Makefile recipe, pytest ``monkeypatch.setenv``). Individual
    ``DEPLODOCK_<K>`` vars take precedence: this only writes a key when
    the per-knob env var is unset, so callers can pin one knob from
    the command line and supply the rest via ``DEPLODOCK_KNOBS``
    without surprise.

    Whitespace is tolerant; empty entries are skipped. A malformed
    entry (no ``=``) raises ``ValueError``. Returns the dict of keys
    actually applied (for tests / diagnostics).

    Runs at module import time on the live process environment. Pass
    ``raw`` explicitly to apply a different aggregate without touching
    ``DEPLODOCK_KNOBS`` (useful in tests).
    """
    if raw is None:
        raw = config.knobs_aggregate()
    applied: dict[str, str] = {}
    if not raw:
        return applied
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"DEPLODOCK_KNOBS entry {entry!r} is missing '=' (expected KEY=VALUE)")
        key, _, value = entry.partition("=")
        key = key.strip().upper()
        value = value.strip()
        if not key:
            raise ValueError(f"DEPLODOCK_KNOBS entry {entry!r} has empty KEY")
        # Individual per-knob env vars win — don't clobber an explicit
        # ``DEPLODOCK_BK=4`` with whatever the aggregate says.
        if config.set_knob(key, value, overwrite=False):
            applied[config.knob_var(key)] = value
    return applied


# Splat ``DEPLODOCK_KNOBS`` once at import so every later per-knob reader
# (``config.knob_raw`` / ``config.int_env`` — knob.py is imported transitively
# by every pipeline pass) sees the individual ``DEPLODOCK_<NAME>`` keys.
apply_knobs_env()


# --- Rendering -------------------------------------------------------------


def format_tuning_knobs(knobs: dict) -> str:
    """Render ``knobs`` as a compact ``key=value`` string, dropping
    pass-marker booleans. Empty after filtering → ``-``.

    A registered ``Knob`` of type ``BOOL`` is treated as a marker and
    dropped; unregistered boolean values are also dropped (forward-compat).
    ``BINMASK`` values are already stored as binary strings in
    ``op.knobs`` (rules stamp via ``Knob.pretty``), so ``str(v)`` here
    round-trips correctly.
    """
    rendered: list[tuple[str, str]] = []
    for k, v in knobs.items():
        knob = get(k)
        if knob is not None and knob.type is KnobType.BOOL:
            continue
        if knob is None and isinstance(v, bool):
            continue
        rendered.append((k, str(v)))
    if not rendered:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(rendered))
