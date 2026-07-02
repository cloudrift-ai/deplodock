"""Knob descriptor — the canonical schema for a tunable / forkable rule parameter.

Each ``Knob`` declares one tuning dimension: its name (also the env-var
key), its type (driving parse + pretty), the autotune candidate hints,
and a short help string. Rules emit knob values into ``TileOp.knobs``
dicts; the env layer reads ``DEPLODOCK_<NAME>`` to pin a knob across
the whole run; pretty-printing routes through the descriptor so
display matches storage.

This module owns the ``Knob`` *descriptor* (the dataclass), the
registry, and the env plumbing — **not** the concrete knob
declarations (those live in :mod:`deplodock.compiler.pipeline.search.space`,
beside the enumeration value grids) and **not** the featurizers (those
live in :mod:`deplodock.compiler.pipeline.search.features`). INVARIANT:
every ``Knob`` *instance* is declared in ``search/space.py``, and rules
import the knobs they decide from there. The :func:`registry`
introspects every loaded module under ``deplodock/`` and collects every
``Knob`` instance — no ``register(...)`` wrapper, no manual bookkeeping
— so a knob declared in ``space.py`` is discovered as soon as a rule
imports it (all rules load at pipeline startup).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from deplodock import config

# The ``deplodock/`` package dir (knob.py → pipeline → compiler → deplodock).
_PKG_ROOT = Path(__file__).resolve().parents[2]

# Reserved prefix for the structural-feature knobs stamped by
# ``loop/stamp/020_stamp_structural_features`` — distinct from any tuning Knob
# name, so ``format_tuning_knobs`` drops them from the tuning view and
# ``knob_features`` passes them through as-is. Declared here (rather than with
# the producing pass, which is loaded under a bare module stem) so every
# consumer can import it.
STRUCT_PREFIX = "S_"

# Reserved prefix for host/hardware-regime features injected from
# :meth:`Context.features` (GPU compute capability + nvcc opt level).
# Treated like ``STRUCT_PREFIX``: dropped from the tuning view, passed straight
# through ``knob_features`` as floats — they describe the regime a row was
# measured in, letting one global prior span every GPU / opt level.
CTX_PREFIX = "H_"


class _Unset:
    """Sentinel for ``Knob.off`` meaning "no OFF value declared" — the knob is
    expected to always be stamped by its owning pass, so
    :func:`apply_off_defaults` never auto-fills it."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


class KnobType(Enum):
    """Knob value type — drives ``Knob.parse`` and ``Knob.pretty``."""

    INT = "int"
    BOOL = "bool"
    BINMASK = "binmask"
    STR = "str"


@dataclass(frozen=True)
class Knob:
    """Schema for one tunable parameter.

    ``name`` is used as the key in ``TileOp.knobs`` dicts and to derive
    the env override ``DEPLODOCK_<NAME>``. ``hints`` is the autotune
    candidate list (a guideline, not a constraint — rules apply their
    own structural validity gates). ``help`` is a short docstring shown
    by ``deplodock eval knobs``."""

    name: str
    type: KnobType
    hints: tuple = ()
    help: str = ""
    # Optional custom featurizer for the learned-prior feature vector: maps this
    # knob's value to a ``dict[str, float]`` of sub-features. Declared at the
    # knob (e.g. ``MMA`` expands an atom kind into physical cell/dtype props) so
    # ``knob_features`` needs no per-knob special-casing. ``None`` → encode by
    # ``type`` (INT → float, BOOL → 0/1, BINMASK → popcount/width/frac).
    features: Callable[[Any], dict[str, float]] | None = None
    # The "unused / declined" OFF value. When a knob doesn't apply to a variant
    # (a tier-foreign knob like ``WM`` on a scalar kernel) or its owning pass
    # declines / is skipped, :func:`apply_off_defaults` stamps this value so the
    # realized variant carries an *explicit* decision rather than an absent key —
    # letting the learned prior tell "decided: unused" (an OFF value) from
    # "not-yet-decided" (still absent → NaN-filled). ``_UNSET`` (the default)
    # means the knob is always stamped by its pass and is never auto-filled.
    off: Any = _UNSET

    @property
    def env(self) -> str:
        return config.knob_var(self.name)

    def raw(self) -> str | None:
        """Live env pin (``DEPLODOCK_<NAME>``); ``None`` when unset."""
        return config.knob_raw(self.name)

    def read_int(self, default: int) -> int:
        """Read this knob's ``DEPLODOCK_<NAME>`` env pin as an int (empty /
        unset / unparseable → ``default``).

        Reads a knob's env override through its canonical descriptor rather
        than re-spelling the name as a bare string, so a rename of the constant
        fails loudly instead of silently reading a dead env var. ``INT`` knobs
        only."""
        if self.type is not KnobType.INT:
            raise ValueError(f"Knob.read_int only valid for INT knobs ({self.name!r} is {self.type})")
        return config.int_env(self.env, default)

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
            low = s.lower()
            if low in {"1", "true", "yes", "on"}:
                return True
            if low in {"", "0", "false", "no", "off"}:
                return False
            # Anything else (a typo like ``ture`` / ``banana``, or a stray ``2``) used to
            # coerce silently to False, disabling the knob with no diagnostic. Fail loudly.
            raise ValueError(f"bad BOOL value for knob {self.name!r}: {raw!r} (expect 1/true/yes/on or 0/false/no/off)")
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
        if self.type is KnobType.STR:
            return s
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
        """Override ``candidates`` with the env pin ``DEPLODOCK_<NAME>``.

        Folds env-driven pinning into the same iteration that produces
        hint-driven candidates, so callers don't enumerate-then-filter:

            reduce_choices = REDUCE.narrow(cands)  # 1-tuple if pinned

        Returns ``tuple(candidates)`` unchanged when the env var is
        absent; otherwise a 1-tuple ``(pinned,)`` — *authoritative*,
        even if ``pinned`` is not in ``candidates``. Hints are guidance,
        not constraint: the user knows what they're doing. Downstream
        structural gates (divisibility, threads-per-CTA budget, etc.)
        still apply, so a structurally invalid pin yields an empty
        enumeration; callers that need a peer-kernel fallback (empty
        after structural gates → invalid pin for *this* shape →
        re-enumerate without pins) implement that policy rule-side.

        ``BINMASK`` isn't supported — it would need ``width``, and no
        rule enumerates BINMASK candidates today."""
        raw = self.raw()
        if raw is None:
            return tuple(candidates)
        if self.type is KnobType.BINMASK:
            raise ValueError(f"Knob.narrow not supported for BINMASK ({self.name!r})")
        pinned = self.parse(raw)
        return (pinned,)

    def narrow_at(self, element: str) -> str | None:
        """The env pin for this knob's ``<NAME>@<element>`` key, falling back to the bare
        ``<NAME>`` pin — the per-element read mirroring :func:`family_value`'s precedence
        (``PLACE@fold`` > bare ``PLACE``). Returns the raw pin string (``None`` when neither is
        set); the caller resolves vocabulary (e.g. the ``PLACE`` family's ``auto`` token). The
        ``@``-suffixed key is not a valid shell var name, so it rides the ``DEPLODOCK_KNOBS``
        aggregate; :func:`parse_knob_spec` canonicalizes its element to lowercase, matching the
        lowercase element spelling producers use."""
        value = config.knob_raw(f"{self.name}@{element}")
        if value is not None:
            return value
        return self.raw()


# --- Axis-named schedule keys ----------------------------------------------
#
# A per-node schedule codec is keyed ``FAMILY@<axis>`` — ``TILE@<k_axis>`` / ``STAGE@<axis>`` /
# ``REDUCE@<axis>`` — so a multi-node kernel (flash) can address each schedule-bearing node by the
# reduce/contraction axis it schedules. The **bare** form (``TILE`` with no suffix) stays first-class:
# it resolves to the unique eligible axis for that family, so the common single-node kernel — and every
# existing pin / recipe / golden — keeps working unchanged (the suffix disambiguates only a kernel with
# two eligible nodes). ``TILE`` / ``STAGE`` / ``REDUCE`` all carry the suffix: the schedule reduce
# partition IS the axis-named reduce decision (there is no separate native ``REDUCE@`` family — the
# reduce/split-K partition is the one reduce family). ``WSPEC`` / ``PLACE`` stay root-global (always
# bare). Readers use :func:`family_value` so a bare and a suffixed key featurize / match identically.

# The per-node schedule codec families that carry an ``@<axis>`` element (``WSPEC`` / ``PLACE`` are
# root-global, always bare).
_AXIS_FAMILIES = ("TILE", "REDUCE", "STAGE")


def family_of(key: str) -> str:
    """The knob family — the part before an ``@<axis>`` suffix (``TILE@d`` → ``TILE``); the whole key
    when unsuffixed."""
    return key.split("@", 1)[0]


def axis_of(key: str) -> str | None:
    """The ``@<axis>`` element of a knob key (``TILE@d`` → ``d``), or ``None`` when bare."""
    return key.split("@", 1)[1] if "@" in key else None


def family_value(knobs: dict, family: str):
    """The value of a per-node schedule codec ``family`` in ``knobs``, keyed bare (``TILE``) or
    axis-suffixed (``TILE@<axis>``). ``None`` when absent. A single-node kernel has exactly one match;
    a multi-node kernel (flash) has one key per node — this returns the first (a pooled read; a
    per-node featurizer reads each node's slice via a group-by-axis loop)."""
    v = knobs.get(family)
    if v is not None:
        return v
    prefix = family + "@"
    for k, val in knobs.items():
        if k.startswith(prefix):
            return val
    return None


def resolve_axis(family: str, key: str, eligible: Sequence[str]) -> str | None:
    """Canonicalize a schedule-knob ``key`` (bare ``TILE`` or suffixed ``TILE@d``) to its
    ``FAMILY@<axis>`` form given the kernel's ``eligible`` axes for that ``family``:

    - already suffixed (``TILE@d``) → returned unchanged (idempotent).
    - bare, exactly one eligible axis → ``TILE@<that axis>`` (the suffix is sugar).
    - bare, no eligible axis → ``None`` (the family doesn't apply — drop / OFF).
    - bare, ≥2 eligible axes → ``ValueError`` naming the candidates (a hand-written pin must
      disambiguate, e.g. ``TILE@d`` vs ``TILE@sk``); enumeration never emits a bare key here, so the
      ambiguous case only arises from a pin."""
    if "@" in key:
        return key
    if not eligible:
        return None
    if len(eligible) == 1:
        return f"{family}@{eligible[0]}"
    cands = " or ".join(f"{family}@{a}" for a in eligible)
    raise ValueError(f"{family} is ambiguous: use {cands}")


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


def apply_off_defaults(knobs: dict, declared: Iterable[Knob]) -> dict:
    """Fill any ``declared`` knob with a defined ``off`` value into ``knobs``
    when it is absent — the "every emitted variant carries an explicit value for
    every declared knob" rule. Mutates and returns ``knobs``.

    Used in two places: the pipeline stamps a pass's declared OFF knobs on the
    realized variant at the pass boundary (so a declined / skipped / no-variant
    pass still records its OFF decision), and the partition planner stamps the
    tier-foreign knobs (``WM``/``WN``/``MMA`` on a scalar row; ``BM``/``BN``/…
    on a warp row) on each enumerated row so the OFF value rides the variant
    identity from enumeration → score → DB → materialized ``TileOp.knobs``.
    Idempotent: a knob already present (real value *or* a prior OFF fill) is left
    untouched; knobs whose ``off`` is ``_UNSET`` are never filled."""
    for knob in declared:
        if knob.off is _UNSET:
            continue
        if knob.name not in knobs:
            knobs[knob.name] = knob.off
    return knobs


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
    for key, value in parse_knob_spec(raw).items():
        # Individual per-knob env vars win — don't clobber an explicit
        # ``DEPLODOCK_BK=4`` with whatever the aggregate says.
        if config.set_knob(key, value, overwrite=False):
            applied[config.knob_var(key)] = value
    return applied


def parse_knob_spec(raw: str) -> dict[str, str]:
    """Parse the shared ``K1=V1,K2=V2`` knob-spec grammar (the ``DEPLODOCK_KNOBS``
    aggregate, ``run --ab``) into an ordered ``{NAME: value}`` dict — whitespace
    tolerated, empty entries skipped. A malformed entry (missing ``=`` / empty
    KEY) raises ``ValueError``. Case is canonicalized per key part: the family
    (before any ``@``) uppercases, the element (after it) lowercases — so
    ``place@FOLD=cut`` pins ``PLACE@fold`` and ``tile@d=…`` keeps its lowercase
    axis element instead of being mangled to ``TILE@D``."""
    out: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"knob spec entry {entry!r} is missing '=' (expected KEY=VALUE)")
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"knob spec entry {entry!r} has empty KEY")
        fam, at, element = key.partition("@")
        key = fam.upper() + (at + element.lower() if at else "")
        out[key] = value.strip()
    return out


# Splat ``DEPLODOCK_KNOBS`` once at import so every later per-knob reader
# (``config.knob_raw`` / ``config.int_env`` — knob.py is imported transitively
# by every pipeline pass) sees the individual ``DEPLODOCK_<NAME>`` keys.
apply_knobs_env()


# --- Rendering -------------------------------------------------------------

# Canonical display order for tuning knobs. The ``FAMILY@element`` keys lead — the structural
# ``PLACE@`` placement, then the axis-named schedule codecs (``TILE@`` / ``REDUCE@`` / ``STAGE@``)
# — each family's keys sorted by element; the bare exact-name knobs follow in ``KNOB_ORDER``,
# unknown knobs last (alpha). Shared by the ``run --bench`` kernel table and the ``deplodock
# eval`` tables so columns read stably.
_FAMILY_ORDER = ("PLACE@", "TILE@", "REDUCE@", "STAGE@")
KNOB_ORDER = ("TILE", "REDUCE", "STAGE", "WSPEC")
_KNOB_RANK = {k: i for i, k in enumerate(KNOB_ORDER)}

# Schedule codec families whose ``@<axis>`` display collapses back to bare when the kernel has a
# single eligible axis (so one-node tables read as ``TILE=…`` / ``REDUCE=…`` / ``STAGE=…``, exactly as
# before the axis-naming, matching the bare golden YAML). All three per-node schedule codecs collapse —
# the schedule reduce partition is the one reduce family, so ``REDUCE@<axis>`` bares out like the rest.
_COLLAPSE_FAMILIES = ("TILE", "REDUCE", "STAGE")


def knob_sort_key(name: str) -> tuple[int, str]:
    """Sort key: native ``MOVE@element`` families first (by :data:`_FAMILY_ORDER`, then
    element), then the codec knobs in :data:`KNOB_ORDER`, unknown knobs last (alpha)."""
    for i, prefix in enumerate(_FAMILY_ORDER):
        if name.startswith(prefix):
            return (i, name[len(prefix) :])
    return (len(_FAMILY_ORDER) + _KNOB_RANK.get(name, len(KNOB_ORDER)), name)


def format_tuning_knobs(knobs: dict) -> str:
    """Render ``knobs`` as a compact ``key=value`` string, dropping
    pass-marker booleans. Empty after filtering → ``-``.

    A registered ``Knob`` of type ``BOOL`` is treated as a marker and
    dropped; unregistered boolean values are also dropped (forward-compat).
    ``BINMASK`` values are already stored as binary strings in
    ``op.knobs`` (rules stamp via ``Knob.pretty``), so ``str(v)`` here
    round-trips correctly. ``STRUCT_PREFIX`` knobs (the structural-feature
    stamp from ``992_stamp_structural_features``) are facts about the kernel, not
    tuning decisions, so they are dropped from this tuning-knob view.
    """
    items = tuning_knob_items(knobs)
    return ", ".join(f"{k}={v}" for k, v in items) if items else "-"


def tuning_knob_items(knobs: dict) -> list[tuple[str, str]]:
    """The filtered, canonically-ordered ``(name, str(value))`` tuning knobs —
    the same view :func:`format_tuning_knobs` renders, but as items so callers can
    build aligned columns. ``STRUCT_PREFIX`` / ``CTX_PREFIX`` features and marker
    booleans are dropped; the rest is sorted by :func:`knob_sort_key`. The unified
    ``TILE`` output-fragment knob is one column for both the scalar and warp tiers
    (the value self-describes), so there are no tier-foreign OFF knobs to hide.

    A per-node schedule codec keyed ``@<axis>`` (``TILE@d`` / ``STAGE@d``) **collapses back to bare**
    (``TILE`` / ``STAGE``) when the kernel has a single eligible axis for that family (one such key),
    so a one-node table reads exactly as it did before axis-naming; a multi-node kernel (flash) keeps
    the ``@<axis>`` suffix to disambiguate."""
    from collections import Counter  # noqa: PLC0415

    fam_counts = Counter(family_of(k) for k in knobs if "@" in k and family_of(k) in _COLLAPSE_FAMILIES)
    rendered: list[tuple[str, str]] = []
    for k, v in knobs.items():
        if k.startswith(STRUCT_PREFIX) or k.startswith(CTX_PREFIX):
            continue
        knob = get(k)
        if knob is not None and knob.type is KnobType.BOOL:
            continue
        if knob is None and isinstance(v, bool):
            continue
        fam = family_of(k)
        disp = fam if ("@" in k and fam in _COLLAPSE_FAMILIES and fam_counts[fam] == 1) else k
        rendered.append((disp, str(v)))
    return sorted(rendered, key=lambda kv: knob_sort_key(kv[0]))
