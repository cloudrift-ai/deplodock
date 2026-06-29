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
from collections.abc import Callable, Iterable
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


def masked_axis_features(*, m: bool = False, n: bool = False, k: bool = False) -> dict[str, float]:
    """The per-role boundary-masked structural features (``S_masked_m/n/k``).

    A tile boundary-masks an output / reduce axis when the extent is symbolic or a
    static non-divisor of the chosen tile — a *consequence* of the shape/tile
    pairing, not a tunable choice — so it belongs with the ``S_`` structural
    identity, not a tuning knob. Masking is only known once the tile geometry is
    chosen, so the producers stamp it at materialize / enumeration time; the
    feature definition lives here, beside ``STRUCT_PREFIX`` and the featurizer that
    reads it (``_geom_feats`` → ``D_neg_masked_*``).

    Replaces the legacy flat ``OVERHANG`` tuple knob, split per role so the prior
    can learn that K-masking (SYNC-pinned, ring-declined) prices differently from
    M / N output masking. Only the masked roles are emitted — an unmasked kernel
    carries none, so its structural identity is unchanged and the featurizer
    defaults a missing flag to ``0.0`` (matching the old ``OVERHANG`` conditional
    stamp). ``S_masked_*`` pass through :func:`knob_features` as raw floats via the
    ``STRUCT_PREFIX`` branch automatically."""
    feats: dict[str, float] = {}
    if m:
        feats[f"{STRUCT_PREFIX}masked_m"] = 1.0
    if n:
        feats[f"{STRUCT_PREFIX}masked_n"] = 1.0
    if k:
        feats[f"{STRUCT_PREFIX}masked_k"] = 1.0
    return feats


class _Unset:
    """Sentinel for ``Knob.off`` meaning "no OFF value declared" — the knob is
    expected to always be stamped by its owning pass (universal knobs like
    ``BN`` / ``BM``), so :func:`apply_off_defaults` never auto-fills it."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()

# --- MMA tier decode -------------------------------------------------------
# The ``MMA`` knob (declared in ``lowering/tile/_enumeration.py``) selects the
# tensor-core "warp" tier: a falsy / empty value is scalar-only, an atom-kind
# name (e.g. ``mma_m16n8k16_f16``) is the warp tier, ``1``/``true`` is the
# pre-enumeration auto control. These live here (not with the knob) so every
# layer — the tile passes, the ``ir/tile/ir.py`` scorer, and ``_tile_features``
# below — shares ONE tier test without importing the pipeline (knob.py has no
# pipeline deps). ``_enumeration.mma_mode`` delegates to :func:`mma_decode`.
_MMA_FALSY = frozenset({"0", "false", "no", "off"})
_MMA_TRUTHY = frozenset({"1", "true", "yes", "on"})


def mma_decode(raw: str | None) -> tuple[bool, str | None]:
    """Decode a raw ``MMA`` value into ``(enabled, pinned_kind)``.

    Unset / empty / truthy → ``(True, None)`` (auto-enumerate); falsy →
    ``(False, None)`` (scalar-only); any other string is an atom-kind name →
    ``(True, name)``."""
    if raw is None:
        return True, None
    s = raw.strip()
    if not s or s.lower() in _MMA_TRUTHY:
        return True, None
    if s.lower() in _MMA_FALSY:
        return False, None
    return True, s


def mma_atom(knobs: dict) -> str | None:
    """The concrete tensor-core atom-kind name carried by ``knobs``, or ``None``
    for the scalar tier (no atom / the ``"scalar"`` decision / the pre-enumeration
    auto control — none of which name an atom).

    Native schema: scan the per-cell ``ATOM@<cell>`` keys and return the first that
    names a real atom kind (``"scalar"`` is the scalar decision, not a kind). A
    legacy ``MMA`` key (golden YAML / pre-migration DB row) is honored as a fallback
    — the one legacy spelling read here, an ingest convenience, not impl vocabulary."""
    for k, v in knobs.items():
        if k.startswith("ATOM@"):
            s = str(v).strip()
            if s and s.lower() != "scalar":
                return s
    v = knobs.get("MMA")  # legacy ingest fallback
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in _MMA_FALSY or s.lower() in _MMA_TRUTHY:
        return None
    return s


def is_warp(knobs: dict) -> bool:
    """True if ``knobs`` is a warp-tier (tensor-core MMA) variant — i.e. it
    names a concrete atom kind. The single tier discriminator shared by the tile
    passes, the scorer, and the featurizer."""
    return mma_atom(knobs) is not None


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
    by future tooling (``deplodock knobs``). ``aliases`` are alternate
    names whose ``DEPLODOCK_<ALIAS>`` env vars also pin this knob (read
    via :meth:`raw`; the primary name wins when both are set)"""

    name: str
    type: KnobType
    hints: tuple = ()
    help: str = ""
    aliases: tuple[str, ...] = ()
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
    # means the knob is always stamped by its pass and is never auto-filled
    # (universal knobs like ``BN`` / ``BM``).
    off: Any = _UNSET

    @property
    def env(self) -> str:
        return config.knob_var(self.name)

    def raw(self) -> str | None:
        """Live env pin: the primary ``DEPLODOCK_<NAME>`` first, then each
        alias in declaration order; ``None`` when nothing is set. Every
        env read of an alias-bearing knob must route through here so the
        alias spelling behaves identically to the primary."""
        value = config.knob_raw(self.name)
        if value is not None:
            return value
        for alias in self.aliases:
            value = config.knob_raw(alias)
            if value is not None:
                return value
        return None

    def read_int(self, default: int) -> int:
        """Read this knob's ``DEPLODOCK_<NAME>`` env pin as an int (empty /
        unset / unparseable → ``default``).

        The heuristic-default path in ``compiler/tuning.py`` uses this to read
        the matmul tile knobs (``BN`` / ``BM`` / ``FM`` / ``FN`` / ``BK`` /
        ``SPLITK``) through their canonical descriptors rather than re-spelling
        the names as bare strings — a rename of the constant now fails loudly
        instead of silently reading a dead env var. ``INT`` knobs only."""
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

            bn_choices = BN.narrow(_TUNE_AXIS_CHOICES)  # 1-tuple if pinned

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
    aggregate, ``run --ab``) into an ordered ``{NAME: value}`` dict — names
    uppercased, whitespace tolerated, empty entries skipped. A malformed entry
    (missing ``=`` / empty KEY) raises ``ValueError``."""
    out: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"knob spec entry {entry!r} is missing '=' (expected KEY=VALUE)")
        key, _, value = entry.partition("=")
        key = key.strip().upper()
        if not key:
            raise ValueError(f"knob spec entry {entry!r} has empty KEY")
        out[key] = value.strip()
    return out


# Splat ``DEPLODOCK_KNOBS`` once at import so every later per-knob reader
# (``config.knob_raw`` / ``config.int_env`` — knob.py is imported transitively
# by every pipeline pass) sees the individual ``DEPLODOCK_<NAME>`` keys.
apply_knobs_env()


# --- Rendering -------------------------------------------------------------

# Canonical display order for tuning knobs. The native ``MOVE@element`` families lead —
# ``SPLIT@`` (free-axis tiles), then ``REDUCE@`` (contraction tower), then ``ATOM@``
# (atomize), then ``PLACE@`` (placement) — each family's keys sorted by element; the
# legacy exact-name knobs (``STAGE``/``TMA``/``CUT``/… still un-folded, plus any legacy
# golden/DB row) follow in the historical order, unknown knobs last (alpha). Shared by the
# ``run --bench`` kernel table and the ``deplodock eval`` tables so columns read stably.
_FAMILY_ORDER = ("SPLIT@", "REDUCE@", "ATOM@", "PLACE@")
KNOB_ORDER = ("TILE", "REDUCE", "BM", "BN", "BK", "BR", "FM", "FN", "FK", "WM", "WN", "SPLITK", "RING", "STAGE", "MMA")
_KNOB_RANK = {k: i for i, k in enumerate(KNOB_ORDER)}


def knob_sort_key(name: str) -> tuple[int, str]:
    """Sort key: native ``MOVE@element`` families first (by :data:`_FAMILY_ORDER`, then
    element), then the legacy names in :data:`KNOB_ORDER`, unknown knobs last (alpha)."""
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


# Tier-foreign knobs hidden from the tuning *display* (not the feature vector,
# the DB row, or the repro env): a scalar kernel shouldn't show the warp knobs
# (their OFF sentinels ``WM=0 WN=0 MMA=0``) and a warp kernel shouldn't show the
# scalar thread-tile knobs (``BM=0 BN=0 BR=0 FK=0``) — those carry OFF values so
# the learned prior sees them, but rendering them is pure noise in the bench /
# eval tables. Kept here as plain name sets (knob.py has no pipeline deps); the
# tier is picked value-based via :func:`is_warp`.
_WARP_TIER_KNOBS = frozenset({"WM", "WN", "MMA"})
# The scalar tier's free-axis tile is the ``TILE`` codec (the legacy ``BM``/``BN``/``BR``/``FK``
# names are kept for back-compat with old golden/DB rows).
_SCALAR_TIER_KNOBS = frozenset({"TILE", "BM", "BN", "BR", "FK"})


def tuning_knob_items(knobs: dict) -> list[tuple[str, str]]:
    """The filtered, canonically-ordered ``(name, str(value))`` tuning knobs —
    the same view :func:`format_tuning_knobs` renders, but as items so callers can
    build aligned columns. ``STRUCT_PREFIX`` / ``CTX_PREFIX`` features and marker
    booleans are dropped, as are the tier-foreign OFF knobs (warp knobs on a
    scalar kernel and vice-versa); the rest is sorted by :func:`knob_sort_key`."""
    foreign = _SCALAR_TIER_KNOBS if is_warp(knobs) else _WARP_TIER_KNOBS
    rendered: list[tuple[str, str]] = []
    for k, v in knobs.items():
        if k.startswith(STRUCT_PREFIX) or k.startswith(CTX_PREFIX):
            continue
        if k in foreign:
            continue
        knob = get(k)
        if knob is not None and knob.type is KnobType.BOOL:
            continue
        if knob is None and isinstance(v, bool):
            continue
        rendered.append((k, str(v)))
    return sorted(rendered, key=lambda kv: knob_sort_key(kv[0]))


# --- Feature vector ---------------------------------------------------------


def _cut_features(knobs: dict) -> dict[str, float]:
    """The engineered ``D_*`` edge-cost feature for the demoted-matmul cut (the
    ``CUT`` knob). A cut materializes the
    demoted operand cone to a **gmem intermediate** — a round-trip the fused keep
    avoids — so the prior needs the materialized volume to price the cut's Σ vs.
    keep's. ``D_cut_roundtrip`` is the cost axis that discriminates the two
    realizations of one decision: positive on a cut fragment (``CUT`` mask
    popcount > 0), **zero on the fused keep** (``CUT="0"``) and absent on a
    never-offered kernel (no ``CUT`` key → the prior's NaN "not considered").

    Coarse like the rest of the ``D_*`` family — sized from the ``S_ext_free_prod``
    product the structural vocabulary carries (a cut producer's free output IS the
    materialized intermediate). Precise per-operand intermediate bytes (split from
    the consumer's own M·N output) and the cross-kernel ``D_cone_fanout`` /
    ``D_recompute_flops`` terms need per-operand shape stamping the coarse
    ``S_ext_*`` skeleton lacks — the deferred §3 follow-up."""
    import math  # noqa: PLC0415

    cut = 1 if str(knobs.get("PLACE@cone", "")) == "cut" else 0  # the materialize-to-gmem decision
    free = float(knobs.get("S_ext_free_prod", 0.0) or 0.0)
    return {"D_cut_roundtrip": math.log2(free) if (cut and free > 1.0) else 0.0}


def knob_features(knobs: dict) -> dict[str, float]:
    """Convert a knob dict into a flat numeric feature vector for the (future)
    learned planner prior — the single featurizer over the whole dict.

    - ``STRUCT_PREFIX`` (``S_``) structural-feature knobs and ``CTX_PREFIX``
      (``H_``) host/hardware-regime knobs pass through as floats: they already
      are the kernel's structural / regime feature set.
    - Registered tuning ``Knob``s are encoded by type: ``INT`` → float, ``BOOL``
      → 0/1, ``BINMASK`` (binary string) → ``{<name>_popcount, _width, _frac}``.
    - A ``Knob`` with a custom ``features`` callable (e.g. ``MMA``, which expands
      an atom kind into physical cell/dtype props) dispatches through it — no
      per-knob special-casing here.
    - Unregistered, non-structural knobs are best-effort float-coerced (skipped
      when non-numeric); other ``STR`` knobs have no generic encoding.
    """
    feats: dict[str, float] = {}
    for name, val in knobs.items():
        if name.startswith(STRUCT_PREFIX) or name.startswith(CTX_PREFIX):
            feats[name] = float(val)
            continue
        knob = get(name)
        if knob is not None and knob.features is not None:
            feats.update(knob.features(val))
            continue
        if knob is None:
            num = _coerce_float(val)
            if num is not None:
                feats[name] = num
            continue
        if knob.type is KnobType.INT:
            feats[name] = float(val)
        elif knob.type is KnobType.BOOL:
            feats[name] = 1.0 if _as_bool(val) else 0.0
        elif knob.type is KnobType.BINMASK:
            s = str(val)
            pop = float(s.count("1"))
            feats[f"{name}_popcount"] = pop
            feats[f"{name}_width"] = float(len(s))
            feats[f"{name}_frac"] = pop / len(s) if s else 0.0
        # STR knobs with no custom featurizer: no generic numeric encoding.
    # Atom (tensor-core cell) features. The legacy ``MMA`` key already routed through
    # its ``features`` callable in the loop above; the native schema names the atom on
    # ``ATOM@<cell>`` instead (no ``MMA`` key), so derive the same ``MMA_*`` features
    # from the native key here. Idempotent for a legacy row (same values).
    atom = mma_atom(knobs)
    if atom is not None:
        mk = get("MMA")
        if mk is not None and mk.features is not None:
            feats.update(mk.features(atom))
    feats.setdefault("MMA_tier", 0.0)  # scalar tier = no atom-kind knob present
    feats.update(_tile_features(knobs))
    # Warp-tier occupancy: the scalar ``_tile_features`` above models the thread
    # tile (``BN·BM``) and skips warp rows, so compute the SAME ``D_*`` family
    # from the warp tile geometry instead — using the atom cell dims the MMA
    # featurizer already put in ``feats`` (``MMA_atom_m`` / ``_n``), since knob.py
    # can't import ``ATOM_REGISTRY``. Shared ``D_*`` names across tiers let the
    # prior learn occupancy / CTA-count uniformly (the signal that picks the
    # skewed-vs-square warp tile per shape — the fp16 mis-pick).
    if is_warp(knobs):
        feats.update(_warp_tile_features(knobs, feats.get("MMA_atom_m"), feats.get("MMA_atom_n")))
    if "PLACE@cone" in knobs:  # the demoted-matmul cut's round-trip cost axis (only at offer sites)
        feats.update(_cut_features(knobs))
    return feats


def _free_slots(knobs: dict) -> tuple[int, int, int, int] | None:
    """Canonical ``(par_n, reg_n, par_m, reg_m)`` for the (≤2) tiled free axes.

    The **scalar** tier sources its free split from the ``TILE`` codec
    (``n<N>[xm<M>]/f<fn>[xf<fm>]`` — the schedule's free-axis output-tile knob); the **warp**
    tier keeps its ``WN``/``FN`` + ``WM``/``FM`` names (not yet rebuilt onto a codec). The two
    axes are canonicalized by ``par`` (the wider parallel binding is the ``n`` / coalesced
    slot, the narrower the ``m`` slot). A single free axis fills the ``n`` slot with a
    degenerate ``(1, 1)`` ``m`` slot. Returns ``None`` when no complete free split is present
    (a non-tiled kernel)."""

    def _nz(name: str) -> int:
        try:
            return int(knobs.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    pairs: list[tuple[int, int]] = []
    if _nz("WN") > 0 or _nz("WM") > 0:  # warp tier — legacy W*/F* names
        for par_key, reg_key in (("WN", "FN"), ("WM", "FM")):
            if par_key in knobs and reg_key in knobs:
                try:
                    pairs.append((int(knobs[par_key]), int(knobs[reg_key])))
                except (TypeError, ValueError):
                    return None
    else:  # scalar tier — the TILE codec
        from deplodock.compiler.ir.tile.schedule import TilePlan  # noqa: PLC0415

        plan = TilePlan.parse(knobs.get("TILE"))
        if not plan.is_tiled:
            return None
        pairs = [(plan.par_n, plan.reg_n), (plan.par_m, plan.reg_m)]
    if not pairs:
        return None
    pairs.sort(key=lambda pr: (pr[0], pr[1]), reverse=True)  # wider par = the n slot
    (par_n, reg_n) = pairs[0]
    (par_m, reg_m) = pairs[1] if len(pairs) >= 2 else (1, 1)
    return par_n, reg_n, par_m, reg_m


@dataclass(frozen=True)
class _Decomp:
    """Local stand-in for the demolished ``_families.Decomp`` — the reduce-axis
    decomposition factors the featurizer reads (``serial``/``fold``/``cta``/``coop``)
    plus the cross-CTA ``finalize`` codec letter."""

    serial: int = 1
    fold: int = 1
    cta: int = 1
    coop: int = 1
    finalize: str = "atomic"


def _reduce_decomp(knobs: dict) -> _Decomp:
    """The primary reduce axis's ``(cta, coop, reg)`` partition factors, decoded from the
    single ``REDUCE`` codec knob (``g<n>`` cta / ``b<n>`` coop / ``r<n>`` reg — the reduce
    tier's one decomposition knob, decided in ``lowering/tile/020_schedule``). The ``serial``
    remainder is derived from the schedule (``ceil(extent / parallel)``), not a knob, so it
    stays the ``_Decomp`` default. No legacy ``BK``/``FK``/``SPLITK``/``BR`` reads — the
    learned prior is refit on the ``REDUCE`` schema."""
    from deplodock.compiler.ir.tile.schedule import ReducePlan  # noqa: PLC0415

    plan = ReducePlan.parse(knobs.get("REDUCE"))
    return _Decomp(fold=plan.reg, cta=plan.cta, coop=plan.coop)


def tile_signature(knobs: dict) -> tuple:
    """Schema-agnostic structural identity of a tile config: the canonical free-axis
    slots, the primary reduce decomposition, and the atom kind — read from native
    ``MOVE@element`` keys or legacy GEMM-letter names alike. Two configs with equal
    signatures are the same kernel variant whichever schema spelled them, so this is
    the bridge for matching a legacy-recorded golden YAML against the native
    enumeration's candidate rows (``scripts/golden_knob_heuristics.py`` /
    ``search/analytic.evaluate_golden``)."""
    return (_free_slots(knobs), _reduce_decomp(knobs), mma_atom(knobs))


def _geom_feats(
    knobs: dict,
    *,
    threads: int,
    cells: int,
    tile_m: int,
    tile_n: int,
    splitk: int,
    bn: int,
    bm: int,
    bk: int,
    br: int,
    free_prod,
    sm: float,
    warp: bool,
    finalize: str = "atomic",
) -> dict[str, float]:
    """The engineered ``D_*`` tile-geometry / occupancy feature family — the
    single featurization the priors rank on. It folds in everything the old
    hand-coded matmul heuristic scored (occupancy waves, tile-area / thread /
    aspect targets, the geometry "bands", K-chunk depth), so a fixed linear model
    over these features (:class:`~deplodock.compiler.pipeline.search.prior.AnalyticPrior`)
    reproduces that heuristic and the learned ``CatBoostPrior`` sees the same
    derived signal a tree can't cheaply reconstruct from raw knobs + the *coarse*
    ``S_ext_*`` extents.

    Tier-aware: the "ideal" tile / thread targets differ between the scalar thread
    tile (256 threads, 8192-elem area) and the warp tile (128 threads = 4 warps,
    64×64 = 4096 area), selected by ``warp``. ``free_prod`` is the output free-dim
    product (``S_ext_free_prod``); when present the occupancy terms are added —
    ``#CTAs ≈ M·N / tile_area · SPLITK`` (ceil-free, needs only the product the
    ``S_*`` features carry, not the per-axis split). The ``BN``/``BM`` band
    features are the OFF sentinel ``0`` on a warp row (so they don't fire there);
    ``BK`` is a real knob on both tiers but pulls opposite ways, so it rides
    tier-split features (``D_*_bk`` scalar vs ``D_w_*_bk`` warp). The rest of the
    warp tier's signal rides the geometry / occupancy terms via the tier-aware
    targets."""
    import math  # noqa: PLC0415

    def l2(x: float) -> float:
        return math.log2(max(float(x), 1.0))

    area = max(tile_m * tile_n, 1)
    reuse = area / (tile_m + tile_n) if (tile_m + tile_n) else 0.0
    aspect = l2(tile_m) - l2(tile_n)
    thr_target = 7.0 if warp else 8.0  # log2 threads: 128 (4-warp) vs 256
    area_target = 12.0 if warp else 13.0  # log2 area: 64×64=4096 vs 8192
    masked_m = float(knobs.get("S_masked_m", 0.0) or 0.0)
    masked_n = float(knobs.get("S_masked_n", 0.0) or 0.0)
    masked_k = float(knobs.get("S_masked_k", 0.0) or 0.0)
    k_ext = float(knobs.get("S_ext_reduce_prod") or 0.0)
    kchunks = max((k_ext / br) / bk, 1.0) if k_ext > 0 else 1.0
    out = {
        # core geometry
        "D_threads": float(threads),
        "D_cells": float(cells),
        "D_tile_m": float(tile_m),
        "D_tile_n": float(tile_n),
        "D_log2_area": l2(area),
        "D_reuse": reuse,
        "D_aspect": aspect,
        # analytic (ex-heuristic) terms — tier-aware targets
        "D_l2_threads": l2(threads),
        "D_near_threads": -abs(l2(threads) - thr_target),
        "D_pow2_threads": 1.0 if threads > 0 and (threads & (threads - 1)) == 0 else 0.0,
        "D_cells_cap": min(float(cells), 128.0),
        "D_near_cells": -abs(float(cells) - 16.0),
        "D_near_area": -abs(l2(area) - area_target),
        "D_square": -abs(aspect),
        "D_l2_reuse": l2(reuse),
        "D_near_intensity": -abs(l2(reuse) - 5.0),
        "D_near_kchunks": -abs(l2(kchunks) - 5.0),
        # Per-role masked-tile penalties (was the single ``D_neg_overhang`` count):
        # split M / N / K so the prior can weight K-masking distinctly. Negative =
        # penalty, preserving the old sign convention.
        "D_neg_masked_m": -masked_m,
        "D_neg_masked_n": -masked_n,
        "D_neg_masked_k": -masked_k,
        # thread-tier geometry bands (raw BN/BM/BK/SPLITK; 0 on a warp row)
        "D_l2_bn": l2(bn),
        "D_l2_bm": l2(bm),
        "D_bn_ge_bm": 1.0 if bn > 0 and bn >= bm else 0.0,
        "D_bn_band": 1.0 if 16 <= bn <= 64 else 0.0,
        "D_bm_band": 1.0 if 8 <= bm <= 16 else 0.0,
        # BK bands are tier-specific: the scalar tile wants deep K-chunks (BK≥32)
        # while the warp / TMA tile wants a shallow pipelined BK≈2 — opposite
        # directions, so they ride separate features (one weight can't serve both).
        "D_l2_bk": 0.0 if warp else l2(bk),
        "D_bk_ge32": 0.0 if warp else (1.0 if bk >= 32 else 0.0),
        "D_w_l2_bk": l2(bk) if warp else 0.0,
        "D_w_near_bk": (-abs(l2(bk) - 1.0)) if warp else 0.0,
        "D_splitk": float(splitk),
        "D_splitk_le2": 1.0 if splitk <= 2 else 0.0,
        # Cross-CTA finalize fold (the REDUCE codec ``c`` field's letter): 1.0 = deferred
        # KERNEL combine (``c<cta>k``), 0.0 = in-place ATOMIC (``c<cta>a`` / bare). Replaces
        # the removed ``NOATOMIC`` knob feature; the analytic prior's split-K gate reads it.
        "D_finalize_kernel": 1.0 if (splitk > 1 and finalize == "kernel") else 0.0,
        "D_tilen_clean": 1.0 if tile_n in (32, 64, 128) else 0.0,
        "D_near_tilen": -abs(l2(tile_n) - 6.0),
    }
    if free_prod:
        ctas = float(free_prod) / area * splitk
        waves = math.log2(max(ctas / sm, 1e-3))
        out["D_log2_ctas"] = l2(ctas)
        out["D_log2_waves"] = waves  # CTAs relative to SM count
        out["D_near_waves"] = -abs(waves - 1.0)  # target ~2 waves
        out["D_ctas_ge_sm"] = 1.0 if ctas >= sm else 0.0
        # Split-K beyond what occupancy needs is pure atomic/combine waste. The free
        # axes alone give ``free_ctas = free_prod/area`` CTAs; split-K is justified
        # only to lift that toward the ~2·SM ``D_near_waves`` target. The terms above
        # fold ``splitk`` straight into ``ctas``, so they CANNOT tell "≈2 waves via a
        # small tile" (golden, free) from "≈2 waves via heavy split-K on a big tile"
        # (atomic-bound) — both score the same waves / ctas≥sm. This credits split-K
        # up to the need and penalizes the excess, the engineered signal the learned
        # prior needs to separate the SPLITK=1/2 goldens from the SPLITK=8/16 tiles
        # the -O1 sweep over-ranks (the analytic prior already gets it via D_splitk_le2).
        free_ctas = float(free_prod) / area
        needed = max(2.0 * sm / max(free_ctas, 1.0), 1.0)
        out["D_splitk_excess"] = math.log2(max(splitk / needed, 1.0))
        # Register-tile intensity × occupancy interaction: a wide per-thread
        # register tile (big FM·FN) is a win only while the grid still covers
        # the SMs — the flat D_cells* terms can't express that, so the big-FM
        # goldens (square.2048's FM=26) rank deep under any sign the fit gives
        # them (2026-06-12 golden-sweep finding 2).
        out["D_l2_cells_occ"] = l2(cells) if ctas >= sm else 0.0
    return out


def _tile_features(knobs: dict) -> dict[str, float]:
    """Scalar thread-tile ``D_*`` features (``BN·BM`` threads, ``BM·FM × BN·FN``
    output). Empty unless the core tile knobs (``BN/BM/FM/FN``) are present, so
    pointwise / non-tiled kernels are unaffected. Warp-tier (tensor-core) rows
    are skipped here — :func:`knob_features` computes their occupancy via
    :func:`_warp_tile_features` (the warp tile is ``WM·WN·32`` threads,
    ``WM·FM·atom_m × WN·FN·atom_n`` output), so the warp ``BM=BN=0`` OFF
    sentinels don't feed a meaningless scalar tile."""
    if is_warp(knobs):
        return {}
    slots = _free_slots(knobs)
    if slots is None:
        return {}
    par_n, reg_n, par_m, reg_m = slots  # (BN, FN, BM, FM)
    d = _reduce_decomp(knobs)
    bn, bm, fm, fn, br, bk, splitk = par_n, par_m, reg_m, reg_n, d.coop, d.serial, d.cta
    return _geom_feats(
        knobs,
        threads=bn * bm * br,
        cells=fm * fn,
        tile_m=bm * fm,
        tile_n=bn * fn,
        splitk=splitk,
        bn=bn,
        bm=bm,
        bk=bk,
        br=br,
        free_prod=knobs.get("S_ext_free_prod"),
        sm=float(knobs.get("H_sm_count") or 170.0),
        warp=False,
        finalize=d.finalize,
    )


def _warp_tile_features(knobs: dict, atom_m: float | None, atom_n: float | None) -> dict[str, float]:
    """Warp-tier (tensor-core MMA) tile ``D_*`` features — the warp analogue of
    :func:`_tile_features`. The CTA runs ``WM·WN`` warps (``·32`` lanes) over a
    ``WM·FM·atom_m × WN·FN·atom_n`` output tile, where ``atom_m/atom_n`` are the
    MMA cell dims the featurizer already derived. Empty if the warp knobs or atom
    dims are missing (so a malformed row degrades gracefully)."""
    slots = _free_slots(knobs)
    if slots is None:
        return {}
    try:
        am, an = int(atom_m), int(atom_n)
    except (TypeError, ValueError):
        return {}
    wn, fn, wm, fm = slots  # (WN, FN, WM, FM) — warp counts in the par slots
    if wm <= 0 or wn <= 0:
        return {}
    d = _reduce_decomp(knobs)
    return _geom_feats(
        knobs,
        threads=wm * wn * 32,
        cells=fm * fn,
        tile_m=wm * fm * am,
        tile_n=wn * fn * an,
        splitk=d.cta,
        bn=0,  # OFF sentinels: the BN/BM bands don't fire on a warp row
        bm=0,
        bk=d.serial,
        br=d.coop,
        free_prod=knobs.get("S_ext_free_prod"),
        sm=float(knobs.get("H_sm_count") or 170.0),
        warp=True,
    )


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "on"}
    return bool(v)


def _coerce_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
