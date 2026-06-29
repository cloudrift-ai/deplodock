"""Unit tests for ``Knob`` parse / pretty round-trips and the registry."""

from __future__ import annotations

import pytest

import deplodock.compiler.pipeline.knob as knob_mod
from deplodock.compiler.pipeline.knob import (
    Knob,
    KnobType,
    apply_knobs_env,
    apply_off_defaults,
    format_tuning_knobs,
    is_warp,
    knob_features,
    mma_atom,
    mma_decode,
)


def test_int_parse():
    k = Knob("BN", KnobType.INT)
    assert k.parse("64") == 64
    assert k.parse("0x40") == 64
    assert k.parse("  128 ") == 128


def test_int_pretty():
    k = Knob("BN", KnobType.INT)
    assert k.pretty(64) == "64"


def test_bool_parse():
    k = Knob("FLAG", KnobType.BOOL)
    for truthy in ("1", "true", "True", "yes", "on", " TRUE "):
        assert k.parse(truthy) is True
    for falsy in ("0", "false", "no", "off", ""):
        assert k.parse(falsy) is False


def test_bool_pretty():
    k = Knob("FLAG", KnobType.BOOL)
    assert k.pretty(True) == "True"
    assert k.pretty(False) == "False"


def test_binmask_parse_binary_string():
    k = Knob("MASK", KnobType.BINMASK)
    # char i = bit i (left-to-right reads as buffer rank 0..n-1)
    assert k.parse("101", width=3) == 0b101
    assert k.parse("000", width=3) == 0
    assert k.parse("111", width=3) == 0b111


def test_binmask_parse_keywords():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.parse("all", width=3) == 0b111
    assert k.parse("all", width=5) == 0b11111
    assert k.parse("none", width=3) == 0


def test_binmask_parse_int_clamps_to_width():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.parse("0xFFFF", width=3) == 0b111
    assert k.parse("5", width=3) == 0b101


def test_binmask_pretty():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.pretty(0b101, width=3) == "101"
    assert k.pretty(0, width=3) == "000"
    assert k.pretty(0b111, width=3) == "111"


def test_binmask_roundtrip():
    k = Knob("MASK", KnobType.BINMASK)
    for mask in range(16):
        assert k.parse(k.pretty(mask, width=4), width=4) == mask


def test_binmask_requires_width():
    k = Knob("MASK", KnobType.BINMASK)
    with pytest.raises(ValueError, match="width"):
        k.parse("101")
    with pytest.raises(ValueError, match="width"):
        k.pretty(5)


def test_env_property():
    assert Knob("BN", KnobType.INT).env == "DEPLODOCK_BN"
    assert Knob("MASK", KnobType.BINMASK).env == "DEPLODOCK_MASK"


# ---------------------------------------------------------------------------
# Knob.narrow — fold env pin into candidate enumeration
# ---------------------------------------------------------------------------


def test_narrow_unpinned_returns_candidates_unchanged(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.delenv("DEPLODOCK_BN", raising=False)
    assert k.narrow((16, 32, 64)) == (16, 32, 64)


def test_narrow_pinned_keeps_matching_candidate(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("DEPLODOCK_BN", "32")
    assert k.narrow((16, 32, 64)) == (32,)


def test_narrow_pinned_out_of_set_is_authoritative(monkeypatch):
    # Hints are guidance, not constraint — an env pin outside the candidate
    # tuple is honored, not silently dropped. Downstream structural gates
    # (divisibility, threads-per-CTA budget, …) still apply.
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("DEPLODOCK_BN", "128")
    assert k.narrow((16, 32, 64)) == (128,)


def test_narrow_accepts_arbitrary_iterable(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("DEPLODOCK_BN", "16")
    # generator, not a tuple
    assert k.narrow(x for x in (8, 16, 32)) == (16,)


def test_narrow_bool(monkeypatch):
    k = Knob("FLAG", KnobType.BOOL)
    monkeypatch.setenv("DEPLODOCK_FLAG", "true")
    assert k.narrow((True, False)) == (True,)
    monkeypatch.setenv("DEPLODOCK_FLAG", "0")
    assert k.narrow((True, False)) == (False,)


def test_narrow_binmask_rejected(monkeypatch):
    k = Knob("MASK", KnobType.BINMASK)
    monkeypatch.setenv("DEPLODOCK_MASK", "111")
    with pytest.raises(ValueError, match="BINMASK"):
        k.narrow((0b000, 0b111))


def test_raw_alias_fallback(monkeypatch):
    """``Knob.raw`` reads the primary ``DEPLODOCK_<NAME>`` first, then each
    alias in declaration order (e.g. ``MMA`` accepting ``ATOM_KIND``)."""
    k = Knob("NEWNAME", KnobType.STR, aliases=("OLDNAME",))
    monkeypatch.delenv("DEPLODOCK_NEWNAME", raising=False)
    monkeypatch.delenv("DEPLODOCK_OLDNAME", raising=False)
    assert k.raw() is None
    monkeypatch.setenv("DEPLODOCK_OLDNAME", "via-alias")
    assert k.raw() == "via-alias"
    monkeypatch.setenv("DEPLODOCK_NEWNAME", "primary")
    assert k.raw() == "primary"


def test_narrow_reads_alias(monkeypatch):
    """``Knob.narrow`` honors an alias-spelled env pin."""
    k = Knob("NEWNAME", KnobType.INT, aliases=("OLDNAME",))
    monkeypatch.delenv("DEPLODOCK_NEWNAME", raising=False)
    monkeypatch.setenv("DEPLODOCK_OLDNAME", "8")
    assert k.narrow((1, 2)) == (8,)


# ---------------------------------------------------------------------------
# OFF defaults + tier (is_warp / mma_atom)
# ---------------------------------------------------------------------------


def test_apply_off_defaults_fills_only_unspecified_off_knobs():
    """``apply_off_defaults`` stamps a declared knob's ``off`` when absent, leaves
    present values (incl. a prior OFF fill) untouched, and never fills a knob
    whose ``off`` is unset (the default)."""
    wm = Knob("WM", KnobType.INT, off=0)
    bk = Knob("BK", KnobType.INT)  # no off → never auto-filled
    knobs = {"BK": 64}
    apply_off_defaults(knobs, [wm, bk])
    assert knobs == {"BK": 64, "WM": 0}  # WM filled to off, BK untouched (no off)
    # Idempotent + respects a present (non-OFF) value.
    knobs2 = {"WM": 2, "BK": 64}
    apply_off_defaults(knobs2, [wm, bk])
    assert knobs2 == {"WM": 2, "BK": 64}


def test_mma_decode_value_semantics():
    """``mma_decode`` maps unset/empty/truthy → auto, falsy → scalar-only, an
    atom name → pinned warp."""
    assert mma_decode(None) == (True, None)
    assert mma_decode("") == (True, None)
    assert mma_decode("1") == (True, None)
    assert mma_decode("0") == (False, None)
    assert mma_decode("false") == (False, None)
    assert mma_decode("mma_m16n8k16_f16") == (True, "mma_m16n8k16_f16")


def test_is_warp_and_mma_atom_tier_discriminator():
    """The ``"0"`` OFF sentinel (and absent / falsy / auto) read as scalar; only a
    concrete atom name is the warp tier. Guards the truthy-string footgun: the
    old ``knobs.get("MMA")`` check misread ``"0"`` as warp."""
    assert not is_warp({}) and mma_atom({}) is None
    assert not is_warp({"MMA": "0"}) and mma_atom({"MMA": "0"}) is None
    assert not is_warp({"MMA": "1"})  # pre-enumeration auto control, not an atom
    assert is_warp({"MMA": "mma_m16n8k16_f16"})
    assert mma_atom({"MMA": "mma_m16n8k16_f16"}) == "mma_m16n8k16_f16"


def test_scalar_tile_features_from_thread_tile():
    """``knob_features`` emits the ``D_*`` occupancy family for a scalar row from its
    ``TILE`` codec free split (``par_n·par_m`` threads, ``par_m·reg_m × par_n·reg_n``
    output) — ``n32xm8`` parallel thread-tile, ``f2xf4`` register sub-tile."""
    sf = knob_features({"TILE": "n32xm8/f2xf4", "MMA": "0", "WM": 0, "WN": 0})
    assert any(k.startswith("D_") for k in sf)
    assert sf["D_threads"] == 32 * 8
    assert sf["D_tile_m"] == 8 * 4 and sf["D_tile_n"] == 32 * 2


def test_warp_tile_features_from_warp_tile():
    """``_warp_tile_features`` builds the ``D_*`` family from the warp tile
    (``WM·WN·32`` threads, ``WM·FM·atom_m × WN·FN·atom_n`` output) — the warp
    ``BM=BN=0`` OFF sentinels never feed a scalar tile. Atom dims (16×8 for
    ``m16n8k16``) come from the MMA featurizer in the real pipeline; here passed
    directly to avoid needing the registry loaded."""
    from deplodock.compiler.pipeline.knob import _warp_tile_features  # noqa: PLC0415

    wf = _warp_tile_features({"WM": 2, "WN": 2, "FM": 2, "FN": 2, "SPLITK": 1, "S_ext_free_prod": 2048 * 2048}, 16.0, 8.0)
    assert wf["D_threads"] == 128.0  # WM·WN·32
    assert wf["D_tile_m"] == 2 * 2 * 16  # WM·FM·atom_m
    assert wf["D_tile_n"] == 2 * 2 * 8  # WN·FN·atom_n
    assert "D_log2_ctas" in wf and "D_log2_waves" in wf  # occupancy present (free_prod given)
    assert _warp_tile_features({"WM": 2, "WN": 2, "FM": 2, "FN": 2}, None, None) == {}  # missing atom dims → empty


# ---------------------------------------------------------------------------
# DEPLODOCK_KNOBS aggregate env var
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_deplodock_env():
    """``apply_knobs_env`` writes ``DEPLODOCK_<K>`` via ``config.set_knob`` —
    a direct ``os.environ`` write monkeypatch can't undo (``delenv`` on an
    absent var records nothing to restore), so splatted pins used to leak
    into later tests in the same xdist worker (pin-sensitive planner tests
    then enumerate under stray ``BK``/``BM``/``BN`` pins). Snapshot + restore
    the whole ``DEPLODOCK_*`` namespace around each test."""
    import os  # noqa: PLC0415

    saved = {k: v for k, v in os.environ.items() if k.startswith("DEPLODOCK_")}
    yield
    for k in [k for k in os.environ if k.startswith("DEPLODOCK_")]:
        if k not in saved:
            del os.environ[k]
    os.environ.update(saved)


def test_apply_knobs_env_splats_into_individual_keys(monkeypatch):
    """Aggregate env var sets ``DEPLODOCK_<K>`` per entry."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    monkeypatch.delenv("DEPLODOCK_BM", raising=False)
    monkeypatch.delenv("DEPLODOCK_BN", raising=False)
    applied = apply_knobs_env("BK=2,BM=16,BN=128")
    assert applied == {"DEPLODOCK_BK": "2", "DEPLODOCK_BM": "16", "DEPLODOCK_BN": "128"}


def test_apply_knobs_env_individual_takes_precedence(monkeypatch):
    """An explicit ``DEPLODOCK_<K>`` wins over the aggregate."""
    monkeypatch.setenv("DEPLODOCK_BK", "4")
    monkeypatch.delenv("DEPLODOCK_BM", raising=False)
    applied = apply_knobs_env("BK=2,BM=16")
    assert "DEPLODOCK_BK" not in applied  # not clobbered
    assert applied == {"DEPLODOCK_BM": "16"}
    import os

    assert os.environ["DEPLODOCK_BK"] == "4"
    assert os.environ["DEPLODOCK_BM"] == "16"


def test_apply_knobs_env_tolerates_whitespace(monkeypatch):
    """Whitespace around keys / values / separators is stripped."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    monkeypatch.delenv("DEPLODOCK_BM", raising=False)
    applied = apply_knobs_env(" BK = 2 ,  BM=16 ")
    assert applied == {"DEPLODOCK_BK": "2", "DEPLODOCK_BM": "16"}


def test_apply_knobs_env_skips_empty_entries(monkeypatch):
    """Empty entries (trailing comma, double comma) are skipped."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    applied = apply_knobs_env("BK=2,,")
    assert applied == {"DEPLODOCK_BK": "2"}


def test_apply_knobs_env_rejects_missing_equals():
    """An entry without ``=`` is malformed and surfaces an error."""
    with pytest.raises(ValueError, match="missing '='"):
        apply_knobs_env("BK=2,BMnoequals")


def test_apply_knobs_env_rejects_empty_key():
    """An entry like ``=4`` has an empty KEY and is rejected."""
    with pytest.raises(ValueError, match="empty KEY"):
        apply_knobs_env("=4")


def test_parse_knob_spec_grammar():
    """``parse_knob_spec`` is the one owner of the ``K1=V1,K2=V2`` grammar
    (``DEPLODOCK_KNOBS`` splat + ``run --ab``): uppercased keys in spec order,
    whitespace tolerated, empties skipped, values kept as raw strings."""
    from deplodock.compiler.pipeline.knob import parse_knob_spec

    assert parse_knob_spec(" bk = 2 ,, BM=16, STAGE=d2/cp ") == {"BK": "2", "BM": "16", "STAGE": "d2/cp"}
    assert parse_knob_spec("") == {}
    with pytest.raises(ValueError, match="missing '='"):
        parse_knob_spec("BK2")


def test_apply_knobs_env_uppercases_key(monkeypatch):
    """Lowercased keys round-trip to the upper-case env-var convention."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    applied = apply_knobs_env("bk=2")
    assert applied == {"DEPLODOCK_BK": "2"}


# ---------------------------------------------------------------------------
# knob_features — knob dict → flat numeric feature vector
# ---------------------------------------------------------------------------


def test_knob_features_struct_passthrough():
    feats = knob_features({"S_n_load": 3.0, "S_ext_free_prod": 512.0})
    assert feats["S_n_load"] == 3.0
    assert feats["S_ext_free_prod"] == 512.0


def test_knob_features_typed_knobs(monkeypatch):
    monkeypatch.setattr(
        knob_mod,
        "_REGISTRY",
        {
            "BN": Knob("BN", KnobType.INT),
            "FLAG": Knob("FLAG", KnobType.BOOL),
            "MASK": Knob("MASK", KnobType.BINMASK),
        },
    )
    feats = knob_features({"BN": 64, "FLAG": True, "MASK": "101"})
    assert feats["BN"] == 64.0
    assert feats["FLAG"] == 1.0
    assert feats["MASK_popcount"] == 2.0
    assert feats["MASK_width"] == 3.0
    assert feats["MASK_frac"] == 2 / 3


def test_knob_features_stage_codec():
    """The ``STAGE`` codec (``d<depth>/sync|cp|tma[/ring]``) featurizes to the ``D_stage_*``
    family; an absent / gmem-direct stage contributes nothing."""
    feats = knob_features({"STAGE": "d3/tma/ring"})
    assert feats["D_stage_depth"] == 3.0
    assert feats["D_stage_async"] == 1.0
    assert feats["D_stage_tma"] == 1.0
    assert feats["D_stage_ring"] == 1.0
    sync = knob_features({"STAGE": "d2/cp"})
    assert sync["D_stage_depth"] == 2.0 and sync["D_stage_async"] == 1.0 and sync["D_stage_tma"] == 0.0
    assert not any(k.startswith("D_stage_") for k in knob_features({"STAGE": ""}))


def test_knob_features_mma_expansion():
    # MMA expansion now dispatches through the MMA Knob's ``features`` callable,
    # so the declaring module must be loaded + present in the registry.
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knobs  # noqa: F401, PLC0415

    knob_mod.reset_registry()
    feats = knob_features({"MMA": "mma_m16n8k16_f16"})
    assert feats["MMA_tier"] == 1.0
    assert (feats["MMA_atom_m"], feats["MMA_atom_n"], feats["MMA_atom_k"]) == (16.0, 8.0, 16.0)
    assert feats["MMA_a_bits"] == 16.0  # f16 operand
    assert feats["MMA_acc_bits"] == 32.0  # f32 accumulator


def test_knob_features_scalar_tier_default():
    feats = knob_features({"S_n_load": 2.0})
    assert feats["MMA_tier"] == 0.0  # no atom selected


def test_knob_features_unregistered_numeric_vs_string():
    feats = knob_features({"weird_num": 7, "weird_str": "not_a_number"})
    assert feats["weird_num"] == 7.0
    assert "weird_str" not in feats


def test_knob_features_differs_by_one_knob():
    a = knob_features({"S_n_load": 2.0, "S_n_write": 1.0})
    b = knob_features({"S_n_load": 3.0, "S_n_write": 1.0})
    assert a["S_n_load"] != b["S_n_load"]
    assert a["S_n_write"] == b["S_n_write"]


def test_knob_features_cut_roundtrip():
    import math

    # A cut fragment (PLACE@cone=cut) carries the materialized round-trip volume,
    # sized from the coarse S_ext_free_prod product; the fused keep (PLACE@cone=inline)
    # carries it as 0.0 — the cost axis that discriminates the two realizations.
    cut = knob_features({"PLACE@cone": "cut", "S_ext_free_prod": 4096.0})
    assert cut["D_cut_roundtrip"] == math.log2(4096.0)
    keep = knob_features({"PLACE@cone": "inline", "S_ext_free_prod": 4096.0})
    assert keep["D_cut_roundtrip"] == 0.0
    # Never-offered kernels (no PLACE@cone key) stay free of the feature — the prior's
    # "not considered" NaN state, never a spurious 0.
    assert "D_cut_roundtrip" not in knob_features({"S_ext_free_prod": 4096.0})


def test_format_tuning_knobs_skips_struct():
    out = format_tuning_knobs({"BN": 64, "S_n_load": 3.0, "S_ext_free_prod": 512.0})
    assert "S_n_load" not in out and "S_ext_free_prod" not in out
    assert "BN=64" in out


def test_format_tuning_knobs_canonical_order():
    """Knobs render in canonical tile-geometry order (``KNOB_ORDER``), not
    alphabetical — shared with the ``deplodock eval`` golden tables. (Uses a
    tier-consistent warp dict so the tier-foreign display filter doesn't drop
    anything — a warp variant carries WM/WN/MMA, not BM/BN.)"""
    out = format_tuning_knobs({"SPLITK": 1, "WN": 4, "WM": 2, "BK": 64, "MMA": "x", "FM": 4})
    assert out == "BK=64, FM=4, WM=2, WN=4, SPLITK=1, MMA=x"


def test_apply_knobs_env_no_raw_falls_back_to_env(monkeypatch):
    """With no ``raw`` argument, the function reads ``DEPLODOCK_KNOBS``."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    monkeypatch.setenv("DEPLODOCK_KNOBS", "BK=8")
    applied = apply_knobs_env()
    assert applied == {"DEPLODOCK_BK": "8"}
