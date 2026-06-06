"""Unit tests for ``Knob`` parse / pretty round-trips and the registry."""

from __future__ import annotations

import pytest

import deplodock.compiler.pipeline.knob as knob_mod
from deplodock.compiler.pipeline.knob import Knob, KnobType, apply_knobs_env, format_tuning_knobs, knob_features


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
    k = Knob("STAGE", KnobType.BINMASK)
    # char i = bit i (left-to-right reads as buffer rank 0..n-1)
    assert k.parse("101", width=3) == 0b101
    assert k.parse("000", width=3) == 0
    assert k.parse("111", width=3) == 0b111


def test_binmask_parse_keywords():
    k = Knob("STAGE", KnobType.BINMASK)
    assert k.parse("all", width=3) == 0b111
    assert k.parse("all", width=5) == 0b11111
    assert k.parse("none", width=3) == 0


def test_binmask_parse_int_clamps_to_width():
    k = Knob("STAGE", KnobType.BINMASK)
    assert k.parse("0xFFFF", width=3) == 0b111
    assert k.parse("5", width=3) == 0b101


def test_binmask_pretty():
    k = Knob("STAGE", KnobType.BINMASK)
    assert k.pretty(0b101, width=3) == "101"
    assert k.pretty(0, width=3) == "000"
    assert k.pretty(0b111, width=3) == "111"


def test_binmask_roundtrip():
    k = Knob("STAGE", KnobType.BINMASK)
    for mask in range(16):
        assert k.parse(k.pretty(mask, width=4), width=4) == mask


def test_binmask_requires_width():
    k = Knob("STAGE", KnobType.BINMASK)
    with pytest.raises(ValueError, match="width"):
        k.parse("101")
    with pytest.raises(ValueError, match="width"):
        k.pretty(5)


def test_env_property():
    assert Knob("BN", KnobType.INT).env == "DEPLODOCK_BN"
    assert Knob("STAGE", KnobType.BINMASK).env == "DEPLODOCK_STAGE"


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
    k = Knob("STAGE", KnobType.BINMASK)
    monkeypatch.setenv("DEPLODOCK_STAGE", "111")
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
            "STAGE": Knob("STAGE", KnobType.BINMASK),
        },
    )
    feats = knob_features({"BN": 64, "FLAG": True, "STAGE": "101"})
    assert feats["BN"] == 64.0
    assert feats["FLAG"] == 1.0
    assert feats["STAGE_popcount"] == 2.0
    assert feats["STAGE_width"] == 3.0
    assert feats["STAGE_frac"] == 2 / 3


def test_knob_features_mma_expansion():
    # MMA expansion now dispatches through the MMA Knob's ``features`` callable,
    # so the declaring module must be loaded + present in the registry.
    from deplodock.compiler.pipeline.passes.lowering.tile import _enumeration  # noqa: F401, PLC0415

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


def test_format_tuning_knobs_skips_struct():
    out = format_tuning_knobs({"BN": 64, "S_n_load": 3.0, "S_ext_free_prod": 512.0})
    assert "S_n_load" not in out and "S_ext_free_prod" not in out
    assert "BN=64" in out


def test_format_tuning_knobs_canonical_order():
    """Knobs render in canonical tile-geometry order (``KNOB_ORDER``), not
    alphabetical — shared with the ``deplodock eval`` golden tables."""
    out = format_tuning_knobs({"SPLITK": 1, "BN": 32, "BM": 8, "BK": 64, "MMA": "x", "FM": 4})
    assert out == "BM=8, BN=32, BK=64, FM=4, SPLITK=1, MMA=x"


def test_apply_knobs_env_no_raw_falls_back_to_env(monkeypatch):
    """With no ``raw`` argument, the function reads ``DEPLODOCK_KNOBS``."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    monkeypatch.setenv("DEPLODOCK_KNOBS", "BK=8")
    applied = apply_knobs_env()
    assert applied == {"DEPLODOCK_BK": "8"}
