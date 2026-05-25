"""Unit tests for ``Knob`` parse / pretty round-trips and the registry."""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.knob import Knob, KnobType, apply_knobs_env


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


def test_narrow_pinned_drops_unmatched(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("DEPLODOCK_BN", "128")
    assert k.narrow((16, 32, 64)) == ()


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


# ---------------------------------------------------------------------------
# DEPLODOCK_KNOBS aggregate env var
# ---------------------------------------------------------------------------


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


def test_apply_knobs_env_no_raw_falls_back_to_env(monkeypatch):
    """With no ``raw`` argument, the function reads ``DEPLODOCK_KNOBS``."""
    monkeypatch.delenv("DEPLODOCK_BK", raising=False)
    monkeypatch.setenv("DEPLODOCK_KNOBS", "BK=8")
    applied = apply_knobs_env()
    assert applied == {"DEPLODOCK_BK": "8"}
