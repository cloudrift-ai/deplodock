"""Unit tests for ``Knob`` parse / pretty round-trips and the registry."""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.knob import Knob, KnobType


def test_int_parse():
    k = Knob("BN", KnobType.INT)
    assert k.parse("64") == 64
    assert k.parse("0x40") == 64
    assert k.parse("  128 ") == 128


def test_int_pretty():
    k = Knob("BN", KnobType.INT)
    assert k.pretty(64) == "64"


def test_bool_parse():
    k = Knob("register_tile", KnobType.BOOL)
    for truthy in ("1", "true", "True", "yes", "on", " TRUE "):
        assert k.parse(truthy) is True
    for falsy in ("0", "false", "no", "off", ""):
        assert k.parse(falsy) is False


def test_bool_pretty():
    k = Knob("register_tile", KnobType.BOOL)
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


