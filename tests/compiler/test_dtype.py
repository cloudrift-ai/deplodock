"""Scalar / structured ``DataType`` hierarchy."""

from __future__ import annotations

from emmy.compiler.dtype import BF16, F16, F32, DataType, F16x2, StructuredType, get


def test_scalars_are_not_structured():
    for dt in (F32, F16, BF16):
        assert not dt.is_structured
        assert not isinstance(dt, StructuredType)


def test_f16x2_is_structured():
    assert F16x2.is_structured
    assert isinstance(F16x2, StructuredType)
    # Still a DataType, still resolvable by canonical name.
    assert isinstance(F16x2, DataType)
    assert get("f16x2") is F16x2


def test_structured_keeps_scalar_carrier_info():
    # The packed type still reports a usable numpy dtype + byte width
    # (one 32-bit register = two fp16).
    assert F16x2.np == F16.np
    assert F16x2.nbytes == 4
