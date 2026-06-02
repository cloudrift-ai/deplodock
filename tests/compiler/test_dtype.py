"""Scalar / structured ``DataType`` hierarchy + ``FragmentType``."""

from __future__ import annotations

from deplodock.compiler.dtype import BF16, F16, F32, DataType, F16x2, FragmentType, StructuredType, get


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


def test_fragment_type_derives_element_from_spec():
    ft = FragmentType("mma_m16n8k16_f16", "a")
    assert ft.is_structured and isinstance(ft, StructuredType)
    assert ft.atom == "mma_m16n8k16_f16"
    assert ft.role == "a"
    # element + nbytes derived from the atom spec's operand dtype (a -> F16).
    assert ft.element is F16
    assert ft.nbytes == F16.nbytes
    assert ft.name == "frag:mma_m16n8k16_f16:a"


def test_fragment_c_role_is_fp32_accumulator():
    c = FragmentType("mma_m16n8k16_f16", "c")
    assert c.element is F32  # the accumulator is always fp32


def test_fragment_roles_distinct():
    a = FragmentType("mma_m16n8k16_f16", "a")
    b = FragmentType("mma_m16n8k16_f16", "b")
    assert a != b
    assert a.role != b.role
    assert hash(a) != hash(b)
