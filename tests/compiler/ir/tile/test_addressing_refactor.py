"""Addressing-mode refactor.

Pure structural change: ``Source.addressing`` is now a stored field of
type ``AffineAddressing | TemplateAddressing`` instead of a property
derived from ``Source.template_index``. ``AffineAddressing`` gains a
per-cache-dim ``block`` multiplier (default ``()`` = trivial). Tests
here pin the new shape directly; the broader byte-clean gate is the
existing ``tests/compiler/`` suite running unchanged.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.tile.ir import (
    AffineAddressing,
    Source,
    TemplateAddressing,
)


def _src(*, addressing: AffineAddressing | TemplateAddressing | None = None) -> Source:
    """16×32 slab on dims (0, 1) — extents 16 and 32. Shared shape for
    every test below so the slab-sizing assertions read in elements."""
    return Source(
        name="a_smem",
        buf="a",
        cache_axes=(Axis("m_r", 16), Axis("k_i", 32)),
        origin=(Literal(0, "int"), Literal(0, "int")),
        addressing=addressing,
    )


def test_default_addressing_is_affine_identity_dims():
    """Omitting ``addressing`` materializes ``AffineAddressing`` with the
    identity cache-axis → source-dim mapping (``dims == range(len)``)."""
    src = _src()
    assert isinstance(src.addressing, AffineAddressing)
    assert src.addressing.dims == (0, 1)
    assert src.addressing.block == ()


def test_explicit_template_addressing_round_trips():
    """Template addressing replaces the old ``template_index=...`` shape
    one-for-one: ``addressing.exprs`` holds the verbatim per-source-dim
    Exprs."""
    exprs = (Var("foo"), Var("bar") + Literal(1, "int"))
    src = _src(addressing=TemplateAddressing(exprs=exprs))
    assert isinstance(src.addressing, TemplateAddressing)
    assert src.addressing.exprs == exprs


def test_alloc_extents_trivial_block_matches_cache_extents():
    """With ``block=()`` the slab is sized to bare cache extents
    (16 × 32 elements) — pre-M2 behavior, the byte-clean gate."""
    src = _src()
    assert src.alloc_extents == (16, 32)


def test_alloc_extents_grows_by_block():
    """A non-trivial ``block`` scales each axis. For an MMA m16n16k16
    operand staging A (m_r × atom_M, k_i × atom_K) the slab is the cell-
    block-sized footprint, not bare cache extents."""
    addr = AffineAddressing(dims=(0, 1), block=(16, 16))
    src = _src(addressing=addr)
    # 16 m_r positions × 16 atom rows = 256 slab rows; same on K.
    assert src.alloc_extents == (16 * 16, 32 * 16)


def test_alloc_extents_pad_adds_after_block():
    """Pad applies after the block multiplier so MMA-shape extents +
    bank-conflict pad compose without re-deriving block math."""
    addr = AffineAddressing(dims=(0, 1), block=(1, 16))
    src = Source(
        name="b_smem",
        buf="b",
        cache_axes=(Axis("n_r", 16), Axis("k_i", 32)),
        origin=(Literal(0, "int"), Literal(0, "int")),
        pad=(0, 1),
        addressing=addr,
    )
    # (16 × 1, 32 × 16) + (0, 1) = (16, 513).
    assert src.alloc_extents == (16, 513)


def test_affine_addressing_block_length_must_match_dims():
    with pytest.raises(ValueError, match="block length"):
        AffineAddressing(dims=(0, 1), block=(1,))


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_affine_addressing_block_entries_must_be_positive_ints(bad):
    with pytest.raises(ValueError, match="block"):
        AffineAddressing(dims=(0,), block=(bad,))


def test_template_addressing_repr_eval_round_trip():
    """``Source.addressing`` shows up as a stored field in repr — this
    pins the shape for the Graph eval-scope serializer."""
    exprs = (Var("foo"),)
    src = _src(addressing=TemplateAddressing(exprs=exprs))
    rep = repr(src)
    assert "addressing=TemplateAddressing(exprs=" in rep


def test_affine_addressing_block_in_repr_when_nontrivial():
    addr = AffineAddressing(dims=(0, 1), block=(1, 16))
    rep = repr(addr)
    assert "block=(1, 16)" in rep
