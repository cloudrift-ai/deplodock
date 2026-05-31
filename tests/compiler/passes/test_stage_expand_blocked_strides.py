"""M4 of ``plans/mma-smem-staging.md`` — block-aware affine decode.

Directly exercises ``affine_decode_per_dim`` and the new
``AffineAddressing.source_index`` helper with non-trivial ``block`` to
pin the per-axis composite stride: the i-th axis's coefficient is
``block[i] · prod((extent[j] · block[j]) for j > i in same dim)``.

Together with the existing ``_stage_expand`` accuracy tests (which run
the unblocked path), this proves the ``block=()`` degeneration leaves
behavior byte-clean while the MMA-shaped ``block != ()`` produces the
expected strides.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.tile.ir import AffineAddressing, affine_decode_per_dim


def _v(name: str) -> Var:
    return Var(name)


def test_decode_single_axis_unit_block():
    """Single axis on dim 0, ``block=()`` → bare coord, stride 1."""
    cache = (Axis("m", 16),)
    coord = {"m": _v("m")}
    out = affine_decode_per_dim(cache, dims=(0,), coord_for=coord)
    assert out[0].pretty() == "m"


def test_decode_two_axes_same_dim_unit_block():
    """Two axes on the same dim with ``block=()`` → composite ``m_w·FM + m_r``.
    Pre-M4 contract; covered by every scalar matmul that staged A's M dim."""
    cache = (Axis("m_w", 2), Axis("m_r", 4))
    coord = {"m_w": _v("m_w"), "m_r": _v("m_r")}
    out = affine_decode_per_dim(cache, dims=(0, 0), coord_for=coord)
    assert out[0].pretty() == "((m_w * 4) + m_r)"


def test_decode_two_axes_same_dim_with_atom_block():
    """MMA-shaped block: ``block=(1, 16)`` on cache ``(m_w extent 2, m_r extent 4)``.
    M4 contract: m_r's stride is its own ``block_m_r = 16``; m_w's stride
    is ``(e_m_r · block_m_r) · block_m_w = (4·16)·1 = 64``."""
    cache = (Axis("m_w", 2), Axis("m_r", 4))
    coord = {"m_w": _v("m_w"), "m_r": _v("m_r")}
    out = affine_decode_per_dim(cache, dims=(0, 0), coord_for=coord, block=(1, 16))
    assert out[0].pretty() == "((m_w * 64) + (m_r * 16))"


def test_decode_unit_block_tuple_matches_empty():
    """``block=(1, 1, …)`` is semantically equivalent to ``block=()``;
    pin so the M3 sentinel ``block_tuple = ()`` and an explicit all-1s
    tuple produce identical decodes if anyone re-emits the latter."""
    cache = (Axis("a", 4), Axis("b", 8))
    coord = {"a": _v("a"), "b": _v("b")}
    bare = affine_decode_per_dim(cache, dims=(0, 0), coord_for=coord)
    explicit = affine_decode_per_dim(cache, dims=(0, 0), coord_for=coord, block=(1, 1))
    assert bare[0].pretty() == explicit[0].pretty()


def test_decode_axes_on_different_dims_with_block():
    """Cross-dim block isolation: axes on different source dims don't
    cross-multiply. ``block=(16, 16)`` on ``(M_r dim 0, K_i dim 1)`` →
    each dim picks up its own atom factor independently."""
    cache = (Axis("M_r", 4), Axis("K_i", 8))
    coord = {"M_r": _v("M_r"), "K_i": _v("K_i")}
    out = affine_decode_per_dim(cache, dims=(0, 1), coord_for=coord, block=(16, 16))
    assert out[0].pretty() == "(M_r * 16)"
    assert out[1].pretty() == "(K_i * 16)"


def test_source_index_includes_origin():
    """``AffineAddressing.source_index`` adds the per-source-dim origin
    on top of the decoded stride term — single source of truth for the
    cooperative producer / gmem revert / MMA fragment load."""
    addr = AffineAddressing(dims=(0, 0), block=(1, 16))
    cache = (Axis("m_w", 2), Axis("m_r", 4))
    coord = {"m_w": _v("m_w"), "m_r": _v("m_r")}
    origin = (_v("k_o") * Literal(32, "int"),)
    full = addr.source_index(cache, coord, origin)
    assert len(full) == 1
    assert full[0].pretty() == "((k_o * 32) + ((m_w * 64) + (m_r * 16)))"


def test_source_index_unswept_dim_carries_origin_only():
    """A source dim with no cache axis mapping → only the origin term,
    matching the pre-M4 ``o if d not in decoded`` branch in
    ``_source_decl_line`` / ``_reconstruct_global_index``."""
    addr = AffineAddressing(dims=(0,))
    cache = (Axis("m", 4),)
    coord = {"m": _v("m")}
    origin = (_v("k_anchor"), Literal(0, "int"))
    full = addr.source_index(cache, coord, origin)
    # dim 0 swept by m; dim 1 unswept → bare origin.
    assert full[0].pretty() == "(k_anchor + m)"
    assert full[1].pretty() == "0"


@pytest.mark.parametrize("block", [(2,), (3, 5), (1, 1, 1)])
def test_block_length_must_match_dims(block):
    """`AffineAddressing` validates length at construction; this test
    pairs each non-trivial block with a matching ``dims`` so we don't
    accidentally drift the validation as future block-bearing knobs
    appear."""
    dims = tuple(range(len(block)))
    addr = AffineAddressing(dims=dims, block=block)
    assert addr.block == block
