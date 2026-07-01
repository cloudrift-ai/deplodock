"""Codec parse validation — degenerate / malformed knob pins raise a clear ``ValueError``.

Before the ``_codec_width`` guard a ``0`` width (``b0`` / ``f0`` / ``n0``) parsed to a level
the plan silently dropped — a no-op pin whose knob column still echoed it — and a missing
number (``g``) threw a bare ``int('')`` error. Each codec now rejects empty / non-numeric /
``< 1`` widths uniformly; a ``1`` width stays legal (the level is off, the identity).
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.schedule import ReducePlan, TilePlan


@pytest.mark.parametrize("spec", ["b0", "r0", "g0", "g", "b", "bx", "g2k/b0"])
def test_reduce_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="REDUCE"):
        ReducePlan.parse(spec)


@pytest.mark.parametrize("spec", ["b1", "r1", "g1k"])  # width 1 = level off — the legal identity
def test_reduce_codec_allows_identity(spec: str) -> None:
    ReducePlan.parse(spec)  # no raise


@pytest.mark.parametrize("spec", ["n0", "f0", "n0x4", "n4x0", "f2x0", "n", "fx"])
def test_tile_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="TILE"):
        TilePlan.parse(spec)


@pytest.mark.parametrize("spec", ["n1", "f1", "n4x4", "f2x2"])
def test_tile_codec_allows_valid(spec: str) -> None:
    TilePlan.parse(spec)  # no raise


@pytest.mark.parametrize(
    "spec",
    [
        "a:mma_m16n8k16_f16/w0x1/f1x1/k1",
        "a:mma_m16n8k16_f16/w1x1/f0x1/k1",
        "a:mma_m16n8k16_f16/w1x1/f1x1/k0",
    ],
)
def test_warp_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="WARP"):
        TilePlan.parse(spec)


def test_warp_codec_allows_identity() -> None:
    TilePlan.parse("a:mma_m16n8k16_f16/w1x1/f1x1/k1")  # all-1 widths — no raise
