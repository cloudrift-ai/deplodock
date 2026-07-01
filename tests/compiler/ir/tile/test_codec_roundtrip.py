"""Byte-identical round-trip of every codec over the canonical golden shape corpus.

A focused unit mirror of ``scripts/migrate_goldens_to_codec.py --check`` (the full byte-identity
oracle over the recorded goldens): each codec's ``parse(s).spell() == s`` for the exact string
shapes that appear on disk. This locks the schema-engine ser/de against the wire format independent
of whether a golden currently happens to use a given shape.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.schedule import ReducePlan, Stage, TilePlan


@pytest.mark.parametrize("spec", ["", "b8", "b16", "b32", "r4", "g2a", "g2k", "g4a/b32", "g2k/b16/r4"])
def test_reduce_round_trip(spec: str) -> None:
    assert ReducePlan.parse(spec).spell() == spec


@pytest.mark.parametrize("spec", ["", "n4", "f2", "n32x16", "n32x16/f2x4", "n4x4/f2x2"])
def test_tile_scalar_round_trip(spec: str) -> None:
    assert TilePlan.parse(spec).spell() == spec


@pytest.mark.parametrize(
    "spec",
    [
        "a:mma_m16n8k16_f16/w1x1/f1x1",
        "a:mma_m16n8k16_f16/w2x1/f1x2/k8",
        "a:mma_m16n8k16_bf16/w2x2/f2x2/k4",
    ],
)
def test_warp_round_trip(spec: str) -> None:
    assert TilePlan.parse(spec).spell() == spec


@pytest.mark.parametrize("spec", ["d1/sync", "d1/cp", "d2/cp/ring", "d3/tma/ring", "d4/cp/ring/p2"])
def test_stage_round_trip(spec: str) -> None:
    assert Stage.parse(spec).spell() == spec
