"""The ``REDUCE`` codec round-trips — including the cross-CTA GRID finalize letter.

``ReducePlan.parse`` / ``.spell`` are the schedule's single reduce-partition codec. The
``g<n>[a|k]`` finalize letter (atomic vs deferred-kernel cross-CTA split) must survive the
round-trip so ``030_split`` can read ``ReducePlan.finalize`` — it was historically parsed
then dropped (``spell`` never re-emitted it), making ``g2a`` and ``g2k`` indistinguishable.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.schedule import ReducePlan


@pytest.mark.parametrize("spec", ["", "b32", "r4", "g2a", "g2k", "g4a/b32", "g2k/b16/r4"])
def test_reduce_codec_round_trips(spec: str) -> None:
    assert ReducePlan.parse(spec).spell() == spec


def test_grid_finalize_letter_decodes() -> None:
    assert ReducePlan.parse("g2a").finalize == "atomic"
    assert ReducePlan.parse("g2k").finalize == "kernel"
    assert ReducePlan.parse("g2").finalize == "kernel"  # default when the letter is omitted
    assert ReducePlan.parse("b32").finalize == "kernel"  # no GRID stage → the default value


def test_needs_split_only_for_grid() -> None:
    assert ReducePlan.parse("g2k").needs_split
    assert not ReducePlan.parse("b128/r4").needs_split
    assert not ReducePlan.parse("").needs_split
