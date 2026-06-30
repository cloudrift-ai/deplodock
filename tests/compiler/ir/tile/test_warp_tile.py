"""Tests for the WarpTile primitive (M1).

Covers the construction-time invariants only: validation in
``TileOp.__post_init__`` and pretty-print. Render + launch-bounds wiring
is exercised in ``test_warp_tile_render.py`` (M2).
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Body, Write
from emmy.compiler.ir.tile.ir import GridTile, ThreadTile, TileOp, WarpTile


def _stub_write() -> Write:
    # Constant-1 write so the body has something legal for TileOp's IO
    # placeholder seeding to find.
    return Write(output="C", index=(Var("m_b"), Var("m_w")), value="one")


def test_warp_tile_pretty_includes_warp_label():
    m_w = Axis("m_w", 2)
    n_w = Axis("n_w", 4)
    tile = WarpTile(axes=(m_w, n_w), body=Body((_stub_write(),)))
    rendered = "\n".join(tile.pretty())
    assert "warp" in rendered
    assert "└ warp" in rendered
    assert "for m_w in 0..2" in rendered
    assert "for n_w in 0..4" in rendered


def test_tileop_accepts_grid_warp_cooperative_tower():
    m_b = Axis("m_b", 4)
    m_w = Axis("m_w", 2)
    body = Body(
        (
            GridTile(
                axes=(m_b,),
                body=Body((WarpTile(axes=(m_w,), body=Body((_stub_write(),))),)),
            ),
        )
    )
    # Construction must succeed — GridTile > WarpTile is the supported tower.
    TileOp(body=body, name="k_warp_smoke")


def test_tileop_rejects_thread_and_warp_in_same_body():
    m_w = Axis("m_w", 2)
    m_t = Axis("m_t", 8)
    body = Body(
        (
            GridTile(
                axes=(Axis("m_b", 4),),
                body=Body(
                    (
                        ThreadTile(axes=(m_t,), body=Body((_stub_write(),))),
                        WarpTile(axes=(m_w,), body=Body((_stub_write(),))),
                    )
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="both a ThreadTile and a WarpTile"):
        TileOp(body=body, name="k_warp_mix")


def test_tileop_rejects_two_top_level_warp_tiles():
    body = Body(
        (
            WarpTile(axes=(Axis("m_w", 2),), body=Body((_stub_write(),))),
            WarpTile(axes=(Axis("n_w", 2),), body=Body((_stub_write(),))),
        )
    )
    with pytest.raises(ValueError, match="at most one outer"):
        TileOp(body=body, name="k_warp_two_top")
