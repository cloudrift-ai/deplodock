"""Tests for ``Role.WARP`` wiring in the partition planner's
``_wrap_tower`` (M3 of ``plans/warptile-primitive.md``).

No planner branch emits ``Role.WARP`` yet — the MMA fragment
factorization consumer plan does. M3's job is to wire
``_layer_kind_for`` / ``_wrap_tower`` so downstream plans can flip a
tier without revisiting the tower mechanics. These tests poke
``_wrap_tower`` directly with a synthetic ``layers`` list.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Write
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile, WarpTile

# ``Role`` / ``_wrap_tower`` / ``_layer_kind_for`` were extracted from the
# partition planner into the shared ``partition._tower`` module (the legacy
# planner imports them back); the move composer reuses the same mechanics.
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _layer_kind_for, _wrap_tower


def _stub_inner() -> tuple:
    """Inner stmt tuple used as the body of the innermost wrap. A bare
    Write keeps the body legal without dragging in IO seeding."""
    return (Write(output="C", index=(Var("m_b"), Var("m_w")), value="one"),)


def test_wrap_tower_warp_role_emits_warp_tile():
    """A single WARP layer wraps into a ``WarpTile``."""
    layers = [
        (Axis("m_w", 2), Role.WARP),  # innermost
        (Axis("m_b", 4), Role.BLOCK),  # outermost
    ]
    out = _wrap_tower(layers, _stub_inner())
    assert len(out) == 1
    grid = out[0]
    assert isinstance(grid, GridTile)
    assert len(grid.body) == 1
    warp = grid.body[0]
    assert isinstance(warp, WarpTile)
    assert tuple(ax.name for ax in warp.axes) == ("m_w",)


def test_wrap_tower_warp_groups_consecutive_warp_axes():
    """Two consecutive WARP axes coalesce into one ``WarpTile`` (mirrors
    the THREAD / BLOCK grouping behaviour)."""
    layers = [
        (Axis("n_w", 4), Role.WARP),  # innermost
        (Axis("m_w", 2), Role.WARP),
        (Axis("m_b", 4), Role.BLOCK),  # outermost
    ]
    out = _wrap_tower(layers, _stub_inner())
    grid = out[0]
    assert isinstance(grid, GridTile)
    warp = grid.body[0]
    assert isinstance(warp, WarpTile)
    # Outer warp axis comes first (innermost-first input → outermost-first
    # wrap order in the resulting axes tuple).
    assert tuple(ax.name for ax in warp.axes) == ("m_w", "n_w")


def test_wrap_tower_thread_and_warp_dont_merge():
    """THREAD and WARP layers stay separate — they bind different inner
    flavors (ThreadTile vs WarpTile) and can't share a wrap, even when
    adjacent. (TileOp validation later rejects this mix; the wrap-tower
    builder just splits them and lets the validator decide.)"""
    layers = [
        (Axis("m_t", 8), Role.THREAD),  # innermost
        (Axis("m_w", 2), Role.WARP),
        (Axis("m_b", 4), Role.BLOCK),  # outermost
    ]
    out = _wrap_tower(layers, _stub_inner())
    grid = out[0]
    assert isinstance(grid, GridTile)
    # Outer warp wraps an inner thread; mixing fails TileOp validation
    # but the planner mechanic itself produces the cleanly nested tower.
    warp = grid.body[0]
    assert isinstance(warp, WarpTile)
    thread = warp.body[0]
    assert isinstance(thread, ThreadTile)


def test_layer_kind_for_warp_returns_warp_string():
    """The kind string drives ``_wrap_tower``'s grouping switch."""
    assert _layer_kind_for(Role.WARP) == "warp"
