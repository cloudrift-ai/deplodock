"""Tests for ``tile/025_swizzle_blocks`` — default-on CTA swizzle.

Covers stamping behaviour (default-on for matmul-shape grids,
env-pin escape, idempotence, skip conditions) and the renderer output
of the swizzled decode (Triton-canonical arithmetic + K_s peel +
symbolic-extent passthrough). Structural-key delta is checked so the
autotune DB caches swizzled and non-swizzled variants separately.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.stmt.base import RenderCtx
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import RuleSkipped

# The pass module file starts with a digit, so it can't be imported with the
# usual ``from … import``. Round-trip via importlib like the pipeline loader.
swizzle = importlib.import_module(
    "deplodock.compiler.pipeline.passes.lowering.tile.025_swizzle_blocks",
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _matmul_grid(m: int = 8, n: int = 4, with_splitk: bool = False, m_extent=None) -> GridTile:
    """Build a minimal matmul-shape GridTile body for the pass / render to chew on.

    The inner ThreadTile is a placeholder so ``GridTile.body`` is non-empty —
    the pass and the renderer ignore it.
    """
    m_axis = Axis("m_b", m if m_extent is None else m_extent)
    n_axis = Axis("n_b", n)
    tt = ThreadTile(axes=(Axis("m_t", 1),), body=Body(()))
    axes: tuple[Axis, ...]
    if with_splitk:
        k_s = Axis("k_s", 2)
        axes = (k_s, m_axis, n_axis)
    else:
        axes = (m_axis, n_axis)
    return GridTile(axes=axes, body=Body((tt,)))


_MATMUL_KNOBS = {"BK": 16, "BR": 1, "SPLITK": 1, "BN": 8, "BM": 8, "FM": 1, "FN": 1}


def _tile_op(grid: GridTile, knobs: dict | None = None) -> TileOp:
    return TileOp(body=Body((grid,)), name="k_test", knobs={**_MATMUL_KNOBS, **(knobs or {})})


class _StubNode:
    def __init__(self, op: TileOp) -> None:
        self.op = op


# ---------------------------------------------------------------------------
# Stamp behaviour
# ---------------------------------------------------------------------------


def test_default_on_stamps_8(monkeypatch):
    """Unset env → pass stamps the Triton/CUTLASS default GROUP_M = 8."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    op = _tile_op(_matmul_grid())
    new = swizzle.rewrite(_StubNode(op))
    grid = new.body[0]
    assert isinstance(grid, GridTile)
    assert grid.swizzle_group_m == 8


def test_env_pin_disables(monkeypatch):
    """DEPLODOCK_GROUP_M=1 → escape hatch fires, pass skips with reason."""
    monkeypatch.setenv("DEPLODOCK_GROUP_M", "1")
    op = _tile_op(_matmul_grid())
    with pytest.raises(RuleSkipped, match="disables CTA swizzle"):
        swizzle.rewrite(_StubNode(op))


def test_env_pin_picks_explicit_value(monkeypatch):
    """DEPLODOCK_GROUP_M=4 → pass stamps 4."""
    monkeypatch.setenv("DEPLODOCK_GROUP_M", "4")
    op = _tile_op(_matmul_grid())
    new = swizzle.rewrite(_StubNode(op))
    assert new.body[0].swizzle_group_m == 4


def test_idempotent(monkeypatch):
    """Already-stamped GridTile → second pass invocation is a skip."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    op = _tile_op(_matmul_grid())
    once = swizzle.rewrite(_StubNode(op))
    with pytest.raises(RuleSkipped, match="no eligible top-level GridTile"):
        swizzle.rewrite(_StubNode(once))


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def test_skip_pointwise_knobs(monkeypatch):
    """Pointwise kernel (BK == 1) → skip — no K-reduce, no L2 A-row reuse story."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    grid = _matmul_grid()
    op = TileOp(body=Body((grid,)), name="k", knobs={**_MATMUL_KNOBS, "BK": 1})
    with pytest.raises(RuleSkipped, match="not a matmul-priority TileOp"):
        swizzle.rewrite(_StubNode(op))


def test_skip_cooperative_reduce_knobs(monkeypatch):
    """Cooperative-reduce kernel (BR > 1) → skip — different axis topology."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    grid = _matmul_grid()
    op = TileOp(body=Body((grid,)), name="k", knobs={**_MATMUL_KNOBS, "BR": 4})
    with pytest.raises(RuleSkipped, match="not a matmul-priority TileOp"):
        swizzle.rewrite(_StubNode(op))


def test_skip_single_axis_grid(monkeypatch):
    """Matmul knobs but a single-block-axis GridTile (degenerate shape)
    → no eligible top-level grid, skip."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    grid = GridTile(axes=(Axis("seq_b", 32),), body=Body((ThreadTile(axes=(Axis("t", 1),), body=Body(())),)))
    op = TileOp(body=Body((grid,)), name="k", knobs=_MATMUL_KNOBS)
    with pytest.raises(RuleSkipped, match="no eligible top-level GridTile"):
        swizzle.rewrite(_StubNode(op))


# ---------------------------------------------------------------------------
# Renderer arithmetic
# ---------------------------------------------------------------------------


def _render_grid(grid: GridTile) -> str:
    return "\n".join(grid.render(RenderCtx(indent=0)))


def test_render_emits_triton_arithmetic():
    """Rendered swizzled decode contains the Triton remap variables and
    binds the GridTile's M/N axis names to the swizzled values.

    Built standalone (not via TileOp) so the body-normalizer's canonical
    ``a0/a1/...`` rename doesn't obscure the axis identities under test —
    this isolates the renderer contract."""
    grid = _matmul_grid(m=8, n=4)
    grid = replace(grid, swizzle_group_m=8)
    src = _render_grid(grid)
    for needle in (
        "int bid = blockIdx.x;",
        "int num_m = 8;",
        "int num_n = 4;",
        "int gsz = 8 * num_n;",
        "int gid = bid / gsz;",
        "int first_m = gid * 8;",
        "int gsize_m =",
    ):
        assert needle in src, f"missing {needle!r} in:\n{src}"
    assert "int m_b = first_m + ((bid % gsz) % gsize_m);" in src
    assert "int n_b = (bid % gsz) / gsize_m;" in src


def test_render_peels_splitk():
    """GridTile(axes=(K_s, M_b, N_b)) → K_s is peeled before the swizzle remap."""
    grid = _matmul_grid(m=8, n=4, with_splitk=True)
    grid = replace(grid, swizzle_group_m=8)
    src = _render_grid(grid)
    # K_s decode comes from the row-major helper applied to (blockIdx.x / (8*4));
    # the swizzle then runs on (blockIdx.x % (8*4)).
    assert "int k_s = (blockIdx.x / (8 * 4));" in src
    assert "int bid = blockIdx.x % (8 * 4);" in src
    assert "int gsz = 8 * num_n;" in src
    assert "int m_b = first_m" in src
    assert "int n_b = (bid % gsz) / gsize_m;" in src


def test_render_symbolic_extent():
    """Symbolic M_b extent (dynamic-shape matmul) renders the symbol in num_m."""
    grid = _matmul_grid(n=4, m_extent=Dim("seq_len"))
    grid = replace(grid, swizzle_group_m=8)
    src = _render_grid(grid)
    assert "int num_m = seq_len;" in src
    assert "int num_n = 4;" in src


def test_default_grid_uses_row_major_decode():
    """Default GridTile (swizzle_group_m=1) still emits the original
    row-major decode — important invariant for the escape hatch.
    N is innermost (modulo); M is outermost (division)."""
    grid = _matmul_grid(m=8, n=4)
    src = _render_grid(grid)
    assert "int n_b = blockIdx.x % 4;" in src
    assert "int m_b = blockIdx.x / (4);" in src
    assert "gsz" not in src


# ---------------------------------------------------------------------------
# Structural-key delta — autotune DB cache key must change
# ---------------------------------------------------------------------------


def test_structural_key_changes_with_swizzle(monkeypatch):
    """Two GridTiles with the same axes but different ``swizzle_group_m``
    produce different ``Body.structural_key()`` values so the autotune
    DB caches them separately."""
    monkeypatch.delenv("DEPLODOCK_GROUP_M", raising=False)
    base = Body((_matmul_grid(),))
    swizzled = swizzle.rewrite(_StubNode(_tile_op(_matmul_grid()))).body
    assert base.structural_key() != swizzled.structural_key()
