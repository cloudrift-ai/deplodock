"""Smoke tests for ``emmy.visualize``. Asserts that the shared theme
exposes the expected token surface, that ``BarChart`` renders self-contained
HTML for both orientations, and that the page shell injects the JS globals
that downstream charts rely on."""

from __future__ import annotations

import pytest

from emmy.visualize import Bar, BarChart, render_bar_chart, render_html
from emmy.visualize.bar_chart import AUTO_HORIZONTAL_THRESHOLD
from emmy.visualize.theme import FONTS, PALETTE_1, PALETTE_2, STATUS, THEMES

_REQUIRED_TOKENS = {
    "fg",
    "muted",
    "axisLine",
    "splitLine",
    "tooltipBg",
    "tooltipText",
    "empty",
    "padCell",
    "focusBorder",
    "labelAccent",
    "rule",
    "surface",
    "opFocus",
    "opFaint",
}


@pytest.mark.parametrize("name", ["dark", "light"])
def test_themes_complete(name: str) -> None:
    assert _REQUIRED_TOKENS <= set(THEMES[name])


def test_palettes_have_32_entries() -> None:
    assert len(PALETTE_1) == 32
    assert len(PALETTE_2) == 32
    # Disjoint by construction so a chart can use both at once.
    assert not (set(PALETTE_1) & set(PALETTE_2))


def test_fonts_and_status() -> None:
    assert "Inter" in FONTS["ui"]
    assert "JetBrains" in FONTS["mono"]
    assert set(STATUS) == {"ok", "warn", "bad"}


def test_render_html_injects_globals() -> None:
    html = render_html(body_html="<div></div>", scripts_js="// noop", title="t")
    assert 'data-theme="dark"' in html
    assert "echarts.min.js" in html
    assert "const THEME =" in html
    assert "const PALETTE_1 =" in html
    assert "const PALETTE_2 =" in html
    assert "const STATUS =" in html
    assert "window.echartsReady = true" in html


def test_render_html_unknown_theme_raises() -> None:
    with pytest.raises(ValueError):
        render_html(body_html="", scripts_js="", theme="solarized")


def test_bar_chart_auto_orientation_vertical() -> None:
    chart = BarChart(
        categories=["a", "b", "c"],
        bars=[Bar("s", [1.0, 2.0, 3.0])],
    )
    assert chart.resolved_orientation() == "vertical"
    html = render_bar_chart(chart)
    assert "echarts" in html and "data-theme" in html


def test_bar_chart_auto_orientation_horizontal() -> None:
    n = AUTO_HORIZONTAL_THRESHOLD + 5
    chart = BarChart(
        categories=[f"c{i}" for i in range(n)],
        bars=[Bar("s", [float(i) for i in range(n)])],
    )
    assert chart.resolved_orientation() == "horizontal"


def test_bar_chart_baseline_emitted() -> None:
    chart = BarChart(
        categories=["a", "b"],
        bars=[Bar("s", [1.0, 2.0])],
        baseline=1.0,
        baseline_label="1×",
    )
    html = render_bar_chart(chart)
    assert "markLine" in html
    assert "1\\u00d7" in html or "1×" in html


def test_bar_chart_tooltip_rows_referenced() -> None:
    chart = BarChart(
        categories=["a", "b"],
        bars=[Bar("s", [1.0, 2.0])],
        tooltip_rows=["<b>row a</b>", "<b>row b</b>"],
    )
    html = render_bar_chart(chart)
    assert "tooltipRows" in html
    assert "row a" in html


def test_image_render_missing_playwright_raises_importerror(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from emmy.visualize.image import render

    with pytest.raises(ImportError, match="playwright is required"):
        render("<html></html>", tmp_path / "x.png")
