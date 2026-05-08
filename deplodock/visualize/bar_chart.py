"""Generic grouped bar chart. One series per ``Bar``; orientation auto-picks
horizontal vs vertical based on category count (override with
``orientation``). Produces a self-contained HTML page via
``deplodock.visualize.page.render_html``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from deplodock.visualize.page import render_html
from deplodock.visualize.theme import THEMES

AUTO_HORIZONTAL_THRESHOLD = 12

# Default colors used when a Bar doesn't pin one. Picked from PALETTE_1 to
# stay categorical and high-contrast on dark/light themes.
DEFAULT_SERIES_COLORS = ["#4dabf7", "#ffd166", "#3ddc84", "#c084fc", "#fb7185", "#67e8f9"]


@dataclass
class Bar:
    name: str
    values: list[float | None]
    color: str | None = None


@dataclass
class BarChart:
    categories: list[str]
    bars: list[Bar]
    value_name: str = ""
    title: str = ""
    subtitle: str = ""
    baseline: float | None = None
    baseline_label: str | None = None
    tooltip_rows: list[str] | None = None
    orientation: Literal["auto", "horizontal", "vertical"] = "auto"
    row_height: int = 22
    margin: dict[str, int] = field(default_factory=dict)

    def resolved_orientation(self) -> Literal["horizontal", "vertical"]:
        if self.orientation != "auto":
            return self.orientation
        return "horizontal" if len(self.categories) > AUTO_HORIZONTAL_THRESHOLD else "vertical"


def _series_for(bar: Bar, idx: int, *, baseline: float | None, baseline_label: str | None, theme: dict[str, Any]) -> dict:
    color = bar.color or DEFAULT_SERIES_COLORS[idx % len(DEFAULT_SERIES_COLORS)]
    series: dict = {
        "name": bar.name,
        "type": "bar",
        "data": [v if v is not None else "-" for v in bar.values],
        "itemStyle": {"color": color},
        "barCategoryGap": "30%",
    }
    if idx == 0 and baseline is not None:
        series["markLine"] = {
            "silent": True,
            "symbol": "none",
            "lineStyle": {"type": "dashed", "color": theme["muted"]},
            "data": [
                {
                    "label": {
                        "formatter": baseline_label or f"{baseline}",
                        "color": theme["muted"],
                    },
                }
            ],
        }
    return series


def _option(chart: BarChart, *, theme_name: str) -> dict:
    t = THEMES[theme_name]
    orient = chart.resolved_orientation()
    is_horizontal = orient == "horizontal"

    cat_axis = {
        "type": "category",
        "data": chart.categories,
        "inverse": is_horizontal,  # horizontal: top-down ordering of caller-supplied list
        "axisLabel": {
            "fontSize": 11,
            "fontFamily": "monospace" if is_horizontal else "inherit",
            "color": t["fg"],
            "interval": 0,
            "rotate": 0 if is_horizontal else 30,
        },
        "axisLine": {"lineStyle": {"color": t["axisLine"]}},
        "axisTick": {"show": False},
    }
    val_axis = {
        "type": "value",
        "name": chart.value_name,
        "nameLocation": "middle",
        "nameGap": 30 if is_horizontal else 40,
        "nameTextStyle": {"color": t["muted"]},
        "axisLine": {"lineStyle": {"color": t["axisLine"]}},
        "axisLabel": {"color": t["fg"]},
        "splitLine": {"lineStyle": {"color": t["splitLine"]}},
    }

    series = []
    for i, bar in enumerate(chart.bars):
        s = _series_for(bar, i, baseline=chart.baseline, baseline_label=chart.baseline_label, theme=t)
        # markLine x/y axis depends on orientation — fix it now.
        if "markLine" in s:
            data0 = s["markLine"]["data"][0]
            data0[("xAxis" if is_horizontal else "yAxis")] = chart.baseline
        series.append(s)

    default_margin = (
        {"left": 200, "right": 50, "top": 70, "bottom": 60} if is_horizontal else {"left": 70, "right": 40, "top": 70, "bottom": 90}
    )
    margin = {**default_margin, **chart.margin}

    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": t["fg"]},
        "title": {
            "text": chart.title,
            "subtext": chart.subtitle,
            "left": 0,
            "top": 0,
            "textStyle": {"color": t["fg"], "fontSize": 14, "fontWeight": "normal"},
            "subtextStyle": {"color": t["muted"], "fontSize": 12},
        },
        "grid": {**margin, "containLabel": False},
        "legend": {
            "top": 36,
            "textStyle": {"color": t["fg"]},
            "data": [b.name for b in chart.bars],
            "icon": "rect",
            "itemWidth": 12,
            "itemHeight": 10,
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow", "shadowStyle": {"color": t["rule"]}},
            "backgroundColor": t["tooltipBg"],
            "borderColor": t["axisLine"],
            "textStyle": {"color": t["tooltipText"]},
        },
        "xAxis": cat_axis if not is_horizontal else val_axis,
        "yAxis": val_axis if not is_horizontal else cat_axis,
        "series": series,
    }


def render_bar_chart(chart: BarChart, *, theme: str = "dark", transparent: bool = True) -> str:
    """Return a self-contained HTML page rendering ``chart``."""
    orient = chart.resolved_orientation()
    is_horizontal = orient == "horizontal"
    option = _option(chart, theme_name=theme)
    payload = {
        "option": option,
        "tooltipRows": chart.tooltip_rows,
        "orientation": orient,
        "rowHeight": chart.row_height,
        "n": len(chart.categories),
        "padTop": option["grid"]["top"],
        "padBot": option["grid"]["bottom"],
    }
    body_html = '<div id="chart" style="width:100%;"></div>\n'
    scripts_js = f"""
const PAYLOAD = {json.dumps(payload)};
const el = document.getElementById('chart');
if (PAYLOAD.orientation === 'horizontal') {{
  el.style.height = (PAYLOAD.n * PAYLOAD.rowHeight + PAYLOAD.padTop + PAYLOAD.padBot) + 'px';
}} else {{
  el.style.height = '520px';
}}
const chart = echarts.init(el, null, {{ renderer: 'canvas' }});
const opt = PAYLOAD.option;
if (PAYLOAD.tooltipRows) {{
  opt.tooltip.formatter = (params) => {{
    const i = params[0].dataIndex;
    return PAYLOAD.tooltipRows[i];
  }};
}}
chart.setOption(opt);
window.addEventListener('resize', () => chart.resize());
"""
    title = chart.title or "deplodock chart"
    return render_html(
        body_html=body_html,
        scripts_js=scripts_js,
        theme=theme,
        title=title,
        transparent=transparent,
        extra_css=("#chart { width: 100%; }\n" if not is_horizontal else ""),
    )
