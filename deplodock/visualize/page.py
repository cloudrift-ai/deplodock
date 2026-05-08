"""Shared HTML page shell for ECharts-based visualizations. Emits the
ECharts script include, exposes ``THEME``/``PALETTE_1``/``PALETTE_2``/
``STATUS``/``FONTS`` as JS globals, and sets ``window.echartsReady`` after
the caller's scripts run so external image renderers know when the canvas
is ready to screenshot."""

from __future__ import annotations

import json

from deplodock.visualize.theme import FONTS, PALETTE_1, PALETTE_2, STATUS, THEMES

ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"


def _root_css(theme: str, *, transparent: bool) -> str:
    t = THEMES[theme]
    bg = "transparent" if transparent else t["surface"]
    return f""":root[data-theme="{theme}"] {{
  --fg: {t["fg"]};
  --muted: {t["muted"]};
  --axis-line: {t["axisLine"]};
  --split-line: {t["splitLine"]};
  --tooltip-bg: {t["tooltipBg"]};
  --tooltip-text: {t["tooltipText"]};
  --rule: {t["rule"]};
  --label-accent: {t["labelAccent"]};
  --surface: {t["surface"]};
}}
html, body {{ margin: 0; background: {bg}; color: var(--fg);
  font-family: {FONTS["ui"]}; }}
"""


def render_html(
    *,
    body_html: str,
    scripts_js: str,
    theme: str = "dark",
    title: str = "",
    extra_css: str = "",
    transparent: bool = True,
) -> str:
    """Wrap ``body_html`` and ``scripts_js`` in the shared page shell.

    Injected JS globals: ``THEME`` (theme dict), ``PALETTE_1`` / ``PALETTE_2``
    (32-color qualitative palettes), ``STATUS`` (ok/warn/bad), ``FONTS``
    (ui/mono CSS strings).
    """
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; expected one of {sorted(THEMES)}")
    globals_js = (
        f"const THEME = {json.dumps(THEMES[theme])};\n"
        f"const PALETTE_1 = {json.dumps(PALETTE_1)};\n"
        f"const PALETTE_2 = {json.dumps(PALETTE_2)};\n"
        f"const STATUS = {json.dumps(STATUS)};\n"
        f"const FONTS = {json.dumps(FONTS)};\n"
    )
    return f"""<!doctype html>
<html lang="en" data-theme="{theme}">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<script src="{ECHARTS_CDN}"></script>
<style>
{_root_css(theme, transparent=transparent)}
{extra_css}
</style>
</head>
<body>
{body_html}
<script>
{globals_js}
{scripts_js}
window.echartsReady = true;
</script>
</body>
</html>
"""
