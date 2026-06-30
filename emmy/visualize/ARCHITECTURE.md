# `emmy/visualize/`

Shared ECharts visualization stack. Owns theme tokens, fonts, qualitative
palettes, the HTML page shell, a generic bar-chart emitter, and HTML→image
rendering.

```
visualize/
  theme.py       THEMES (dark/light), FONTS, STATUS, PALETTE_1, PALETTE_2
  page.py        render_html — page shell, ECharts <script>, JS globals
  bar_chart.py   Bar / BarChart / render_bar_chart — auto orientation
  image.py       render — PNG/JPEG/WebP/PDF/SVG via Playwright
```

## Theme tokens

`THEMES["dark" | "light"]` is the single source of truth for foreground
text, axis lines, tooltip surface, etc. `page.render_html` lifts every
token into `:root[data-theme=…]` CSS variables (`--fg`, `--muted`,
`--axis-line`, `--split-line`, `--tooltip-bg`, `--tooltip-text`, `--rule`,
`--label-accent`, `--surface`) and exposes the same dict to JS as
`window.THEME`.

`FONTS["ui" | "mono"]` ship as JS globals too — use `FONTS.mono` for any
monospace label so axis ticks line up across charts.

`STATUS = {ok, warn, bad}` is the universal pass/borderline/fail palette
(green/orange/red).

## Palettes

Two 32-color qualitative palettes live in `theme.py`:

- `PALETTE_1` — cool/rainbow hues (cyans, greens, blues, purples).
- `PALETTE_2` — warm hues (yellows, oranges, reds, browns).

They're designed to stay visually disjoint, so a chart that has two
independent color dimensions (e.g. `color = bank` and `color = address` on
the same page) can pick one palette per dimension without confusion. Don't
introduce chart-specific aliases — reference the generic palettes directly
and document the mapping at the call site.

## Page shell

`render_html(*, body_html, scripts_js, theme="dark", title="",
extra_css="", transparent=True)` builds a self-contained page:

- Includes ECharts from a CDN.
- Sets `data-theme` on `<html>` and emits CSS variables.
- Injects `THEME`, `PALETTE_1`, `PALETTE_2`, `STATUS`, `FONTS` as JS
  globals.
- Sets `window.echartsReady = true` after the caller's `scripts_js`
  finishes — used by `image.render` to know when the canvas is paintable.

Pass `transparent=False` for standalone images that need an opaque
backdrop; the default suits embedding in dark pages and READMEs.

## Bar chart

`BarChart` carries `categories`, parallel `bars` (each a `Bar(name,
values, color?)`), an optional `baseline` reference line, and optional
pre-formatted `tooltip_rows` (HTML strings indexed by category).

Orientation policy:

- `len(categories) <= AUTO_HORIZONTAL_THRESHOLD` (12) → vertical.
- otherwise → horizontal.
- Override via `orientation = "horizontal" | "vertical"`.

Vertical = bars grow upward, categories on x-axis; horizontal = bars grow
rightward, categories on y-axis (long lists, monospace labels). Same
option scaffolding either way — only axis assignments differ.

## Image rendering

`render(html, out_path, *, width, height, selector, transparent)` detects
format from `out_path.suffix`:

| suffix | backend | transparent? |
|---|---|---|
| `.png`, `.webp` | Playwright `element.screenshot` | ✓ |
| `.jpg`/`.jpeg`  | Playwright `element.screenshot(type="jpeg")` | always opaque |
| `.pdf`          | Playwright `page.pdf` | via `print_background` |
| `.svg`          | concatenated `chart.renderToSVGString()` | ECharts only — no surrounding HTML |

Playwright is an optional dep:

```
pip install -e '.[visualize]'
playwright install chromium
```

Without it `image.render` raises `ImportError` with the install command.
HTML emission has zero deps beyond stdlib.

## Adding a new chart

1. Build an option dict that pulls colors from `THEMES[theme]` and
   typography from `FONTS`. Don't hardcode hex literals — every shared
   token belongs in `theme.py`.
2. If the chart shape is general (bars, lines, scatter), extend the
   relevant module. If it's bespoke layout (e.g. the smem punchcard with
   ladders + legends), keep the layout at the call site and pass the
   bespoke CSS/HTML/JS into `render_html` via `extra_css` / `body_html` /
   `scripts_js`.
3. Pick palettes by name (`PALETTE_1`, `PALETTE_2`) — never inline an
   ad-hoc list.
