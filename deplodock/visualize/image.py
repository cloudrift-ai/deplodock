"""Render an HTML page (typically one produced by ``deplodock.visualize``)
to an image file. Format is auto-detected from the output path's suffix.
Backed by Playwright headless Chromium — installed via the optional
``[visualize]`` extra (``pip install -e '.[visualize]' && playwright install
chromium``)."""

from __future__ import annotations

import tempfile
from pathlib import Path

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".svg"}

_PLAYWRIGHT_HINT = (
    "playwright is required for image rendering. Install with:\n    pip install -e '.[visualize]' && playwright install chromium"
)


def _import_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(_PLAYWRIGHT_HINT) from e
    return sync_playwright


def render(
    html: str,
    out_path: Path | str,
    *,
    width: int = 1400,
    height: int = 900,
    selector: str = "body",
    transparent: bool = True,
) -> None:
    """Render ``html`` to ``out_path``. Format is detected from the suffix
    (case-insensitive). Supported: ``.png``, ``.jpg``/``.jpeg``, ``.webp``,
    ``.pdf``, ``.svg``.

    PNG and WebP honor ``transparent``. JPEG always renders on white.
    PDF uses ``page.pdf`` (``print_background = not transparent``). SVG
    extracts each ECharts instance via ``chart.renderToSVGString()`` and
    concatenates them; this requires charts to use the SVG renderer or
    the page to be re-rendered, so the function temporarily swaps the
    renderer at runtime.
    """
    out = Path(out_path)
    suffix = out.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported image suffix {suffix!r}; expected one of {sorted(SUPPORTED)}")

    sync_playwright = _import_playwright()
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tmp:
        tmp.write(html)
        tmp_path = tmp.name

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.goto(f"file://{tmp_path}")
            page.wait_for_function("window.echartsReady === true", timeout=10_000)

            if suffix == ".pdf":
                page.pdf(path=str(out), print_background=not transparent)
            elif suffix == ".svg":
                svg = page.evaluate(
                    """() => {
                        const insts = [];
                        document.querySelectorAll('div').forEach(el => {
                            const inst = echarts.getInstanceByDom(el);
                            if (inst) insts.push(inst);
                        });
                        return insts.map(i => i.renderToSVGString()).join('\\n');
                    }"""
                )
                if not svg.strip():
                    raise RuntimeError(
                        "no ECharts instances found on page — cannot render SVG. "
                        "Note: SVG export only captures ECharts canvases, not surrounding HTML."
                    )
                Path(out).write_text(svg)
            else:
                el = page.locator(selector).first
                kwargs: dict = {"path": str(out)}
                if suffix in (".jpg", ".jpeg"):
                    kwargs["type"] = "jpeg"
                    kwargs["quality"] = 92
                else:
                    kwargs["type"] = "png" if suffix == ".png" else "webp"
                    kwargs["omit_background"] = transparent
                el.screenshot(**kwargs)
        finally:
            browser.close()
            Path(tmp_path).unlink(missing_ok=True)
