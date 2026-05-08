"""Shared ECharts theming, page shell, generic bar chart, and HTML→image
rendering. See ``ARCHITECTURE.md``."""

from deplodock.visualize.bar_chart import AUTO_HORIZONTAL_THRESHOLD, Bar, BarChart, render_bar_chart
from deplodock.visualize.image import SUPPORTED as IMAGE_SUPPORTED
from deplodock.visualize.image import render as render_image
from deplodock.visualize.page import render_html
from deplodock.visualize.theme import FONTS, PALETTE_1, PALETTE_2, PALETTES, STATUS, THEMES

__all__ = [
    "AUTO_HORIZONTAL_THRESHOLD",
    "Bar",
    "BarChart",
    "FONTS",
    "IMAGE_SUPPORTED",
    "PALETTES",
    "PALETTE_1",
    "PALETTE_2",
    "STATUS",
    "THEMES",
    "render_bar_chart",
    "render_html",
    "render_image",
]
