"""Theme tokens, fonts, and qualitative palettes shared by every chart in
``deplodock.visualize``. Generic by design — palettes are named by hue
character (not by use site) so any chart can pick whichever fits."""

from __future__ import annotations

from typing import Any

THEMES: dict[str, dict[str, Any]] = {
    "dark": {
        "fg": "#e8eaed",
        "muted": "#6b7280",
        "axisLine": "#2a2d33",
        "splitLine": "#1c212b",
        "tooltipBg": "#0e1014",
        "tooltipText": "#e8eaed",
        "empty": "#2a2d33",
        "padCell": "#3a3f48",
        "focusBorder": "#ffffff",
        "labelAccent": "#7dd3fc",
        "rule": "rgba(255,255,255,.06)",
        "surface": "#0b0d11",
        "opFocus": 1.0,
        "opFaint": 0.45,
    },
    "light": {
        "fg": "#1f2937",
        "muted": "#4b5563",
        "axisLine": "#9ca3af",
        "splitLine": "#d1d5db",
        "tooltipBg": "#ffffff",
        "tooltipText": "#0f172a",
        "empty": "#b8bec7",
        "padCell": "#64748b",
        "focusBorder": "#0f172a",
        "labelAccent": "#0369a1",
        "rule": "rgba(0,0,0,.08)",
        "surface": "#ffffff",
        "opFocus": 1.0,
        "opFaint": 0.45,
    },
}

FONTS = {
    "ui": "'Inter',system-ui,-apple-system,'Segoe UI',sans-serif",
    "mono": "'JetBrains Mono',ui-monospace,monospace",
}

STATUS = {"ok": "#3ddc84", "warn": "#ffb454", "bad": "#ff5c7a"}

# 32 distinct cool/rainbow hues. Use when each color stands for a discrete
# category (e.g. one bank id per color). Saturated and well-spaced so two
# adjacent entries are visually distinguishable.
PALETTE_1 = [
    "#7dd3fc",
    "#3ddc84",
    "#ffb454",
    "#ff5c7a",
    "#c084fc",
    "#fcd34d",
    "#67e8f9",
    "#fb923c",
    "#a3e635",
    "#f472b6",
    "#60a5fa",
    "#34d399",
    "#fde047",
    "#fb7185",
    "#818cf8",
    "#facc15",
    "#0ea5e9",
    "#16a34a",
    "#d97706",
    "#dc2626",
    "#9333ea",
    "#ca8a04",
    "#0891b2",
    "#ea580c",
    "#65a30d",
    "#db2777",
    "#1d4ed8",
    "#059669",
    "#a16207",
    "#be123c",
    "#4338ca",
    "#b45309",
]

# 32 warm hues (yellows → oranges → reds → browns). Designed to stay visually
# disjoint from PALETTE_1 so a reader can tell at a glance which dimension a
# color belongs to when both palettes appear on the same page.
PALETTE_2 = [
    "#fef08a",
    "#fed7aa",
    "#fca5a5",
    "#fbbf24",
    "#f59e0b",
    "#fdba74",
    "#f97316",
    "#ef4444",
    "#fef9c3",
    "#fde68a",
    "#fee2e2",
    "#eab308",
    "#c2410c",
    "#b91c1c",
    "#7f1d1d",
    "#92400e",
    "#fef3c7",
    "#ffedd5",
    "#ffe4e6",
    "#9a3412",
    "#991b1b",
    "#78350f",
    "#7c2d12",
    "#881337",
    "#fbcfe8",
    "#fff7ed",
    "#fef2f2",
    "#713f12",
    "#44403c",
    "#f43f5e",
    "#f87171",
    "#ea661a",
]

PALETTES = (PALETTE_1, PALETTE_2)
