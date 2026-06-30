"""Tiny ANSI-aware console-table helper shared by the CLI tables.

One renderer for every column-aligned table the CLI prints — the ``deplodock eval``
golden / knob tables and the ``run`` / ``tune`` latency and per-kernel tables. A cell
is a plain ``str`` or a ``(text, colour)`` tuple; column width is measured by the
*visible* text length so embedded ANSI colour codes never throw off the alignment.
Colours activate only on a tty (piped / logged output stays plain).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from deplodock.compiler.pipeline.knob import knob_sort_key

# ANSI colours, only when stdout is a tty (piped / logged output stays plain).
TTY = sys.stdout.isatty()
GREEN, YELLOW, RED, RESET = ("\033[32m", "\033[33m", "\033[31m", "\033[0m") if TTY else ("", "", "", "")

# A cell is a plain ``str`` or a ``(text, colour)`` tuple (an empty colour renders plain).
Cell = "str | tuple[str, str]"


@dataclass(frozen=True)
class Col:
    """A table column: ``name`` is the header label, ``align`` is ``"l"`` or ``"r"``."""

    name: str
    align: str = "l"


def _text(cell) -> str:
    return cell[0] if isinstance(cell, tuple) else cell


def col_widths(columns: list[Col], rows: list[list], min_widths: list[int] | None = None) -> list[int]:
    """Per-column width: the widest of the header name, every visible cell, and (if given)
    ``min_widths[i]`` — a floor used to reserve room for content rendered outside ``rows``
    (e.g. an error row's left column)."""
    widths = [max(len(c.name), m) for c, m in zip(columns, min_widths or [0] * len(columns), strict=True)]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(_text(cell)))
    return widths


def _pad(cell, width: int, align: str) -> str:
    """Render one cell to ``width`` visible columns; colour is applied but not measured."""
    text = _text(cell)
    color = cell[1] if isinstance(cell, tuple) else ""
    body = f"{color}{text}{RESET}" if color else text
    pad = " " * (width - len(text))
    return pad + body if align == "r" else body + pad


def render_table(
    columns: list[Col], rows: list[list], *, rule: bool = False, indent: str = "", gap: str = "  ", min_widths: list[int] | None = None
) -> list[str]:
    """Render a table to a list of lines: a header row, an optional ``---`` rule, then one
    line per row. Each cell is padded to its column's width (by visible length) and aligned
    per :attr:`Col.align`; trailing whitespace is stripped and ``indent`` prefixes every line.
    All number / percent formatting happens at the call site — cells are final strings.
    ``min_widths`` floors per-column widths (see :func:`col_widths`); the resolved widths are
    available via ``col_widths(columns, rows, min_widths)`` for callers that render extra rows
    (e.g. error rows) aligned to the same columns."""
    widths = col_widths(columns, rows, min_widths)

    def line(cells: list) -> str:
        body = gap.join(_pad(c, w, col.align) for c, w, col in zip(cells, widths, columns, strict=True))
        return (indent + body).rstrip()

    lines = [line([c.name for c in columns])]
    if rule:
        lines.append(indent + "-" * (sum(widths) + len(gap) * (len(widths) - 1)))
    lines += [line(r) for r in rows]
    return lines


def knob_columns(rows: list[dict[str, tuple[str, bool]]]) -> tuple[list[Col], list[list]]:
    """Build the right-aligned knob columns for a set of rows. Each row is a
    ``{knob_name: (value_text, red?)}`` mapping — ``value_text`` carries no ``NAME=``
    prefix (the name is the column header). Columns are the union of knob names in
    canonical order (:func:`knob_sort_key`); a row missing a knob gets a blank cell,
    ``red`` cells are coloured. Returns ``(columns, per_row_cells)`` to splice in after a
    caller's own leading columns / cells."""
    keys = sorted({k for r in rows for k in r}, key=knob_sort_key)
    cols = [Col(k, "r") for k in keys]
    cells = [[((r[k][0], RED if r[k][1] else "") if k in r else "") for k in keys] for r in rows]
    return cols, cells
