"""Shared tty-coloured, column-aligned knob rendering for the CLI tables.

Both ``deplodock eval`` (the golden ``found/golden`` tables) and ``deplodock run
--bench`` (the per-kernel table, greedy pick vs a benched golden) render knob
columns the same way: the union of knobs across the rows in canonical order
(:func:`~deplodock.compiler.pipeline.knob.knob_sort_key`), each padded to its
widest cell so columns line up vertically, mismatches in red, blank where a row
lacks a knob. :func:`align_knob_columns` is that one renderer; the ANSI colours
activate only on a tty (piped / logged output stays plain).
"""

from __future__ import annotations

import sys

from deplodock.compiler.pipeline.knob import knob_sort_key

# ANSI colours, only when stdout is a tty (piped / logged output stays plain).
TTY = sys.stdout.isatty()
GREEN, YELLOW, RED, RESET = ("\033[32m", "\033[33m", "\033[31m", "\033[0m") if TTY else ("", "", "", "")


def align_knob_columns(rows: list[dict[str, tuple[str, bool]]]) -> list[str]:
    """Align a set of knob rows into shared columns. Each row is a
    ``{knob_name: (visible_text, red?)}`` mapping (``visible_text`` already the
    rendered cell, e.g. ``"BN=32"`` or ``"BN=16/32"``). Returns one string per
    row: the union of knobs in canonical order, each padded to its widest cell,
    ``red`` cells wrapped in ANSI red, blanks where a row lacks the knob."""
    keys = sorted({k for r in rows for k in r}, key=knob_sort_key)
    width = {k: max(len(r[k][0]) for r in rows if k in r) for k in keys}
    lines = []
    for r in rows:
        cells = []
        for k in keys:
            if k not in r:
                cells.append(" " * width[k])
                continue
            text, red = r[k]
            body = f"{RED}{text}{RESET}" if red else text
            cells.append(body + " " * (width[k] - len(text)))
        lines.append("  ".join(cells).rstrip())
    return lines
