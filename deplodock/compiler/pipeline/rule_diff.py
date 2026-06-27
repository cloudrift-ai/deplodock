"""Diff-style renderer for per-rule ``-vv`` output in the rewrite engine.

The engine snapshots each rule's matched subgraph before and after the
rewrite and hands both to :func:`render_rule_diff`, which emits a unified
diff bracketed by ``>>> NNN_rulename`` / ``<<< NNN_rulename`` markers.
The bracketing makes it trivial to slice one rule out of a long ``-vv``
log with ``awk '/^>>> 005_/,/^<<< 005_/'``.

Color (ANSI) is applied only inside the diff body so the markers stay
plain ASCII and ``awk`` matches reliably even on colored output. Color
follows ``--color {auto,always,never}`` and honors the standard
``NO_COLOR`` environment variable when ``auto``.
"""

from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass
from typing import IO

_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_CYAN = "\x1b[36m"
_RESET = "\x1b[0m"

# Single-letter pass shorthands. Single source of truth — the CLI's
# ``--passes d,o,l`` shortcut expander imports the inverse mapping from
# here. Used in ``-vv`` diff markers (e.g. ``>>> t:005_blockify_launch``)
# so a reader can slice an entire pass with ``awk '/^>>> t:/,/^<<< t:/'``
# or one rule with ``awk '/^>>> t:005/,/^<<< t:005/'``.
PASS_SHORTHAND = {
    "frontend/decomposition": "d",
    "frontend/optimization": "o",
    "loop/lifting": "l",
    "loop/fusion": "f",
    "loop/stamp": "s",
    "lowering/tile": "t",
    "lowering/kernel": "k",
    "lowering/cuda": "c",
}


def display_name(pass_name: str | None, rule_name: str) -> str:
    """``<shorthand>:<rule>`` if the pass is known, else just ``<rule>``."""
    short = PASS_SHORTHAND.get(pass_name or "")
    return f"{short}:{rule_name}" if short else rule_name


@dataclass
class RuleRenderConfig:
    """How to render per-rule ``-vv`` output.

    ``color`` is the resolved boolean (already accounts for ``auto`` /
    ``NO_COLOR`` / tty detection). ``context`` is the unified-diff context
    line count. ``max_lines`` caps a single rule's diff before falling
    back to a full pre/post listing inside the same markers.
    """

    color: bool = False
    context: int = 2
    max_lines: int = 200


_config = RuleRenderConfig()


def set_config(cfg: RuleRenderConfig) -> None:
    """Install the active render config (called once from the CLI)."""
    global _config
    _config = cfg


def get_config() -> RuleRenderConfig:
    return _config


def should_use_color(stream: IO, mode: str) -> bool:
    """Resolve ``--color {auto,always,never}`` against a stream + env.

    ``auto`` → true iff ``stream`` is a tty and ``NO_COLOR`` is unset.
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def render_rule_diff(
    name: str,
    before: str,
    after: str,
    *,
    header: str = "",
    cfg: RuleRenderConfig | None = None,
) -> str:
    """Render one rule application as a marker-bracketed unified diff.

    Output shape:

        >>> NNN_rulename
        @@ <header> @@         (only when ``header`` is non-empty)
        @@ -a,b +c,d @@
         context
        -removed
        +added
         context
        <<< NNN_rulename

    When the diff body would exceed ``cfg.max_lines``, falls back to
    printing the full ``before:`` / ``after:`` blocks inside the same
    markers with a leading suppression note.
    """
    cfg = cfg or _config
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            n=cfg.context,
            lineterm="",
        )
    )
    # Drop the unified_diff file headers (``--- ``/``+++ ``); we use our
    # own ``>>> name``/``<<< name`` markers instead.
    diff_body = [ln for ln in diff_lines if not (ln.startswith("--- ") or ln.startswith("+++ "))]

    out: list[str] = [f">>> {name}"]
    if header:
        out.append(f"@@ {header} @@")

    if len(diff_body) > cfg.max_lines:
        out.append(f"# diff suppressed: {len(diff_body)} lines > --diff-max-lines {cfg.max_lines}")
        out.append("before:")
        out.extend(f"  {ln}" for ln in before_lines)
        out.append("after:")
        out.extend(f"  {ln}" for ln in after_lines)
    else:
        out.extend(_colorize(ln, cfg.color) for ln in diff_body)

    out.append(f"<<< {name}")
    return "\n".join(out)


def emit(text: str) -> None:
    """Write a per-rule diff or skipped one-liner to stdout.

    Diffs go to stdout (not via ``logger.debug``) so users can pipe
    ``compile -vv`` through ``grep`` / ``awk`` without needing
    ``2>&1``. The IR result also lands on stdout but appears after the
    pipeline finishes — the marker-bracketed range patterns slice
    cleanly around it.
    """
    print(text, flush=True)


def format_skipped(name: str, root: str, reason: str) -> str:
    """One-liner for rules that matched but raised ``RuleSkipped``.

    Uses ``---`` to stay visually distinct from ``>>>``/``<<<`` blocks
    and to make it greppable on its own (``grep '^--- 004_'``).
    """
    return f"--- {name} skipped at {root}: {reason}"


def _colorize(line: str, color: bool) -> str:
    if not color or not line:
        return line
    if line.startswith("+"):
        return f"{_GREEN}{line}{_RESET}"
    if line.startswith("-"):
        return f"{_RED}{line}{_RESET}"
    if line.startswith("@@"):
        return f"{_CYAN}{line}{_RESET}"
    return line


__all__ = [
    "PASS_SHORTHAND",
    "RuleRenderConfig",
    "display_name",
    "format_skipped",
    "get_config",
    "render_rule_diff",
    "set_config",
    "should_use_color",
]


# Convenience: when this module is imported, default color resolution
# considers stdout (where ``emit()`` writes the diff text).
_config.color = should_use_color(sys.stdout, "auto")
