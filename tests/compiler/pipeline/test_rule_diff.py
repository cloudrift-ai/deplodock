"""Unit tests for the per-rule diff renderer used at ``compile -vv``."""

from __future__ import annotations

import io

from deplodock.compiler.pipeline.rule_diff import (
    PASS_SHORTHAND,
    RuleRenderConfig,
    display_name,
    format_skipped,
    render_rule_diff,
    should_use_color,
)


def test_render_rule_diff_brackets_with_markers():
    out = render_rule_diff("004_launch", "a\nb\nc\n", "a\nB\nc\n", cfg=RuleRenderConfig(color=False))
    lines = out.splitlines()
    assert lines[0] == ">>> 004_launch"
    assert lines[-1] == "<<< 004_launch"
    assert any(ln.startswith("-b") for ln in lines)
    assert any(ln.startswith("+B") for ln in lines)
    # No file-header noise from unified_diff.
    assert not any(ln.startswith(("--- ", "+++ ")) for ln in lines)


def test_render_rule_diff_includes_header_when_provided():
    out = render_rule_diff("001_x", "a\n", "b\n", header="matched at root7", cfg=RuleRenderConfig(color=False))
    assert "@@ matched at root7 @@" in out


def test_render_rule_diff_color_wraps_diff_lines():
    cfg = RuleRenderConfig(color=True)
    out = render_rule_diff("001_x", "a\n", "b\n", cfg=cfg)
    assert "\x1b[31m-a\x1b[0m" in out
    assert "\x1b[32m+b\x1b[0m" in out
    # Markers stay plain ASCII so awk slicing keeps working.
    assert "\x1b" not in out.splitlines()[0]
    assert "\x1b" not in out.splitlines()[-1]


def test_render_rule_diff_max_lines_falls_back_to_full_listing():
    cfg = RuleRenderConfig(color=False, context=2, max_lines=3)
    before = "\n".join(f"old{i}" for i in range(20))
    after = "\n".join(f"new{i}" for i in range(20))
    out = render_rule_diff("002_big", before, after, cfg=cfg)
    assert "diff suppressed" in out
    assert "before:" in out
    assert "after:" in out
    # Markers still present so slicing is unaffected.
    assert out.startswith(">>> 002_big")
    assert out.endswith("<<< 002_big")


def test_format_skipped_uses_dashed_marker():
    assert format_skipped("004_coop_reduce", "rootX", "no reduce loop") == "--- 004_coop_reduce skipped at rootX: no reduce loop"


def test_display_name_prefixes_known_pass():
    assert display_name("lowering/tile", "005_launch_geometry") == "t:005_launch_geometry"
    assert display_name("loop/fusion", "010_merge_loop_ops") == "f:010_merge_loop_ops"
    # Unknown / missing pass falls back to the bare rule name.
    assert display_name(None, "001_x") == "001_x"
    assert display_name("not/a/pass", "001_x") == "001_x"


def test_pass_shorthand_matches_cli_shortcut_inverse():
    # Single source of truth: commands/compile.py builds its --passes
    # shortcut expander by inverting PASS_SHORTHAND.
    from deplodock.commands.compile import _PASS_SHORTCUTS

    assert _PASS_SHORTCUTS == {short: full for full, short in PASS_SHORTHAND.items()}


def test_should_use_color_modes(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    tty = io.StringIO()
    tty.isatty = lambda: True  # type: ignore[method-assign]
    notty = io.StringIO()
    notty.isatty = lambda: False  # type: ignore[method-assign]

    assert should_use_color(tty, "auto") is True
    assert should_use_color(notty, "auto") is False
    assert should_use_color(notty, "always") is True
    assert should_use_color(tty, "never") is False

    monkeypatch.setenv("NO_COLOR", "1")
    assert should_use_color(tty, "auto") is False
    assert should_use_color(tty, "always") is True  # explicit override
