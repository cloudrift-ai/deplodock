"""Layering guards — static import checks that lock in dialect boundaries.

These tests don't exercise behavior. They scan source files for forbidden
import patterns that would silently undo earlier refactors. Fast, no
GPU, no test fixtures — they just open files and grep.

Update / extend when adding a new layered subsystem; add a new test in
this file rather than scattering one-off greps elsewhere.
"""

from __future__ import annotations

import pathlib

# Repo root: tests/architecture/ → tests/ → repo
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_lowering_tile_does_not_import_kernel_ir() -> None:
    """``lowering/tile/*.py`` may not import from ``ir.kernel.ir``.

    The Tile-IR / Kernel-IR boundary is: Tile passes encode scheduling
    *decisions* (``StagePolicy``, ``AsyncWait``, ``WarpSpecialize``);
    Kernel-IR carries hardware *primitives* (``Sync``, ``Smem``,
    ``Mbarrier*``, ``SetMaxNReg``, …) that the materializer
    (``100_materialize_tile``) emits. A Tile pass that imports
    Kernel-IR types is fabricating Kernel stmts inside a Tile body —
    exactly what PR #166 did to ``085_warp_specialize`` and what the
    WarpSpecialize Stmt refactor undid.

    If this test fires, either:
    1. you genuinely need a new Tile-level abstraction (add a Stmt to
       ``ir/tile/ir.py``, lower it in ``100_materialize_tile``), or
    2. you're in the wrong directory — Kernel-IR-emitting passes live
       under ``lowering/kernel/``.
    """
    tile_dir = _REPO_ROOT / "deplodock" / "compiler" / "pipeline" / "passes" / "lowering" / "tile"
    assert tile_dir.is_dir(), f"lowering/tile/ not found at {tile_dir}"
    forbidden = "from deplodock.compiler.ir.kernel"
    offenders: list[str] = []
    for py in sorted(tile_dir.glob("*.py")):
        text = py.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if forbidden in line:
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "lowering/tile/*.py must not import from ir.kernel — layering violation.\n" + "\n".join(offenders)
