"""Layering guards — static import checks that lock in dialect boundaries.

These tests don't exercise behavior. They scan source files for forbidden
import patterns that would silently undo earlier refactors. Fast, no
GPU, no test fixtures — they just open files and grep.

Update / extend when adding a new layered subsystem; add a new test in
this file rather than scattering one-off greps elsewhere.
"""

from __future__ import annotations

import pathlib
import re

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


def test_lowering_tile_does_not_import_kernel_passes() -> None:
    """``lowering/tile/**/*.py`` may not import from the ``lowering/kernel`` pass layer.

    The tile layer (``enumeration`` + ``assembly`` + ``split``) runs ABOVE the
    kernel pass layer; a tile pass importing ``lowering.kernel`` is a back-edge in
    the pass DAG — a tile pass depending on a downstream kernel pass's internals.
    Structural predicates the two layers share (``is_matmul_reduce``,
    ``segmentable_k_extent``, ``reduce_body_has_coupled_accum``,
    ``classify_fragment_epilogue``, the fused-edge ``map_transform`` /
    ``split_monoid_producer``) live in ``lowering/_predicates.py`` — pure ``ir.*``
    queries imported by both layers.

    If this fires: move the shared helper into ``lowering/_predicates`` and import
    it there from both layers, rather than reaching down into ``lowering/kernel``.
    """
    tile_dir = _REPO_ROOT / "deplodock" / "compiler" / "pipeline" / "passes" / "lowering" / "tile"
    assert tile_dir.is_dir(), f"lowering/tile/ not found at {tile_dir}"
    forbidden = "deplodock.compiler.pipeline.passes.lowering.kernel"
    offenders: list[str] = []
    for py in sorted(tile_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden in line and "import" in line:
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "lowering/tile/**/*.py must not import from lowering/kernel — back-edge in the pass DAG.\n"
        "Shared structural predicates live in lowering/_predicates.\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Fix 1 firewall — keep the
# assemble→TileOp boundary clean: every benched scheduling choice lives in the
# enumeration ``Schedule`` (above assemble); every kernel pass is *mechanical*
# (it lowers the stamped ``TileOp``, it never re-derives a scheduling decision).
# These two guards are the plan's "structural firewall": without them Fix 2's
# "mechanical kernel passes" rule is unenforceable and a new kernel pass can
# silently re-accrete the schedule-lives-in-the-tree-shape brittleness the
# block-DAG refactor removed upstream.
# ---------------------------------------------------------------------------

_KERNEL_DIR = _REPO_ROOT / "deplodock" / "compiler" / "pipeline" / "passes" / "lowering" / "kernel"

# The enumeration / split move-composer layer: where the *forks* live — offer
# enumeration, knob schema, the algebra classifiers, the edge-placement cut. A
# kernel pass importing any of these is reaching above assemble to (re-)make a
# scheduling decision instead of reading a stamped fact off the ``TileOp``.
_FORBIDDEN_IMPORTS = (
    "lowering.tile.enumeration",
    "lowering.tile.split",
)

# Schedule-decision functions (offer enumeration / algebra classifiers / the
# edge-placement cut). A kernel pass must not *call* one — not even a re-rolled
# local copy. ``classify_fragment_epilogue`` (``lowering/_predicates.py``) is the
# one allowed ``classify*`` name: it is a mechanical fragment-epilogue lowering
# analysis shared by both layers, not a ranked scheduling choice.
_FORBIDDEN_CALL_RE = re.compile(
    r"\b("
    r"\w*_offers"  # thread_offers / map_reg_offers / warp_bk_offers / coop_reduce_offers / …
    r"|cut_offers|stage_candidates|legal_decomps|eligible_atoms"
    r"|classify_algebra|classify_matmul_operands"
    r")\b"
)
_ALLOWED_CALL_NAMES = frozenset({"classify_fragment_epilogue"})


def _kernel_pass_files() -> list[pathlib.Path]:
    assert _KERNEL_DIR.is_dir(), f"lowering/kernel/ not found at {_KERNEL_DIR}"
    return sorted(_KERNEL_DIR.glob("*.py"))


def test_lowering_kernel_does_not_import_enumeration() -> None:
    """``lowering/kernel/*.py`` may not import from the enumeration / split layer.

    Fix 1's import half of the firewall. The enumeration passes
    (``tile/enumeration/`` + ``tile/split/``) own the *forks* — the offer/knob/
    move/classifier modules the search benches over. assemble renders the chosen
    ``Schedule`` into a ``TileOp``; from there every kernel pass is mechanical.
    A kernel pass that imports an offer / knob / cut / classifier module is
    making (or re-deriving) a scheduling decision below assemble — the exact leak
    this fix closes.

    If this fires: the fact you need is a *scheduling decision* — stamp it on the
    ``TileOp`` node in ``assembly/`` and read the attribute here, rather than
    importing the enumeration module that decides it.
    """
    offenders: list[str] = []
    for py in _kernel_pass_files():
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if any(mod in line for mod in _FORBIDDEN_IMPORTS):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "lowering/kernel/*.py must not import from the enumeration/split fork layer — "
        "a kernel pass reads stamped TileOp attributes, it does not re-decide the schedule.\n" + "\n".join(offenders)
    )


def test_lowering_kernel_calls_no_schedule_classifier() -> None:
    """No ``lowering/kernel/*.py`` calls a schedule classifier / offer fn.

    Fix 1's call half of the firewall — defense in depth over the import guard,
    and it also catches a re-rolled *local* copy of a classifier (no import to
    flag). A kernel pass may read a stamped ``TileOp`` attribute (``StageBundle.
    policy``, ``SerialTile.kind``, ``WarpSpecialize.ring_depth``, …) but must not
    invoke the offer enumeration / algebra classification / edge-placement cut
    that produced those facts — those are mechanical-pass discipline violations.
    """
    offenders: list[str] = []
    for py in _kernel_pass_files():
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            for m in _FORBIDDEN_CALL_RE.finditer(line):
                if m.group(1) not in _ALLOWED_CALL_NAMES:
                    offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
                    break
    assert not offenders, (
        "lowering/kernel/*.py must not call a schedule classifier / offer fn — "
        "read the stamped TileOp attribute instead of re-deriving the decision.\n" + "\n".join(offenders)
    )
