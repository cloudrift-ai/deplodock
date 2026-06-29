#!/usr/bin/env python
"""One-shot migration: rewrite the golden YAMLs' legacy GEMM-letter knobs into the native
output-fragment codecs.

The tile-IR rebuild replaced the legacy knob vocabulary (``BM``/``BN``/``FM``/``FN``/``BK``/
``SPLITK``/``BR``/``FK``, ``WM``/``WN``/``MMA``, binmask ``STAGE`` + ``RING`` + ``TMA``) with the
unified ``TILE`` output-fragment codec (scalar ``n../f..`` OR warp ``a:.../w../f../k..``), the
``REDUCE`` partition codec, and the ``STAGE`` operand-pipeline codec. This script converts each
golden config's ``knobs`` dict in place, preserving the rest of the file byte-for-byte (only the
``knobs: {...}`` line of each entry is rewritten).

:func:`legacy_to_codec` is deterministic and idempotent (a native dict round-trips unchanged), and
delegates to the codec ``*.spell()`` builders so the mapping cannot drift from the parsers. The
recorded ``deplodock_us`` / ``cublas_us`` latencies are left untouched — they are pre-rebuild
measurements; re-tuning the set on hardware (``scripts/tune_golden_set.py`` / the ``tune-golden``
skill) is a separate follow-up.

``WARPSPEC`` (warp-specialization) has no codec yet — the ``WarpSpec`` schedule slot is reserved —
so it is carried through as a passthrough boolean knob (it featurizes as a plain 0/1) until the
warp-spec codec lands.

Usage::

    python scripts/migrate_goldens_to_codec.py              # rewrite all goldens/*.yaml in place
    python scripts/migrate_goldens_to_codec.py --check      # report drift, write nothing (exit 1 if any)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from deplodock.compiler.ir.tile.atom import atom_for
from deplodock.compiler.ir.tile.schedule import ReducePlan, Stage, TilePlan, WarpTile

_GOLDENS_DIR = Path(__file__).resolve().parents[1] / "deplodock" / "compiler" / "pipeline" / "search" / "goldens"

# Legacy keys whose presence (or a binary-string ``STAGE``) marks a not-yet-migrated dict.
_LEGACY_KEYS = frozenset({"BN", "BM", "FM", "FN", "BK", "SPLITK", "BR", "FK", "WN", "WM", "MMA", "RING", "TMA"})


def _is_legacy(knobs: dict) -> bool:
    """True iff ``knobs`` still speaks the legacy vocabulary — any legacy key, or a binmask
    ``STAGE`` (all-``0``/``1`` digits, e.g. ``"11"``) rather than the native ``d<depth>/…`` codec."""
    if _LEGACY_KEYS & set(knobs):
        return True
    stage = str(knobs.get("STAGE", ""))
    return bool(stage) and all(c in "01" for c in stage)


def legacy_to_codec(knobs: dict) -> dict:
    """Convert a legacy golden ``knobs`` dict to the native codec schema. Idempotent: a dict that
    already speaks codecs (no legacy keys) is returned unchanged."""
    if not _is_legacy(knobs):
        return dict(knobs)

    def _int(name: str, default: int = 1) -> int:
        return int(knobs.get(name, default))

    out: dict = {}
    # Output fragment → the unified ``TILE`` codec. A legacy ``MMA`` key marks the warp fragment;
    # otherwise the scalar register sub-tile (``BN``/``BM``/``FM``/``FN``).
    if "MMA" in knobs:
        wt = WarpTile(
            atom=atom_for(str(knobs["MMA"])),
            warps=(_int("WM"), _int("WN")),
            reg=(_int("FM"), _int("FN")),
            bk=_int("BK"),
        )
        out["TILE"] = wt.spell()
    else:
        # slot mapping mirrors knob._free_slots / _tile_features: par_n=BN, reg_n=FN, par_m=BM, reg_m=FM.
        # An all-1 tile (per-cell — e.g. a cooperative row-reduce golden) spells empty → omit it.
        tile_spec = TilePlan(par_n=_int("BN"), reg_n=_int("FN"), par_m=_int("BM"), reg_m=_int("FM")).spell()
        if tile_spec:
            out["TILE"] = tile_spec
    # Reduce partition → the ``REDUCE`` codec (cross-CTA split-K ``g``, cooperative ``b``, ILP reg
    # ``r``). Legacy ``BK`` is the serial K-chunk — now schedule-derived, not a knob — so it is
    # dropped. ``finalize="atomic"`` is the historical additive-matmul split-K default. Empty
    # (all factors 1) → omitted.
    reduce_spec = ReducePlan.of(cta=_int("SPLITK"), coop=_int("BR"), reg=_int("FK"), finalize="atomic").spell()
    if reduce_spec:
        out["REDUCE"] = reduce_spec
    # Operand staging → the ``STAGE`` codec. The legacy binmask ``STAGE`` ("11") meant staging on;
    # the ring depth came from ``RING`` and the transport from ``TMA`` (tma vs cp.async).
    if str(knobs.get("STAGE", "")):
        ring = _int("RING")
        transport = "tma" if _int("TMA", 0) else "cp.async"  # canonical value; .spell() emits the cp/tma token
        out["STAGE"] = Stage(depth=ring, transport=transport, ring=ring >= 2).spell()
    # Warp-specialization has no codec yet — carry it through as a passthrough boolean.
    if "WARPSPEC" in knobs:
        out["WARPSPEC"] = bool(knobs["WARPSPEC"])
    return out


def _emit_flow(knobs: dict) -> str:
    """Render a knobs dict as a compact YAML flow mapping, matching the goldens' style: string
    values single-quoted (codec strings carry ``/`` and ``:``), bools lowercased, ints bare."""
    parts = []
    for k, v in knobs.items():
        if isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, str):
            parts.append(f"{k}: '{v}'")
        else:
            parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


_KNOBS_RE = re.compile(r"^(?P<indent>\s*)knobs:\s*(?P<map>\{.*\})\s*$")


def migrate_text(text: str) -> tuple[str, int]:
    """Rewrite every ``knobs: {...}`` line in ``text`` to native codecs. Returns the new text and
    the count of lines that actually changed."""
    out_lines, changed = [], 0
    for line in text.splitlines(keepends=True):
        m = _KNOBS_RE.match(line.rstrip("\n"))
        if not m:
            out_lines.append(line)
            continue
        legacy = yaml.safe_load(m.group("map"))
        native = legacy_to_codec(legacy)
        nl = "\n" if line.endswith("\n") else ""
        new_line = f"{m.group('indent')}knobs: {_emit_flow(native)}{nl}"
        if new_line != line:
            changed += 1
        out_lines.append(new_line)
    return "".join(out_lines), changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report drift, write nothing (exit 1 if any)")
    args = ap.parse_args()

    paths = sorted(_GOLDENS_DIR.glob("*.yaml"))
    if not paths:
        print(f"no golden YAMLs under {_GOLDENS_DIR}", file=sys.stderr)
        return 1
    total = 0
    for path in paths:
        text = path.read_text()
        new_text, changed = migrate_text(text)
        total += changed
        if changed and not args.check:
            path.write_text(new_text)
        verb = "would rewrite" if args.check else "rewrote"
        print(f"{verb} {changed:3d} knobs in {path.name}")
    if args.check and total:
        print(f"\n{total} golden entries still on the legacy schema", file=sys.stderr)
        return 1
    print(f"\n{'drift' if args.check else 'migrated'}: {total} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
