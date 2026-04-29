"""Test helpers for Tile-IR tests.

Re-exports ``lower_naive`` from the ``001_lower_loopop`` rule. The rule
file's name starts with a digit so it can't be imported via ``import``;
this module loads it via ``importlib`` and forwards the symbol so test
files can keep doing ``lower_naive(loop_op, kernel_name=...)`` directly
without graph wrapping.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RULE_PATH = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/tile/001_lower_loopop.py"
_spec = importlib.util.spec_from_file_location("_lower_loopop_rule", _RULE_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

lower_naive = _mod.lower_naive

__all__ = ["lower_naive"]
