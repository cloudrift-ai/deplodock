"""Test helpers for Tile-IR tests.

Re-exports ``tileify`` from the ``001_tileify`` rule. The rule file's
name starts with a digit so it can't be imported via ``import``; this
module loads it via ``importlib`` and forwards the symbol so test files
can keep doing ``tileify(loop_op, kernel_name=...)`` directly without
graph wrapping.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RULE_PATH = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/tile/001_tileify.py"
_spec = importlib.util.spec_from_file_location("_tileify_rule", _RULE_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

tileify = _mod.tileify

__all__ = ["tileify"]
