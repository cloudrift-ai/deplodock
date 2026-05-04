"""Conftest for ``tests/compiler/passes/``.

Exposes the ``recording_dump`` fixture: a duck-typed ``CompilerDump``
that records the names of every rule that produced a rewrite, with the
numeric ordering prefix stripped (``002_split_matmul_k`` →
``split_matmul_k``, ``004_cooperative_reduce`` →
``cooperative_reduce``). Pass it into ``run_pipeline`` / ``run_pass``
to assert which rules fired without coupling tests to file ordering.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.engine import _strip_rule_prefix as strip_rule_prefix


class RecordingDump:
    """Minimal ``CompilerDump`` substitute that records fired rule names.

    The engine calls ``on_rule(pass_idx, pass_name, rule_name, record, text)``
    once per applied rewrite and ``on_pass(idx, name, graph)`` after each
    pass. Only ``on_rule`` is interesting here; ``on_pass`` is a no-op.
    """

    def __init__(self) -> None:
        self.fired: list[tuple[str, str]] = []  # (pass_name, stripped_rule_name)

    def on_rule(self, pass_idx, pass_name, rule_name, record, text) -> None:
        self.fired.append((pass_name, strip_rule_prefix(rule_name)))

    def on_pass(self, idx, pass_name, graph) -> None:
        pass

    def fired_rules(self, pass_name: str | None = None) -> set[str]:
        if pass_name is None:
            return {r for _, r in self.fired}
        return {r for p, r in self.fired if p == pass_name}


@pytest.fixture
def recording_dump() -> RecordingDump:
    return RecordingDump()
