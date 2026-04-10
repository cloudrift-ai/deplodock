"""Structured compiler trace for AI-in-the-loop optimization.

Captures every stage of the compile → run pipeline as JSON-serializable
data: input graph, each pass with rules applied, generated CUDA source,
execution results, and performance metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class RuleApplication:
    """Record of a single rule firing."""

    rule_name: str
    matched_at: str  # root node id
    bindings: dict[str, str]
    captured_constraints: dict[str, str | int]

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "matched_at": self.matched_at,
            "bindings": self.bindings,
            "captured_constraints": {k: str(v) for k, v in self.captured_constraints.items()},
        }


@dataclass
class PassTrace:
    """Record of a single pass execution."""

    name: str
    graph_before: dict | None = None
    graph_after: dict | None = None
    rules_applied: list[RuleApplication] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pass": self.name,
            "rules_applied": [r.to_dict() for r in self.rules_applied],
            "graph_before": self.graph_before,
            "graph_after": self.graph_after,
        }


@dataclass
class ExecutionResult:
    """Record of kernel execution."""

    output: list[float] | None = None
    expected: list[float] | None = None
    correct: bool | None = None
    max_error: float | None = None
    kernel_time_ms: float | None = None
    dimensions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"dimensions": self.dimensions}
        if self.output is not None:
            d["output"] = self.output
        if self.expected is not None:
            d["expected"] = self.expected
        if self.correct is not None:
            d["correct"] = self.correct
        if self.max_error is not None:
            d["max_error"] = self.max_error
        if self.kernel_time_ms is not None:
            d["kernel_time_ms"] = self.kernel_time_ms
        return d


@dataclass
class CompilerTrace:
    """Full trace of a compile-and-run cycle."""

    input_graph: dict | None = None
    passes: list[PassTrace] = field(default_factory=list)
    generated_code: str | None = None
    execution: ExecutionResult | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.input_graph is not None:
            d["input_graph"] = self.input_graph
        d["passes"] = [p.to_dict() for p in self.passes]
        if self.generated_code is not None:
            d["generated_code"] = self.generated_code
        if self.execution is not None:
            d["execution"] = self.execution.to_dict()
        if self.error is not None:
            d["error"] = self.error
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full trace to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
