"""Pass-based graph rewrite engine.

Rules declare a ``GRAMMAR`` — a chain grammar (list of ``Production`` /
``Group``) — and a ``rewrite(graph, match)`` function. The engine
matches the grammar against the graph via ``match_grammar()`` and calls
the rewrite function for each match. Each pass runs to fixed point.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import ChainGrammar, match_grammar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace dataclasses — capture what each pass did for debugging/dump
# ---------------------------------------------------------------------------


@dataclass
class RuleApplication:
    """Record of a single rule firing."""

    rule_name: str
    matched_at: str

    def to_dict(self) -> dict:
        return {"rule": self.rule_name, "matched_at": self.matched_at}


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
class Rule:
    """A single rewrite rule: grammar + rewrite function."""

    name: str
    grammar: ChainGrammar
    rewrite: object  # Callable[[Graph, ChainMatch], Graph]

    @staticmethod
    def from_file(path: Path) -> Rule:
        """Load a rule from a Python file exporting ``GRAMMAR`` and ``rewrite()``."""
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load rule from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        grammar = getattr(module, "GRAMMAR", None)
        rewrite_fn = getattr(module, "rewrite", None)
        if grammar is None:
            raise ValueError(f"Rule {path} missing GRAMMAR")
        if rewrite_fn is None:
            raise ValueError(f"Rule {path} missing rewrite() function")

        return Rule(name=path.stem, grammar=grammar, rewrite=rewrite_fn)


@dataclass
class Pass:
    """An ordered collection of rewrite rules applied to fixed point."""

    name: str
    rules: list[Rule] = field(default_factory=list)

    @staticmethod
    def from_directory(path: Path) -> Pass:
        """Load all .py rule files from a directory, sorted by filename."""
        rule_files = sorted(f for f in path.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
        rules = [Rule.from_file(f) for f in rule_files]
        return Pass(name=path.name, rules=rules)

    def apply(self, graph: Graph, trace: PassTrace | None = None) -> Graph:
        """Apply rules in order with restart-on-match until fixed point.

        A rule signals "no change" by returning the same ``Graph`` object
        (identity comparison). The loop restarts from the first rule when
        any rewrite produces a new graph.
        """
        if trace is not None:
            trace.graph_before = graph.to_dict()
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                matches = match_grammar(graph, rule.grammar)
                if not matches:
                    continue
                rule_changed = False
                for match in matches:
                    logger.debug("Rule %s matched at %s", rule.name, match.root_node_id)
                    new_graph = rule.rewrite(graph, match)
                    if new_graph is not graph:
                        if trace is not None:
                            trace.rules_applied.append(RuleApplication(rule_name=rule.name, matched_at=match.root_node_id))
                        graph = new_graph
                        rule_changed = True
                if rule_changed:
                    changed = True
                    break
        if trace is not None:
            trace.graph_after = graph.to_dict()
        return graph


DEFAULT_PASS_ORDER = ["decomposition", "optimization"]


class Rewriter:
    """Run passes sequentially, each to fixed point."""

    def __init__(self, passes: list[Pass] | None = None) -> None:
        self.passes: list[Pass] = passes or []

    @staticmethod
    def from_directory(rules_dir: Path, pass_order: list[str] | None = None) -> Rewriter:
        """Load passes from named subdirectories in the given order."""
        order = pass_order if pass_order is not None else DEFAULT_PASS_ORDER
        passes: list[Pass] = []
        for name in order:
            pass_dir = rules_dir / name
            if not pass_dir.is_dir():
                raise FileNotFoundError(f"Pass directory not found: {pass_dir}")
            if not any(pass_dir.glob("*.py")):
                continue
            passes.append(Pass.from_directory(pass_dir))
        return Rewriter(passes=passes)

    def apply(self, graph: Graph, pass_traces: list[PassTrace] | None = None) -> Graph:
        """Apply all passes in order."""
        for p in self.passes:
            logger.debug("Running pass: %s", p.name)
            trace = None
            if pass_traces is not None:
                trace = PassTrace(name=p.name)
                pass_traces.append(trace)
            graph = p.apply(graph, trace=trace)
        return graph
