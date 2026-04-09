"""Pass-based graph rewrite engine."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.pattern import Pattern, parse_pattern
from deplodock.compiler.trace import PassTrace, RuleApplication

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single rewrite rule: pattern + rewrite function."""

    name: str
    pattern: list[Pattern]  # alternatives
    rewrite: RewriteFn

    @staticmethod
    def from_file(path: Path) -> Rule:
        """Load a rule from a Python file exporting PATTERN and rewrite()."""
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load rule from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        pattern_text = getattr(module, "PATTERN", None)
        rewrite_fn = getattr(module, "rewrite", None)
        if pattern_text is None:
            raise ValueError(f"Rule {path} missing PATTERN string")
        if rewrite_fn is None:
            raise ValueError(f"Rule {path} missing rewrite() function")

        return Rule(
            name=path.stem,
            pattern=parse_pattern(pattern_text),
            rewrite=rewrite_fn,
        )


# Type alias for rewrite functions.
RewriteFn = type(lambda graph, match: graph)  # noqa: E731


@dataclass
class Pass:
    """An ordered collection of rewrite rules applied to fixed point."""

    name: str
    rules: list[Rule] = field(default_factory=list)

    @staticmethod
    def from_directory(path: Path) -> Pass:
        """Load all .py rule files from a directory, sorted by filename."""
        rule_files = sorted(path.glob("*.py"))
        rules = [Rule.from_file(f) for f in rule_files]
        return Pass(name=path.name, rules=rules)

    def apply(self, graph: Graph, trace: PassTrace | None = None) -> Graph:
        """Apply rules in order with restart-on-match until fixed point."""
        if trace is not None:
            trace.graph_before = graph.to_dict()
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                matches = match_pattern(graph, rule.pattern)
                if matches:
                    for match in matches:
                        logger.debug("Rule %s matched at %s", rule.name, match.root_node_id)
                        if trace is not None:
                            trace.rules_applied.append(
                                RuleApplication(
                                    rule_name=rule.name,
                                    matched_at=match.root_node_id,
                                    bindings=dict(match.bindings),
                                    captured_constraints=dict(match.captured_constraints),
                                )
                            )
                        graph = rule.rewrite(graph, match)
                    changed = True
                    break  # restart from first rule
        if trace is not None:
            trace.graph_after = graph.to_dict()
        return graph


class Rewriter:
    """Run passes sequentially, each to fixed point."""

    def __init__(self, passes: list[Pass] | None = None) -> None:
        self.passes: list[Pass] = passes or []

    @staticmethod
    def from_directory(rules_dir: Path) -> Rewriter:
        """Load passes from subdirectories of rules_dir, sorted by name."""
        pass_dirs = sorted(d for d in rules_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))
        passes = [Pass.from_directory(d) for d in pass_dirs if any(d.glob("*.py"))]
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
