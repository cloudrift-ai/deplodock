"""Graph pattern matching engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler import ops as ops_module
from deplodock.compiler.ir import Graph
from deplodock.compiler.pattern import (
    Pattern,
    PatternNode,
    PatternVar,
    PatternWildcard,
)


@dataclass
class Match:
    """Result of a successful pattern match against a graph."""

    root_node_id: str
    bindings: dict[str, str] = field(default_factory=dict)
    captured_constraints: dict[str, str | int] = field(default_factory=dict)


def match_pattern(graph: Graph, alternatives: list[Pattern]) -> list[Match]:
    """Find all non-overlapping matches of any alternative pattern in graph.

    Tries each node in reverse topological order. For each node, tries each
    alternative; first alternative that matches wins. Already-matched nodes
    are skipped to ensure non-overlapping.
    """
    matched_nodes: set[str] = set()
    results: list[Match] = []

    for node_id in reversed(graph.topological_order()):
        if node_id in matched_nodes:
            continue
        for alt in alternatives:
            bindings: dict[str, str] = {}
            captures: dict[str, str | int] = {}
            if _match_node(graph, node_id, alt, bindings, captures):
                # Record all nodes involved in this match as consumed.
                consumed = set(bindings.values()) | {node_id}
                matched_nodes.update(consumed)
                results.append(
                    Match(
                        root_node_id=node_id,
                        bindings=bindings,
                        captured_constraints=captures,
                    )
                )
                break
    return results


def _match_node(
    graph: Graph,
    node_id: str,
    pattern: Pattern,
    bindings: dict[str, str],
    captures: dict[str, str | int],
) -> bool:
    """Recursively try to unify a graph node against a pattern element."""
    if node_id not in graph.nodes:
        return False
    node = graph.nodes[node_id]

    # --- PatternVar: capture or check backreference ---
    if isinstance(pattern, PatternVar):
        if pattern.name in bindings:
            return bindings[pattern.name] == node_id
        bindings[pattern.name] = node_id
        return True

    # --- PatternWildcard: match any single node (or subgraph if greedy) ---
    if isinstance(pattern, PatternWildcard):
        return True

    # --- PatternNode: match op type, constraints, and inputs ---
    if not isinstance(pattern, PatternNode):
        return False

    # Check op class via isinstance against the actual op module.
    op_cls = getattr(ops_module, pattern.op_class, None)
    if op_cls is None or not isinstance(node.op, op_cls):
        return False

    # Check constraints.
    for field_name, expected in pattern.constraints.items():
        actual = getattr(node.op, field_name, None)
        if actual is None:
            return False
        if expected.startswith("$"):
            # Capture variable — bind or check.
            cap_name = expected[1:]
            if cap_name in captures:
                if captures[cap_name] != actual:
                    return False
            else:
                captures[cap_name] = actual
        else:
            # Literal match.
            if str(actual) != expected:
                return False

    # Check inputs — must match in order and count.
    if len(pattern.inputs) != len(node.inputs):
        return False
    for pat_input, node_input_id in zip(pattern.inputs, node.inputs, strict=True):
        if not _match_node(graph, node_input_id, pat_input, bindings, captures):
            return False

    return True
