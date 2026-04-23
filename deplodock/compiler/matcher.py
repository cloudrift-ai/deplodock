"""Simple chain matcher.

A pattern is a list of ``Pattern`` entries — one per node in the chain.
Matching walks forward from each topo-ordered seed along fan-out-1
consumer edges: the seed must match ``pattern[0]``, its sole consumer
must match ``pattern[1]``, and so on. Multi-node patterns therefore
only match a producer→consumer chain when the producer has exactly one
consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.graph import Graph


@dataclass
class Pattern:
    """One node in a chain-match pattern.

    ``constraints`` is a dict of ``field_name → expected_value`` checks
    applied to ``node.op`` (e.g. ``{"fn": "softmax"}``).
    """

    name: str
    op_type: type
    constraints: dict = field(default_factory=dict)


@dataclass
class Match:
    """Result of matching a pattern against a graph.

    ``nodes`` maps each pattern entry's name to the node id it matched. ``consumed`` and ``output`` may be overwritten by the
    rewrite function to control which nodes the rewriter removes and
    which node its edges get redirected to. ``output`` defaults to
    ``root_node_id`` when left as ``None``.
    """

    root_node_id: str
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None


def match_pattern(graph: Graph, pattern: list[Pattern]) -> list[Match]:
    """Return every pattern match rooted at a topo-ordered node.

    Matches may overlap — e.g. both ``{A, B}`` and ``{B, C}`` for a
    two-node pattern. The rewriter breaks after the first
    successful ``rewrite`` per pass iteration, so overlap is only a
    candidate-enumeration concern.
    """
    results: list[Match] = []
    for nid in graph.topological_order():
        m = _match_at(graph, nid, pattern)
        if m is not None:
            results.append(m)
    return results


def _match_at(graph: Graph, start: str, pattern: list[Pattern]) -> Match | None:
    if start not in graph.nodes:
        return None
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    for prod in pattern:
        if cursor is None:
            return None
        node = graph.nodes.get(cursor)
        if node is None or not isinstance(node.op, prod.op_type):
            return None
        if not _check_constraints(node, prod):
            return None
        nodes[prod.name] = cursor
        consumed.add(cursor)
        cursor = _sole_consumer(graph, cursor)
    return Match(root_node_id=start, nodes=nodes, consumed=consumed)


def _check_constraints(node, prod: Pattern) -> bool:
    for field_name, expected in prod.constraints.items():
        actual = getattr(node.op, field_name, None)
        if actual is None or str(actual) != str(expected):
            return False
    return True


def _sole_consumer(graph: Graph, nid: str) -> str | None:
    consumers = graph.consumers(nid)
    return consumers[0] if len(consumers) == 1 else None
