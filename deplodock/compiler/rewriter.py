"""Pattern-based graph rewriter.

A pass is a directory of rule modules. Each rule module declares a
``PATTERN = [Pattern(...), ...]`` list and a ``rewrite(graph, match) ->
Graph | None`` function. ``run_pass(graph, pass_dir)`` loads every rule
in the directory and applies them to fixed point — each rule is driven
to its own fixed point before moving on, then the whole sequence is
re-scanned until no rule makes further progress.

Rules return a ``Graph`` fragment (the replacement subgraph, with
``InputOp`` placeholders for external edges) or ``None`` (no-op /
in-place mutation). Returned fragments are spliced into the main graph
by ``_apply_replacement``.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.matcher import Match, Pattern, match_pattern

logger = logging.getLogger(__name__)


@dataclass
class _Rule:
    name: str
    pattern: list[Pattern]
    rewrite: Callable[[Graph, Match], Graph | None]


def run_pass(graph: Graph, pass_dir: Path) -> Graph:
    """Load all rule modules in ``pass_dir`` and apply them to fixed point."""
    return _apply_rules(graph, _load_rules(pass_dir))


def run_rule(graph: Graph, rule_path: Path) -> Graph:
    """Load a single rule module and apply it to fixed point."""
    return _apply_rules(graph, [_load_rule(rule_path)])


def _apply_rules(graph: Graph, rules: list[_Rule]) -> Graph:
    outer_changed = True
    rule_stats: dict[str, tuple[int, float, float]] = {}  # name -> (applied, match_s, rewrite_s)
    while outer_changed:
        outer_changed = False
        for rule in rules:
            while True:
                t0 = time.monotonic()
                matches = match_pattern(graph, rule.pattern)
                match_time = time.monotonic() - t0
                if not matches:
                    prev = rule_stats.get(rule.name, (0, 0.0, 0.0))
                    rule_stats[rule.name] = (prev[0], prev[1] + match_time, prev[2])
                    break
                rule_changed = False
                applied = 0
                rewrite_time = 0.0
                for match in matches:
                    if any(nid not in graph.nodes for nid in match.consumed):
                        continue
                    logger.debug("Rule %s matched at %s", rule.name, match.root_node_id)
                    t1 = time.monotonic()
                    fragment = rule.rewrite(graph, match)
                    if fragment is None:
                        rewrite_time += time.monotonic() - t1
                        continue
                    graph = _apply_replacement(graph, match, fragment)
                    rewrite_time += time.monotonic() - t1
                    applied += 1
                    rule_changed = True
                    outer_changed = True
                prev = rule_stats.get(rule.name, (0, 0.0, 0.0))
                rule_stats[rule.name] = (prev[0] + applied, prev[1] + match_time, prev[2] + rewrite_time)
                if not rule_changed:
                    break
    for name, (n, mt, rt) in sorted(rule_stats.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        if n or mt > 0.01:
            logger.info("  rule %-30s applied=%4d  match=%5.2fs  rewrite=%5.2fs", name, n, mt, rt)
    return graph


def _load_rules(pass_dir: Path) -> list[_Rule]:
    rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
    return [_load_rule(f) for f in rule_files]


def _load_rule(path: Path) -> _Rule:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rule from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pattern = getattr(module, "PATTERN", None)
    rewrite_fn = getattr(module, "rewrite", None)
    if pattern is None:
        raise ValueError(f"Rule {path} missing PATTERN")
    if rewrite_fn is None:
        raise ValueError(f"Rule {path} missing rewrite() function")
    return _Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn)


# ---------------------------------------------------------------------------
# Splice: insert a fragment into the main graph
# ---------------------------------------------------------------------------


def _apply_replacement(graph: Graph, match: Match, fragment: Graph) -> Graph:
    """Splice a replacement fragment into the graph.

    1. Add fragment's non-InputOp nodes to the graph (with fresh IDs).
    2. Wire the fragment's output to replace the match's output.
    3. Remove consumed nodes and orphaned constants.

    Mutates ``graph`` in place and returns the same object.
    """
    g = graph

    id_map: dict[str, str] = {}
    for frag_id in fragment.topological_order():
        frag_node = fragment.nodes[frag_id]
        if isinstance(frag_node.op, InputOp):
            id_map[frag_id] = frag_id  # references existing graph node
            continue
        mapped_inputs = [id_map.get(inp, inp) for inp in frag_node.inputs]
        new_id = g.add_node(
            op=frag_node.op,
            inputs=mapped_inputs,
            output=Tensor(frag_node.output.name, frag_node.output.shape, frag_node.output.dtype),
        )
        if frag_node.hints:
            g.nodes[new_id].hints = frag_node.hints
        id_map[frag_id] = new_id

    new_output = id_map[fragment.outputs[0]]
    old_output = match.output or match.root_node_id
    g.replace_node(old_output, new_output)

    for nid in match.consumed:
        orig = graph.nodes.get(nid)
        if orig is not None and orig.hints:
            g.nodes[new_output].hints.merge(orig.hints)

    for nid in match.consumed:
        if nid in g.nodes and nid != old_output:
            g.remove_node(nid)
    if old_output in g.nodes:
        g.remove_node(old_output)

    _remove_orphans(g)
    return g


def _remove_orphans(graph: Graph) -> None:
    """Remove nodes with zero consumers that aren't graph outputs."""
    output_set = set(graph.outputs)
    input_set = set(graph.inputs)

    def _is_protected(nid: str) -> bool:
        if nid in output_set or nid in input_set:
            return True
        node = graph.nodes.get(nid)
        return node is not None and isinstance(node.op, InputOp)

    consumer_count: dict[str, int] = dict.fromkeys(graph.nodes, 0)
    for node in graph.nodes.values():
        for inp in set(node.inputs):
            if inp in consumer_count:
                consumer_count[inp] += 1

    queue: list[str] = [nid for nid, c in consumer_count.items() if c == 0 and not _is_protected(nid)]
    while queue:
        nid = queue.pop()
        if nid not in graph.nodes:
            continue
        node = graph.nodes[nid]
        for inp in set(node.inputs):
            if inp in consumer_count:
                consumer_count[inp] -= 1
                if consumer_count[inp] == 0 and not _is_protected(inp):
                    queue.append(inp)
        graph.remove_node(nid)
