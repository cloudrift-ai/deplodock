"""Pass-based graph rewrite engine.

Rules declare a ``GRAMMAR`` and a ``rewrite(graph, match)`` function that
returns a ``Graph`` fragment (the replacement subgraph) or ``None`` (no-op).

The engine matches the grammar, calls the rewrite function, and splices the
returned fragment into the main graph: adds new nodes, wires the fragment's
output to replace the consumed region, and cleans up.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.matcher import ChainGrammar, ChainMatch, match_grammar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RuleApplication:
    rule_name: str
    matched_at: str

    def to_dict(self) -> dict:
        return {"rule": self.rule_name, "matched_at": self.matched_at}


@dataclass
class PassTrace:
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


# ---------------------------------------------------------------------------
# Rule / Pass / Rewriter
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """A single rewrite rule: grammar + rewrite function."""

    name: str
    grammar: ChainGrammar
    rewrite: object  # Callable[[Graph, ChainMatch], Graph | None]

    @staticmethod
    def from_file(path: Path) -> Rule:
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
        rule_files = sorted(f for f in path.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
        rules = [Rule.from_file(f) for f in rule_files]
        return Pass(name=path.name, rules=rules)

    def apply(self, graph: Graph, trace: PassTrace | None = None) -> Graph:
        """Apply rules to fixed point. Rules return Graph fragment or None.

        Each rule is driven to its own fixed point (all matches applied per
        scan, rescanned until empty) before moving on to the next rule. An
        outer loop re-runs the sequence if any cross-rule dependency
        introduced work. Match results within one scan are guaranteed
        non-overlapping by the matcher, so they are safe to batch; stale
        matches (nodes removed as orphans by an earlier sibling rewrite)
        are skipped defensively.
        """
        if trace is not None:
            trace.graph_before = graph.to_dict()
        outer_changed = True
        rule_stats: dict[str, tuple[int, float, float]] = {}  # name -> (applied, match_s, rewrite_s)
        while outer_changed:
            outer_changed = False
            for rule in self.rules:
                while True:
                    t0 = time.monotonic()
                    matches = match_grammar(graph, rule.grammar)
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
                        if trace is not None:
                            trace.rules_applied.append(RuleApplication(rule_name=rule.name, matched_at=match.root_node_id))
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
        if trace is not None:
            trace.graph_after = graph.to_dict()
        return graph


DEFAULT_PASS_ORDER = ["decomposition", "optimization", "fusion"]


class Rewriter:
    """Run passes sequentially, each to fixed point."""

    def __init__(self, passes: list[Pass] | None = None) -> None:
        self.passes: list[Pass] = passes or []

    @staticmethod
    def from_directory(rules_dir: Path, pass_order: list[str] | None = None) -> Rewriter:
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
        for p in self.passes:
            logger.debug("Running pass: %s", p.name)
            trace = None
            if pass_traces is not None:
                trace = PassTrace(name=p.name)
                pass_traces.append(trace)
            graph = p.apply(graph, trace=trace)
        return graph


# ---------------------------------------------------------------------------
# Splice: insert a fragment into the main graph
# ---------------------------------------------------------------------------


def _apply_replacement(graph: Graph, match: ChainMatch, fragment: Graph) -> Graph:
    """Splice a replacement fragment into the graph.

    1. Add fragment's non-InputOp nodes to the graph (with fresh IDs).
    2. Wire the fragment's output to replace the match's output.
    3. Remove consumed nodes and orphaned constants.

    Mutates ``graph`` in place and returns the same object. ``apply``'s
    batched-match loop already guards against stale matches from sibling
    rewrites via the ``consumed`` check, so per-match atomicity is not
    needed; copying the whole graph per match was O(N) × O(matches).
    """
    g = graph

    # Add fragment nodes, mapping fragment IDs to new graph IDs.
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

    # Wire fragment output to replace the match output.
    new_output = id_map[fragment.outputs[0]]
    old_output = match.output or match.root_node_id
    g.replace_node(old_output, new_output)  # also updates LoopOp internals

    # Merge hints from consumed nodes into the new output.
    for nid in match.consumed:
        orig = graph.nodes.get(nid)
        if orig is not None and orig.hints:
            g.nodes[new_output].hints.merge(orig.hints)

    # Remove consumed nodes.
    for nid in match.consumed:
        if nid in g.nodes and nid != old_output:
            g.remove_node(nid)
    if old_output in g.nodes:
        g.remove_node(old_output)

    # Remove orphaned nodes (zero consumers, not a graph output).
    _remove_orphans(g)

    return g


def _remove_orphans(graph: Graph) -> None:
    """Remove nodes with zero consumers that aren't graph outputs.

    Linear-time: computes a consumer-count map once and propagates
    decrements as orphans are removed. The prior implementation called
    ``graph.consumers(nid)`` (O(N)) inside a ``while changed`` loop over
    all nodes, which blew up to O(N^2 * orphans) on large graphs.
    """
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
