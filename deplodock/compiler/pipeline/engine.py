"""Pattern-based rewrite engine and compile-pipeline entry point.

Public surface:

- ``Pattern`` / ``Match`` / ``match_pattern`` — chain matcher: each
  ``Pattern`` matches one node by ``op_type`` + field constraints;
  ``match_pattern(graph, pattern)`` walks forward from every
  topo-ordered seed along fan-out-1 consumer edges.
- ``run_rule`` / ``run_pass`` — apply one rule module / every rule
  module in a directory to fixed point. Rule modules declare
  ``PATTERN = [Pattern(...), ...]`` and a ``rewrite(graph, match) ->
  Graph | None`` function. Returned fragments are spliced into the
  graph; ``None`` means no-op / in-place mutation.
- ``run_pipeline(graph, passes, dump=None)`` — run each named pass
  directory in order, dispatching ``dump.on_pass`` after each.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.dump import CompilerDump

_PASSES_DIR = Path(__file__).parent / "passes"
_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


logger = logging.getLogger(__name__)


class RuleSkipped(Exception):
    """Raised by a rule's ``rewrite()`` to signal that the match was
    considered but skipped, with a human-readable reason for why no
    rewrite was applied. The engine catches it, logs the reason at
    DEBUG (visible at ``compile -vv``), and treats the result the same
    as ``return None`` with no in-place mutation. Use this in place of
    a bare ``return None`` whenever the skip reason would help debug
    why a rule didn't fire on a given match."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Chain matcher
# ---------------------------------------------------------------------------


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

    ``nodes`` maps each pattern entry's name to the node id it matched.
    ``consumed`` and ``output`` may be overwritten by the rewrite
    function to control which nodes the rewriter removes and which node
    its edges get redirected to. ``output`` defaults to ``root_node_id``
    when left as ``None``.
    """

    root_node_id: str
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None


def match_pattern(graph: Graph, pattern: list[Pattern]) -> list[Match]:
    """Return every pattern match rooted at a topo-ordered node.

    Matches may overlap — e.g. both ``{A, B}`` and ``{B, C}`` for a
    two-node pattern. The rewriter breaks after the first successful
    ``rewrite`` per pass iteration, so overlap is only a candidate-
    enumeration concern.
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


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    """Loaded rule module — pattern + rewrite plus the rewrite's param list.

    ``param_names`` is captured at load time so the dispatcher can bind
    each rewrite param via signature inspection. The binding rules:

    - ``graph`` — the current ``Graph``
    - ``match`` — the full ``Match`` (escape hatch for advanced rewrites)
    - ``root`` — ``graph.nodes[match.root_node_id]`` (the matched ``Node``)
    - ``out`` — ``root.output`` (the produced ``Tensor``)
    - any ``Pattern.name`` declared in ``PATTERN`` — that pattern entry's
      matched ``Node``
    - anything else — bound positionally to the input ``Node`` at slot
      ``i`` (i.e. ``graph.nodes[root.inputs[i]]``) where ``i`` is the
      param's position among non-reserved / non-pattern params; ``None``
      when ``i ≥ len(root.inputs)`` or the source node was deleted.

    The "anything else" rule lets rewrites read input slots straight off
    the signature::

        def rewrite(graph, inp_x, inp_w, inp_b, out):
            # inp_x = graph.nodes[root.inputs[0]]            (Node)
            # inp_w = graph.nodes[root.inputs[1]]            (Node)
            # inp_b = graph.nodes[root.inputs[2]] or None    (Node | None)
            # out   = root.output                            (Tensor)
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | None]
    param_names: tuple[str, ...]


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
    param_names = tuple(inspect.signature(rewrite_fn).parameters.keys())
    return _Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn, param_names=param_names)


def _build_rewrite_kwargs(rule: _Rule, graph: Graph, match: Match) -> dict | None:
    """Bind each ``rewrite`` param to its source.

    Reserved-name params (``graph`` / ``match`` / ``root`` / ``out``) and
    ``PATTERN``-name params bind by name; every remaining param binds
    positionally to ``root.inputs[i]`` (in declaration order, ``None``
    when the position exceeds the available inputs).

    Returns ``None`` when a pattern-name param's matched node has been
    deleted from the graph between match enumeration and rewrite — a
    safety net for the case the outer ``match.consumed`` check misses.
    """
    pattern_names = {p.name for p in rule.pattern}
    nid = match.root_node_id
    root_node = graph.nodes.get(nid)
    if root_node is None:
        return None

    kwargs: dict = {}
    input_slot = 0
    for pname in rule.param_names:
        if pname == "graph":
            kwargs[pname] = graph
        elif pname == "match":
            kwargs[pname] = match
        elif pname == "root":
            kwargs[pname] = root_node
        elif pname == "out":
            kwargs[pname] = root_node.output
        elif pname in pattern_names:
            mid = match.nodes.get(pname)
            if mid is None or mid not in graph.nodes:
                return None
            kwargs[pname] = graph.nodes[mid]
        else:
            if input_slot < len(root_node.inputs):
                kwargs[pname] = graph.nodes.get(root_node.inputs[input_slot])
            else:
                kwargs[pname] = None
            input_slot += 1
    return kwargs


# ---------------------------------------------------------------------------
# Rewrite loop
# ---------------------------------------------------------------------------


def run_pass(
    graph: Graph,
    pass_dir: Path,
    dump: CompilerDump | None = None,
    pass_idx: int | None = None,
    pass_name: str | None = None,
    select: Iterable[str] | None = None,
) -> Graph:
    """Load all rule modules in ``pass_dir`` and apply them to fixed
    point. ``select``, if given, restricts the run to rules whose name
    (with or without the numeric ordering prefix, e.g. ``tileify`` or
    ``001_tileify``) appears in the iterable — useful for isolating a
    single rule's behavior in tests."""
    rules = _load_rules(pass_dir)
    if select is not None:
        wanted = set(select)
        rules = [r for r in rules if r.name in wanted or _strip_rule_prefix(r.name) in wanted]
    return _apply_rules(graph, rules, dump=dump, pass_idx=pass_idx, pass_name=pass_name)


def run_rule(graph: Graph, rule_path: Path) -> Graph:
    """Load a single rule module and apply it to fixed point."""
    return _apply_rules(graph, [_load_rule(rule_path)])


def _apply_rules(
    graph: Graph,
    rules: list[_Rule],
    dump: CompilerDump | None = None,
    pass_idx: int | None = None,
    pass_name: str | None = None,
) -> Graph:
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
                    t1 = time.monotonic()
                    # Snapshot matched-node ops before rewrite. Always
                    # captured: needed for in-place change detection
                    # (counting toward ``applied``). At -vv or when a
                    # dump is set, the same snapshot also drives the
                    # before/after render text.
                    debug_on = logger.isEnabledFor(logging.DEBUG)
                    pre_ops = {nid: graph.nodes[nid].op for nid in match.consumed if nid in graph.nodes}
                    # Output-name → pre-rewrite node id so the formatter
                    # can follow renames (rules like ``tileify`` mutate
                    # the op AND ``rename_node`` to a friendlier id).
                    pre_names = {graph.nodes[nid].output.name: nid for nid in pre_ops if nid in graph.nodes}
                    kwargs = _build_rewrite_kwargs(rule, graph, match)
                    if kwargs is None:
                        continue
                    try:
                        fragment = rule.rewrite(**kwargs)
                    except RuleSkipped as exc:
                        if debug_on:
                            from deplodock.compiler.pipeline.rule_diff import display_name, emit, format_skipped

                            emit(format_skipped(display_name(pass_name, rule.name), match.root_node_id, exc.reason))
                        rewrite_time += time.monotonic() - t1
                        continue
                    if fragment is None:
                        # Detect both in-place op rebinds AND rule-driven
                        # rename/delete (e.g. ``tileify`` calls
                        # ``graph.rename_node``; the old id disappears
                        # from ``graph.nodes`` even though the rule did
                        # apply).
                        if any(nid not in graph.nodes or graph.nodes[nid].op is not pre_ops[nid] for nid in pre_ops):
                            text = _format_inplace_application(rule.name, graph, match, pre_ops, pass_name=pass_name, pre_names=pre_names)
                            if debug_on:
                                from deplodock.compiler.pipeline.rule_diff import emit

                                emit(text)
                            if dump is not None and pass_idx is not None and pass_name is not None:
                                record = _record_inplace_application(graph, match, pre_ops)
                                dump.on_rule(pass_idx, pass_name, rule.name, record, text)
                            applied += 1
                        rewrite_time += time.monotonic() - t1
                        continue
                    text = _format_rule_application(rule.name, graph, match, fragment, pass_name=pass_name)
                    if debug_on:
                        from deplodock.compiler.pipeline.rule_diff import emit

                        emit(text)
                    if dump is not None and pass_idx is not None and pass_name is not None:
                        record = _record_rule_application(graph, match, fragment)
                        dump.on_rule(pass_idx, pass_name, rule.name, record, text)
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


# ---------------------------------------------------------------------------
# Per-rule snapshot formatting (used at DEBUG, i.e. ``compile -vv``)
# ---------------------------------------------------------------------------


def _format_rule_application(name: str, graph: Graph, match: Match, fragment: Graph, *, pass_name: str | None = None) -> str:
    """Render a one-rule-application snapshot as a unified diff bracketed
    by ``>>> name`` / ``<<< name`` markers (see ``rule_diff``). Kernel
    ops (LoopOp/TileOp/KernelOp/CudaOp) are pretty-printed via their
    dedicated printers rather than dumped as a body repr."""
    from deplodock.compiler.pipeline.rule_diff import display_name, render_rule_diff

    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    matched_nodes = [graph.nodes[nid] for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes]
    before = _format_nodes(matched_nodes, graph)
    frag_nodes = [fragment.nodes[nid] for nid in fragment.topological_order()]
    after = _format_nodes(frag_nodes, fragment)
    return render_rule_diff(display_name(pass_name, name), before, after, header=f"matched at {match.root_node_id}")


def _format_inplace_application(
    name: str,
    graph: Graph,
    match: Match,
    pre_ops: dict,
    *,
    pass_name: str | None = None,
    pre_names: dict[str, str] | None = None,
) -> str:
    """Snapshot for rules that mutate ``node.op`` in place and return
    None (e.g. lowering/tile). Renders before/after of the mutated
    nodes as a unified diff by swapping their op temporarily for the
    "before" view.

    Some rules (notably ``tileify``) mutate the op AND rename the node
    to a friendlier id. ``pre_names`` (output-name → pre-rewrite id)
    lets the formatter resolve a renamed-away id to its post-rewrite
    counterpart by matching on output name, which is preserved across
    ``rename_node``."""
    from deplodock.compiler.pipeline.rule_diff import display_name, render_rule_diff

    # Build a "current id" for each pre_ops entry, following any rename
    # that happened during the rewrite.
    name_to_post_id = {graph.nodes[nid].output.name: nid for nid in graph.nodes} if pre_names else {}
    pre_to_post: dict[str, str] = {}
    for nid in pre_ops:
        if nid in graph.nodes:
            pre_to_post[nid] = nid
        elif pre_names is not None:
            for out_name, old_id in pre_names.items():
                if old_id == nid and out_name in name_to_post_id:
                    pre_to_post[nid] = name_to_post_id[out_name]
                    break

    mutated_pre = [nid for nid in pre_ops if nid in pre_to_post and graph.nodes[pre_to_post[nid]].op is not pre_ops[nid]]
    post_ids = [pre_to_post[nid] for nid in mutated_pre]
    post_ops = {pid: graph.nodes[pid].op for pid in post_ids}
    # Render "before": swap each post-id node's op back to the pre op.
    for nid in mutated_pre:
        graph.nodes[pre_to_post[nid]].op = pre_ops[nid]
    before_nodes = [graph.nodes[pid] for pid in graph.topological_order() if pid in post_ids]
    before = _format_nodes(before_nodes, graph)
    # Restore.
    for pid in post_ids:
        graph.nodes[pid].op = post_ops[pid]
    after_nodes = [graph.nodes[pid] for pid in graph.topological_order() if pid in post_ids]
    after = _format_nodes(after_nodes, graph)
    return render_rule_diff(display_name(pass_name, name), before, after, header=f"matched at {match.root_node_id} (in-place)")


def _record_rule_application(graph: Graph, match: Match, fragment: Graph) -> dict:
    """Structured analog of ``_format_rule_application`` for JSON dumps.

    Captures the matched-subgraph nodes and the fragment's nodes as plain
    dicts so post-hoc scripts (and the article-side analysis) can iterate
    rule applications without re-parsing the text snapshot.
    """
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    return {
        "root": match.root_node_id,
        "matched_pattern_nodes": dict(match.nodes),
        "before": [_node_to_dict(graph.nodes[nid]) for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes],
        "after": [_node_to_dict(fragment.nodes[nid]) for nid in fragment.topological_order()],
    }


def _record_inplace_application(graph: Graph, match: Match, pre_ops: dict) -> dict:
    mutated = [nid for nid, prev in pre_ops.items() if nid in graph.nodes and graph.nodes[nid].op is not prev]
    return {
        "root": match.root_node_id,
        "in_place": True,
        "before": [{"id": nid, "op_class": type(pre_ops[nid]).__name__} for nid in mutated],
        "after": [_node_to_dict(graph.nodes[nid]) for nid in mutated],
    }


def _node_to_dict(node) -> dict:
    return {
        "id": node.output.name,
        "op_class": type(node.op).__name__,
        "inputs": list(node.inputs),
        "output_shape": list(node.output.shape),
        "output_dtype": node.output.dtype,
    }


def _format_nodes(nodes: list, graph: Graph) -> str:
    """Render a list of nodes as readable text. Kernel-IR ops use their
    own ``pretty_body``; everything else falls back to a ``name: ClsName(args)``
    one-liner. Scalar ``ConstantOp`` inputs are inlined as literals (same
    treatment as ``format_kernels`` — see ``_inline_scalar_loads``)."""
    from deplodock.compiler.graph import _fmt_op
    from deplodock.compiler.ir.base import ConstantOp, InputOp
    from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs

    lines: list[str] = []
    for node in nodes:
        op = node.op
        if isinstance(op, (InputOp, ConstantOp)):
            continue
        body = op.pretty_body()
        if body is None:
            lines.append(f"{node.output.name} = {_fmt_op(node, graph)}")
            continue
        arg_names = [graph.nodes[inp].output.name for inp in node.inputs if inp in graph.nodes]
        lines.append(f"{node.output.name} = {type(op).__name__}({', '.join(arg_names)})")
        scalar_inputs = _scalar_constant_inputs(graph, node, ConstantOp)
        if scalar_inputs:
            body = _inline_scalar_loads(body, scalar_inputs)
        lines.extend(f"  {line}" for line in body.splitlines())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fragment splicer
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
        # Preserve the fragment's id when it doesn't collide. Lifting / fusion
        # fragments use stable names (``lift_<nid>``, ``merged_<nid>``) that
        # don't clash with the surrounding graph because the original node is
        # already consumed. Keeping the id stable means buf names inside the
        # LoopOp body (``Load.source`` / ``Write.output``, both buf names)
        # remain consistent with the surrounding graph.
        preferred_id = frag_id if frag_id not in g.nodes else None
        new_id = g.add_node(
            op=frag_node.op,
            inputs=mapped_inputs,
            output=Tensor(frag_node.output.name, frag_node.output.shape, frag_node.output.dtype),
            node_id=preferred_id,
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


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    graph: Graph,
    passes: list[str],
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
) -> Graph:
    """Run each named pass directory in order; dispatch ``dump.on_pass``
    after each. ``select`` is forwarded to :func:`run_pass` for every
    pass — only rules whose name matches will run."""
    t_start = time.monotonic()
    select_set = set(select) if select is not None else None
    for idx, name in enumerate(passes, start=1):
        t0 = time.monotonic()
        n_before = len(graph.nodes)
        graph = run_pass(graph, _PASSES_DIR / name, dump=dump, pass_idx=idx, pass_name=name, select=select_set)
        logger.info("compile: %-18s %.2fs (%d -> %d nodes)", name, time.monotonic() - t0, n_before, len(graph.nodes))
        if dump is not None:
            dump.on_pass(idx, name, graph)
    logger.info("compile: total %.2fs", time.monotonic() - t_start)
    return graph
