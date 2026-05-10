"""Pattern-based rewrite engine and compile-pipeline entry point.

Public surface:

- ``Pattern`` / ``Match`` / ``match_pattern`` — chain matcher: each
  ``Pattern`` matches one node by ``op_type`` + field constraints;
  ``match_pattern(graph, pattern)`` walks forward from every
  topo-ordered seed along fan-out-1 consumer edges.
- ``run_rule`` / ``run_pass`` — apply one rule module / every rule
  module in a directory to fixed point. Rule modules declare
  ``PATTERN = [Pattern(...), ...]`` and a ``rewrite(...)`` function
  whose return type discriminates the rewrite flavor:
  * ``Graph`` — functional fragment, spliced in place of the match.
  * ``Op`` — in-place rebind of ``root.op`` (id, inputs, hints kept).
  * ``list[Graph | Op]`` — autotuning fork: engine applies option 0
    inline and pushes one ``Candidate`` per remaining option onto the
    search queue.
  Raise ``RuleSkipped`` to decline a match.
- ``Candidate`` / ``Search`` / ``run_pipeline`` — the autotune driver.
  ``run_pipeline`` yields ``Candidate``s; for deterministic rules
  (no list returns) it yields exactly one.

Rule contract: rules MUST be idempotent on their own output. The engine
re-runs the full pipeline on every popped candidate, relying on each
rule's "already applied" guard (often implicit via op-type change) to
skip work that's already done."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node, Tensor, _fmt_op
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from deplodock.compiler.pipeline.rule_diff import display_name, emit, format_skipped, render_rule_diff

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

    ``graph`` is the graph this match was built against (rules access
    it via ``match.graph`` for ad-hoc lookups). ``nodes`` maps each
    pattern entry's name to the matched node id. ``consumed`` and
    ``output`` may be overwritten by the rewrite function to control
    which nodes the rewriter removes and which node its edges get
    redirected to. ``output`` defaults to ``root_node_id`` when left
    as ``None``.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None
    # Snapshot of id(Node) at match time for every consumed node. The
    # ``is_alive`` check uses this to detect the case where an earlier
    # match in the same batch removed a consumed node and a different
    # node was added at the same id (e.g. splicer auto-rename hitting
    # a recently-freed name). Pure id-existence wouldn't catch that.
    _identities: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def root(self) -> Node:
        """The root ``Node`` (matched by the first ``Pattern`` entry)."""
        return self.graph.nodes[self.root_node_id]

    def node(self, name_or_id: str) -> Node:
        """Resolve a pattern name (e.g. ``"producer"``) OR a raw node id
        to the current ``Node`` in ``graph``. Raises ``KeyError`` if the
        node has been removed."""
        nid = self.nodes.get(name_or_id, name_or_id)
        return self.graph.nodes[nid]

    def input(self, i: int) -> Node | None:
        """Root's ``i``-th input as a ``Node``, or ``None`` when ``i``
        exceeds the input count or the input node was removed."""
        root = self.root
        if i >= len(root.inputs):
            return None
        return self.graph.nodes.get(root.inputs[i])

    def is_alive(self) -> bool:
        """``True`` when every consumed node still resolves to the same
        ``Node`` object captured at match time. Catches both removal and
        the "removed-then-re-added under same id" case."""
        for nid in self.consumed:
            n = self.graph.nodes.get(nid)
            if n is None or id(n) != self._identities.get(nid):
                return False
        return True


# ---------------------------------------------------------------------------
# Autotune surface: Candidate, TraceEntry, Search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceEntry:
    """One rule application in a candidate's history. ``choice_idx`` is
    the option picked at fork points (0 for deterministic single-option
    rules)."""

    rule_name: str
    choice_idx: int = 0


@dataclass
class Cursor:
    """Pipeline resume state for a ``Candidate``.

    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass."""

    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    def advance_rule(self) -> None:
        self.rule_idx += 1

    def restart_scan(self) -> None:
        self.rule_idx = 0
        self.n_applied = 0

    def advance_pass(self) -> None:
        self.pass_idx += 1
        self.restart_scan()

    def record_functional(self) -> None:
        self.n_applied += 1


@dataclass
class Candidate:
    """A single point in the search space. The engine pops a candidate,
    advances it by one rule application attempt, pushes the resulting
    successor(s) back onto the search queue, and yields the candidate
    when ``cursor.pass_idx`` reaches the end of the pipeline.

    ``graph`` is owned by this candidate (deep-copied on multi-option
    forks; mutated in place on single-option steps). ``ctx`` is shared
    by reference. ``trace`` is the immutable history of rule
    applications on this branch. ``cursor`` is the pipeline cursor."""

    graph: Graph
    ctx: Context
    trace: tuple[TraceEntry, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)


class Search(Protocol):
    """Search-strategy hook. The engine pushes spawned candidates here
    and pops the next one to expand. Implementations choose the
    ordering — DFS / BFS / priority / MCTS / whatever."""

    def push(self, c: Candidate) -> None: ...
    def pop(self) -> Candidate | None: ...  # None when exhausted


class DFSSearch:
    """Depth-first stack. With deterministic (non-list-returning) rules,
    no fork ever happens, so this yields exactly one candidate — same
    observable behavior as the old single-graph pipeline driver. Used
    as the default by ``run_pipeline``; pass an explicit ``search=`` to
    ``run_autotune`` for any other strategy."""

    def __init__(self) -> None:
        self._stack: list[Candidate] = []

    def push(self, c: Candidate) -> None:
        self._stack.append(c)

    def pop(self) -> Candidate | None:
        return self._stack.pop() if self._stack else None


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
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
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
        identities[cursor] = id(node)
        cursor = _sole_consumer(graph, cursor)
    return Match(graph=graph, root_node_id=start, nodes=nodes, consumed=consumed, _identities=identities)


def _check_constraints(node, prod: Pattern) -> bool:
    return all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items())


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

        def rewrite(inp_x, inp_w, inp_b, out):
            # inp_x = graph.nodes[root.inputs[0]]            (Node)
            # inp_w = graph.nodes[root.inputs[1]]            (Node)
            # inp_b = graph.nodes[root.inputs[2]] or None    (Node | None)
            # out   = root.output                            (Tensor)

    Rules that need ad-hoc graph-wide lookups take ``match`` and use
    ``match.graph`` / ``match.node(id)`` — there's no ``graph`` reserved
    kwarg.
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | Op | None]
    param_names: tuple[str, ...]


def _load_rules(pass_dir: Path) -> list[_Rule]:
    rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
    return [_load_rule(f) for f in rule_files]


def _load_rule(path: Path) -> _Rule:
    import sys

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rule from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any ``@dataclass`` defined in the rule
    # module can resolve its own module via ``sys.modules`` —
    # ``dataclasses._is_type`` looks up ``cls.__module__`` there to
    # check for ``KW_ONLY`` and raises ``AttributeError`` on a missing
    # entry.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    pattern = getattr(module, "PATTERN", None)
    rewrite_fn = getattr(module, "rewrite", None)
    if pattern is None:
        raise ValueError(f"Rule {path} missing PATTERN")
    if rewrite_fn is None:
        raise ValueError(f"Rule {path} missing rewrite() function")
    param_names = tuple(inspect.signature(rewrite_fn).parameters.keys())
    return _Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn, param_names=param_names)


def _build_rewrite_kwargs(rule: _Rule, match: Match, ctx: Context | None) -> dict:
    """Bind each ``rewrite`` param to its source.

    Reserved-name params (``match`` / ``root`` / ``out`` / ``ctx``) and
    ``PATTERN``-name params bind by name; every remaining param binds
    positionally to ``root.inputs[i]`` (in declaration order, ``None``
    when the position exceeds the available inputs)."""
    pattern_names = {p.name for p in rule.pattern}
    root_node = match.root
    graph = match.graph
    kwargs: dict = {}
    input_slot = 0
    for pname in rule.param_names:
        if pname == "match":
            kwargs[pname] = match
        elif pname == "root":
            kwargs[pname] = root_node
        elif pname == "out":
            kwargs[pname] = root_node.output
        elif pname == "ctx":
            kwargs[pname] = ctx
        elif pname in pattern_names:
            kwargs[pname] = match.node(pname)
        else:
            if input_slot < len(root_node.inputs):
                kwargs[pname] = graph.nodes.get(root_node.inputs[input_slot])
            else:
                kwargs[pname] = None
            input_slot += 1
    return kwargs


def _try_rewrite(
    rule: _Rule,
    match: Match,
    ctx: Context | None,
    *,
    debug_on: bool,
    pass_name: str | None,
) -> list | None:
    """Run ``rule.rewrite`` against ``match`` and return its options.

    Returns ``None`` (caller should ``continue``) when the match is
    stale, the rule raises ``RuleSkipped``, or it returns no options.
    Otherwise returns a non-empty list of ``Op``/``Graph`` options."""
    if not match.is_alive():
        return None
    kwargs = _build_rewrite_kwargs(rule, match, ctx)
    try:
        result = rule.rewrite(**kwargs)
    except RuleSkipped as exc:
        if debug_on:
            emit(format_skipped(display_name(pass_name, rule.name), match.root_node_id, exc.reason))
        return None
    options = list(result) if isinstance(result, (list, tuple)) else [result]
    return options or None


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
    ctx: Context | None = None,
) -> Graph:
    """Load all rule modules in ``pass_dir`` and apply them to fixed
    point. ``select``, if given, restricts the run to rules whose name
    (with or without the numeric ordering prefix, e.g. ``tileify`` or
    ``001_tileify``) appears in the iterable — useful for isolating a
    single rule's behavior in tests.

    Single-graph helper: use ``run_pipeline`` for autotuning. Discards
    any fork ``Candidate``s a multi-option rule might want to spawn."""
    if ctx is None:
        ctx = Context.probe()
    rules = _filter_rules(_load_rules(pass_dir), set(select) if select is not None else None)
    search = DFSSearch()
    search.push(Candidate(graph=graph, ctx=ctx))
    return next(_search_loop(search, [rules], [pass_name or ""], ctx, dump)).graph


def run_rule(graph: Graph, rule_path: Path, ctx: Context | None = None) -> Graph:
    """Load a single rule module and apply it to fixed point. Discards
    fork siblings — for autotuning use the full ``run_pipeline`` driver."""
    if ctx is None:
        ctx = Context.probe()
    search = DFSSearch()
    search.push(Candidate(graph=graph, ctx=ctx))
    return next(_search_loop(search, [[_load_rule(rule_path)]], [""], ctx, None)).graph


def _search_loop(
    search: Search,
    rules_per_pass: list[list[_Rule]],
    pass_names: list[str],
    ctx: Context | None,
    dump: CompilerDump | None,
) -> Iterator[Candidate]:
    """The unified search-driven driver. Each iteration: pop a
    candidate, try one rule application (or end-of-pass bookkeeping),
    push successor(s). Yields when a candidate reaches the end of the
    pipeline (``cursor.pass_idx >= len(pass_names)``).

    Used by every engine entry point — ``run_autotune`` (full pipeline),
    ``run_pass`` (one pass), ``run_rule`` (one rule). They differ only
    in the rules-per-pass list and the ``Search`` instance supplied."""
    while (cand := search.pop()) is not None:
        cur = cand.cursor
        if cur.pass_idx >= len(pass_names):
            yield cand
            continue
        rules = rules_per_pass[cur.pass_idx]
        # End-of-scan bookkeeping: did any rewrite fire in this scan?
        # If yes, restart from rule 0 (something changed; earlier rules
        # may have new opportunities). If no, advance to next pass.
        if cur.rule_idx >= len(rules):
            if cur.n_applied > 0:
                cur.restart_scan()
            else:
                name = pass_names[cur.pass_idx]
                if name:
                    logger.info("compile: %-18s done (%d nodes)", name, len(cand.graph.nodes))
                if dump is not None and name:
                    dump.on_pass(cur.pass_idx + 1, name, cand.graph)
                cur.advance_pass()
            search.push(cand)
            continue
        # One iteration: try one rule application.
        rule = rules[cur.rule_idx]
        pass_idx_arg = cur.pass_idx + 1 if pass_names[cur.pass_idx] else None
        pass_name_arg = pass_names[cur.pass_idx] or None
        next_cands = _try_one_rule(cand, rule, ctx, dump, pass_idx_arg, pass_name_arg)
        if next_cands is None:
            # Rule had nothing to do — advance rule_idx and push back.
            cur.advance_rule()
            search.push(cand)
        else:
            # Rule applied (possibly with forks). cursor.rule_idx unchanged
            # so the next iteration re-tries the same rule (more matches
            # may exist; idempotent rules will RuleSkipped on already-
            # processed roots). ``_try_one_rule`` already incremented
            # ``cursor.n_applied`` on every successor.
            for c in next_cands:
                search.push(c)


def _apply_one(graph: Graph, match: Match, result: Graph | Op, *, rule_name: str) -> Graph:
    """Apply one rewrite outcome to ``graph``. ``Op`` rebinds
    ``root.op`` in place (id, inputs, hints kept); ``Graph`` is a
    fragment spliced via ``_apply_replacement``. Returns the (possibly
    same, possibly new) graph."""
    if isinstance(result, Op):
        graph.nodes[match.root_node_id].op = result
        return graph
    assert isinstance(result, Graph), f"rule {rule_name} returned {type(result).__name__}; expected Graph, Op, list, or RuleSkipped"
    return _apply_replacement(graph, match, result)


def _try_one_rule(
    cand: Candidate,
    rule: _Rule,
    ctx: Context | None,
    dump: CompilerDump | None,
    pass_idx: int | None,
    pass_name: str | None,
) -> list[Candidate] | None:
    """One iteration: enumerate ``rule``'s matches once and apply each
    live match (with non-skipped rewrite) in batch. Match enumeration
    happens ONCE per call — staged matches that get invalidated by an
    earlier application in the batch are filtered via ``is_alive()``
    rather than re-walking the graph. Per-rule batch semantics are what
    downstream rules (lift / fusion / staging) depend on for
    deterministic structure.

    Returns:
    - ``None`` — no live match yielded a non-skipped rewrite; the rule
      had nothing to do at this state.
    - ``[c]`` — applied without forking; ``cand`` with extended trace
      and incremented ``cursor.n_applied``.
    - ``[*forks, c]`` — multi-option fork(s) encountered during the
      batch; ``c`` (inline branch) is at the *end* so a LIFO ``Search``
      pops it first; forks carry deep-copied graphs with their alt
      applied at the fork point.

    Caller (``run_autotune``) handles the ``cursor.rule_idx`` / ``cursor.pass_idx``
    bookkeeping based on whether this returns ``None`` or ``[...]``."""
    debug_on = logger.isEnabledFor(logging.DEBUG)
    dump_on = dump is not None and pass_idx is not None and pass_name is not None
    need_text = debug_on or dump_on

    matches = match_pattern(cand.graph, rule.pattern)
    forks: list[Candidate] = []
    fired = False
    functional_fired = False
    for match in matches:
        options = _try_rewrite(rule, match, ctx, debug_on=debug_on, pass_name=pass_name)
        if options is None:
            continue
        chosen = options[0]
        fragment = _wrap_op_as_fragment(cand.graph, match.root_node_id, chosen) if isinstance(chosen, Op) else chosen
        text = _format_rule_application(rule.name, cand.graph, match, fragment, pass_name=pass_name) if need_text else None
        if debug_on:
            emit(text)
        if dump_on:
            record = _record_rule_application(cand.graph, match, fragment)
            dump.on_rule(pass_idx, pass_name, rule.name, record, text)
        # Fork branches: each alt gets a deep-copy of cand.graph at
        # this point in the batch (after prior matches' option-0
        # applications, before this match's). The fork resumes into
        # the same rule batch from the next match — when it's popped,
        # ``_try_one_rule`` will re-enumerate against its alt graph
        # and process whatever matches remain there.
        if len(options) > 1:
            snapshot = copy.deepcopy(cand.graph)
            for alt_idx, alt in enumerate(options[1:], start=1):
                fg = copy.deepcopy(snapshot)
                fm = _remap_match_to(fg, match)
                fg = _apply_one(fg, fm, alt, rule_name=rule.name)
                fork_cursor = Cursor(
                    pass_idx=cand.cursor.pass_idx,
                    rule_idx=cand.cursor.rule_idx,
                    n_applied=cand.cursor.n_applied + (0 if isinstance(alt, Op) else 1),
                )
                forks.append(
                    Candidate(
                        graph=fg,
                        ctx=cand.ctx,
                        trace=(*cand.trace, TraceEntry(rule.name, alt_idx)),
                        cursor=fork_cursor,
                    )
                )
        cand.graph = _apply_one(cand.graph, match, chosen, rule_name=rule.name)
        cand.trace = (*cand.trace, TraceEntry(rule.name, 0))
        # ``cursor.n_applied`` counts only FUNCTIONAL (Graph) fires — the
        # end-of-scan restart fires when these accumulate. In-place (Op)
        # fires don't restart the scan: they only mutate ``root.op`` and
        # the same pattern won't re-match the new op for most rules;
        # rules whose op type stays the same rely on per-rule idempotence
        # guards (and their downstream rule order is intentional).
        if not isinstance(chosen, Op):
            cand.cursor.record_functional()
            functional_fired = True
        fired = True
    if not fired:
        return None
    # If only in-place fires happened in this batch, advance
    # ``cursor.rule_idx`` — re-running the same rule on the post-mutation
    # state would either RuleSkipped (idempotent guard) or re-fire
    # (non-idempotent in-place rule), so don't re-scan it.
    if not functional_fired:
        cand.cursor.advance_rule()
    # Forks first, inline branch (cand) last — LIFO ``Search`` pops
    # cand next, driving the current branch deep before backtracking.
    return [*forks, cand]


def _remap_match_to(forked_graph: Graph, match: Match) -> Match:
    """Build a fresh ``Match`` against ``forked_graph`` (a deep copy)
    that mirrors ``match``'s ids. Re-snapshot ``_identities`` against
    the forked nodes so the new match's ``is_alive`` check works."""
    identities = {nid: id(forked_graph.nodes[nid]) for nid in match.consumed if nid in forked_graph.nodes}
    return Match(
        graph=forked_graph,
        root_node_id=match.root_node_id,
        nodes=dict(match.nodes),
        consumed=set(match.consumed),
        output=match.output,
        _identities=identities,
    )


# ---------------------------------------------------------------------------
# Per-rule snapshot formatting (used at DEBUG, i.e. ``compile -vv``)
# ---------------------------------------------------------------------------


def _format_rule_application(name: str, graph: Graph, match: Match, fragment: Graph, *, pass_name: str | None = None) -> str:
    """Render a one-rule-application snapshot as a unified diff bracketed
    by ``>>> name`` / ``<<< name`` markers (see ``rule_diff``). Kernel
    ops (LoopOp/TileOp/KernelOp/CudaOp) are pretty-printed via their
    dedicated printers rather than dumped as a body repr."""
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    matched_nodes = [graph.nodes[nid] for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes]
    before = _format_nodes(matched_nodes, graph)
    frag_nodes = [fragment.nodes[nid] for nid in fragment.topological_order()]
    after = _format_nodes(frag_nodes, fragment)
    return render_rule_diff(display_name(pass_name, name), before, after, header=f"matched at {match.root_node_id}")


def _wrap_op_as_fragment(graph: Graph, root_id: str, new_op: Op) -> Graph:
    """Build a single-node fragment that mirrors ``graph.nodes[root_id]``
    with ``new_op`` substituted. Lets the engine render an in-place op
    rebind through the same diff/dump path as a functional fragment splice
    (the engine then assigns ``root.op = new_op`` directly, bypassing the
    splicer — node id, inputs list, hints, and output Tensor are kept)."""
    root = graph.nodes[root_id]
    frag = Graph()
    for inp_id in root.inputs:
        if inp_id in frag.nodes:
            continue
        inp = graph.nodes.get(inp_id)
        shape = inp.output.shape if inp is not None else ()
        dtype = inp.output.dtype if inp is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    out_id = frag.add_node(new_op, list(root.inputs), root.output, node_id=root.id)
    frag.outputs = [out_id]
    return frag


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
    treatment as ``format_kernels`` — see ``_inline_scalar_loads``).

    The leading ``kernel <name>  inputs: ...  outputs: ...`` header that
    ``TileOp.pretty_body`` prepends is stripped here: this path already
    emits ``<output> = TileOp(<inputs>)`` one line above, so the kernel
    header would just duplicate the same info and shift the body's
    indent by 4 spaces, ballooning the diff."""
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
        body_lines = body.splitlines()
        if body_lines and body_lines[0].lstrip().startswith("kernel ") and " inputs: " in body_lines[0] and " outputs: " in body_lines[0]:
            body_lines = [_dedent(ln, 4) for ln in body_lines[1:]]
        lines.extend(f"  {line}" for line in body_lines)
    return "\n".join(lines)


def _dedent(line: str, n: int) -> str:
    """Strip up to ``n`` leading spaces from ``line``."""
    i = 0
    while i < n and i < len(line) and line[i] == " ":
        i += 1
    return line[i:]


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
        orig = g.nodes.get(nid)
        if orig is None:
            continue
        if orig.hints:
            g.nodes[new_output].hints.merge(orig.hints)
        if nid != old_output:
            g.remove_node(nid)
    if old_output in g.nodes:
        g.remove_node(old_output)

    # Promote the new node's id to its friendly output.name once consumed
    # nodes are gone — keeps kernel buf names (which embed the node id)
    # readable. Falls back silently if the friendly name is taken.
    desired = g.nodes[new_output].output.name
    if desired and desired != new_output and desired not in g.nodes:
        g.rename_node(new_output, desired)

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
    ctx: Context | None = None,
) -> Graph:
    """Run each named pass directory in order; dispatch ``dump.on_pass``
    after each. Single-candidate convenience wrapper around
    :func:`run_autotune` (uses ``DFSSearch``) — returns the graph of the
    first terminal candidate. With deterministic rules that's the only
    candidate.

    ``ctx`` is built once (probing the live device if not provided)
    and passed to every rule that takes a ``ctx`` parameter."""
    return next(run_autotune(graph, passes, search=DFSSearch(), dump=dump, select=select, ctx=ctx)).graph


def run_autotune(
    graph: Graph,
    passes: list[str],
    *,
    search: Search,
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
) -> Iterator[Candidate]:
    """Drive the autotune search. Yields one terminal ``Candidate`` per
    fully-explored branch. With deterministic rules (no list-returning
    rewrites) the search yields exactly one — same shape as
    ``run_pipeline``.

    The loop is fully search-driven: pop a candidate, advance it by one
    rule application via :func:`_try_step`, push successor(s) back to
    ``search``. When no rule fires in the current pass, advance the
    candidate's ``cursor.pass_idx`` and push it back. When ``cursor.pass_idx``
    reaches the end of ``passes``, the candidate is terminal and gets
    yielded.

    ``search`` chooses the order: ``DFSSearch`` for depth-first
    (current-branch-first) traversal — the default in ``run_pipeline``;
    any other ``Search`` implementation for priority / MCTS / beam.

    ``ctx`` is built once (probing the live device if not provided)
    and shared by every candidate."""
    if ctx is None:
        ctx = Context.probe()
    select_set = set(select) if select is not None else None
    rules_per_pass = [_filter_rules(_load_rules(_PASSES_DIR / name), select_set) for name in passes]
    t_start = time.monotonic()

    search.push(Candidate(graph=graph, ctx=ctx))

    yield from _search_loop(search, rules_per_pass, passes, ctx, dump)
    logger.info("compile: total %.2fs", time.monotonic() - t_start)


def _filter_rules(rules: list[_Rule], select_set: set[str] | None) -> list[_Rule]:
    if select_set is None:
        return rules
    return [r for r in rules if r.name in select_set or _strip_rule_prefix(r.name) in select_set]
