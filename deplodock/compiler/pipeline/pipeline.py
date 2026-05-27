"""Pipeline value types and compile driver: ``Pattern``, ``Match``,
``Rule``, ``RuleSkipped``, ``Pass``, ``Cursor``, ``Pipeline``.

Bundled together because they form a tight chain — ``Pattern`` defines
what a rule matches, ``Rule`` carries the pattern + rewrite, ``Pass``
groups rules, ``Pipeline`` holds passes and drives matching, ``Cursor``
tracks per-candidate resume state, and ``Match`` carries ``Rule`` (which
backref-resolves to ``Pass``) + ``Pipeline`` so callers can read all
pass / rule / dump metadata off one field.

``Pipeline`` also owns the compile entry points — :meth:`build`,
:meth:`run`, :meth:`tune`, :meth:`search`. The per-rule logging,
rewrite-kwarg dispatch, and snapshot rendering live on
:class:`Candidate` (see :mod:`..search.candidate`).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.pipeline.knob import format_tuning_knobs

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.candidate import Candidate
    from deplodock.compiler.pipeline.search.db import SearchDB
    from deplodock.compiler.pipeline.search.policy import Search

logger = logging.getLogger("deplodock.compiler.pipeline")


_PASSES_DIR = Path(__file__).resolve().parent / "passes"
_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


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
class Rule:
    """One rewrite rule loaded from a ``passes/<dir>/NNN_<name>.py``
    module.

    * ``name`` — the file stem (engine display + dump filenames).
    * ``pattern`` — the chain-match pattern the rule fires on.
    * ``rewrite`` — the rule's ``rewrite`` function. ``None`` for the
      no-rewrite stubs :meth:`Pipeline.from_pattern` builds for
      pattern-matching-only callers.
    * ``param_names`` — captured at load time so the dispatcher can
      bind each rewrite param via signature inspection. The binding
      rules (kept here so docstring + dataclass live together):

      - ``graph`` — the current ``Graph``
      - ``match`` — the full ``Match`` (escape hatch)
      - ``root`` — ``graph.nodes[match.root_node_id]``
      - ``out`` — ``root.output``
      - ``ctx`` — the engine's ``Context``
      - any ``Pattern.name`` declared in ``pattern`` — that pattern
        entry's matched ``Node``
      - anything else — bound positionally to the input ``Node`` at
        slot ``i``, ``None`` past the input count or for deleted
        source nodes.

    * ``pass_`` — backref to the owning ``Pass``. Stamped by ``Pass``
      at construction time; ``None`` only on stray ``Rule`` instances
      built outside a pipeline (none exist in production paths).
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | Op | None] | None = None
    param_names: tuple[str, ...] = field(default_factory=tuple)
    pass_: Pass | None = field(default=None, repr=False, compare=False)


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


@dataclass(frozen=True)
class Fork:
    """A deferred fork option in the search tree.

    Two flavors share the same shape:

    - **Branch Fork** (``is_leaf=False``) — produced explicitly by a rule's
      ``rewrite()`` to spawn a hierarchical fork point. ``expand()`` returns
      the next level of options (more Forks, concrete leaves, or a mix);
      the search loop drives this via :meth:`LazyCandidate.expand`.
    - **Leaf Fork** (``is_leaf=True``) — produced by
      :meth:`LazyCandidate.from_op` / :meth:`LazyCandidate.from_graph` to
      uniformly wrap concrete ``Op`` / ``Graph`` rewrites. ``expand()``
      returns ``[option]`` (one element); :meth:`LazyCandidate.resolve`
      invokes it once at resolve time to retrieve the leaf and apply it.

    Sharing one shape lets ``LazyCandidate.pending`` carry just ``Fork``
    (no tagged union) — the search loop branches on ``Fork.is_leaf`` to
    decide expand-vs-resolve, and :func:`_best_fork` reads ``Fork.knobs``
    polymorphically without isinstance.

    ``knobs`` is the knob-delta this Fork pins (used by :func:`_best_fork`
    to match the lowering DB without firing the thunk). ``score`` is the
    MCTS prior for the unvisited-sibling tiebreak — typically the score
    of the best-reachable leaf under this branch."""

    knobs: dict
    expand: Callable[[], list[Op | Graph | Fork]]
    score: float = 0.0
    is_leaf: bool = False


@dataclass
class Pass:
    """One pipeline pass: a named, indexed list of rules.

    * ``name`` — pass directory name (e.g. ``"frontend/decomposition"``),
      or ``""`` for empty / nameless pass slots (early-exit stubs).
    * ``rules`` — the rules in this pass, in load order.
    * ``index`` — 0-based position in the pipeline.

    Stamps each rule's ``pass_`` backref on construction so a ``Match``
    that carries a ``Rule`` can resolve pass metadata without holding a
    separate index.
    """

    name: str
    rules: list[Rule]
    index: int = 0

    def __post_init__(self) -> None:
        for r in self.rules:
            r.pass_ = self

    @classmethod
    def load(cls, name: str, index: int, select: set[str] | None = None) -> Pass:
        """Discover, import, and (optionally) filter the rule modules
        under ``passes/<name>/``. ``select``, when given, keeps only
        rules whose file stem — or stem with the numeric prefix
        stripped — appears in the set."""
        pass_dir = _PASSES_DIR / name
        rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
        rules: list[Rule] = []
        for path in rule_files:
            if select is not None and path.stem not in select and _strip_rule_prefix(path.stem) not in select:
                continue
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load rule from {path}")
            module = importlib.util.module_from_spec(spec)
            # Register before exec so any ``@dataclass`` defined in the
            # rule module can resolve its own module via ``sys.modules``
            # — ``dataclasses._is_type`` looks up ``cls.__module__``
            # there to check for ``KW_ONLY`` and raises
            # ``AttributeError`` on a missing entry.
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            pattern = getattr(module, "PATTERN", None)
            rewrite_fn = getattr(module, "rewrite", None)
            if pattern is None:
                raise ValueError(f"Rule {path} missing PATTERN")
            if rewrite_fn is None:
                raise ValueError(f"Rule {path} missing rewrite() function")
            param_names = tuple(inspect.signature(rewrite_fn).parameters.keys())
            rules.append(Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn, param_names=param_names))
        return cls(name=name, rules=rules, index=index)


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

    ``pipeline`` + ``rule`` locate this match in the run; the rewriter
    reaches pass metadata via ``match.rule.pass_`` and the dump sink
    via ``match.pipeline.dump``. Both stamped by :meth:`Pipeline.match`
    at construction time. ``is_last`` is stamped on the last live
    match returned by :meth:`Pipeline.match` so
    ``Candidate.try_rewrite`` knows when to advance the cursor.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    pipeline: Pipeline
    rule: Rule
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    # ``str`` redirects one node's consumers to the fragment's sole output;
    # ``dict[old_id, frag_output_id]`` redirects several at once (multi-output
    # splice — see ``Graph.splice``). ``None`` defaults to ``root_node_id``.
    output: str | dict[str, str] | None = None
    is_last: bool = False
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

    def remap(self, graph: Graph) -> Match:
        """Build a fresh ``Match`` against ``graph`` (a copy of the
        original) that mirrors this match's ids. Re-snapshots
        ``_identities`` against the new graph's nodes so ``is_alive``
        still works after the copy. Used when materializing a lazy
        fork — the fork copies the parent's snapshot, then needs a
        match anchored on the copy."""
        identities = {nid: id(graph.nodes[nid]) for nid in self.consumed if nid in graph.nodes}
        return Match(
            graph=graph,
            root_node_id=self.root_node_id,
            pipeline=self.pipeline,
            rule=self.rule,
            nodes=dict(self.nodes),
            consumed=set(self.consumed),
            output=self.output,
            is_last=self.is_last,
            _identities=identities,
        )


@dataclass
class Cursor:
    """Pipeline resume state for a candidate. Owns the entire advance
    logic: ``advance(graph)`` moves past the current rule batch,
    wrapping to the next pass — logging "compile: <pass> done" and
    flushing ``pipeline.dump.on_pass`` — when the scan completes with
    no functional rewrites.

    * ``pipeline`` — the pipeline being driven; needed to look up the
      current pass / rule by index.
    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass.
    """

    pipeline: Pipeline
    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    @property
    def is_done(self) -> bool:
        return self.pass_idx >= len(self.pipeline.passes)

    @property
    def current_pass(self) -> Pass:
        assert not self.is_done, f"cursor is done (pass_idx={self.pass_idx} >= {len(self.pipeline.passes)})"
        return self.pipeline.passes[self.pass_idx]

    @property
    def current_rule(self) -> Rule:
        pass_ = self.current_pass
        assert self.rule_idx < len(pass_.rules), f"rule_idx={self.rule_idx} out of range for pass {pass_.name!r} ({len(pass_.rules)} rules)"
        return pass_.rules[self.rule_idx]

    def advance(self, graph: Graph) -> None:
        """Move past the just-finished rule batch. Wraps to the next
        pass (logging done + flushing ``pipeline.dump.on_pass``) when
        the scan completes with no functional rewrites; otherwise
        restarts the scan from rule 0 to apply newly-spawned matches.
        ``graph`` is the candidate's current graph — passed in so the
        on-pass dump and node-count debug line have something to
        report. Raises if the cursor is already done."""
        pass_ = self.current_pass  # asserts not is_done
        self.rule_idx += 1
        if self.rule_idx < len(pass_.rules):
            return
        finished = self.n_applied == 0
        self.rule_idx = 0
        self.n_applied = 0
        if finished:
            if pass_.name:
                logger.debug("compile: %-18s done (%d nodes)", pass_.name, len(graph.nodes))
                if self.pipeline.dump is not None:
                    self.pipeline.dump.on_pass(pass_, graph)
            self.pass_idx += 1


@dataclass(frozen=True)
class Pipeline:
    """Frozen per-run layout of the rewrite pipeline.

    :meth:`match` is the only entry point for pattern matching: it
    walks the graph for one rule and stamps the rule onto every Match.
    Tests / standalone callers that just want pattern matching can
    build a one-rule Pipeline via :meth:`from_pattern`.

    ``dump`` is the optional artifact collector — when set,
    :class:`Candidate` routes per-rule diffs through ``dump.on_rule``
    inside :meth:`Candidate._log_apply` and :class:`Cursor` routes
    post-pass graph dumps through ``dump.on_pass`` inside
    :meth:`Cursor.advance`. Living on Pipeline lets both read it off
    one shared reference (reached via ``match.pipeline.dump`` and
    ``cursor.pipeline.dump``) instead of threading it through every
    helper.
    """

    passes: list[Pass]
    dump: CompilerDump | None = None

    def match(self, graph: Graph, rule: Rule) -> list[Match]:
        """Enumerate every live pattern match for ``rule`` against
        ``graph``. Stamps ``is_last=True`` on the last surviving match
        so the rewriter knows which apply closes out the rule batch
        (cursor advance flows through ``Candidate.try_rewrite`` /
        ``Candidate.apply``). Drops matches that fail
        :meth:`Match.is_alive` — an earlier match in the same batch
        may have removed a consumed node."""
        results: list[Match] = []
        for nid in graph.topological_order():
            m = _match_at(graph, nid, self, rule)
            if m is not None and m.is_alive():
                results.append(m)
        if results:
            results[-1].is_last = True
        return results

    @classmethod
    def from_pattern(cls, pattern: list[Pattern]) -> Pipeline:
        """Test/standalone helper: build a single-pass, single-rule
        Pipeline whose only rule wraps ``pattern`` (no ``rewrite``).
        Lets pattern-matching tests drive :meth:`match` without
        setting up the full engine pipeline."""
        rule = Rule(name="__test__", pattern=pattern)
        return cls(passes=[Pass(name="__test__", rules=[rule], index=0)])

    @classmethod
    def build(cls, passes: list[str], *, select: Iterable[str] | None = None, dump: CompilerDump | None = None) -> Pipeline:
        """Load each named pass directory into a :class:`Pass` and
        assemble them into a Pipeline. ``select``, when given, filters
        rules whose stem (with or without numeric prefix) appears in
        the set."""
        select_set = set(select) if select is not None else None
        return cls(passes=[Pass.load(name, i, select_set) for i, name in enumerate(passes)], dump=dump)

    def search(self, search: Search, ctx: Context | None, *, db: SearchDB | None = None) -> Iterator[Candidate]:
        """The unified search-driven driver. Each iteration: pop a
        candidate, run one rule's batch of matches against its graph,
        push successor(s). Yields when a candidate reaches the end of
        the pipeline (``cursor.pass_idx >= len(self.passes)``).

        Per-rule batch semantics: enumerate matches via :meth:`match`
        (which stamps ``rule`` on each Match plus ``is_last`` on the
        last live one), call ``Candidate.try_rewrite`` for each.
        Single-option matches apply inline; the first multi-option
        match spawns one ``LazyCandidate`` per option and breaks the
        loop. Cursor advance for the rule batch is owned by
        :meth:`Cursor.advance`, fired from ``Candidate.apply`` on
        ``match.is_last`` (and directly here for batches that produced
        no live matches) even when the rewrite skipped or yielded no
        valid options."""
        from deplodock.compiler.pipeline.search.candidate import LazyCandidate  # noqa: PLC0415

        while (popped := search.pop()) is not None:
            # Thunk-bearing fork: expand before resolving. Each expansion
            # spawns the next level of ``LazyCandidate``s (more thunks or
            # concrete options) sharing the same ``inner`` and ``match`` —
            # cursor advance is deferred until a leaf actually resolves.
            if popped.is_expandable():
                children = popped.expand()
                best = _best_fork(children, popped.inner, db) if db is not None else None
                search.push(children[0], *children[1:], best=best)
                continue
            cand = popped.resolve()
            cur = cand.cursor
            if cur.is_done:
                yield cand
                continue
            pass_ = cur.current_pass
            # Empty pass (e.g. all rules filtered out) OR no live
            # matches → no apply fires → advance the cursor directly
            # so the search loop doesn't re-pop the same rule batch
            # forever. ``advance`` handles both cases uniformly: with
            # ``n_applied == 0`` it wraps to the next pass and fires
            # the post-pass log + dump.
            if not pass_.rules:
                cur.advance(cand.graph)
                search.push(cand.lazy())
                continue
            matches = self.match(cand.graph, cur.current_rule)
            if not matches:
                cur.advance(cand.graph)
                search.push(cand.lazy())
                continue

            forks: list[LazyCandidate] | None = None
            for match in matches:
                options = cand.try_rewrite(match)
                if options is None:
                    continue
                # Multi-option fork point: spawn one ``LazyCandidate`` per
                # option (option-0 included as the primary). Each shares
                # ``cand`` as ``inner`` so siblings don't duplicate the
                # snapshot. ``from_option`` lifts concrete ``Op``/``Graph``
                # options into leaf Forks so every LazyCandidate's pending
                # carries a uniform Fork shape. The fork's apply on
                # resolve advances the cursor when ``match.is_last`` (the
                # cursor advance the eager loop didn't do here, since we
                # deferred to forks).
                forks = [LazyCandidate.from_option(inner=cand, cursor=replace(cur), match=match, option=opt) for opt in options]
                break

            if forks is not None:
                best = _best_fork(forks, cand, db) if db is not None else None
                search.push(forks[0], *forks[1:], best=best)
                continue

            search.push(cand.lazy())

    def run(self, graph: Graph, *, ctx: Context | None = None, backend=None, db: SearchDB | None = None) -> Graph:
        """Single-shot greedy compile — run each pass in order, picking
        option 0 at every fork point. Convenience wrapper around
        :meth:`tune` that yields the first terminal.

        ``ctx`` is built once (probing the live device if not provided)
        and passed to every rule that takes a ``ctx`` parameter.

        ``backend`` (typically :class:`CudaBackend`) opts the run into
        real GPU measurement: every terminal graph's per-kernel latency
        is recorded to ``db`` and attributed to every ancestor along
        the ``Op.source`` chain. ``db`` defaults to a fresh in-memory
        store; pass an explicit :class:`SearchDB` to persist
        measurements across runs.

        For exhaustive autotuning, call :meth:`tune` directly with
        :class:`TuningSearch` and iterate every yielded candidate."""
        from deplodock.compiler.pipeline.search.policy import GreedySearch  # noqa: PLC0415

        return next(self.tune(graph, search=GreedySearch(db=db), ctx=ctx, backend=backend, db=db)).graph

    def tune(
        self,
        graph: Graph,
        *,
        search: Search,
        ctx: Context | None = None,
        backend=None,
        db: SearchDB | None = None,
    ) -> Iterator[Candidate]:
        """Drive the autotune search. Yields one terminal ``Candidate``
        per fully-explored branch. With deterministic rules (no
        list-returning rewrites) the search yields exactly one — same
        shape as :meth:`run`.

        ``search`` chooses both the order and the stopping condition:
        :class:`GreedySearch` for single-shot compiles (stops at the
        first terminal); :class:`TuningSearch` for ``deplodock tune`` (runs the
        queue dry, exploring every fork).

        When ``search`` exposes a ``tree: SearchTree``
        (:class:`TuningSearch` does), each yielded terminal candidate
        has its ``CudaOp`` nodes recorded to ``db`` and the tree via
        :func:`record_terminal` before being yielded — so subsequent
        candidates see the updated priority signal. Pass a ``Backend``
        (typically :class:`CudaBackend`) via ``backend=`` to record
        real GPU-event latencies; omit it to skip persistence entirely
        (terminals yield a stub ``latency_us=1.0`` but nothing is
        written to ``db``).

        ``ctx`` is built once (probing the live device if not provided)
        and shared by every candidate."""
        from deplodock.compiler import provenance  # noqa: PLC0415
        from deplodock.compiler.context import Context as _Context  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.candidate import Candidate as _Candidate  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.db import SearchDB as _SearchDB  # noqa: PLC0415

        # Seed op provenance on the input graph before any pass runs — the one
        # universal entry both ``run`` and ``tune`` funnel through. Idempotent,
        # so a graph reloaded mid-pipeline keeps whatever prov it carried.
        provenance.seed(graph)

        if ctx is None:
            ctx = _Context.probe()
        backend_name = getattr(backend, "name", "cuda")
        if ctx.backend_name != backend_name:
            ctx = replace(ctx, backend_name=backend_name)
        t_start = time.monotonic()

        search.push(_Candidate(ctx=ctx, graph=graph, cursor=Cursor(pipeline=self)).lazy())

        if db is None:
            db = _SearchDB()
        n_terminals = 0
        for cand in self.search(search, ctx, db=db):
            n_terminals += 1
            if backend is not None:
                knob_strs = [
                    s
                    for nid in cand.graph.topological_order()
                    if (k := getattr(cand.graph.nodes[nid].op, "knobs", None)) and (s := format_tuning_knobs(k)) != "-"
                ]
                label = " | ".join(knob_strs) if knob_strs else "option-0"
                logger.info("[tune] variant #%d  [%s]", n_terminals, label)
            stats, status = _bench_terminal(cand, backend=backend, db=db)
            search.observe(stats, status)
            yield cand
        logger.info("compile: total %.2fs (%d terminal(s))", time.monotonic() - t_start, n_terminals)


def _match_at(graph: Graph, start: str, pipeline: Pipeline, rule: Rule) -> Match | None:
    nid: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
    matched_nodes: list[Node] = []
    for prod in rule.pattern:
        if nid is None:
            return None
        node = graph.nodes.get(nid)
        if node is None or not isinstance(node.op, prod.op_type):
            return None
        if not all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items()):
            return None
        nodes[prod.name] = nid
        consumed.add(nid)
        identities[nid] = id(node)
        matched_nodes.append(node)
        consumers = graph.consumers(nid)
        nid = consumers[0] if len(consumers) == 1 else None
    # Snap ``op.inputs`` / ``op.outputs`` to the surrounding graph for
    # every matched node so rule rewrites can read per-buffer Tensors
    # straight off the op without re-querying the graph.
    for node in matched_nodes:
        node.op.populate_io(graph, node)
    return Match(
        graph=graph,
        root_node_id=start,
        pipeline=pipeline,
        rule=rule,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


# ---------------------------------------------------------------------------
# Bench + DB persistence for autotune terminals (used by Pipeline.tune)
# ---------------------------------------------------------------------------


def _best_fork(forks, parent_cand, db):
    """Pick the fork whose pending option matches the lowering DB's
    best-known child for the parent op. Returns ``None`` when no row
    exists (untuned site), the parent op has no cache key, or no fork
    matches the recorded child key (variant drifted).

    The comparison is on ``op_cache_key`` of the post-apply option:
    :meth:`Candidate.apply` merges ``parent.knobs`` into ``option.knobs``
    on resolve so the source-chain child carries the cumulative knob
    set, and the recorded ``child_key`` was hashed against that merged
    set. Pre-apply (here) the option only has the rule's local delta,
    so we replicate the merge before hashing — otherwise the cache key
    of a freshly-emitted option never matches what tune recorded for
    the same fork."""
    from dataclasses import replace  # noqa: PLC0415

    from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

    first = forks[0]
    if first.pending is None:
        return None
    match, _ = first.pending
    parent_node = parent_cand.graph.nodes.get(match.root_node_id)
    if parent_node is None:
        return None
    parent_key = op_cache_key(parent_node.op)
    if parent_key is None:
        return None
    row = db.lookup_lowering(parent_key)
    if row is None:
        return None
    parent_knobs = getattr(parent_node.op, "knobs", None) or {}
    for f in forks:
        if f.pending is None:
            continue
        _, fork = f.pending  # always a Fork after the constructor unification
        opt_knobs = fork.knobs
        if fork.is_leaf:
            # Leaf Fork wrapping a concrete Op/Graph — match by structural
            # cache key. Fire the (trivial) thunk to retrieve the wrapped
            # leaf, then compare against the DB's child_key. The thunk
            # for a leaf is ``lambda: [op]``, constant-time.
            leaves = fork.expand()
            if not leaves:
                continue
            leaf = leaves[0]
            merged = {**parent_knobs, **opt_knobs}
            probe = replace(leaf, knobs=merged) if merged != opt_knobs else leaf
            if op_cache_key(probe) == row.child_key:
                return f
        else:
            # Branch Fork — match by knobs-delta against the recorded
            # row's knobs. The fork "wins" iff every knob it pins appears
            # in ``row.knobs`` with the same value AND at least one of
            # those knobs is constrained by the row; otherwise every
            # sibling is equally consistent and we fall back to option-0.
            # No rule currently emits branch Forks (the partition planner's
            # hierarchical conversion was reverted as it degraded MCTS
            # efficiency on small-to-medium shapes); kept here as
            # documented infrastructure for future hierarchical rules.
            if not opt_knobs:
                continue
            constraining = [k for k in opt_knobs if k in row.knobs]
            if constraining and all(row.knobs.get(k) == v for k, v in opt_knobs.items()):
                return f
    return None


def _bench_terminal(cand, *, backend, db):
    """Bench every ``CudaOp`` in ``cand.graph``, persist per-kernel
    ``perf`` / inventory / lowering rows, and return ``(stats, status)``
    where ``stats`` is the per-kernel ``PerfStats`` summed across the
    graph (total terminal latency)."""
    import json as _json  # noqa: PLC0415
    import statistics as _statistics  # noqa: PLC0415

    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.db import PerfStats  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.keys import (  # noqa: PLC0415
        _is_kernel_bearing,
        dialect_of,
        op_cache_key,
        source_chain,
    )

    def _point_stats(us: float) -> PerfStats:
        return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)

    def _stats_from_launch(lt) -> PerfStats:
        if lt.samples and len(lt.samples) >= 1:
            us = [s * 1000.0 for s in lt.samples]
            return PerfStats(
                median=_statistics.median(us),
                min=min(us),
                max=max(us),
                mean=_statistics.fmean(us),
                variance=_statistics.pvariance(us) if len(us) > 1 else 0.0,
                n_samples=len(us),
            )
        return _point_stats(lt.time_ms * 1000.0)

    def _body_json(op, dialect: str) -> str:
        return _json.dumps(
            {
                "dialect": dialect,
                "name": getattr(op, "name", None) or getattr(op, "kernel_name", None) or "?",
                "body_repr": repr(op.body),
            },
            default=str,
        )

    def _record_op_inventory(op) -> None:
        from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
        from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

        key = op_cache_key(op)
        if key is None:
            return
        if isinstance(op, CudaOp):
            db.record_cuda_op(
                key,
                kernel_source=op.kernel_source,
                arg_order=list(op.arg_order),
                grid=list(op.grid),
                block=list(op.block),
                smem_bytes=op.smem_bytes,
                pretty=op.kernel_source,
            )
        elif isinstance(op, KernelOp):
            db.record_kernel_op(key, _body_json(op, "kernel"), op.pretty_body())
        elif isinstance(op, TileOp):
            db.record_tile_op(key, _body_json(op, "tile"), op.pretty_body())
        elif isinstance(op, LoopOp):
            db.record_loop_op(key, _body_json(op, "loop"), op.pretty_body())

    def _persist(cuda_op, *, stats: PerfStats, status: str, backend_name: str) -> None:
        cuda_key = op_cache_key(cuda_op)
        if cuda_key is None:
            return
        chain = [op for op in source_chain(cuda_op) if _is_kernel_bearing(op)]
        for op in chain:
            _record_op_inventory(op)
        for parent_op, child_op in zip(chain[1:], chain[:-1], strict=False):
            p_dialect = dialect_of(parent_op)
            c_dialect = dialect_of(child_op)
            if p_dialect is None or c_dialect is None:
                continue
            p_key = op_cache_key(parent_op)
            c_key = op_cache_key(child_op)
            if p_key is None or c_key is None:
                continue
            p_knobs = getattr(parent_op, "knobs", None) or {}
            c_knobs = getattr(child_op, "knobs", None) or {}
            knobs_delta = {k: v for k, v in c_knobs.items() if p_knobs.get(k) != v}
            db.record_lowering(
                p_key,
                p_dialect,
                c_key,
                c_dialect,
                knobs=knobs_delta,
                measured_median_us=stats.median if status == "ok" else None,
            )
        knobs = getattr(cuda_op, "knobs", None) or {}
        db.record_perf(context_key, cuda_key, backend=backend_name, status=status, stats=stats, knobs=knobs)
        logger.info("[tune]   %s @ %.2f us  (%s)", getattr(cuda_op, "kernel_name", "?"), stats.median, status)

    def _accumulate(acc: PerfStats | None, s: PerfStats) -> PerfStats:
        if acc is None:
            return s
        return PerfStats(
            median=acc.median + s.median,
            min=acc.min + s.min,
            max=acc.max + s.max,
            mean=acc.mean + s.mean,
            variance=acc.variance + s.variance,
            n_samples=min(acc.n_samples, s.n_samples) if acc.n_samples and s.n_samples else (acc.n_samples or s.n_samples),
        )

    graph = cand.graph
    context_key = cand.ctx.structural_key()
    cuda_nodes = [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]
    if not cuda_nodes:
        return _point_stats(0.0), "ok"

    backend_name = getattr(backend, "name", "stub")

    # Cache lookup: if every CudaOp already has a perf row for this
    # (context, backend), skip the benchmark entirely and rebuild the
    # aggregate stats from the DB. Per-kernel partial caching isn't
    # useful here because ``backend.benchmark`` runs the whole graph.
    cached_rows = []
    for node in cuda_nodes:
        key = op_cache_key(node.op)
        row = db.lookup_perf(context_key, key, backend=backend_name) if key is not None else None
        if row is None:
            cached_rows = None
            break
        cached_rows.append(row)
    if cached_rows is not None:
        logger.info("[tune] cache hit for %d kernel(s) — skipping bench", len(cuda_nodes))
        agg: PerfStats | None = None
        status = "ok"
        for row in cached_rows:
            if row.status != "ok":
                status = row.status
            agg = _accumulate(agg, row.stats)
            logger.info("[tune]   %s @ %.2f us  (%s, cached)", row.op_key[:12], row.stats.median, row.status)
        return agg or _point_stats(0.0), status

    status = "ok"
    agg = None

    if backend is None:
        # No real measurement → do NOT persist. Writing the 1.0us stub
        # to a shared DB used to clobber tuned ``best_median_us`` values
        # (record_lowering / record_perf keep the minimum), so any plain
        # ``deplodock run`` (which routes through ``Pipeline.run`` without
        # a backend) was overwriting real autotune rows with 1.0us stubs.
        # Tests that need lowering edges in stub mode should pass an
        # explicit stub backend.
        for _node in cuda_nodes:
            s = _point_stats(1.0)
            agg = _accumulate(agg, s)
    else:
        logger.info("[tune] benching %d kernel(s) in graph", len(cuda_nodes))
        try:
            result = backend.benchmark(graph, num_iters="auto")
        except Exception as exc:  # noqa: BLE001
            fail_us = float(backend.bench_run_timeout_s) * 1_000_000.0
            logger.warning(
                "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d kernel(s)",
                exc,
                fail_us,
                len(cuda_nodes),
            )
            s = _point_stats(fail_us)
            for node in cuda_nodes:
                _persist(node.op, stats=s, status="bench_fail", backend_name=backend_name)
                agg = _accumulate(agg, s)
            status = "bench_fail"
        else:
            per_launch = result.per_launch or []
            if len(per_launch) != len(cuda_nodes):
                logger.warning(
                    "[tune] per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
                    len(per_launch),
                    len(cuda_nodes),
                )
                avg_us = (result.time_ms * 1000.0) / max(len(cuda_nodes), 1)
                s = _point_stats(avg_us)
                for node in cuda_nodes:
                    _persist(node.op, stats=s, status="ok", backend_name=backend_name)
                    agg = _accumulate(agg, s)
            else:
                for node, lt in zip(cuda_nodes, per_launch, strict=True):
                    s = _stats_from_launch(lt)
                    _persist(node.op, stats=s, status="ok", backend_name=backend_name)
                    agg = _accumulate(agg, s)
            try:
                import cupy as _cp  # noqa: PLC0415

                _cp.cuda.runtime.deviceSynchronize()
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    return agg or _point_stats(0.0), status


__all__ = ["Match", "Pass", "Pattern", "Pipeline", "Rule", "RuleSkipped"]
