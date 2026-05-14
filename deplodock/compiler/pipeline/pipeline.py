"""Pipeline value types: ``Pattern``, ``Match``, ``Rule``,
``RuleSkipped``, ``Pass``, ``Pipeline``.

Bundled together because they form a tight chain — ``Pattern`` defines
what a rule matches, ``Rule`` carries the pattern + rewrite,
``Pass`` groups rules, ``Pipeline`` holds passes and drives matching,
and ``Match`` carries ``Rule`` (which backref-resolves to ``Pass``) so
the driver can read pass / rule metadata off one field. Splitting them
across modules forced ``TYPE_CHECKING`` shuffles for backrefs without
buying anything; one module keeps the cycle visible.

The driver (rule loader, rewrite dispatcher, search loop, public
``run_pipeline`` / ``run_autotune``) lives in :mod:`.driver`.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Node

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.candidate import Candidate
    from deplodock.compiler.pipeline.search.db import SearchDB
    from deplodock.compiler.pipeline.search.policy import Search

logger = logging.getLogger(__name__)


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
    pipeline: Pipeline | None = None
    rule: Rule | None = None
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None
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


@dataclass(frozen=True)
class Pipeline:
    """Frozen per-run layout of the rewrite pipeline.

    :meth:`match` is the only entry point for pattern matching: it
    walks the graph for one rule and stamps the rule onto every Match.
    Tests / standalone callers that just want pattern matching can
    build a one-rule Pipeline via :meth:`from_pattern`.

    ``dump`` is the optional artifact collector — when set, the driver
    routes per-rule diffs and post-pass graph dumps through it. Living
    on Pipeline lets the driver read it off one shared reference
    instead of threading it through every helper.
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

    def search(self, search: Search, ctx: Context | None) -> Iterator[Candidate]:
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
        ``Candidate.apply`` / ``_advance_batch``, fired on
        ``match.is_last`` even when the rewrite skipped or yielded no
        valid options."""
        from deplodock.compiler.pipeline.search.candidate import LazyCandidate  # noqa: PLC0415

        while (popped := search.pop()) is not None:
            cand = popped.resolve()
            cur = cand.cursor
            if cur.pass_idx >= len(self.passes):
                yield cand
                continue
            pass_ = self.passes[cur.pass_idx]
            # Empty pass (e.g. all rules filtered out): nothing to do, skip.
            if not pass_.rules:
                cur.pass_idx += 1
                search.push(cand.lazy())
                continue
            rule = pass_.rules[cur.rule_idx]
            matches = self.match(cand.graph, rule)
            if not matches:
                # No live matches → no apply will fire → advance the cursor
                # ourselves with a synthetic last-match sentinel so the
                # transition still flows through ``Candidate.apply`` /
                # ``_advance_batch``.
                sentinel = Match(graph=cand.graph, root_node_id="", pipeline=self, rule=rule, is_last=True)
                cand._advance_if_last(sentinel)
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
                # snapshot. The fork's apply on resolve advances the
                # cursor when ``match.is_last`` (the cursor advance the
                # eager loop didn't do here, since we deferred to forks).
                forks = [LazyCandidate(inner=cand, cursor=dc_replace(cur), chain=[(match, opt)]) for opt in options]
                break

            if forks is not None:
                search.push(forks[0], *forks[1:])
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
        first terminal); :class:`TuningSearch` for ``--tune`` (runs the
        queue dry, exploring every fork).

        When ``search`` exposes a ``tree: SearchTree``
        (:class:`TuningSearch` does), each yielded terminal candidate
        has its ``CudaOp`` nodes recorded to ``db`` and the tree via
        :func:`record_terminal` before being yielded — so subsequent
        candidates see the updated priority signal. Pass a ``Backend``
        (typically :class:`CudaBackend`) via ``backend=`` to record
        real GPU-event latencies; omit it to record the stub
        ``latency_us=1.0``.

        ``ctx`` is built once (probing the live device if not provided)
        and shared by every candidate."""
        from deplodock.compiler.context import Context as _Context  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.candidate import Candidate as _Candidate  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.db import SearchDB as _SearchDB  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.recorder import TuneAborted, record_terminal  # noqa: PLC0415

        if ctx is None:
            ctx = _Context.probe()
        backend_name = getattr(backend, "name", "cuda")
        if ctx.backend_name != backend_name:
            ctx = dc_replace(ctx, backend_name=backend_name)
        t_start = time.monotonic()

        search.push(_Candidate(ctx=ctx, graph=graph).lazy())

        tree = getattr(search, "tree", None)
        if db is None:
            db = _SearchDB()
        n_terminals = 0
        for cand in self.search(search, ctx):
            n_terminals += 1
            if backend is not None:
                knob_strs: list[str] = []
                for nid in cand.graph.topological_order():
                    op = cand.graph.nodes[nid].op
                    k = getattr(op, "knobs", None) or {}
                    if k:
                        knob_strs.append(", ".join(f"{kk}={vv}" for kk, vv in sorted(k.items())))
                label = " | ".join(knob_strs) if knob_strs else "option-0"
                logger.info("[tune] variant #%d  [%s]", n_terminals, label)
            try:
                record_terminal(cand.graph, db, tree, cand.ctx.structural_key(), backend=backend)
            except TuneAborted as exc:
                # A bench failure left GPU work queued; running another
                # variant would block in cupy's ``_allocate``. Yield this
                # terminal (its measurements are already recorded as
                # bench_fail) and stop the sweep so the caller can pick a
                # winner from whatever ok variants we've got.
                logger.warning("[tune] %s — stopping after %d terminal(s)", exc, n_terminals)
                yield cand
                break
            yield cand
        logger.info("compile: total %.2fs (%d terminal(s))", time.monotonic() - t_start, n_terminals)


def _match_at(graph: Graph, start: str, pipeline: Pipeline, rule: Rule) -> Match | None:
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
    for prod in rule.pattern:
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
    return Match(
        graph=graph,
        root_node_id=start,
        pipeline=pipeline,
        rule=rule,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


def _check_constraints(node: Node, prod: Pattern) -> bool:
    return all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items())


def _sole_consumer(graph: Graph, nid: str) -> str | None:
    consumers = graph.consumers(nid)
    return consumers[0] if len(consumers) == 1 else None


__all__ = ["Match", "Pass", "Pattern", "Pipeline", "Rule", "RuleSkipped"]
