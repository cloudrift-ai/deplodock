"""Pipeline value types and compile driver: ``Pattern``, ``Match``,
``Rule``, ``RuleSkipped``, ``Pass``, ``Cursor``, ``Pipeline``, ``Run``.

Bundled together because they form a tight chain — ``Pattern`` defines
what a rule matches, ``Rule`` carries the pattern + rewrite, ``Pass``
groups rules, ``Pipeline`` is the frozen pass layout + matcher, ``Run``
owns ONE drive of that layout (ctx / search / db / backend / dump /
rejections + the engine loop), ``Cursor`` tracks per-candidate resume
state inside a Run, and ``Match`` carries ``Rule`` (which backref-resolves
to ``Pass``). ``Run`` exposes two entry points over one shared rule-batch
body (``Run._step``): ``drive`` (exploration — a ``Search`` policy ranks
the fork frontier) and ``resolve`` (deterministic resolution — a
``decide`` callback picks at each ``ForkPoint`` and the fold returns the
terminal graph plus a ``Decision`` trace).

``Pipeline`` also owns the compile entry points — :meth:`build`,
:meth:`run`, :meth:`tune` (each constructs a :class:`Run` and drives it).
The per-rule logging, rewrite-kwarg dispatch, and snapshot rendering live
on :class:`Candidate` (see :mod:`..search.candidate`).
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
from deplodock.compiler.pipeline.fork import Fork, OptionFork
from deplodock.compiler.pipeline.knob import Knob, apply_off_defaults, format_tuning_knobs

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.candidate import Candidate
    from deplodock.compiler.pipeline.search.db import SearchDB
    from deplodock.compiler.pipeline.search.policy import Search

logger = logging.getLogger("deplodock.compiler.pipeline")

# Greedy compile validity-fallback cap: how many times ``Pipeline.run``
# re-resolves blocklisting a tile that failed ``validate(ctx)``. Each retry
# blocks ≥1 fresh tile or stops, so this only bounds pathological cases (every
# sibling unviable).
_MAX_GREEDY_RETRIES = 8


_PASSES_DIR = Path(__file__).resolve().parent / "passes"
_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


def variant_label(graph: Graph) -> str:
    """A human label for one tuned variant: the ``|``-joined per-op tuning
    knobs across the graph (``tile=128 | warps=4``), or ``"option-0"`` when no
    op carries knobs. Shared by :meth:`Pipeline.tune`'s per-variant log line and
    the ``tune`` progress bar so both render the same knob string."""
    knob_strs = [
        s
        for nid in graph.topological_order()
        if (k := getattr(graph.nodes[nid].op, "knobs", None)) and (s := format_tuning_knobs(k)) != "-"
    ]
    return " | ".join(knob_strs) if knob_strs else "option-0"


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


class LoweringError(Exception):
    """Raised by :meth:`Pipeline.run` when a deterministic (greedy)
    compile finishes with a node left un-lowered because every option of
    its only lowering rule failed ``validate(ctx)`` — e.g. a tile shape
    whose materialized kernel exceeds the device smem cap.

    This converts the old silent leak (an un-lowered ``TileOp`` surviving
    every pass until ``CudaBackend`` raises the cryptic ``non-CudaOp``
    ``TypeError``) into an actionable, early error that names the node,
    the pass that declined it, and the ``validate`` reason. The
    fork-pruning path under ``tune`` is unaffected: there the dropped
    branch is a legitimate dead end and sibling branches carry other
    shapes, so no sink is installed and nothing is raised."""


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
    # Every ``Knob`` declared (or imported) by this pass's rule modules — the
    # knobs this pass "owns". Populated by :meth:`load` (scans each rule
    # module's ``vars()`` for ``Knob`` instances, so imported knobs like the
    # planner's ``_enumeration`` set count too). :meth:`Cursor.advance` stamps
    # any of these with a defined ``off`` onto the variant at the pass boundary.
    declared_knobs: tuple[Knob, ...] = ()

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
        declared: dict[str, Knob] = {}  # knob name → Knob, deduped across rule modules
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
            # Collect the knobs this rule module declares OR imports (e.g. the
            # planner imports the ``_enumeration`` tier knobs) — ``Cursor.advance``
            # uses them to OFF-fill the pass's variants.
            for v in vars(module).values():
                if isinstance(v, Knob):
                    declared.setdefault(v.name, v)
        return cls(name=name, rules=rules, index=index, declared_knobs=tuple(declared.values()))


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

    ``rule`` locates this match in the pipeline; the rewriter reaches
    pass metadata via ``match.rule.pass_``. Stamped by
    :meth:`Pipeline.match` at construction time. Run-scoped sinks (dump,
    rejections) live on the :class:`Run` — reached through the candidate,
    not the match. ``is_last`` is stamped on the last live match returned
    by :meth:`Pipeline.match` so ``Candidate.try_rewrite`` knows when to
    advance the cursor.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
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
            rule=self.rule,
            nodes=dict(self.nodes),
            consumed=set(self.consumed),
            output=self.output,
            is_last=self.is_last,
            _identities=identities,
        )


def _off_fill_pass(graph: Graph, pass_: Pass) -> None:
    """Stamp every OFF-declared knob of ``pass_`` that a variant left unspecified
    onto that variant — the "every emitted variant carries an explicit value for
    every knob the pass declares" rule, realized once at the pass boundary (so
    all the pass's rules — including a declined / no-variant rule — have had
    their turn). Rebuilds the op via :func:`dataclasses.replace` (a fresh
    ``knobs`` dict, never an in-place mutation) so a structurally shared op isn't
    corrupted across sibling candidates. Only ops that already carry tuning
    knobs (a realized kernel variant) are touched — inputs / constants with an
    empty ``knobs`` are left alone."""
    if not pass_.declared_knobs:
        return
    for node in graph.nodes.values():
        knobs = getattr(node.op, "knobs", None)
        if not knobs:
            continue
        filled = apply_off_defaults(dict(knobs), pass_.declared_knobs)
        if filled != knobs:
            node.op = replace(node.op, knobs=filled)


@dataclass
class Cursor:
    """Pipeline resume state for a candidate. Owns the entire advance
    logic: ``advance(graph)`` moves past the current rule batch,
    wrapping to the next pass — logging "compile: <pass> done" and
    flushing ``run.dump.on_pass`` — when the scan completes with
    no functional rewrites.

    * ``run`` — the :class:`Run` being driven; resolves the pipeline
      (current pass / rule by index) and the per-run dump sink.
    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass.
    """

    run: Run
    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    @property
    def is_done(self) -> bool:
        return self.pass_idx >= len(self.run.pipeline.passes)

    @property
    def current_pass(self) -> Pass:
        assert not self.is_done, f"cursor is done (pass_idx={self.pass_idx} >= {len(self.run.pipeline.passes)})"
        return self.run.pipeline.passes[self.pass_idx]

    @property
    def current_rule(self) -> Rule:
        pass_ = self.current_pass
        assert self.rule_idx < len(pass_.rules), f"rule_idx={self.rule_idx} out of range for pass {pass_.name!r} ({len(pass_.rules)} rules)"
        return pass_.rules[self.rule_idx]

    def advance(self, graph: Graph) -> None:
        """Move past the just-finished rule batch. Wraps to the next
        pass (logging done + flushing ``run.dump.on_pass``) when
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
                _off_fill_pass(graph, pass_)
                logger.debug("compile: %-18s done (%d nodes)", pass_.name, len(graph.nodes))
                if self.run.dump is not None:
                    self.run.dump.on_pass(pass_, graph)
            self.pass_idx += 1


@dataclass(frozen=True)
class Pipeline:
    """Frozen, shareable pass layout of the rewrite pipeline — nothing
    run-scoped lives here (that's :class:`Run`), so one Pipeline can
    drive any number of concurrent runs.

    :meth:`match` is the only entry point for pattern matching: it
    walks the graph for one rule and stamps the rule onto every Match.
    Tests / standalone callers that just want pattern matching can
    build a one-rule Pipeline via :meth:`from_pattern`.
    """

    passes: list[Pass]

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
            m = _match_at(graph, nid, rule)
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
    def build(cls, passes: list[str], *, select: Iterable[str] | None = None) -> Pipeline:
        """Load each named pass directory into a :class:`Pass` and
        assemble them into a Pipeline. ``select``, when given, filters
        rules whose stem (with or without numeric prefix) appears in
        the set."""
        select_set = set(select) if select is not None else None
        return cls(passes=[Pass.load(name, i, select_set) for i, name in enumerate(passes)])

    def run(
        self,
        graph: Graph,
        *,
        ctx: Context | None = None,
        backend=None,
        db: SearchDB | None = None,
        dump: CompilerDump | None = None,
    ) -> Graph:
        """Single-shot greedy compile — a deterministic resolution
        (:meth:`Run.resolve`) with the greedy pick
        (:func:`~deplodock.compiler.pipeline.search.policy.greedy.greedy_decide`):
        at every fork point, flatten to complete leaves and take the
        ``Prior``'s ``mean_scores`` argmin. Not a search — no frontier,
        no tree, no benching (it can only *use* a prior trained earlier
        by ``tune``, never train one); exploration (PUCT) stays in
        :meth:`tune`. The input ``graph`` is copied once per attempt and
        resolved in place — no per-fork graph copies.

        ``ctx`` is built once (probing the live device if not provided)
        and passed to every rule that takes a ``ctx`` parameter.

        ``backend`` (typically :class:`CudaBackend`) opts the run into
        real GPU measurement: the terminal graph's per-kernel latency is
        recorded to ``db`` (via :func:`_bench_terminal`, once after the
        resolution settles) and attributed to every ancestor along the
        ``Op.source`` chain. ``db`` defaults to a fresh in-memory store;
        pass an explicit :class:`SearchDB` to persist measurements
        across runs.

        Retries are ``decide`` wrappers over a deterministic re-resolve
        — cheap non-chronological backtracking with no graph snapshots
        or undo log, since every other choice replays identically:

        * **Validity fallback** — the prior ranks by predicted latency
          and can rank a tile that fails ``validate(ctx)`` (smem /
          thread budget) first; ``tune`` benches-and-skips it, but
          greedy benches nothing, so on a left-un-lowered node we
          blocklist its tile and re-resolve, falling back to the next
          prior-ranked leaf. Bounded retries (each adds ≥1 block or
          stops).
        * **Structural retirement** — a resolution that took a
          *structural* pick (a prior-priced kernel-set change; the trace
          contains a ``Graph`` decision) gets one coarser fallback
          first: any lowering failure retires structural picks wholesale
          (``price_structural=False``) and re-resolves down the
          keep-fused branch, since a fragment kernel's failure can't be
          blocklisted at the fork site (the splice minted fresh node
          ids).

        Installs a per-run ``rejections`` sink (on the :class:`Run`) so
        :func:`Candidate.try_rewrite` records any rewrite whose every
        option failed ``validate(ctx)``. After the resolution settles,
        :func:`_raise_on_unlowered` turns a recorded rejection that left
        its node un-lowered into a loud :class:`LoweringError` instead
        of a downstream ``CudaBackend`` mystery."""
        from deplodock.compiler.context import Context as _Context  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.db import SearchDB as _SearchDB  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.policy.greedy import greedy_decide  # noqa: PLC0415

        if ctx is None:
            ctx = _Context.probe()
        backend_name = getattr(backend, "name", "cuda")
        if ctx.backend_name != backend_name:
            ctx = replace(ctx, backend_name=backend_name)
        db = db if db is not None else _SearchDB()
        t_start = time.monotonic()

        blocked: dict[str, set[frozenset]] = {}
        allow_structural = True
        for _attempt in range(_MAX_GREEDY_RETRIES):
            rejections: list[tuple[str, str, str]] = []
            run = Run(pipeline=self, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            decide = greedy_decide(blocked=blocked, price_structural=allow_structural)
            terminal, trace = run.resolve(graph.copy(), decide)
            failed = _unlowered_tiles(terminal, rejections)
            if not failed:
                break
            if allow_structural and any(d.chosen_kind == "graph" for d in trace):
                allow_structural = False
                continue
            new = {nid: ident for nid, ident in failed.items() if ident not in blocked.get(nid, set())}
            if not new:  # no fresh info to retry on → report below
                break
            for nid, ident in new.items():
                blocked.setdefault(nid, set()).add(ident)
        # The prior-ranked tiles all overflowed ``validate(ctx)`` within the
        # retry budget — a *learned* prior can extrapolate a large tile onto a
        # small shape (e.g. one trained on big square matmuls picking an
        # over-smem-cap tile for a tiny projection), and the blocklist retry
        # exhausts before reaching an in-budget leaf. Before giving up, take the
        # conservative emission-order pick (``greedy_decide(prior=None)`` =
        # option-0): it ignores the prior, and the planner emits a budget-safe
        # tile first, so it lowers whenever any in-budget tile exists. When even
        # option-0 overflows (the over-budget rule genuinely has no in-budget
        # option — e.g. the single-option guardrail) the re-resolve stays
        # un-lowered and ``_raise_on_unlowered`` fires below, exactly as before.
        if _unlowered_tiles(terminal, rejections):
            rejections = []
            run = Run(pipeline=self, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            terminal, _ = run.resolve(graph.copy(), greedy_decide(prior=None, price_structural=False))
        _raise_on_unlowered(terminal, rejections, ctx)
        logger.info("compile: total %.2fs (deterministic resolve)", time.monotonic() - t_start)
        return terminal

    def _new_run(self, graph: Graph, *, search, ctx, backend, db, dump, rejections) -> Run:
        """Build the :class:`Run` shared by :meth:`tune_async` (and the deterministic
        :meth:`run`):
        seed provenance, probe / align ``ctx``, and wire the run-scoped sinks."""
        from deplodock.compiler import provenance  # noqa: PLC0415
        from deplodock.compiler.context import Context as _Context  # noqa: PLC0415
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
        return Run(
            pipeline=self,
            ctx=ctx,
            search=search,
            db=db if db is not None else _SearchDB(),
            backend=backend,
            dump=dump,
            rejections=rejections,
        )

    async def tune_async(
        self,
        graph: Graph,
        *,
        search: Search,
        ctx: Context | None = None,
        backend=None,
        db: SearchDB | None = None,
        dump: CompilerDump | None = None,
        rejections: list[tuple[str, str, str]] | None = None,
    ):
        """Async-generator mirror of :meth:`tune` for the multi-GPU tune driver.

        The lowering (``run.drive``) stays a synchronous generator — only the
        per-terminal bench is awaited (:func:`_bench_terminal_async`), so N
        kernels' benches overlap across device-pinned workers on one event loop
        while the (light) Python lowering runs cooperatively between awaits."""
        run = self._new_run(graph, search=search, ctx=ctx, backend=backend, db=db, dump=dump, rejections=rejections)
        t_start = time.monotonic()
        n_terminals = 0
        for token, cand in run.drive(graph):
            n_terminals += 1
            if backend is not None:
                logger.info("[tune] variant #%d  [%s]", n_terminals, variant_label(cand.graph))
            stats, status = await _bench_terminal_async(cand, backend=backend, db=run.db)
            search.observe(token, stats, status, candidate=cand)
            if backend is not None and getattr(search, "last_o3_worthy", False):
                o3_us = await _rebench_o3_async(cand, backend)
                if o3_us is not None:
                    search.observe_o3(token, o3_us)
            yield cand
        dropped = run._dropped_candidates
        logger.info(
            "compile: total %.2fs (%d terminal(s)%s)",
            time.monotonic() - t_start,
            n_terminals,
            f", {dropped} un-lowerable candidate(s) dropped" if dropped else "",
        )


@dataclass
class ForkPoint:
    """What a :meth:`Run.resolve` ``decide`` callback sees at one
    multi-option rewrite: the live :class:`Match`, the raw ``options``
    list exactly as ``Candidate.try_rewrite`` returned it (concrete
    ``Op``/``Graph`` leaves and lazy ``Fork``s — branch Forks included,
    unexpanded), the pre-decision root op, and the run's ``ctx``. No
    ``LazyCandidate`` wrapping: ``resolve`` holds one live graph and
    applies the chosen option in place.

    ``score`` is the decide callback's one output channel besides its
    return value: a decide that ranks options with a prior stamps the
    chosen option's predicted µs here, and ``resolve`` copies it onto the
    fork's :class:`Decision` trace entry (where e.g. the structural
    pricing probe reads a kernel's price off the partition fork)."""

    match: Match
    options: list
    root_op: Op
    ctx: Context
    structural: bool
    score: float | None = None

    @property
    def node_id(self) -> str:
        """The graph node this fork is rewriting — the blocklist / trace key."""
        return self.match.root_node_id


@dataclass(frozen=True)
class Decision:
    """One :meth:`Run.resolve` trace entry — what a deterministic
    resolution decided at one fork point. The trace is the resolution's
    only output channel besides the terminal graph: process facts
    (structural picks taken, per-fork predicted cost) are trace queries,
    never policy-object state.

    * ``rule_name`` / ``node_id`` — where the fork was offered.
    * ``chosen_kind`` — ``"graph"`` for a structural (``Graph``-splicing)
      pick, ``"op"`` for an in-place rebind.
    * ``knob_delta`` — the chosen option's knob identity: a ``Fork``'s
      pinned row, an ``Op``'s own knobs, a ``Graph``'s decision-knob delta
      vs the offer op (:func:`_option_decision`).
    * ``score`` — the decide callback's predicted µs for the pick
      (``None`` when the decide didn't rank, e.g. option-0 fallback).
    * ``n_options`` — raw option count at the fork (a lazy fork tree
      counts as one — its leaves are the decide callback's to expand)."""

    rule_name: str
    node_id: str
    chosen_kind: str
    knob_delta: dict
    score: float | None
    n_options: int


@dataclass
class Run:
    """Mutable per-run state of ONE drive of a pipeline — everything
    scoped to a single compile / tune invocation lives here, so
    :class:`Pipeline` stays a frozen, shareable pass layout and nothing
    run-scoped is ever smuggled onto shared objects.

    * ``pipeline`` — the frozen pass layout being driven.
    * ``ctx`` — the resolved hardware context, shared by every candidate
      (reached as ``cand.ctx``).
    * ``search`` — the policy ordering an exploration (:meth:`drive`);
      ``None`` for a deterministic resolution (:meth:`resolve`), which
      has no frontier to rank.
    * ``db`` — the autotune store ``_bench_terminal`` persists into (the
      training data for the learned prior).
    * ``backend`` — optional measurement backend (``None`` = stub bench,
      no persistence).
    * ``dump`` — optional artifact collector: :meth:`Candidate._log_apply`
      routes per-rule diffs through ``dump.on_rule``, :meth:`Cursor.advance`
      routes post-pass graphs through ``dump.on_pass``.
    * ``rejections`` — optional sink for rewrites whose every option
      failed ``validate(ctx)`` (installed by :meth:`Pipeline.run` so
      greedy compiles can raise :class:`LoweringError`; absent under
      tune, where a pruned fork is a legitimate dead end).

    Candidates and cursors hold a back-reference to their Run, so
    engine-adjacent code reads run state off the object at hand
    (``cand.run.dump``) instead of threading six arguments around."""

    pipeline: Pipeline
    ctx: Context
    search: Search | None = None
    db: SearchDB | None = None
    backend: object | None = None
    dump: CompilerDump | None = None
    rejections: list[tuple[str, str, str]] | None = None
    # Count of search candidates dropped by :meth:`drive`'s per-variant
    # containment (un-lowerable forks that raised during lowering). Tune-only;
    # stays 0 on the deterministic greedy path.
    _dropped_candidates: int = 0

    def _step(self, cand: Candidate) -> tuple[Match, list, bool] | None:
        """Run one rule batch against ``cand`` — the per-candidate engine
        body shared by :meth:`drive` and :meth:`resolve`. Single-option
        rewrites apply inline (via ``Candidate.try_rewrite``), empty /
        quiescent batches advance the cursor, and a structural fork whose
        offer site was already decided on this trajectory replays that
        side inline (:func:`_replay_structural_decision`). Returns ``None``
        when the batch completed with nothing left to decide, or
        ``(match, options, structural)`` at the first undecided
        multi-option fork — selection is the caller's job (``drive``
        spawns ``LazyCandidate`` siblings for its search; ``resolve``
        asks its ``decide`` callback)."""
        cur = cand.cursor
        pass_ = cur.current_pass
        # Empty pass (e.g. all rules filtered out) OR no live matches →
        # no apply fires → advance the cursor directly so the caller's
        # loop doesn't re-run the same rule batch forever. ``advance``
        # handles both cases uniformly: with ``n_applied == 0`` it wraps
        # to the next pass and fires the post-pass log + dump.
        if not pass_.rules:
            cur.advance(cand.graph)
            return None
        matches = self.pipeline.match(cand.graph, cur.current_rule)
        if not matches:
            cur.advance(cand.graph)
            return None
        for match in matches:
            options = cand.try_rewrite(match)
            if options is None:
                continue
            # The fork is classified here, where the raw ``options`` list
            # is concrete (no thunk fired): any ``Graph`` option makes the
            # fork **structural** (kernel-set-changing); pure ``Op``
            # rebinds (and the partition planner's branch Forks) are
            # op-variant.
            structural = any(_is_structural_option(o) for o in options)
            # A structurally identical offer site already decided on this
            # trajectory takes the same side inline (no fork), so the
            # search tree stays linear in *unique* kernels instead of
            # ``2^sites`` (e.g. one decision for 28 identical per-layer
            # splits). The earlier decision is read off the candidate's
            # own graph — see :func:`_replay_structural_decision` — so no
            # side-table state is threaded through resolves.
            if structural and (chosen := _replay_structural_decision(cand.graph, match.root.op, options)) is not None:
                cand.apply(match, chosen)
                continue
            return match, options, structural
        return None

    def resolve(self, graph: Graph, decide: Callable[[ForkPoint], object]) -> tuple[Graph, list[Decision]]:
        """Deterministic resolution — fold the pipeline over ``graph``
        IN PLACE, asking ``decide`` at every undecided fork point, and
        return ``(terminal_graph, trace)``. The counterpart of
        :meth:`drive` for callers with no frontier to rank (greedy
        compile, structural pricing probes, assembled-graph lowering):
        one live graph, no ``LazyCandidate`` sibling snapshots, no
        per-fork graph copies — the returned terminal IS the seeded
        ``graph`` object.

        ``decide`` receives a :class:`ForkPoint` and returns the option
        to apply — a concrete ``Op`` / ``Graph`` from the fork's raw
        options, or a **leaf** ``Fork`` (a decide that wants a lazy fork
        tree's complete rows expands branch Forks itself; returning a
        branch Fork is an error). It may stamp ``fp.score`` with the
        pick's predicted µs — copied onto the trace entry.

        The trace (one :class:`Decision` per decided fork, in resolution
        order) is the only output channel besides the terminal graph:
        "did this compile take a structural pick", "what did the
        partition fork predict for this kernel" are trace queries, not
        accumulated policy state. Inline replays of an already-decided
        structural offer site (see :meth:`_step`) are not decisions and
        don't trace."""
        from deplodock.compiler import provenance  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.candidate import Candidate  # noqa: PLC0415

        provenance.seed(graph)
        cand = Candidate(run=self, graph=graph, cursor=Cursor(run=self))
        trace: list[Decision] = []
        while not cand.cursor.is_done:
            step = self._step(cand)
            if step is None:
                continue
            match, options, structural = step
            root_op = match.root.op  # read before apply rebinds it
            fp = ForkPoint(match=match, options=options, root_op=root_op, ctx=self.ctx, structural=structural)
            choice = decide(fp)
            option = _concrete_option(choice)
            if option is None:
                raise ValueError(f"decide returned a branch Fork at {match.rule.name!r} — return a concrete option or a leaf Fork")
            knob_delta = _choice_knobs(choice, option, root_op)
            cand.apply(match, option)
            trace.append(
                Decision(
                    rule_name=match.rule.name,
                    node_id=fp.node_id,
                    chosen_kind="graph" if isinstance(option, Graph) else "op",
                    knob_delta=knob_delta,
                    score=fp.score,
                    n_options=len(options),
                )
            )
        return cand.graph, trace

    def drive(self, graph: Graph) -> Iterator[tuple[object | None, Candidate]]:
        """Seed ``graph`` as the root candidate and drive the search to
        every terminal. Each iteration: pop a ``(token, candidate)``
        pair, run one rule's batch of matches against the candidate's
        graph, push successor(s) under ``parent=token``. Yields
        ``(token, candidate)`` when a candidate reaches the end of the
        pipeline (``cursor.is_done``) — the caller passes the token to
        ``search.observe`` so the measurement lands on the terminal's
        own lineage (no "most recently popped" hidden state).

        Per-rule batch semantics live in :meth:`_step` (shared with
        :meth:`resolve`): single-option matches apply inline; the first
        undecided multi-option match comes back as ``(match, options,
        structural)`` and spawns one ``LazyCandidate`` per option, in
        rule-emission order. Selection is the search's job (tuning
        explores every fork and ranks the unvisited frontier with its
        learned prior). Siblings share ``cand`` as ``inner`` so they
        don't duplicate the snapshot; ``from_option`` lifts concrete
        ``Op``/``Graph`` options into leaf Forks so every LazyCandidate's
        pending carries a uniform Fork shape. Cursor advance for the rule
        batch is owned by :meth:`Cursor.advance`, fired from
        ``Candidate.apply`` on ``match.is_last`` (the fork's apply on
        resolve fires it for deferred forks) or directly in ``_step`` for
        batches that produced no live matches. The ``structural`` flag
        rides ``Search.push`` so policies can treat kernel-set decisions
        specially."""
        from deplodock.compiler.pipeline.search.candidate import Candidate, LazyCandidate  # noqa: PLC0415

        search = self.search
        assert search is not None, "Run.drive needs a search policy; use Run.resolve for deterministic resolution"
        # Seed candidate: no parent token — the policy roots it itself.
        search.push(Candidate(run=self, graph=graph, cursor=Cursor(run=self)).lazy())

        while (popped := search.pop()) is not None:
            token, lc = popped
            # Thunk-bearing fork: expand before resolving. Each expansion
            # spawns the next level of ``LazyCandidate``s (more thunks or
            # concrete options) sharing the same ``inner`` and ``match`` —
            # cursor advance is deferred until a leaf actually resolves.
            if lc.is_expandable():
                children = lc.expand()
                search.push(*children, parent=token)
                continue
            cand = lc.resolve()
            if cand.cursor.is_done:
                yield token, cand
                continue
            # Per-variant containment: a search-explored candidate can reach an
            # un-lowerable shape that a *deterministic* lowering pass raises on
            # (e.g. a sibling-cell-fused slab fill the single-Write hoisted-compute
            # materializer can't represent, or an orphan AtomTile at render). The
            # deployable greedy pick (``Run.resolve``) never reaches these forks, so
            # under tune they are legitimate dead ends — exactly like a branch whose
            # every option fails ``validate(ctx)``. Drop the candidate's subtree and
            # keep driving the rest of the search instead of aborting the whole tune.
            # (``RuleSkipped`` is handled inside ``try_rewrite``; control-flow
            # exceptions propagate.)
            try:
                step = self._step(cand)
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:  # noqa: BLE001 — broad by design; this is the tune dead-end sink
                self._dropped_candidates += 1
                logger.warning(
                    "[tune] dropped un-lowerable candidate (%s: %s) — pruning branch, continuing search",
                    type(exc).__name__,
                    exc,
                )
                continue
            if step is None:
                search.push(cand.lazy(), parent=token)
                continue
            match, options, structural = step
            forks = [LazyCandidate.from_option(inner=cand, cursor=replace(cand.cursor), match=match, option=opt) for opt in options]
            search.push(*forks, parent=token, structural=structural)


def _is_structural_option(option: object) -> bool:
    """Classify one raw rewrite option by its effect (see
    ``plans/structural-forks-in-two-level.md`` step 1): a ``Graph`` splice
    changes which ops exist — **structural**; an ``Op`` rebind is in-place —
    **op-variant**. The Op/Graph return type IS the classification; rules wrap
    a Graph option in a leaf :class:`OptionFork` (e.g. ``tile/005_split_demoted``),
    whose ``option`` is readable without firing any thunk. A *branch* ``Fork``
    reads op-variant: the sole branch-Fork emitter today is the partition
    planner (all ``TileOp`` leaves), and typing it would require ``expand()`` —
    the body-normalizing build the lazy tree exists to avoid."""
    return isinstance(option, Graph) or (isinstance(option, OptionFork) and isinstance(option.option, Graph))


def _concrete_option(option: object) -> object | None:
    """Unwrap one raw rewrite option to the concrete ``Op`` / ``Graph`` a
    replayed structural decision can apply inline: leaf Forks fire their
    single-element thunk, concrete options pass through, and a *branch* Fork
    returns ``None`` (un-applyable without a full expand — the caller falls
    back to forking normally; no structural branch Fork exists today)."""
    if isinstance(option, Fork):
        return option.expand()[0] if option.is_leaf else None
    return option


def _option_decision(option: object, root_knobs: dict) -> dict | None:
    """The decision-knob delta one raw structural-fork option would stamp vs
    the offer op: new non-``S_*`` knob keys on the option's op / fork knobs (a
    ``Graph`` option reads the union over its nodes' op knobs — fragment
    kernels restamp their own ``S_*``, which describe the child bodies, not
    the decision). ``None`` when the option stamps nothing new."""
    if isinstance(option, Graph):
        knobs: dict = {}
        for node in option.nodes.values():
            knobs.update(getattr(node.op, "knobs", None) or {})
    else:
        knobs = getattr(option, "knobs", None) or {}
    delta = {k: v for k, v in knobs.items() if k not in root_knobs and not k.startswith("S_")}
    return delta or None


def _choice_knobs(choice: object, option: object, root_op) -> dict:
    """The chosen option's knob identity for a :class:`Decision` trace entry:
    a ``Fork``'s pinned row when it carries one, a ``Graph``'s decision-knob
    delta vs the offer op (:func:`_option_decision`), an ``Op``'s own knobs.
    ``choice`` is what ``decide`` returned (possibly a leaf Fork); ``option``
    is its unwrapped concrete ``Op`` / ``Graph``."""
    if isinstance(choice, Fork) and choice.knobs:
        return dict(choice.knobs)
    if isinstance(option, Graph):
        return _option_decision(option, root_op.knobs) or {}
    return dict(getattr(option, "knobs", None) or {})


def _replay_structural_decision(graph: Graph, root_op, options: list) -> object | None:
    """The concrete option a structurally identical, already-decided offer
    site on this trajectory took — or ``None`` (undecided / unmatchable →
    fork normally).

    The earlier decision is read off the candidate's own graph, not a
    side-table: a decided site leaves its evidence in the IR by contract —
    every structural rule stamps its decision knob onto the surviving ops
    (the ``SPLIT_CONE`` considered-vs-declined idiom), and those ops chain to
    the pre-decision offer op via the engine-owned ``Op.source`` (stamped
    unconditionally on rebinds, stamped across loop-dialect splices,
    preserved by ``_rename_buf_in_op``). So: find any op carrying every
    decision knob whose source chain contains an op structurally identical
    to this offer (same ``op_cache_key``), and replay the option whose delta
    matches its stamped values. Matching by decision-knob agreement (not a
    stored option index) survives rules reordering their emissions."""
    from deplodock.compiler.pipeline.search.keys import op_cache_key, source_chain  # noqa: PLC0415

    key = op_cache_key(root_op)
    if key is None:
        return None
    deltas = [(opt, _option_decision(opt, root_op.knobs)) for opt in options]
    decision_keys = {k for _, d in deltas if d for k in d}
    if not decision_keys:
        return None
    for node in graph.nodes.values():
        knobs = getattr(node.op, "knobs", None)
        if not knobs or not decision_keys <= set(knobs):
            continue  # undecided or unrelated op — the cheap pre-filter
        chain = source_chain(node.op)
        next(chain)  # the op itself — its key differs from the pre-decision key by the stamp
        if not any(op_cache_key(anc) == key for anc in chain):
            continue
        found = {k: knobs[k] for k in decision_keys}
        for opt, delta in deltas:
            if delta == found:
                return _concrete_option(opt)
        return None  # decided, but no option matches (emission drift) — fork normally
    return None


def _unlowered_tiles(graph: Graph, rejections: list[tuple[str, str, str]]) -> dict[str, frozenset]:
    """``{node_id: tile_identity}`` for every node a ``validate(ctx)`` rejection
    left un-lowered (still a ``LoopOp`` / ``TileOp`` at the terminal). The
    ``tile_identity`` is the offending tile's planner knobs — what ``Pipeline.run``
    blocklists so the greedy retry falls back to the next prior-ranked sibling."""
    if not rejections:
        return {}
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.policy.greedy import tile_identity  # noqa: PLC0415

    out: dict[str, frozenset] = {}
    for nid, _pass_label, _reason in rejections:
        node = graph.nodes.get(nid)
        if node is not None and isinstance(node.op, (LoopOp, TileOp)):
            out[nid] = tile_identity(node.op.knobs)
    return out


def _raise_on_unlowered(graph: Graph, rejections: list[tuple[str, str, str]], ctx: Context) -> None:
    """Fail a greedy compile loudly when a recorded ``validate(ctx)``
    rejection (see :func:`Candidate.try_rewrite`) left its node un-lowered.

    ``rejections`` is ``[(node_id, pass_label, reason), ...]``. A node is
    "stuck" iff it still holds a pre-final dialect op (``LoopOp`` /
    ``TileOp``) at the terminal — if a later rule lowered it anyway the
    op is a ``CudaOp`` and we stay silent (the rejection was a harmless
    intermediate filter). Only nodes with a recorded rejection are
    checked, so partial pipelines that legitimately terminate at the loop
    / tile stage (no lowering pass to drop anything) never trip this."""
    if not rejections:
        return
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    # Last recorded reason / pass wins (the final pass that tried to lower it).
    reason_by_node: dict[str, str] = {}
    pass_by_node: dict[str, str] = {}
    for nid, pass_label, reason in rejections:
        reason_by_node[nid] = reason
        pass_by_node[nid] = pass_label

    stuck = [nid for nid in reason_by_node if (node := graph.nodes.get(nid)) is not None and isinstance(node.op, (LoopOp, TileOp))]
    if not stuck:
        return
    lines = [f"  - {nid!r}: {pass_by_node[nid]} rejected its only lowering — {reason_by_node[nid]}" for nid in stuck]
    raise LoweringError(
        f"compile: {len(stuck)} node(s) left un-lowered — the chosen tile shape produced a kernel that "
        f"failed validate(ctx) and the deterministic compile had no fallback:\n"
        + "\n".join(lines)
        + "\nPin a fitting tile via DEPLODOCK_KNOBS, raise the smem budget, or adjust tile-geometry "
        "scoring so an in-budget variant ranks first."
    )


def _match_at(graph: Graph, start: str, rule: Rule) -> Match | None:
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
        rule=rule,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


# ---------------------------------------------------------------------------
# Bench + DB persistence for autotune terminals (used by Pipeline.tune_async)
# ---------------------------------------------------------------------------


async def _rebench_o3_async(cand, backend):
    """Re-bench an already-lowered tune winner at ``-Xcicc -O3`` (deployable codegen)
    for a clean prior sample, awaiting the device-pinned worker. Returns the -O3
    median latency in µs, or ``None`` when the sweep is already at -O3 or the bench
    errors (best-effort — a re-bench hiccup must never abort the sweep). The winner
    already benched OK at -O1, so the only added cost is one -O3 compile (cubin-cached)."""
    from deplodock import config  # noqa: PLC0415

    if "-O3" in config.nvcc_flags():
        return None
    try:
        result = await backend.benchmark_async(cand.graph, nvcc_flags="-Xcicc -O3")
    except Exception:  # noqa: BLE001 — a re-bench failure is non-fatal to tuning
        return None
    return result.time_ms * 1000.0 if result.time_ms else None


class _TerminalBench:
    """Shared machinery for benching one terminal candidate's ``CudaOp``s and
    persisting per-kernel ``perf`` / inventory / lowering rows.

    :func:`_bench_terminal_async` drives it: the no-cuda / cache-hit / stub
    short-circuits (:meth:`prelude`) and every DB write (:meth:`finalize_result` /
    :meth:`finalize_exc`) live here, so the only awaited step is the device bench."""

    def __init__(self, cand, *, backend, db) -> None:
        from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

        self.backend = backend
        self.db = db
        self.graph = cand.graph
        self.context_key = cand.ctx.structural_key()
        self.cuda_nodes = [self.graph.nodes[nid] for nid in self.graph.topological_order() if isinstance(self.graph.nodes[nid].op, CudaOp)]
        self.backend_name = getattr(backend, "name", "stub")

    @staticmethod
    def _point_stats(us: float):
        from deplodock.compiler.pipeline.search.db import PerfStats  # noqa: PLC0415

        return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)

    @classmethod
    def _stats_from_launch(cls, lt):
        import statistics as _statistics  # noqa: PLC0415

        from deplodock.compiler.pipeline.search.db import PerfStats  # noqa: PLC0415

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
        return cls._point_stats(lt.time_ms * 1000.0)

    @staticmethod
    def _body_json(op, dialect: str) -> str:
        import json as _json  # noqa: PLC0415

        return _json.dumps(
            {
                "dialect": dialect,
                "name": getattr(op, "name", None) or getattr(op, "kernel_name", None) or "?",
                "body_repr": repr(op.body),
            },
            default=str,
        )

    def _record_op_inventory(self, op) -> None:
        from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
        from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
        from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

        key = op_cache_key(op)
        if key is None:
            return
        if isinstance(op, CudaOp):
            self.db.record_cuda_op(
                key,
                kernel_source=op.kernel_source,
                arg_order=list(op.arg_order),
                grid=list(op.grid),
                block=list(op.block),
                smem_bytes=op.smem_bytes,
                pretty=op.kernel_source,
            )
        elif isinstance(op, KernelOp):
            self.db.record_kernel_op(key, self._body_json(op, "kernel"), op.pretty_body())
        elif isinstance(op, TileOp):
            self.db.record_tile_op(key, self._body_json(op, "tile"), op.pretty_body())
        elif isinstance(op, LoopOp):
            self.db.record_loop_op(key, self._body_json(op, "loop"), op.pretty_body())

    def _persist(self, cuda_op, *, stats, status: str, captured: bool = False, error: str | None = None) -> None:
        from deplodock.compiler.pipeline.search.keys import (  # noqa: PLC0415
            _is_kernel_bearing,
            dialect_of,
            op_cache_key,
            source_chain,
        )

        cuda_key = op_cache_key(cuda_op)
        if cuda_key is None:
            return
        chain = [op for op in source_chain(cuda_op) if _is_kernel_bearing(op)]
        for op in chain:
            self._record_op_inventory(op)
        for parent_op, child_op in zip(chain[1:], chain[:-1], strict=False):
            p_dialect = dialect_of(parent_op)
            c_dialect = dialect_of(child_op)
            if p_dialect is None or c_dialect is None:
                continue
            if p_dialect == c_dialect == "loop":
                # loop→loop source hops are structural/decision hops, not
                # lowering rewrites: the splice attribution stamped by
                # ``Candidate.apply`` (a decomposition's kernels → the
                # pre-split op), 005's keep-vs-split rebind, name stamps.
                # A ``lowering`` row holds ONE best child per parent, so
                # recording a multi-kernel decomposition's hops would let
                # ``best_per_op_time``'s chain walk resolve the pre-split
                # op to a single fragment kernel's median — half the work
                # masquerading as the whole op. The decomposition's cost
                # is a Σ, owned by the two-level tuner, never this table.
                continue
            p_key = op_cache_key(parent_op)
            c_key = op_cache_key(child_op)
            if p_key is None or c_key is None:
                continue
            p_knobs = getattr(parent_op, "knobs", None) or {}
            c_knobs = getattr(child_op, "knobs", None) or {}
            knobs_delta = {k: v for k, v in c_knobs.items() if p_knobs.get(k) != v}
            self.db.record_lowering(
                p_key,
                p_dialect,
                c_key,
                c_dialect,
                knobs=knobs_delta,
                measured_median_us=stats.median if status == "ok" else None,
            )
        knobs = getattr(cuda_op, "knobs", None) or {}
        self.db.record_perf(
            self.context_key, cuda_key, backend=self.backend_name, status=status, stats=stats, knobs=knobs, captured=captured, error=error
        )
        logger.info("[tune]   %s @ %.2f us  (%s)", getattr(cuda_op, "kernel_name", "?"), stats.median, status)

    def _accumulate(self, acc, s):
        from deplodock.compiler.pipeline.search.db import PerfStats  # noqa: PLC0415

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

    def prelude(self):
        """Resolve everything that needs no live bench. Returns ``("done", (stats,
        status))`` when no measurement is needed (no CudaOps / full cache hit /
        stub backend), else ``("bench", None)`` — the caller obtains a
        ``BenchmarkResult`` and calls :meth:`finalize_result` / :meth:`finalize_exc`."""
        from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

        if not self.cuda_nodes:
            return "done", (self._point_stats(0.0), "ok")

        # Cache lookup: if every CudaOp already has a perf row for this
        # (context, backend), skip the benchmark entirely and rebuild the
        # aggregate stats from the DB. Per-kernel partial caching isn't
        # useful here because ``backend.benchmark`` runs the whole graph.
        cached_rows = []
        for node in self.cuda_nodes:
            key = op_cache_key(node.op)
            row = self.db.lookup_perf(self.context_key, key, backend=self.backend_name) if key is not None else None
            if row is None:
                cached_rows = None
                break
            cached_rows.append(row)
        if cached_rows is not None:
            logger.info("[tune] cache hit for %d kernel(s) — skipping bench", len(self.cuda_nodes))
            agg = None
            status = "ok"
            for row in cached_rows:
                if row.status != "ok":
                    status = row.status
                agg = self._accumulate(agg, row.stats)
                logger.info("[tune]   %s @ %.2f us  (%s, cached)", row.op_key[:12], row.stats.median, row.status)
            return "done", (agg or self._point_stats(0.0), status)

        if self.backend is None:
            # No real measurement → do NOT persist. Writing the 1.0us stub
            # to a shared DB used to clobber tuned ``best_median_us`` values
            # (record_lowering / record_perf keep the minimum), so any plain
            # ``deplodock run`` (which routes through ``Pipeline.run`` without
            # a backend) was overwriting real autotune rows with 1.0us stubs.
            # Tests that need lowering edges in stub mode should pass an
            # explicit stub backend.
            agg = None
            for _node in self.cuda_nodes:
                agg = self._accumulate(agg, self._point_stats(1.0))
            return "done", (agg or self._point_stats(0.0), "ok")

        logger.info("[tune] benching %d kernel(s) in graph", len(self.cuda_nodes))
        return "bench", None

    def finalize_exc(self, exc):
        fail_us = float(self.backend.bench_run_timeout_s) * 1_000_000.0
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d kernel(s)",
            exc,
            fail_us,
            len(self.cuda_nodes),
        )
        s = self._point_stats(fail_us)
        agg = None
        for node in self.cuda_nodes:
            self._persist(node.op, stats=s, status="bench_fail", error=f"{type(exc).__name__}: {exc}")
            agg = self._accumulate(agg, s)
        return agg or self._point_stats(0.0), "bench_fail"

    def finalize_result(self, result):
        agg = None
        per_launch = result.per_launch or []
        if len(per_launch) != len(self.cuda_nodes):
            logger.warning(
                "[tune] per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
                len(per_launch),
                len(self.cuda_nodes),
            )
            avg_us = (result.time_ms * 1000.0) / max(len(self.cuda_nodes), 1)
            s = self._point_stats(avg_us)
            for node in self.cuda_nodes:
                self._persist(node.op, stats=s, status="ok", captured=result.captured)
                agg = self._accumulate(agg, s)
        else:
            for node, lt in zip(self.cuda_nodes, per_launch, strict=True):
                s = self._stats_from_launch(lt)
                self._persist(node.op, stats=s, status="ok", captured=result.captured)
                agg = self._accumulate(agg, s)
        try:
            import cupy as _cp  # noqa: PLC0415

            _cp.cuda.runtime.deviceSynchronize()
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        return agg or self._point_stats(0.0), "ok"


async def _bench_terminal_async(cand, *, backend, db):
    """Bench every ``CudaOp`` in ``cand.graph``, persist per-kernel ``perf`` /
    inventory / lowering rows, and return ``(stats, status)`` where ``stats`` is the
    per-kernel ``PerfStats`` summed across the graph (total terminal latency). The
    only ``await`` is the device-pinned bench, so N kernels' benches overlap on one
    event loop; cache-hit / stub / persistence semantics live in :class:`_TerminalBench`."""
    b = _TerminalBench(cand, backend=backend, db=db)
    kind, payload = b.prelude()
    if kind == "done":
        return payload
    try:
        result = await backend.benchmark_async(b.graph, num_iters="auto")
    except Exception as exc:  # noqa: BLE001
        return b.finalize_exc(exc)
    return b.finalize_result(result)


__all__ = ["Decision", "ForkPoint", "LoweringError", "Match", "Pass", "Pattern", "Pipeline", "Rule", "RuleSkipped"]
