"""Search-tree data classes — :class:`Candidate` (concrete graph state)
and :class:`LazyCandidate` (a parent + an optional pending rewrite
that materializes via :meth:`resolve`).

Sibling forks at multi-option rewrite points share a single ``inner``
``Candidate`` (the parent's snapshot) by reference; each fork holds
its own ``pending = (match, option)`` pair. ``resolve()`` is the
single entry point that turns a lazy candidate into a concrete one —
copy the inner's graph once, replay ``pending`` through
``Candidate.apply``, then drop ``pending``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor, _fmt_op
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from deplodock.compiler.pipeline.pipeline import Cursor, Fork, OptionFork
from deplodock.compiler.pipeline.rule_diff import display_name, render_rule_diff

# Use the engine logger so the existing debug-emit toggles (rule-
# skipped lines under ``compile -vv``) keep working without callers
# having to also bump this module's level.
_logger = logging.getLogger("deplodock.compiler.pipeline")

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.pipeline import Match


@dataclass
class Candidate:
    """A concrete point in the search space — owns a real ``graph``.

    ``ctx`` is shared by reference across siblings. ``cursor`` tracks
    pipeline resume state. The candidate reads any logging / dump sink
    off ``match.pipeline.dump`` inside :meth:`apply` and the
    end-of-pass hook inside :meth:`_advance_batch` — no separate
    callback wiring is needed.

    :class:`LazyCandidate` is the deferred-apply counterpart used for
    autotune fork siblings; both expose :meth:`resolve` so the search
    loop can treat them uniformly."""

    ctx: Context
    graph: Graph
    cursor: Cursor

    def resolve(self) -> Candidate:
        """Identity — already concrete. Provided so callers can resolve
        any candidate uniformly."""
        return self

    def lazy(self) -> LazyCandidate:
        """Wrap in a no-op :class:`LazyCandidate` (``pending=None``). The search
        layer always handles ``LazyCandidate``; this helper lifts a
        concrete cand back into that interface (e.g. before pushing
        the rollout's current cand back to ``Search.push``)."""
        return LazyCandidate(inner=self, cursor=self.cursor, pending=None)

    def try_rewrite(self, match: Match) -> list[Op | Graph] | None:
        """Eager mode (called by the search loop): invoke
        ``match.rule.rewrite`` against this candidate's graph,
        validate the result, and either apply the single chosen
        option or — for a multi-option fork — return the option list
        for the caller to spawn ``LazyCandidate`` siblings from.
        Returns ``None`` when no rewrite was applied (``RuleSkipped``,
        empty options after validation, or single option applied
        successfully).

        Cursor advance is unconditional on ``match.is_last`` — even
        when the rewrite skipped or produced no valid option — so the
        search loop terminates on quiescent batches where every match
        is skipped by the rule's own idempotence guard. The
        multi-option return path is the one exception: the cursor
        advance is left to the eventual fork's apply on resolve."""
        from deplodock.compiler.pipeline.pipeline import RuleSkipped  # noqa: PLC0415
        from deplodock.compiler.pipeline.rule_diff import emit, format_skipped  # noqa: PLC0415

        if not match.is_alive():
            # Earlier applies in this batch invalidated the match's
            # consumed nodes. Skip the rewrite, but still advance the
            # cursor when this was the last match — otherwise the
            # search loop would re-pop the same rule batch forever.
            self._advance_if_last(match)
            return None
        rule = match.rule
        try:
            result = rule.rewrite(**_build_rewrite_kwargs(rule, match, self.ctx))
        except RuleSkipped as exc:
            if _logger.isEnabledFor(logging.DEBUG):
                emit(format_skipped(display_name(rule.pass_.name if rule.pass_ else None, rule.name), match.root_node_id, exc.reason))
            self._advance_if_last(match)
            return None
        raw_options = list(result) if isinstance(result, (list, tuple)) else [result]
        # ``Fork`` options pass through unconditionally — they're deferred
        # expansions with no graph to validate yet. Concrete ``Op`` leaves
        # still get the per-ctx validate filter; ``Graph`` splices are
        # validated at splice time, not here.
        options = [o for o in raw_options if not isinstance(o, Op) or o.validate(self.ctx)]
        if not options:
            # Validation-filtered rewrite: the rule produced output but
            # every option failed ``validate(ctx)`` — most commonly the
            # ``KernelOp.validate`` smem-cap check after
            # ``100_materialize_tile`` produces a kernel that exceeds
            # ``ctx.max_dynamic_smem``. In a *fork* this is legitimate
            # pruning (sibling branches carry other tile shapes), but in a
            # deterministic single-path compile it leaves the node
            # un-lowered with no recourse — so we both (a) emit a debug
            # "filtered" line and (b) record the rejection into the
            # pipeline's optional sink. ``Pipeline.run`` (greedy) reads the
            # sink after the terminal settles and raises a loud
            # ``LoweringError`` if any recorded node is still un-lowered,
            # turning the old "CudaBackend: non-CudaOp TileOp" mystery into
            # an actionable error. The sink is absent under ``tune`` so the
            # fork-pruning path keeps its zero-overhead silent behavior.
            sink = getattr(match.pipeline, "_lowering_rejections", None)
            if raw_options and (sink is not None or _logger.isEnabledFor(logging.DEBUG)):
                rejected = [o for o in raw_options if isinstance(o, Op)]
                reasons = [r for r in (_validate_reason(o, self.ctx) for o in rejected) if r]
                reason_str = "; ".join(reasons) if reasons else "validate(ctx)=False"
                pass_label = display_name(rule.pass_.name if rule.pass_ else None, rule.name)
                if sink is not None:
                    sink.append((match.root_node_id, pass_label, reason_str))
                if _logger.isEnabledFor(logging.DEBUG):
                    emit(
                        format_skipped(pass_label, match.root_node_id, f"all {len(rejected)} option(s) failed validate(ctx): {reason_str}")
                    )
            self._advance_if_last(match)
            return None
        if len(options) > 1 or isinstance(options[0], Fork):
            # Defer to a fork — caller spawns ``LazyCandidate`` siblings.
            # Single-option ``Fork`` also goes through this path: the
            # search loop needs to dispatch on ``is_expandable()`` to
            # invoke the thunk, which can't happen via the inline apply
            # path below. Cursor advance for both cases happens via the
            # eventual leaf's apply on resolve.
            return options
        self.apply(match, options[0])
        return None

    def apply(self, match: Match, option: Op | Graph) -> None:
        """Lazy mode (called by ``LazyCandidate.resolve`` and
        internally by :meth:`try_rewrite` for single-option matches):
        apply the specific ``option`` to this candidate's graph.
        Mutates the graph, logs the rewrite (debug diff +
        ``pipeline.dump.on_rule`` snapshot, both read off
        ``match.pipeline``), bumps cursor ``n_applied`` for functional
        splices, and advances the rule-batch cursor when
        ``match.is_last``.

        ``Op`` rebinds ``root.op`` (id / inputs / hints kept);
        ``Graph`` is a fragment spliced via ``Graph.splice``. On the
        ``Op`` path the chain ``Op.source`` is stamped with the op
        being replaced (unless the rule already set it) and the
        predecessor's ``knobs`` are merged forward — so the rewrite
        chain threads through every in-place rebind for free."""
        self._log_apply(match, option)
        if isinstance(option, Op):
            old_op = self.graph.nodes[match.root_node_id].op
            if option is not old_op and option.source is None:
                option.source = old_op
                option.knobs = {**old_op.knobs, **option.knobs}
            self.graph.nodes[match.root_node_id].op = option
        else:
            assert isinstance(option, Graph), f"expected Graph or Op; got {type(option).__name__}"
            # Decomposition expands one op into many distinct pieces (mint);
            # every other fragment splice aggregates the consumed pieces.
            pass_ = match.rule.pass_
            mint_pieces = pass_ is not None and pass_.name.startswith("frontend/decomposition")
            self.graph.splice(
                option,
                consumed=match.consumed,
                output=match.output or match.root_node_id,
                mint_pieces=mint_pieces,
            )
            self.cursor.n_applied += 1
        self._advance_if_last(match)

    def _log_apply(self, match: Match, option: Op | Graph) -> None:
        """Render a per-rule diff at DEBUG and route a structured
        record to ``match.pipeline.dump`` when set. Returns early when
        neither sink is active."""
        from deplodock.compiler.pipeline.rule_diff import emit  # noqa: PLC0415

        rule = match.rule
        pass_ = rule.pass_
        dump = match.pipeline.dump
        debug_on = _logger.isEnabledFor(logging.DEBUG)
        if not (debug_on or dump is not None):
            return
        fragment = _wrap_op_as_fragment(self.graph, match.root_node_id, option) if isinstance(option, Op) else option
        pass_name = pass_.name if pass_ is not None else None
        text = _format_rule_application(rule.name, self.graph, match, fragment, pass_name=pass_name)
        if debug_on:
            emit(text)
        if dump is not None and pass_ is not None and pass_.name:
            record = _record_rule_application(self.graph, match, fragment)
            dump.on_rule(pass_, rule, record, text)

    def _advance_if_last(self, match: Match) -> None:
        if match.is_last:
            self.cursor.advance(self.graph)


@dataclass
class LazyCandidate:
    """Deferred-apply counterpart of :class:`Candidate`. Holds a parent
    ``inner`` Candidate (whose ``graph`` is the snapshot to clone from
    and whose ``ctx`` propagates onto the resolved Candidate) and an
    optional ``pending`` ``(match, fork)`` pair to replay on resolve.

    Sibling forks at the same rewrite point share ``inner`` by reference
    — only one snapshot is ever held in memory per fork point. Each
    sibling's ``pending`` carries its own :class:`Fork` (branch or leaf;
    see :class:`deplodock.compiler.pipeline.pipeline.Fork`).

    The constructor classmethods (:meth:`from_op` / :meth:`from_graph` /
    :meth:`from_fork` / :meth:`from_option`) are the supported way to
    spawn a non-trivial LazyCandidate — they handle the Op/Graph-to-Fork
    leaf-wrapping uniformly so callers don't have to.

    ``cursor`` is the lazy candidate's own pipeline cursor (typically a
    copy of the parent's cursor at fork-creation time)."""

    inner: Candidate
    cursor: Cursor
    pending: tuple[Match, Fork] | None

    @classmethod
    def from_op(cls, *, inner: Candidate, cursor: Cursor, match: Match, op: Op) -> LazyCandidate:
        """Wrap a concrete ``Op`` rewrite as a leaf-Fork-pending LazyCandidate.
        Validation has already happened upstream (in ``try_rewrite``'s
        filter) — the constructor just lifts the option into the Fork
        shape so resolve / expand / _best_fork can treat it uniformly."""
        knobs = dict(getattr(op, "knobs", None) or {})
        leaf = OptionFork(option=op, knobs=knobs)
        return cls(inner=inner, cursor=cursor, pending=(match, leaf))

    @classmethod
    def from_graph(cls, *, inner: Candidate, cursor: Cursor, match: Match, graph: Graph) -> LazyCandidate:
        """Wrap a ``Graph`` fragment splice as a leaf-Fork-pending LazyCandidate."""
        leaf = OptionFork(option=graph)
        return cls(inner=inner, cursor=cursor, pending=(match, leaf))

    @classmethod
    def from_fork(cls, *, inner: Candidate, cursor: Cursor, match: Match, fork: Fork) -> LazyCandidate:
        """Direct wrap for an explicit branch ``Fork`` produced by a rule."""
        return cls(inner=inner, cursor=cursor, pending=(match, fork))

    @classmethod
    def from_option(cls, *, inner: Candidate, cursor: Cursor, match: Match, option: Op | Graph | Fork) -> LazyCandidate:
        """Dispatch on the option's type. The single entry point used by
        ``Pipeline.search``'s fork-spawn site so every option gets lifted
        consistently into a Fork-pending LazyCandidate."""
        if isinstance(option, Fork):
            return cls.from_fork(inner=inner, cursor=cursor, match=match, fork=option)
        if isinstance(option, Op):
            return cls.from_op(inner=inner, cursor=cursor, match=match, op=option)
        return cls.from_graph(inner=inner, cursor=cursor, match=match, graph=option)

    def is_expandable(self) -> bool:
        """``True`` iff ``pending`` carries a *branch* :class:`Fork` —
        one whose ``expand()`` produces the next level of options. Leaf
        Forks (constructed from a concrete Op/Graph via :meth:`from_op`
        / :meth:`from_graph`) are NOT expandable — they resolve directly
        via :meth:`resolve`. ``False`` also for the no-pending wrapper
        produced by :meth:`Candidate.lazy`."""
        return self.pending is not None and not self.pending[1].is_leaf

    def expand(self) -> list[LazyCandidate]:
        """Fire the pending branch :class:`Fork`'s thunk and lift each
        returned option into a sibling ``LazyCandidate`` via
        :meth:`from_option`. Children share ``inner`` by reference (same
        pattern as the flat-fork spawn site in ``pipeline.py``), carry
        an independent cursor copy, and thread the same ``match`` — so
        ``match.is_last`` only fires the cursor advance once a leaf
        actually resolves.

        Raises if called when :meth:`is_expandable` would return False —
        the search loop dispatches on that predicate."""
        from dataclasses import replace  # noqa: PLC0415

        assert self.pending is not None and not self.pending[1].is_leaf, "expand() requires branch Fork pending"
        match, fork = self.pending
        children_options = fork.expand()
        return [
            LazyCandidate.from_option(inner=self.inner, cursor=replace(self.cursor), match=match, option=opt) for opt in children_options
        ]

    def resolve(self) -> Candidate:
        """Materialize: copy ``inner.graph``, build a fresh Candidate
        carrying our cursor, replay the pending leaf Fork by invoking
        its thunk (returns ``[op]`` or ``[graph]``) and applying that
        single option through ``Candidate.apply``, drop ``pending`` so
        a second resolve is a no-op. Multiple sibling ``LazyCandidate``
        instances pointing at the same ``inner`` each get their own
        copy — the snapshot is shared only across siblings, not across
        resolve calls.

        Caller must ensure :meth:`is_expandable` is False before
        resolving — branch Forks expand into children that are
        eventually leaves themselves."""
        if self.pending is None:
            return self.inner
        match, fork = self.pending
        assert fork.is_leaf, "resolve() called on branch Fork; use expand() first"
        leaves = fork.expand()
        assert len(leaves) == 1, f"leaf Fork must expand to a single option, got {len(leaves)}"
        option = leaves[0]
        resolved = Candidate(ctx=self.inner.ctx, graph=self.inner.graph.copy(), cursor=self.cursor)
        resolved.apply(match.remap(resolved.graph), option)
        self.pending = None
        self.inner = resolved
        return resolved

    @property
    def fork(self) -> Fork | None:
        """The pending :class:`Fork`, or ``None`` for a no-pending
        wrapper. Scoring/ranking is search policy — the policies read
        ``Search.score_of(cand.fork)`` (which owns the value-keyed
        cache and the why-prior-only rationale); this accessor is just
        the unwrap."""
        return self.pending[1] if self.pending is not None else None


# ---------------------------------------------------------------------------
# Per-rule snapshot rendering (used at DEBUG, i.e. ``compile -vv``, and
# routed to ``pipeline.dump.on_rule`` when a dump sink is set).
# Module-private helpers used only by :meth:`Candidate._log_apply`.
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
    """Render a list of nodes as readable text. Body-carrying ops use
    their own ``pretty_body``; everything else falls back to a
    ``name: ClsName(args)`` one-liner. Scalar ``ConstantOp`` inputs are
    inlined as literals (same treatment as ``format_kernels`` — see
    ``_inline_scalar_loads``). The surrounding
    ``<output> = TileOp(<inputs>)`` label is emitted here, one line
    above the body — ``BodyOp.pretty_body`` no longer prepends its own
    kernel-name / I/O header to keep the two from duplicating."""
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


def _build_rewrite_kwargs(rule, match: Match, ctx: Context | None) -> dict:
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


def _validate_reason(op: Op, ctx: Context) -> str:
    """Best-effort introspection of *why* ``op.validate(ctx)`` returned
    ``False``. Returns a short reason like ``smem 106496 > cap 101376``
    when the op exposes the right introspection hooks (``KernelOp``
    today). Empty string when no reason can be derived — the caller
    treats that as a generic ``validate(ctx)=False`` line."""
    # Best-effort: keep the import local so a missing kernel-IR module
    # (e.g. minimal harness in tests) doesn't break the engine itself.
    try:
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile  # noqa: PLC0415
    except ImportError:
        return ""
    if not isinstance(op, KernelOp):
        return ""
    reasons: list[str] = []
    for s in op.body:
        if isinstance(s, GridTile):
            ctas = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in s.axes)
            # _MAX_CTAS lives next to KernelOp.validate; keep the compare local.
            for child in s.body:
                if isinstance(child, ThreadTile):
                    threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in child.axes)
                    if threads > ctx.max_threads_per_cta:
                        reasons.append(f"threads {threads} > max_threads_per_cta {ctx.max_threads_per_cta}")
            _ = ctas  # _MAX_CTAS comparison kept inside validate
        elif isinstance(s, ThreadTile):
            threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in s.axes)
            if threads > ctx.max_threads_per_cta:
                reasons.append(f"threads {threads} > max_threads_per_cta {ctx.max_threads_per_cta}")
    try:
        smem = op.smem_bytes()
        if smem > ctx.max_dynamic_smem:
            reasons.append(f"smem {smem} > max_dynamic_smem {ctx.max_dynamic_smem}")
    except Exception:  # noqa: BLE001 — best-effort introspection
        pass
    return "; ".join(reasons)
