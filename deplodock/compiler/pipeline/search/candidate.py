"""Search-tree data classes — :class:`Candidate` (concrete graph state)
and :class:`LazyCandidate` (a parent + a chain of pending applications
that materializes via :meth:`resolve`).

Sibling forks at multi-option rewrite points share a single ``inner``
``Candidate`` (the parent's snapshot) by reference; each fork holds its
own one-element chain. ``resolve()`` is the single entry point that
turns a lazy candidate into a concrete one — copy the inner's graph
once, replay the chain of ``(match, option)`` pairs through
``Candidate.apply``, drop the chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor, _fmt_op
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from deplodock.compiler.pipeline.pipeline import Cursor
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
        """Wrap in a zero-chain :class:`LazyCandidate`. The search
        layer always handles ``LazyCandidate``; this helper lifts a
        concrete cand back into that interface (e.g. before pushing
        the rollout's current cand back to ``Search.push``)."""
        return LazyCandidate(inner=self, cursor=self.cursor, chain=[])

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
        options = list(result) if isinstance(result, (list, tuple)) else [result]
        options = [o for o in options if not isinstance(o, Op) or o.validate(self.ctx)]
        if not options:
            self._advance_if_last(match)
            return None
        if len(options) > 1:
            # Defer to a fork — caller spawns ``LazyCandidate``
            # siblings. Cursor advance happens via the fork's apply on
            # resolve, so we leave it alone here.
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
            self.graph.splice(option, consumed=match.consumed, output=match.output or match.root_node_id)
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
    and whose ``ctx`` propagates onto the resolved Candidate) and a
    ``chain`` of ``(match, option)`` pairs to replay on resolve.

    Sibling forks at the same rewrite point share ``inner`` by
    reference — only one snapshot is ever held in memory per fork
    point. Each fork's chain is its own short list (typically a single
    pair carrying the alt option for that fork's match site).

    ``cursor`` is the lazy candidate's own pipeline cursor (typically a
    copy of the parent's cursor at fork-creation time)."""

    inner: Candidate
    cursor: Cursor
    chain: list[tuple[Match, Op | Graph | None]]

    def resolve(self) -> Candidate:
        """Materialize: copy ``inner.graph``, build a fresh Candidate
        carrying our cursor, replay the chain through its ``apply``,
        drop the chain so a second resolve is a no-op (returns the
        cached resolved Candidate). Multiple sibling ``LazyCandidate``
        instances pointing at the same ``inner`` each get their own
        copy — the snapshot is shared only across siblings, not across
        resolve calls."""
        if not self.chain:
            return self.inner
        resolved = Candidate(ctx=self.inner.ctx, graph=self.inner.graph.copy(), cursor=self.cursor)
        for match, option in self.chain:
            resolved.apply(match.remap(resolved.graph), option)
        self.chain = []
        self.inner = resolved
        return resolved


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
