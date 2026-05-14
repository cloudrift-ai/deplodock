"""Compile-pipeline driver — rule loading, rewrite dispatch, search loop,
and the public ``run_pipeline`` / ``run_autotune`` entry points.

The matcher value types (``Pattern`` / ``Match``) live in
:mod:`.match`; ``Pass`` / ``Rule`` / ``Pipeline`` live in their own
sibling modules. This module wires them together: loads rules off disk
into a :class:`Pipeline`, runs the search loop one rule batch at a
time, fires per-rule diff / dump hooks, and records terminal candidates.

Rule contract: rules MUST be idempotent on their own output. The engine
re-runs the full pipeline on every popped candidate, relying on each
rule's "already applied" guard (often implicit via op-type change) to
skip work that's already done."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor, _fmt_op
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from deplodock.compiler.pipeline.engine.pipeline import Match, Pass, Pipeline, Rule
from deplodock.compiler.pipeline.rule_diff import display_name, emit, render_rule_diff
from deplodock.compiler.pipeline.search.candidate import Candidate, LazyCandidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.policy import GreedySearch, Search
from deplodock.compiler.pipeline.search.recorder import TuneAborted, record_terminal

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.policy.mcts import SearchTree

_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def _load_rules(pass_dir: Path) -> list[Rule]:
    rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
    return [_load_rule(f) for f in rule_files]


def _load_rule(path: Path) -> Rule:
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
    return Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn, param_names=param_names)


def _build_rewrite_kwargs(rule: Rule, match: Match, ctx: Context | None) -> dict:
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


def _filter_rules(rules: list[Rule], select_set: set[str] | None) -> list[Rule]:
    if select_set is None:
        return rules
    return [r for r in rules if r.name in select_set or _strip_rule_prefix(r.name) in select_set]


# ---------------------------------------------------------------------------
# Search driver — pop a candidate, run one rule's batch, push successors.
# ``run_pipeline`` / ``run_autotune`` are the public entry points.
# ---------------------------------------------------------------------------


_PASSES_DIR = Path(__file__).resolve().parent.parent / "passes"


def _make_apply_logger(
    *,
    debug_on: bool,
    dump: CompilerDump | None,
) -> Callable[[Graph, Match, Op | Graph], None] | None:
    """Build the ``Candidate.on_apply`` callback that renders a rule
    application's diff at debug / writes its dump record. Returns
    ``None`` when neither sink is active so ``Candidate.apply`` can
    skip the hook entirely. Rule + pass metadata is read off
    ``match.rule`` (stamped by ``Pipeline.match``). One logger instance
    serves every rule batch in the pipeline."""
    if not (debug_on or dump is not None):
        return None

    def _on_apply(graph_before: Graph, match: Match, option: Op | Graph) -> None:
        fragment = _wrap_op_as_fragment(graph_before, match.root_node_id, option) if isinstance(option, Op) else option
        rule = match.rule
        pass_ = match.pass_
        rule_name = rule.name if rule is not None else ""
        text = _format_rule_application(rule_name, graph_before, match, fragment, pass_name=match.pass_name)
        if debug_on:
            emit(text)
        if dump is not None and rule is not None and pass_ is not None and pass_.name:
            record = _record_rule_application(graph_before, match, fragment)
            dump.on_rule(pass_, rule, record, text)

    return _on_apply


def _search_loop(
    search: Search,
    pipeline: Pipeline,
    ctx: Context | None,
    dump: CompilerDump | None,
) -> Iterator[Candidate]:
    """The unified search-driven driver. Each iteration: pop a
    candidate, run one rule's batch of matches against its graph, push
    successor(s). Yields when a candidate reaches the end of the
    pipeline (``cursor.pass_idx >= pipeline.n_passes()``).

    Per-rule batch semantics: enumerate matches via
    :meth:`Pipeline.match` (which stamps ``rule`` on each Match plus
    ``is_last`` on the last live one), call ``Candidate.try_rewrite``
    for each. Single-option matches apply inline; the first multi-
    option match spawns one ``LazyCandidate`` per option and breaks
    the loop. Cursor advance for the rule batch is owned by
    ``Candidate.apply`` / ``_advance_batch``, fired on ``match.is_last``
    even when the rewrite skipped or yielded no valid options."""
    while (popped := search.pop()) is not None:
        cand = popped.resolve()
        cur = cand.cursor
        if cur.pass_idx >= pipeline.n_passes():
            yield cand
            continue
        n_rules = pipeline.n_rules(cur.pass_idx)
        # Empty pass (e.g. all rules filtered out): nothing to do, skip.
        if n_rules == 0:
            cur.pass_idx += 1
            search.push(cand.lazy())
            continue
        rule = pipeline.rule_at(cur.pass_idx, cur.rule_idx)
        matches = pipeline.match(cand.graph, rule)
        if not matches:
            # No live matches → no apply will fire → advance the cursor
            # ourselves with a synthetic last-match sentinel so the
            # transition still flows through ``Candidate.apply`` /
            # ``_advance_batch``.
            sentinel = Match(graph=cand.graph, root_node_id="", rule=rule, is_last=True)
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
            forks = [LazyCandidate(inner=cand, cursor=replace(cur), chain=[(match, opt)]) for opt in options]
            break

        if forks is not None:
            search.push(forks[0], *forks[1:])
            continue

        search.push(cand.lazy())


def run_pipeline(
    graph: Graph,
    passes: list[str],
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
    backend=None,
    db: SearchDB | None = None,
) -> Graph:
    """Single-shot greedy compile — run each named pass directory in
    order, picking option 0 at every fork point. Convenience wrapper
    around :func:`run_autotune` that yields the first terminal.

    ``ctx`` is built once (probing the live device if not provided)
    and passed to every rule that takes a ``ctx`` parameter.

    ``backend`` (typically :class:`CudaBackend`) opts the run into real
    GPU measurement: every terminal graph's per-kernel latency is
    recorded to ``db`` and attributed to every ancestor along the
    ``Op.source`` chain. ``db`` defaults to a fresh in-memory store;
    pass an explicit :class:`SearchDB` to persist measurements
    across runs.

    For exhaustive autotuning, call :func:`run_autotune` directly with
    :class:`TuningSearch` and iterate every yielded candidate."""
    search = GreedySearch(db=db)
    return next(run_autotune(graph, passes, search=search, dump=dump, select=select, ctx=ctx, backend=backend, db=db)).graph


def run_autotune(
    graph: Graph,
    passes: list[str],
    *,
    search: Search,
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
    backend=None,
    db: SearchDB | None = None,
) -> Iterator[Candidate]:
    """Drive the autotune search. Yields one terminal ``Candidate`` per
    fully-explored branch. With deterministic rules (no list-returning
    rewrites) the search yields exactly one — same shape as
    ``run_pipeline``.

    ``search`` chooses both the order and the stopping condition:
    :class:`GreedySearch` for single-shot compiles (stops at the first
    terminal); :class:`TuningSearch` for ``--tune`` (runs the queue dry,
    exploring every fork).

    When ``search`` exposes a ``tree: SearchTree`` (``TuningSearch``
    does), each yielded terminal candidate has its ``CudaOp`` nodes
    recorded to ``db`` and the tree via :func:`record_terminal` before
    being yielded — so subsequent candidates see the updated priority
    signal. Pass a ``Backend`` (typically :class:`CudaBackend`) via
    ``backend=`` to record real GPU-event latencies; omit it to record
    the stub ``latency_us=1.0``.

    ``ctx`` is built once (probing the live device if not provided) and
    shared by every candidate."""
    if ctx is None:
        ctx = Context.probe()
    backend_name = getattr(backend, "name", "cuda")
    if ctx.backend_name != backend_name:
        ctx = replace(ctx, backend_name=backend_name)
    select_set = set(select) if select is not None else None
    pass_list = [
        Pass(name=name, rules=_filter_rules(_load_rules(_PASSES_DIR / name), select_set), index=i) for i, name in enumerate(passes)
    ]
    pipeline = Pipeline(passes=pass_list)
    t_start = time.monotonic()

    # The on_apply hook (debug diff + dump.on_rule) propagates from the
    # root Candidate to every fork via ``cand.on_apply`` in the loop, so
    # one logger built here serves the whole pipeline run — the search
    # loop never has to (re)install it.
    on_apply = _make_apply_logger(debug_on=logger.isEnabledFor(logging.DEBUG), dump=dump)

    def on_pass_finish(pass_: Pass, graph_after: Graph) -> None:
        logger.debug("compile: %-18s done (%d nodes)", pass_.name, len(graph_after.nodes))
        if dump is not None:
            dump.on_pass(pass_, graph_after)

    search.push(Candidate(ctx=ctx, graph=graph, on_apply=on_apply, on_pass_finish=on_pass_finish).lazy())

    tree: SearchTree | None = getattr(search, "tree", None)
    if db is None:
        db = SearchDB()
    n_terminals = 0
    for cand in _search_loop(search, pipeline, ctx, dump):
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


__all__ = ["run_autotune", "run_pipeline"]
