"""Autotune driver — the search-loop that walks the rewrite tree.

``_search_loop`` is the unified driver: pop a candidate, advance it by
one rule application (via ``engine._try_one_rule``), push successor(s)
back. ``run_autotune`` wraps it with backend measurement / DB recording
per terminal; ``run_pipeline`` is the single-shot greedy convenience
wrapper used by ``deplodock compile`` without ``--tune``."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.policy import GreedySearch, Search
from deplodock.compiler.pipeline.search.recorder import TuneAborted, record_terminal

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.policy.mcts import SearchTree

_PASSES_DIR = Path(__file__).resolve().parent.parent / "passes"

logger = logging.getLogger(__name__)


def _search_loop(
    search: Search,
    rules_per_pass: list,
    pass_names: list[str],
    ctx: Context | None,
    dump: CompilerDump | None,
) -> Iterator[Candidate]:
    """The unified search-driven driver. Each iteration: pop a
    candidate, try one rule application (or end-of-pass bookkeeping),
    push successor(s). Yields when a candidate reaches the end of the
    pipeline (``cursor.pass_idx >= len(pass_names)``).

    Used by every engine entry point — ``run_autotune`` (full pipeline)
    and ``run_pass`` (one pass; pass ``select=[stem]`` to run a single
    rule). They differ only in the rules-per-pass list and the
    ``Search`` instance supplied."""
    # Local import to break the engine ↔ driver cycle (engine.run_pass
    # calls _search_loop, and _try_one_rule lives in engine).
    from deplodock.compiler.pipeline.engine import _try_one_rule  # noqa: PLC0415

    tree: SearchTree | None = getattr(search, "tree", None)
    while (cand := search.pop()) is not None:
        cur = cand.cursor
        if cur.pass_idx >= len(pass_names):
            yield cand
            continue
        rules = rules_per_pass[cur.pass_idx]
        # Empty pass (e.g. all rules filtered out): nothing to do, skip.
        if not rules:
            cur.pass_idx += 1
            search.push(cand)
            continue
        rule = rules[cur.rule_idx]
        pass_idx_arg = cur.pass_idx + 1 if pass_names[cur.pass_idx] else None
        pass_name_arg = pass_names[cur.pass_idx] or None
        result = _try_one_rule(cand, rule, ctx, dump, pass_idx_arg, pass_name_arg, tree=tree)

        def _on_pass_finish(idx: int) -> None:
            name = pass_names[idx]
            if name:
                logger.debug("compile: %-18s done (%d nodes)", name, len(cand.graph.nodes))
            if dump is not None and name:
                dump.on_pass(idx + 1, name, cand.graph)

        cur.advance(result, n_rules=len(rules), on_pass_finish=_on_pass_finish)
        # Push ``cand`` and its sibling forks together so the policy
        # sees them as one fork-point group. ``GreedySearch`` discards
        # the forks (or picks a DB-preferred one via the rewrite site
        # recorded on ``cand.last_rewritten``); ``TuningSearch``
        # registers all of them (forks first internally so the
        # ``_just_popped`` check still fires on ``cand``, preserving
        # the rollout semantics).
        search.push(cand, *result.forks)


def run_pipeline(
    graph: Graph,
    passes: list[str],
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
    backend=None,
    db: SearchDB | None = None,
) -> Graph:
    """Run each named pass directory in order; dispatch ``dump.on_pass``
    after each. Single-candidate convenience wrapper around
    :func:`run_autotune` using :class:`GreedySearch` — stops at the
    first terminal so autotune forks beyond option 0 are never explored.

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

    The loop is fully search-driven: pop a candidate, advance it by one
    rule application via :func:`_try_step`, push successor(s) back to
    ``search``. When no rule fires in the current pass, advance the
    candidate's ``cursor.pass_idx`` and push it back. When ``cursor.pass_idx``
    reaches the end of ``passes``, the candidate is terminal and gets
    yielded.

    ``search`` chooses both the order and the stopping condition:
    :class:`GreedySearch` for single-shot compiles (stops at the first
    terminal); :class:`TuningSearch` for ``--tune`` (runs the queue
    dry, exploring every fork).

    When ``search`` exposes a ``tree: SearchTree`` (both built-in
    searches do), each yielded terminal candidate has its ``CudaOp``
    nodes recorded to ``db`` and the tree via :func:`record_terminal`
    before being yielded — so subsequent candidates see the updated
    priority signal. Pass a ``Backend`` (typically
    :class:`CudaBackend`) via ``backend=`` to record real GPU-event
    latencies; omit it to record the stub ``latency_us=1.0``.

    ``ctx`` is built once (probing the live device if not provided)
    and shared by every candidate."""
    from deplodock.compiler.pipeline.engine import _filter_rules, _load_rules  # noqa: PLC0415

    if ctx is None:
        ctx = Context.probe()
    # Stamp the backend identifier so DB lookups (``GreedySearch._select``)
    # pull perf rows from the matching backend. ``"cuda"`` is the
    # canonical autotune target — keep it when no live backend is
    # supplied so a previously-tuned DB still drives the choice.
    backend_name = getattr(backend, "name", "cuda")
    if ctx.backend_name != backend_name:
        from dataclasses import replace as _replace  # noqa: PLC0415

        ctx = _replace(ctx, backend_name=backend_name)
    select_set = set(select) if select is not None else None
    rules_per_pass = [_filter_rules(_load_rules(_PASSES_DIR / name), select_set) for name in passes]
    t_start = time.monotonic()

    search.push(Candidate(ctx=ctx, _graph=graph))

    tree: SearchTree | None = getattr(search, "tree", None)
    if db is None:
        db = SearchDB()
    n_terminals = 0
    for cand in _search_loop(search, rules_per_pass, passes, ctx, dump):
        n_terminals += 1
        if backend is not None:
            # Collect knobs from every terminal kernel in the graph so
            # the log line reflects the *actual* autotune choices that
            # produced this variant, not just the rule:choice indices.
            knob_strs: list[str] = []
            for nid in cand.graph.topological_order():
                op = cand.graph.nodes[nid].op
                k = getattr(op, "knobs", None) or {}
                if k:
                    knob_strs.append(", ".join(f"{kk}={vv}" for kk, vv in sorted(k.items())))
            label = " | ".join(knob_strs) if knob_strs else "option-0"
            logger.info("[tune] variant #%d  [%s]", n_terminals, label)
        try:
            # ``recorder.record_terminal`` is a no-op when backend is
            # ``None`` (or writes 1.0us stubs to the in-memory DB) and
            # gracefully handles ``tree is None`` (greedy search). The
            # call is unconditional so any backend-driven measurement
            # always lands in the perf / inventory tables.
            record_terminal(cand.graph, db, tree, cand.ctx.structural_key(), backend=backend)
        except TuneAborted as exc:
            # A bench failure left GPU work queued; running another
            # variant would block in cupy's ``_allocate``. Yield
            # this terminal (its measurements are already recorded
            # as bench_fail) and stop the sweep so the caller can
            # pick a winner from whatever ok variants we've got.
            logger.warning("[tune] %s — stopping after %d terminal(s)", exc, n_terminals)
            yield cand
            break
        yield cand
    logger.info("compile: total %.2fs (%d terminal(s))", time.monotonic() - t_start, n_terminals)
