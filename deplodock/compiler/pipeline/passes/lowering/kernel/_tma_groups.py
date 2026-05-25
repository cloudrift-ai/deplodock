"""TMA pipeline-unit grouping analysis for ``100_materialize_tile``.

Read-only analysis of a (flattened) Tile body — mirrors the
``Body.coordination`` precedent: one walk produces the per-stage /
per-wait group assignments the materializer's TMA emit closures consume
to pick mbarrier names, arrive counts, and issuer threads. No IR is
rewritten here.

The leading-underscore module name keeps the pass loader (globs
``*.py``, skips ``_``-prefixed files) from mistaking this for a rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt import Stmt
from deplodock.compiler.ir.tile.ir import AsyncWait, SerialTile, TmaBufferedStage


@dataclass(frozen=True)
class TmaGroups:
    """Result of :func:`partition_tma_groups`. Stage / wait → group-id
    lookups use ``id(...)`` keys because the keyed nodes aren't hashable;
    the materializer passes the exact body nodes it walks, so the ids
    match.

    - ``stage_group``: ``id(TmaBufferedStage)`` → group id
    - ``wait_group``: ``id(AsyncWait)`` → group id
    - ``group_stage_names``: group id → distinct stage (smem) names
    - ``group_buffer_count``: group id → max ``buffer_count`` in the group
    - ``issuer_tid``: stage name → issuer thread index within its group
    """

    stage_group: dict[int, int] = field(default_factory=dict)
    wait_group: dict[int, int] = field(default_factory=dict)
    group_stage_names: tuple[frozenset[str], ...] = ()
    group_buffer_count: tuple[int, ...] = ()
    issuer_tid: dict[str, int] = field(default_factory=dict)

    @property
    def has_tma(self) -> bool:
        return bool(self.group_stage_names)

    def mbar_name(self, gid: int) -> str:
        """Per-group mbarrier array name. Single-group tiles keep the
        unsuffixed ``tma_mbar`` name for readability."""
        return f"tma_mbar_{gid}" if len(self.group_stage_names) > 1 else "tma_mbar"

    def arrive_count(self, gid: int) -> int:
        """Distinct stages in the group — the mbarrier's arrive count."""
        return len(self.group_stage_names[gid])


def partition_tma_groups(body: tuple[Stmt, ...]) -> TmaGroups:
    """Partition this Tile body's TMA stages + waits into pipeline-unit
    groups. Each K-loop containing TmaBufferedStages is one group, plus
    any prologue stages immediately before the loop and the immediately-
    following epilogue AsyncWait. Synchronous (pre-pipelining) stages
    (a TmaBufferedStage at body level paired with a trailing AsyncWait)
    each form a singleton group.

    The per-stage issuer thread is the position of the stage name within
    its (sorted) group, so stages in the same group issue their
    arrive+TMA from distinct elected threads (tid 0, tid 1, ...) rather
    than serializing on tid 0.
    """
    stage_group_by_id: dict[int, int] = {}
    wait_group_by_id: dict[int, int] = {}
    group_stage_names: list[set[str]] = []
    group_buffer_count: list[int] = []

    def _new_group() -> int:
        group_stage_names.append(set())
        group_buffer_count.append(0)
        return len(group_stage_names) - 1

    def _add_stage(gid: int, stage: TmaBufferedStage) -> None:
        stage_group_by_id[id(stage)] = gid
        # 050_use_tma promotes single-source stages only; group_stage_names
        # tracks per-Source smem name so issuer-tid allocation distributes
        # the arrive+TmaLoad across distinct elected threads.
        for src in stage.sources:
            group_stage_names[gid].add(src.name)
        group_buffer_count[gid] = max(group_buffer_count[gid], stage.buffer_count)

    pending_prologue: list[TmaBufferedStage] = []
    last_pipeline_group: int | None = None

    def _flush_prologues_as_singletons() -> None:
        for st in pending_prologue:
            gid = _new_group()
            _add_stage(gid, st)
        pending_prologue.clear()

    for stmt in body:
        if isinstance(stmt, TmaBufferedStage):
            pending_prologue.append(stmt)
            continue
        if isinstance(stmt, SerialTile) and any(isinstance(s, TmaBufferedStage) for s in stmt.body):
            gid = _new_group()
            for st in pending_prologue:
                _add_stage(gid, st)
            pending_prologue.clear()
            for s in stmt.body:
                if isinstance(s, TmaBufferedStage):
                    _add_stage(gid, s)
                elif isinstance(s, AsyncWait):
                    wait_group_by_id[id(s)] = gid
            last_pipeline_group = gid
            continue
        if isinstance(stmt, AsyncWait):
            # Trailing epilogue wait pairs with the most recent
            # pipeline-unit group (080_pipeline_stages always emits one
            # epilogue AsyncWait after its main loop). If there's no
            # pipeline group in scope but pending synchronous stages
            # exist, the wait pairs them with the next-to-be-flushed
            # singleton group (whichever stage was most recently added).
            if last_pipeline_group is not None and not pending_prologue:
                wait_group_by_id[id(stmt)] = last_pipeline_group
            elif pending_prologue:
                # Synchronous stage(s) followed by their wait — collapse
                # into one group with arrive count = num pending.
                gid = _new_group()
                for st in pending_prologue:
                    _add_stage(gid, st)
                pending_prologue.clear()
                wait_group_by_id[id(stmt)] = gid
                last_pipeline_group = gid
            continue
        # Any other statement breaks the prologue chain — flush.
        _flush_prologues_as_singletons()
        last_pipeline_group = None

    _flush_prologues_as_singletons()

    # Per-stage issuer thread = position of the stage name within its group.
    issuer_tid: dict[str, int] = {}
    for names in group_stage_names:
        for idx, name in enumerate(sorted(names)):
            issuer_tid[name] = idx

    return TmaGroups(
        stage_group=stage_group_by_id,
        wait_group=wait_group_by_id,
        group_stage_names=tuple(frozenset(s) for s in group_stage_names),
        group_buffer_count=tuple(group_buffer_count),
        issuer_tid=issuer_tid,
    )
