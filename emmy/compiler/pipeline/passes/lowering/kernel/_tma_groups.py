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

from deplodock.compiler.ir.stmt import Cond, Stmt
from deplodock.compiler.ir.tile.ir import AsyncWait, SerialTile, StageBundle, StagePolicy, WarpSpecialize


def _is_tma_bundle(s) -> bool:
    return isinstance(s, StageBundle) and s.policy == StagePolicy.TMA


@dataclass(frozen=True)
class TmaGroups:
    """Result of :func:`partition_tma_groups`.

    - ``stage_group``: **smem name** → group id. Keyed by Source.name
      (not bundle id) because the materializer's ``Body.map`` walk
      rebuilds StageBundle wrappers as it descends — id(bundle) at
      partition time differs from id(bundle) at emit time. Source
      identity (smem name) is stable across rewrites and uniquely
      identifies each member Stage within its group.
    - ``wait_group``: ``id(AsyncWait)`` → group id. AsyncWait has no
      nested body so ``Body.map`` preserves its identity — safe to
      keep using id().
    - ``group_stage_names``: group id → distinct smem names across all
      member stages of the bundle(s) in the group
    - ``group_buffer_count``: group id → max ``buffer_count`` in the group
    - ``issuer_tid``: smem name → issuer thread index within its group
    """

    stage_group: dict[str, int] = field(default_factory=dict)
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


def _is_kouter(stmt: Stmt) -> bool:
    """A SerialTile is the K_o loop iff it has a TMA bundle directly in
    its body AND no nested SerialTile that itself contains a TMA bundle.
    The second clause distinguishes a real K_o from an outer wrapper
    (e.g., SDPA's ``for a5 in 0..1`` with K_o ``for a7`` nested inside)."""
    if not isinstance(stmt, SerialTile):
        return False
    direct = False
    for s in stmt.body:
        if _is_tma_bundle(s):
            direct = True
        if isinstance(s, SerialTile):
            for s2 in s.body.iter():
                if _is_tma_bundle(s2):
                    return False  # nested K_o — current stmt is a wrapper
    return direct


def partition_tma_groups(body: tuple[Stmt, ...]) -> TmaGroups:
    """Partition this Tile body's TMA stages + waits into pipeline-unit
    groups. Each K-loop containing TmaBufferedStages is one group, plus
    any prologue stages immediately before the loop and the immediately-
    following epilogue AsyncWait. Synchronous (pre-pipelining) stages
    (a TmaBufferedStage at body level paired with a trailing AsyncWait)
    each form a singleton group.

    Walks recursively: outer SerialTile wrappers that don't directly own
    TMA bundles (but have them deeper) pass through, so SDPA-style
    layouts with an extra wrapping loop around the K_o still get their
    inner SerialTile recognised as K_o. State (pending prologues, last
    pipeline group) is shared across recursion levels so a prologue
    bundle at the outer level still flushes into the inner K_o's group.

    The per-stage issuer thread is the position of the stage name within
    its (sorted) group, so stages in the same group issue their
    arrive+TMA from distinct elected threads (tid 0, tid 1, ...) rather
    than serializing on tid 0.
    """
    stage_group_by_smem: dict[str, int] = {}
    wait_group_by_id: dict[int, int] = {}
    group_stage_names: list[set[str]] = []
    group_buffer_count: list[int] = []

    def _new_group() -> int:
        group_stage_names.append(set())
        group_buffer_count.append(0)
        return len(group_stage_names) - 1

    def _add_bundle(gid: int, bundle: StageBundle) -> None:
        # group_stage_names tracks per-Source smem name across the bundle so
        # issuer-tid allocation distributes the arrive+TmaLoad across distinct
        # elected threads.
        for src in bundle.sources:
            stage_group_by_smem[src.name] = gid
            group_stage_names[gid].add(src.name)
        group_buffer_count[gid] = max(group_buffer_count[gid], bundle.buffer_count)

    # Shared mutable state across recursion levels — single Python list/int
    # holders so the inner closures see updates done at any depth.
    pending_prologue: list[StageBundle] = []
    last_pipeline_group: list[int | None] = [None]

    def _flush_prologues_as_singletons() -> None:
        for st in pending_prologue:
            gid = _new_group()
            _add_bundle(gid, st)
        pending_prologue.clear()

    def _walk(body_: tuple[Stmt, ...]) -> None:
        for stmt in body_:
            if _is_tma_bundle(stmt):
                pending_prologue.append(stmt)
                continue
            if isinstance(stmt, SerialTile):
                if _is_kouter(stmt):
                    gid = _new_group()
                    for st in pending_prologue:
                        _add_bundle(gid, st)
                    pending_prologue.clear()
                    for s in stmt.body:
                        if _is_tma_bundle(s):
                            _add_bundle(gid, s)
                        elif isinstance(s, AsyncWait):
                            wait_group_by_id[id(s)] = gid
                    last_pipeline_group[0] = gid
                    continue
                # Outer wrapper or unrelated SerialTile — recurse into
                # its body so inner K_o loops still get discovered.
                _walk(tuple(stmt.body))
                continue
            if isinstance(stmt, WarpSpecialize):
                # 085_warp_specialize packs producer (TMA bundles) and
                # consumer (AsyncWaits, reduce) into the two branches of
                # a WarpSpecialize marker. Recurse with shared state so
                # the bundles in producer_body and the AsyncWaits in
                # consumer_body join the same pipeline-unit group.
                _walk(tuple(stmt.producer_body))
                _walk(tuple(stmt.consumer_body))
                continue
            if isinstance(stmt, Cond):
                # Pre-WarpSpecialize 085 emitted producer/consumer as a
                # bare Cond; left in place for any other rule that may
                # emit a Cond around StageBundles / AsyncWaits.
                _walk(tuple(stmt.body))
                _walk(tuple(stmt.else_body))
                continue
            if isinstance(stmt, AsyncWait):
                # Trailing epilogue wait pairs with the most recent
                # pipeline-unit group (080_pipeline_stages always emits one
                # epilogue AsyncWait after its main loop). If there's no
                # pipeline group in scope but pending synchronous stages
                # exist, the wait pairs them with the next-to-be-flushed
                # singleton group (whichever stage was most recently added).
                if last_pipeline_group[0] is not None and not pending_prologue:
                    wait_group_by_id[id(stmt)] = last_pipeline_group[0]
                elif pending_prologue:
                    # Synchronous stage(s) followed by their wait — collapse
                    # into one group with arrive count = num pending.
                    gid = _new_group()
                    for st in pending_prologue:
                        _add_bundle(gid, st)
                    pending_prologue.clear()
                    wait_group_by_id[id(stmt)] = gid
                    last_pipeline_group[0] = gid
                continue
            # Flush any orphan prologue bundles (no following K_o), but
            # KEEP ``last_pipeline_group`` intact — non-TMA stmts (Smem,
            # SetMaxNReg, Loop, Load, …) that sit between a K_o and its
            # epilogue AsyncWait shouldn't sever the pairing. Pre-WS
            # this was a "reset on any other stmt" check; with WS-emitted
            # Conds putting SetMaxNReg inside the consumer branch (right
            # before the K_o-containing subtree), that reset would
            # detach the epilogue AsyncWaits from their producer-side
            # group.
            _flush_prologues_as_singletons()

    _walk(tuple(body))
    _flush_prologues_as_singletons()

    # Per-stage issuer thread = position of the stage name within its group.
    issuer_tid: dict[str, int] = {}
    for names in group_stage_names:
        for idx, name in enumerate(sorted(names)):
            issuer_tid[name] = idx

    return TmaGroups(
        stage_group=stage_group_by_smem,
        wait_group=wait_group_by_id,
        group_stage_names=tuple(frozenset(s) for s in group_stage_names),
        group_buffer_count=tuple(group_buffer_count),
        issuer_tid=issuer_tid,
    )
