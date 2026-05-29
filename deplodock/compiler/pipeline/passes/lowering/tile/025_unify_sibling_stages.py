"""Drop redundant per-sibling staging of an input buffer already staged
by an earlier sibling scope.

When ``020_stage_inputs`` runs on a Tile body whose ``ThreadTile``
contains multiple sibling reduce ``SerialTile`` scopes that each read
the same input — for example the fused RMSNorm + linear case where the
prologue mean-reduce K-tower and the matmul K-tower both ``Load x[m,
k]`` — 020 forms a separate ``Source`` per consumer scope. Each is
genuinely independent (the producer reloads fresh content per its own
K_o iteration), so the slabs *cannot* meaningfully share content at
runtime: the prologue ends with its last-iter data, the matmul wants
its own iter's data, and re-using one slab requires re-issuing the
load anyway. The cleanest answer is to **stop staging the later
consumer** and let it read from gmem — exactly what the legacy
non-blocked path did (which is what those tests pin).

This pass walks the TileOp body top-down, maintaining a set of ``buf``
names already staged at any prior scope. When it visits a
``StageBundle``, every ``Source`` whose ``buf`` is already in the set
is dropped and the consumer ``Load``s inside the bundle body that
referenced that Source's slab name are rewritten back to read from
gmem at the original index (reconstructed from ``Source.origin`` +
``cache_dims``, or directly from ``Source.template_index`` when the
addressing was template). If every Source in a Stage gets dropped the
Stage is removed; if every Stage in the bundle goes the bundle is
unwrapped (its body inlined at the parent scope).

Runs between ``020_stage_inputs`` and ``030_hoist_invariant_compute``
so every subsequent pass (ring-buffer / TMA promotion / pad / pipeline)
sees only the surviving Stages and operates on a clean shape.

Idempotent — a body with no buf-overlap raises ``RuleSkipped``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import Source, Stage, StageBundle, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    state = _State()
    new_body = _walk(root.op.body, state=state)
    if not state.dropped_any:
        raise RuleSkipped("no StageBundle stages a buffer already staged in a prior scope")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


@dataclass
class _State:
    """Top-down walk state: bufs ever staged anywhere we've already
    visited, and a flag the rule entry uses to decide ``RuleSkipped``."""

    bufs_staged: set[str]
    dropped_any: bool = False

    def __init__(self) -> None:
        self.bufs_staged = set()
        self.dropped_any = False


def _walk(body: Body, *, state: _State) -> Body:
    """Walk one body level, splicing-out any ``StageBundle`` whose every
    Source was dropped (its consumer body inlines into the sibling list).
    Sibling order is preserved; the ``state.bufs_staged`` set is mutated
    in pre-order so a later sibling sees the kept Sources from earlier
    siblings."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, StageBundle):
            kept, inline_body = _visit_bundle(s, state=state)
            if kept is not None:
                out.append(kept)
            else:
                # Whole bundle dropped — inline its (already-reverted) body.
                out.extend(inline_body)
            continue
        nested = s.nested()
        if nested:
            new_bodies = tuple(_walk(b, state=state) for b in nested)
            out.append(s.with_bodies(new_bodies))
        else:
            out.append(s)
    return Body(out)


def _visit_bundle(bundle: StageBundle, *, state: _State) -> tuple[Stmt | None, Body]:
    """Partition the bundle's Sources into kept (first occurrence of each
    buf) and dropped (subsequent). Rewrite the bundle body's Loads
    targeting dropped slabs back to gmem.

    Returns ``(rebuilt_bundle, Body(()))`` when at least one Source
    survives — the caller appends the bundle to its sibling list.
    Returns ``(None, reverted_body)`` when every Source was dropped —
    the caller splices ``reverted_body`` into the sibling list (the
    bundle wrapper carried only staging policy + sync glue, no longer
    needed)."""
    revert: dict[str, Source] = {}
    new_stages: list[Stage] = []
    for stage in bundle.stages:
        kept_sources: list[Source] = []
        for src in stage.sources:
            if src.buf in state.bufs_staged:
                revert[src.name] = src
                state.dropped_any = True
                continue
            state.bufs_staged.add(src.buf)
            kept_sources.append(src)
        if kept_sources:
            new_stages.append(replace(stage, sources=tuple(kept_sources)))

    # Recurse into the bundle body first so any nested StageBundle (a
    # bulk-prefetch pipeline emits sibling bundles inside a serial_outer
    # loop, for example) sees the updated ``bufs_staged`` set.
    new_inner = _walk(bundle.body, state=state)
    if revert:
        new_inner = _revert_loads(new_inner, revert)

    if not new_stages:
        return None, new_inner
    return replace(bundle, stages=tuple(new_stages), body=new_inner), Body(())


def _reconstruct_global_index(src: Source) -> tuple[Expr, ...]:
    """Build the gmem ``Load.index`` that 020 would have emitted had it
    not promoted this Source to smem. ``cache_dims`` carries the
    affine mapping (``origin[d] + Var(axis_name)`` for each cache dim);
    ``template_index`` is the verbatim non-affine form when 020 set it.
    Inverse of 020's ``smem_index = tuple(Var(ax.name) for ax in cache_axes)``."""
    if src.template_index is not None:
        return src.template_index
    cache_var_by_dim = {cd.source_dim: Var(cd.axis.name) for cd in src.cache_dims}
    return tuple((o + cache_var_by_dim[d]) if d in cache_var_by_dim else o for d, o in enumerate(src.origin))


def _revert_loads(body: Body, revert: dict[str, Source]) -> Body:
    """Rewrite every ``Load(input=src.name, …)`` inside ``body`` back to
    ``Load(input=src.buf, index=reconstruct(src))``. Recurses through
    every wrapper via ``Body.map`` — a Load deep inside a nested
    SerialTile / RegisterTile / Cond / sibling StageBundle gets the
    revert too. The dtype field is preserved (the gmem buffer has the
    same element type as the smem slab — 030_stamp_types runs later
    anyway and rewrites both from ``graph.nodes[buf].output.dtype``)."""
    if not revert:
        return body

    def fn(s: Stmt) -> Stmt:
        if not isinstance(s, Load):
            return s
        src = revert.get(s.input)
        if src is None:
            return s
        return Load(names=s.names, input=src.buf, index=_reconstruct_global_index(src), dtype=s.dtype)

    return body.map(fn)
