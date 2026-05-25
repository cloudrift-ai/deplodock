"""Promote SYNC StageBundle to BUFFERED with a double-buffered ring.

For a Tile body containing ``SerialTile(K_o, kind="serial_outer",
body=[StageBundle(policy=SYNC, ...)])``, this pass promotes the SYNC
bundle to ``policy=BUFFERED`` carrying ``buffer_count = 2`` and
``phase = Var(K_o.name) % 2``. Loads inside ``bundle.body`` that read
from any of the bundle's staged smem names get the phase prepended as
a leading index dimension. The materializer doubles the smem
allocation, prepends the phase to the cooperative-load write, and
drops the leading ``__syncthreads`` (consecutive iterations write
distinct physical slabs).

Trigger:

- A ``SerialTile`` with ``kind="serial_outer"`` and ``extent >= 2``.
- Its body contains ≥ 1 ``StageBundle`` with ``policy == SYNC``.
- The bundle's body carries a ``SerialTile(kind="stage_inner",
  is_reduce=True)`` — the K_i reduce. No non-Accum stmt inside that
  reduce reads a sibling Accum's running value (rejects in-loop
  online-softmax-style merges where running-value reads would compound
  fp32 drift under the rotated-slot rewrite).
- Smem budget: ``2 * sum(bundle.smem_bytes) <= ctx.max_dynamic_smem``.

Idempotence: any K_o whose bundles are already non-SYNC is left alone.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import SerialTile, StageBundle, StagePolicy, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_BUFFER_COUNT = 2


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    body = root.op.body
    if any(isinstance(s, StageBundle) and s.policy != StagePolicy.SYNC for s in body.iter()):
        raise RuleSkipped("double-buffer already applied (non-SYNC bundle present)")

    new_body, changed = _walk(body, smem_budget=ctx.max_dynamic_smem)
    if not changed:
        raise RuleSkipped("no K-outer StageBundle eligible for double-buffering")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body, *, smem_budget: int) -> tuple[Body, bool]:
    """Recursive descent: visit every wrapper looking for ``SerialTile(serial_outer)``
    whose body holds an eligible SYNC bundle — promote and stop descending into
    the promoted subtree (no further serial_outer is expected below the bundle
    for the matmul shape)."""
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            promoted = _maybe_promote_kouter(s, smem_budget=smem_budget)
            if promoted is not None:
                out.append(promoted)
                changed = True
                continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b, smem_budget=smem_budget)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _maybe_promote_kouter(kouter: SerialTile, *, smem_budget: int) -> SerialTile | None:
    # Ring buffers need ≥2 K iterations to prologue + steady-state. Symbolic
    # K can be either — defer the promotion (no per-iter overlap) rather than
    # speculatively allocating a multi-buffer slab.
    if not kouter.axis.extent.is_static or kouter.axis.extent.as_static() < 2:
        return None
    promote_ids: set[int] = set()
    total_bytes = 0
    for s in kouter.body:
        if not isinstance(s, StageBundle):
            continue
        if s.policy != StagePolicy.SYNC:
            return None  # mixed (already-promoted + SYNC) — leave alone
        if not _has_stage_inner_reduce(s.body):
            return None
        if not _accums_independent_in(s.body):
            return None
        promote_ids.add(id(s))
        # SYNC bundle: smem_bytes has no buffer factor; we'll apply ×_BUFFER_COUNT.
        total_bytes += s.smem_bytes
    if not promote_ids:
        return None
    if _BUFFER_COUNT * total_bytes > smem_budget:
        return None

    phase = Var(kouter.axis.name) % Literal(_BUFFER_COUNT, "int")
    new_kouter_body: list[Stmt] = []
    for s in kouter.body:
        if isinstance(s, StageBundle) and id(s) in promote_ids:
            staged_names = set(s.local_decls())
            new_inner = s.body.map(_make_phase_load_rewriter(staged_names, phase))
            new_kouter_body.append(
                StageBundle(
                    stages=s.stages,
                    body=new_inner,
                    policy=StagePolicy.BUFFERED,
                    buffer_count=_BUFFER_COUNT,
                    phase=phase,
                )
            )
        else:
            new_kouter_body.append(s)
    return SerialTile(
        axis=kouter.axis,
        body=Body(tuple(new_kouter_body)),
        kind=kouter.kind,
        unroll=kouter.unroll,
    )


def _has_stage_inner_reduce(body: Body) -> bool:
    for s in body.iter():
        if isinstance(s, SerialTile) and s.kind == "stage_inner" and s.is_reduce:
            return True
    return False


def _accums_independent_in(body: Body) -> bool:
    """For each reduce SerialTile inside ``body``, reject if any non-Accum
    stmt reads a sibling Accum's running value (online-softmax merge shape)."""
    for nested in body.iter():
        if isinstance(nested, SerialTile) and nested.is_reduce:
            for c in nested.body:
                if isinstance(c, Accum):
                    continue
                if any(isinstance(d, Accum) for d in nested.body.deps_of(c) if d is not None):
                    return False
    return True


def _make_phase_load_rewriter(staged_names: set[str], phase):
    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.input in staged_names:
            return Load(name=s.name, input=s.input, index=(phase, *s.index))
        return s

    return fn
