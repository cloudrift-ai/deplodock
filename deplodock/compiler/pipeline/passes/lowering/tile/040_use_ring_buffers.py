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

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import SerialTile, StageBundle, StagePolicy, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

# Ring-buffer depth fork. 2 = classic double-buffer (compute current slot while
# the next async copy fills the other). 3-4 = multi-stage pipeline (CUTLASS-style)
# — only viable when the per-stage smem footprint × N fits ``max_dynamic_smem``;
# variants that don't fit are pruned at fork time. Same value flows to
# ``StageBundle.buffer_count`` and downstream ``pipeline_stages`` reads it from
# the bundle (no separate PIPE knob — ring depth and pipeline depth are coupled
# by construction in CUTLASS-style multistage matmul).
BUFCNT = Knob(
    "BUFCNT",
    KnobType.INT,
    hints=(2, 3, 4),
    help="Ring-buffer depth (and pipeline stages) for BUFFERED/ASYNC/TMA staged K-outer loops",
)


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    body = root.op.body
    if BUFCNT.name in root.op.knobs:
        raise RuleSkipped("ring buffers already applied (idempotence via knob)")
    if any(isinstance(s, StageBundle) and s.policy != StagePolicy.SYNC for s in body.iter()):
        raise RuleSkipped("double-buffer already applied (non-SYNC bundle present)")

    candidates = BUFCNT.narrow(BUFCNT.hints)
    # Pin detection: if the user pinned ``DEPLODOCK_BUFCNT=N`` (or
    # ``DEPLODOCK_KNOBS=BUFCNT=N``), a silent fall-through to SYNC when
    # ``N`` doesn't fit the smem cap drops every downstream transport
    # promotion (060_use_async_copy, 050_use_tma, 080_pipeline_stages
    # all gate on a BUFFERED/ASYNC/TMA bundle) — kernel runs as plain
    # SYNC with no staging benefit, looks ~50 % slower than the auto
    # variant. Surface the constraint instead of swallowing it.
    pinned = config.knob_raw(BUFCNT.name) is not None
    variants: list[TileOp] = []
    fail_reasons: list[str] = []
    for buf_count in candidates:
        new_body, changed = _walk(body, smem_budget=ctx.max_dynamic_smem, buffer_count=buf_count)
        if not changed:
            fail_reasons.append(
                f"BUFCNT={buf_count} not promotable (smem cap {ctx.max_dynamic_smem} B / K_o extent / SYNC bundle eligibility)"
            )
            continue
        variants.append(
            TileOp(
                body=new_body,
                name=root.op.name,
                knobs={**root.op.knobs, BUFCNT.name: buf_count},
            )
        )
    if not variants:
        msg = "; ".join(fail_reasons) if fail_reasons else "no candidate fit"
        if pinned:
            raise ValueError(f"DEPLODOCK_BUFCNT={candidates[0]} pinned but cannot fire: {msg}")
        raise RuleSkipped(f"no K-outer StageBundle fits any BUFCNT candidate ({msg})")
    return variants


def _walk(body: Body, *, smem_budget: int, buffer_count: int) -> tuple[Body, bool]:
    """Recursive descent: visit every wrapper looking for ``SerialTile(serial_outer)``
    whose body holds an eligible SYNC bundle — promote and stop descending into
    the promoted subtree (no further serial_outer is expected below the bundle
    for the matmul shape)."""
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            promoted = _maybe_promote_kouter(s, smem_budget=smem_budget, buffer_count=buffer_count)
            if promoted is not None:
                out.append(promoted)
                changed = True
                continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b, smem_budget=smem_budget, buffer_count=buffer_count)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _maybe_promote_kouter(kouter: SerialTile, *, smem_budget: int, buffer_count: int) -> SerialTile | None:
    # Ring buffers need ≥``buffer_count`` K iterations to prologue + steady-state.
    # Symbolic K can be either — defer the promotion (no per-iter overlap) rather
    # than speculatively allocating a multi-buffer slab.
    if not kouter.axis.extent.is_static or kouter.axis.extent.as_static() < buffer_count:
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
        # SYNC bundle: smem_bytes has no buffer factor; we'll apply ×buffer_count.
        total_bytes += s.smem_bytes
    if not promote_ids:
        return None
    if buffer_count * total_bytes > smem_budget:
        return None

    phase = Var(kouter.axis.name) % Literal(buffer_count, "int")
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
                    buffer_count=buffer_count,
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
