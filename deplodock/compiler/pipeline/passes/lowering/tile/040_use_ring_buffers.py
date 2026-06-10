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
from deplodock.compiler.pipeline.knob import Knob, KnobType, is_warp

PATTERN = [Pattern("root", TileOp)]

# Ring-buffer depth fork. 2 = classic double-buffer (compute current slot while
# the next async copy fills the other). 3-4 = multi-stage pipeline (CUTLASS-style)
# — only viable when the per-stage smem footprint × N fits ``max_dynamic_smem``;
# variants that don't fit are pruned at fork time. Same value flows to
# ``StageBundle.buffer_count`` and downstream ``pipeline_stages`` reads it from
# the bundle (no separate PIPE knob — ring depth and pipeline depth are coupled
# by construction in CUTLASS-style multistage matmul).
BUFFER_COUNT = Knob(
    "RING",
    KnobType.INT,
    hints=(2, 3, 4),
    help="Ring-buffer depth (and pipeline stages) for BUFFERED/ASYNC/TMA staged K-outer loops",
    aliases=("BUFFER_COUNT",),
    off=1,  # single buffer = no ring (the value 040 already stamps when no ring fits)
)


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    body = root.op.body
    if BUFFER_COUNT.name in root.op.knobs:
        raise RuleSkipped("ring buffers already applied (idempotence via knob)")
    if any(isinstance(s, StageBundle) and s.policy != StagePolicy.SYNC for s in body.iter()):
        raise RuleSkipped("double-buffer already applied (non-SYNC bundle present)")

    candidates = BUFFER_COUNT.narrow(BUFFER_COUNT.hints)
    # Pin detection: if the user pinned ``DEPLODOCK_BUFFER_COUNT=N`` (or
    # ``DEPLODOCK_KNOBS=BUFFER_COUNT=N``), a silent fall-through to SYNC when
    # ``N`` doesn't fit the smem cap drops every downstream transport
    # promotion (060_use_async_copy, 050_use_tma, 080_pipeline_stages
    # all gate on a BUFFERED/ASYNC/TMA bundle) — kernel runs as plain
    # SYNC with no staging benefit, looks ~50 % slower than the auto
    # variant. Surface the constraint instead of swallowing it.
    pinned = BUFFER_COUNT.raw() is not None
    # Real per-buffer element bytes, keyed by gmem source name. At this tile-stage
    # ``Source.dtype`` is unstamped (``030_stamp_types`` is a downstream kernel
    # pass), so ``Source.smem_bytes`` falls back to the fp32-assuming
    # ``BYTES_PER_ELEM=4`` and 2×-over-counts fp16 slabs — which wrongly prunes
    # BUFFER_COUNT 3-4 for fp16 matmul tiles that actually fit (e.g. a 64×256 fp16
    # tile is 20 KB/stage real, fits depth 4 at 80 KB, but the over-count reports
    # 40 KB/stage → 160 KB and rejects). Pull the true dtype off the TileOp's input
    # tensors so the budget check matches the materializer's real allocation.
    input_nbytes = {buf: t.dtype.nbytes for buf, t in root.op.inputs.items() if getattr(t, "dtype", None) is not None}
    variants: list[tuple[int, TileOp]] = []
    fail_reasons: list[str] = []
    for buf_count in candidates:
        new_body, changed = _walk(body, smem_budget=ctx.max_dynamic_smem, buffer_count=buf_count, input_nbytes=input_nbytes)
        if not changed:
            fail_reasons.append(
                f"BUFFER_COUNT={buf_count} not promotable (smem cap {ctx.max_dynamic_smem} B / K_o extent / SYNC bundle eligibility)"
            )
            continue
        variants.append(
            (
                buf_count,
                TileOp(
                    body=new_body,
                    name=root.op.name,
                    knobs={**root.op.knobs, BUFFER_COUNT.name: buf_count},
                ),
            )
        )
    if not variants:
        msg = "; ".join(fail_reasons) if fail_reasons else "no candidate fit"
        if pinned:
            raise ValueError(f"{BUFFER_COUNT.env}={candidates[0]} pinned but cannot fire: {msg}")
        # No buffered ring fits (smem cap / K-outer extent / no eligible SYNC
        # bundle). Record the declined decision as BUFFER_COUNT=1 (single buffer
        # = no ring) on the unchanged SYNC body so the realized config carries a
        # uniform knob set — the knob is metadata for the prior / cache key and
        # never touches the SYNC bundle. Downstream transport passes (050/060/080)
        # then see SYNC and likewise record their off decisions.
        return [TileOp(body=body, name=root.op.name, knobs={**root.op.knobs, BUFFER_COUNT.name: 1})]

    # Occupancy-optimal ordering for the GREEDY default. ``GreedySearch`` keeps
    # only option-0 of a rule's fork and drops the rest (it does NOT re-score the
    # buffer variants), so the *order* here is what greedy picks at an untuned
    # site. A deeper ring hides the TMA/cp.async load latency, but only pays when
    # the deeper slab still leaves ≥ 2 CTA-blocks resident on one SM — past that
    # the kernel drops to 1 block/SM and runs slower (measured 2048² fp16: 128×128
    # depth 3 = 113 µs at 2 blocks vs depth 4 = 141 µs at 1 block). So front-load
    # the deepest depth that keeps 2 blocks; if none do (a fat tile where even
    # depth 2 is already 1 block), keep the shallowest first to preserve occupancy.
    # The autotuner still sees every variant — this only fixes the greedy prior.
    #
    # Gate to single-staged-bundle kernels — a pure GEMM, where the ring slab IS
    # the kernel's entire dynamic-smem footprint, so the ``keeps_two`` test below
    # is exact. A fused multi-matmul kernel (SDPA chains QK then P@V → two
    # ``StageBundle``s) also allocates an intermediate cross-bundle workspace
    # whose smem *dominates* the materialized footprint (measured: per-stage ring
    # ≈ 2 KB but the depth-3 kernel materializes ≈ 104 KB of workspace + rings).
    # No ring-byte budget can predict that, so front-loading a deep ring there can
    # pick a depth whose materialized smem overflows the cap — and greedy keeps
    # only option-0 with no fallback, so the deterministic compile hard-errors
    # (``100_materialize_tile rejected its only lowering``). Multi-bundle kernels
    # keep the shallow-first default (the pre-occupancy order, which always fit
    # downstream); the autotuner still explores their deeper rings (it tolerates a
    # variant that fails ``validate``, unlike the no-fallback greedy default).
    # Warp-tier mma.sync override: depth-2 unlocks ``WARP_SPECIALIZE`` (085
    # requires ``pipeline_depth == 2``), and WS + depth-2 is the measured greedy
    # optimum for the fp16 tensor-core path (2048²: 94 µs / 1.05× cuBLAS vs
    # 115 µs for the deeper-ring no-WS kernel; depth-2 *without* WS is no worse
    # than depth-4). So front-load depth-2 for the greedy default on these
    # tiles, ahead of the occupancy ordering below (which front-loads the
    # deepest 2-block ring — right for scalar / 128×128, but here it picks a
    # depth that disqualifies the better WS kernel). The autotuner still sees
    # every depth.
    if is_warp(root.op.knobs):
        variants.sort(key=lambda bv: (bv[0] != 2, bv[0]))
        return [v for _, v in variants]

    per_stage = _kouter_sync_bytes(body, input_nbytes)
    n_bundles = sum(1 for s in body.iter() if isinstance(s, StageBundle))
    if per_stage > 0 and n_bundles == 1:

        def _occ_key(bc: int) -> tuple[int, int]:
            keeps_two = 2 * bc * per_stage <= ctx.max_dynamic_smem
            # keeps-2-blocks first (deepest among them); then 1-block (shallowest).
            return (0, -bc) if keeps_two else (1, bc)

        variants.sort(key=lambda bv: _occ_key(bv[0]))
    return [v for _, v in variants]


def _kouter_sync_bytes(body: Body, input_nbytes: dict[str, int]) -> int:
    """Single-slot real-dtype smem bytes of the first ``SerialTile(serial_outer)``
    SYNC bundle eligible for promotion — the per-stage footprint a ring multiplies
    by ``buffer_count``. Used to order the buffer-count variants by occupancy. 0
    when no eligible bundle is found (caller skips the reorder)."""
    for s in body.iter():
        if isinstance(s, StageBundle) and s.policy == StagePolicy.SYNC:
            return _bundle_real_bytes(s, input_nbytes)
    return 0


def _walk(body: Body, *, smem_budget: int, buffer_count: int, input_nbytes: dict[str, int]) -> tuple[Body, bool]:
    """Recursive descent: visit every wrapper looking for ``SerialTile(serial_outer)``
    whose body holds an eligible SYNC bundle — promote and stop descending into
    the promoted subtree (no further serial_outer is expected below the bundle
    for the matmul shape)."""
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            promoted = _maybe_promote_kouter(s, smem_budget=smem_budget, buffer_count=buffer_count, input_nbytes=input_nbytes)
            if promoted is not None:
                out.append(promoted)
                changed = True
                continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b, smem_budget=smem_budget, buffer_count=buffer_count, input_nbytes=input_nbytes)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _maybe_promote_kouter(kouter: SerialTile, *, smem_budget: int, buffer_count: int, input_nbytes: dict[str, int]) -> SerialTile | None:
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
        # Use the real per-source dtype (off the TileOp inputs) instead of the
        # unstamped ``Source.smem_bytes`` fp32 fallback — see ``input_nbytes``
        # note in ``rewrite``. Falls back to the property when a source's buf
        # isn't a known input (intermediate / handwritten test fixture).
        total_bytes += _bundle_real_bytes(s, input_nbytes)
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
                    sources=s.sources,
                    body=new_inner,
                    compute=s.compute,
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


def _bundle_real_bytes(bundle: StageBundle, input_nbytes: dict[str, int]) -> int:
    """Single-slot smem bytes of ``bundle``, using the real per-source dtype
    bytes from ``input_nbytes`` (keyed by gmem source name) where known. Each
    source's element count is ``prod(alloc_extents)``; multiply by the true
    dtype byte width (2 for fp16) instead of the unstamped fp32 fallback that
    ``Source.smem_bytes`` uses pre-``030_stamp_types``. Sources whose ``buf``
    isn't a known input fall back to the property (still fp32-safe)."""
    total = 0
    for src in bundle.sources:
        nbytes = input_nbytes.get(src.buf)
        if nbytes is None:
            total += src.smem_bytes
            continue
        count = 1
        for e in src.alloc_extents:
            count *= e
        total += count * nbytes
    return total


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
