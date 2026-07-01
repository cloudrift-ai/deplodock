"""The **single home for every ``Knob`` declaration** — the tunable / forkable fork points the
pipeline offers.

**INVARIANT: every ``Knob`` instance is declared here, and nowhere else.** :mod:`~deplodock.compiler.pipeline.knob`
owns the ``Knob`` *descriptor* (the dataclass), the registry, and the featurizers; this module owns
the concrete *declarations*. A rule that decides a knob imports it from here (``from
deplodock.compiler.pipeline.forks import VECTORIZE_LOADS``) rather than declaring its own — so the whole
tunable surface is visible in one file, and there is one place to look for any knob's type / hints /
``off`` value. ``knob.registry()`` still discovers them via ``knob._walk_modules`` (any package module
with module-level ``Knob`` attributes is walked, and this module is imported at pipeline startup by the
rules that consume its knobs). When adding a knob, declare it here and import it into the owning rule.

Two groups:

- **Schedule codec knobs** (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) — the tile-lowering schedule
  fork points that spell the ir schedule codecs (:mod:`deplodock.compiler.ir.schedule`). Decided in
  ``lowering/tile/010_recognize`` (the ``_schedule`` helper) and materialized in
  ``lowering/kernel/010_materialize``. Each is the **ephemeral** codec spelling: it resolves into a
  schedule slice (``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec``) and rides on ``TileOp.knobs``
  so the learned prior featurizes / tunes the decision. ``off=""`` (the conservative serial / per-cell /
  gmem-direct / uniform default) is auto-stamped on kernels the pass doesn't schedule.
- **Kernel-lowering policy knobs** (``VECTORIZE_LOADS`` / ``INTERLEAVE_LOADS``) — boolean codegen
  policies recorded on the kernel op (idempotence + env override), on by default and not search
  dimensions (``hints=(True,)``).
"""

from __future__ import annotations

from deplodock.compiler.pipeline.knob import Knob, KnobType

# --- Schedule codec knobs ---------------------------------------------------

# The reduce-axis partition codec. ``off=""`` = the scalar serial fold.
REDUCE = Knob(
    "REDUCE",
    KnobType.STR,
    help="Reduce-axis partition codec (g<n> cta / b<n> coop / r<n> reg; empty=serial). "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# The free-axis output tile — the **unified output-fragment** knob. A contraction's output tile is
# *either* the scalar register sub-tile (``n<N>[x<M>]`` parallel thread-tile / ``f<fn>[x<fm>]``
# register sub-tile) *or* the tensor-core warp mma tile (``a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>``),
# never both. The value self-discriminates: an ``a:<atom>`` token selects the warp fragment (see
# ``schedule.is_warp_codec``); otherwise the scalar fragment. Only a ``CONTRACTION`` tiles its output
# today; ``off=""`` auto-stamps everything else. The codec is the sole on-dict spelling — the
# learned-prior featurizer (``mma_atom`` / ``is_warp`` / ``_free_slots`` / ``tile_signature``) parses
# it directly (no legacy ``WM``/``WN``/``MMA`` keys).
TILE = Knob(
    "TILE",
    KnobType.STR,
    help="Output-fragment codec — scalar tile (n<N>[x<M>]/f<fn>[x<fm>]) OR warp mma tile "
    "(a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>, selected by the a:<atom> token); empty=per-cell. "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Operand staging — the reused gmem operands (matmul A/B, a fused prologue's read) ride a
# shared-memory slab + double-buffered producer (``sync`` plain copy / ``cp.async`` / ``tma``) over
# the serial reduce loop, instead of the gmem-direct register baseline. Resolved into the schedule's
# :class:`Stage` (``None`` = gmem-direct). Pin-only this cut (the prior auto-fork is a follow-up, as
# ``TILE``'s is). Composes with both fragments of the unified ``TILE`` knob.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/sync|cp|tma[/ring][/p<reg_depth>]; empty=gmem-direct). "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Warp specialization — the worker-mapping sibling of ``REDUCE``/``TILE``/``STAGE`` and ORTHOGONAL to
# all three: the pipeline (what's staged, the mma tile, the reduce partition) is fixed by those pins;
# ``WSPEC`` only splits the warps that run it into roles (``p<np>`` producer warps drive the ``STAGE``
# load half; the compute warps stay on the mma). ``off=""`` is uniform SIMT (every warp does both
# halves). Resolved into the schedule's :class:`WarpSpec` (``None`` = uniform). **Pin-only this cut,
# and gated** on a warp ``TILE`` + a ``STAGE``; the producer/consumer materialization is a follow-up.
WSPEC = Knob(
    "WSPEC",
    KnobType.STR,
    help="Warp-specialization codec — role→warp split over the fixed pipeline "
    "(p<np> producer[:q<window>], s<ns> sfu, …; compute warps implicit = TilePlan.units; empty=uniform SIMT). "
    "Decided in lowering/tile/020_schedule; materialization reserved (TODO).",
    off="",
)


# --- Kernel-lowering policy knobs -------------------------------------------
#
# Boolean codegen policies recorded on the kernel op — on by default, not search dimensions
# (``hints=(True,)``); a rule records its knob for idempotence and honors the ``DEPLODOCK_<NAME>``
# env override. Consumed by ``lowering/kernel/050_vectorize_loads`` / ``095_interleave_loads``.

VECTORIZE_LOADS = Knob(
    "VECTORIZE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Fold runs of consecutive scalar Loads into one wide vector Load (float4 / __half2).",
    off=False,
)

INTERLEAVE_LOADS = Knob(
    "INTERLEAVE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Sink each Load to just before its first SSA-consumer in flat compute blocks.",
    off=False,
)
