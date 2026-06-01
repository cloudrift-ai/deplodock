"""Register-tier software pipelining for the mma.sync K reduce — double-buffered
``ldmatrix`` prefetch (``plans/mma-register-pipeline.md``).

After ``005_lower_atom_tile`` emits the ``ldmatrix + mma.sync`` chain and
``010_split_register_axes`` replicates it per (M_r, N_r) cell, the K_i reduce
body is a flat ``[ldmatrix×(FM+FN), mma.sync×(FM·FN)]`` sequence — **load then
compute, serially**. The ``mma.sync`` chain can't begin until the operand
``ldmatrix`` loads land, and the next K-step's loads don't issue until the
current mma chain drains, so the tensor-core pipeline empties at every K-step
boundary (the ``barrier`` + ``short_scoreboard`` stall window).

cuBLAS/CUTLASS double-buffer the operand **register fragments**: they issue the
*next* K-step's ``ldmatrix`` into a second fragment buffer concurrently with the
*current* K-step's ``mma.sync``, so loads overlap compute and the mma pipeline
never drains. ``095_interleave_loads`` is the *scalar*-FMA analog (sinks
``Load``+``Assign`` clusters within one iteration) but it does **not** touch the
``LdmatrixLoad``/``MmaSyncPtx`` chain and does **not** double-buffer across
iterations — so register-tier prefetch genuinely does not exist for the mma.sync
path until this pass.

This is the structural mirror of ``tile/080_pipeline_stages`` (prologue / steady
/ epilogue peeling + ring-index rotation) applied one tier down — on the
``RegFragment`` operand chain instead of the smem ``StageBundle`` ring. The
accumulator ``c_frag`` stays single-buffered (it accumulates across all K); only
the ``a``/``b`` operand frags double-buffer, which is the register cost.

**The knob.** ``DEPLODOCK_REG_PIPELINE`` (BOOL, default off) gates the whole
feature. Register pipelining is *not* universally faster — it costs registers
(occupancy) and only pays when the mma chain is long enough to hide the prefetch
— so it is a measured autotune fork, never forced on. Default **off** (first
hint) keeps the greedy / DB-less path and every existing test byte-identical;
the autotuner forks on it (``True``/``False``) like ``PAD_SMEM`` /
``HOIST_COMPUTE`` and the goldens record whichever wins per shape. A pin
(``DEPLODOCK_REG_PIPELINE=1``) forces it for bring-up / A-B benching.

Run order: AFTER ``012_fuse_sibling_register_cells`` (so the per-cell frags are
concrete and sibling-cell Conds are already fused) and BEFORE
``020_place_inits`` / ``050_vectorize_loads`` / ``095_interleave_loads`` (which
operate on the post-peel chain shape). The ``LdmatrixLoad`` / ``MmaSyncPtx``
chain is untouched by ``095`` (it sinks only scalar ``Load``+``Assign``), so the
two passes don't interact.

Idempotence: keyed on the ``REG_PIPELINE`` knob — once stamped into
``op.knobs`` the pass skips on a second visit. M0 is an **identity no-op** for
both polarities (knob plumbing only); M1 peels the within-tile K_i reduce and
M2 spans the K_o smem-stage boundary.
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import SerialTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

# Default off (first hint): the greedy / DB-less path and every existing test
# stay byte-identical. The autotuner forks on (False, True) and measures both;
# a pin ``DEPLODOCK_REG_PIPELINE=1`` forces it on for bring-up / A-B benching.
REG_PIPELINE = Knob(
    "REG_PIPELINE",
    KnobType.BOOL,
    hints=(False, True),
    help=(
        "Software-pipeline the mma.sync K reduce: double-buffer the operand "
        "ldmatrix fragments so the next K-step's loads overlap the current "
        "step's mma.sync. Costs ~32 regs/thread (occupancy) — measured fork, "
        "off by default. 1 forces it on."
    ),
)


def rewrite(root: Node) -> list[TileOp] | None:
    if REG_PIPELINE.name in root.op.knobs:
        raise RuleSkipped("reg-pipeline already applied (idempotence via knob)")
    if not _has_mma_reduce(root.op.body):
        raise RuleSkipped("no mma.sync K reduce to pipeline (ldmatrix + mma.sync chain absent)")

    variants: list[TileOp] = []
    for polarity in REG_PIPELINE.narrow((False, True)):
        # M0: identity for both polarities — the body is unchanged; only the
        # knob is stamped so the autotuner enumerates the fork. M1/M2 will peel
        # the reduce when ``polarity`` is True.
        new_body = root.op.body
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, REG_PIPELINE.name: polarity}))
    if not variants:
        raise RuleSkipped("REG_PIPELINE env pin produced no matching variants")
    return variants


def _has_mma_reduce(body: Body) -> bool:
    """True iff ``body`` holds a ``SerialTile`` whose (possibly nested) body
    contains both a ``LdmatrixLoad`` and a ``MmaSyncPtx`` — the K reduce the
    mma.sync path emits. Walks every nested body so the reduce is found inside
    the K_o ring / prologue / epilogue structure ``080_pipeline_stages`` leaves."""
    for s in body:
        if isinstance(s, SerialTile) and _chain_in(s.body):
            return True
        for sub in s.nested():
            if _has_mma_reduce(sub):
                return True
    return False


def _chain_in(body: Body) -> bool:
    """True iff ``body`` (recursively) contains both an ``LdmatrixLoad`` and a
    ``MmaSyncPtx``."""
    has_ldm = False
    has_mma = False
    for s in body.iter():
        if isinstance(s, LdmatrixLoad):
            has_ldm = True
        elif isinstance(s, MmaSyncPtx):
            has_mma = True
        if has_ldm and has_mma:
            return True
    return False
