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
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx, RegFragment
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import SerialTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

# Suffix for the second operand-fragment buffer. The accumulator (``c`` role)
# is never aliased — it accumulates in place across all K, so it stays single-
# buffered. Only the ``a`` / ``b`` operand frags double-buffer.
_BUF1 = "__rp1"

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
        # ``True`` peels the K_i reduce into a double-buffered prologue / steady
        # / epilogue (M1); ``False`` leaves the body untouched (byte-identical
        # to the pre-pass path). The accumulator stays single-buffered.
        new_body = _pipeline(root.op.body) if polarity else root.op.body
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, REG_PIPELINE.name: polarity}))
    if not variants:
        raise RuleSkipped("REG_PIPELINE env pin produced no matching variants")
    return variants


# ---------------------------------------------------------------------------
# M1 — within-tile operand-fragment double-buffer
# ---------------------------------------------------------------------------


def _pipeline(body: Body) -> Body:
    """Peel every double-bufferable mma.sync K_i reduce in ``body`` and declare
    the second operand-fragment buffer.

    Two scopes are touched, both reached by one recursive walk:

    - the ``RegFragment`` operand decls (``a`` / ``b`` roles) at the warp-tile
      body head get a ``_BUF1``-suffixed twin emitted right after them, and
    - each reduce ``SerialTile`` (flat body of ``LdmatrixLoad`` + ``MmaSyncPtx``,
      static extent ≥ 2) is unrolled into the prologue / steady / epilogue
      double-buffer shape.

    The operand frags are shared across every reduce site (the main-loop K_i
    reduce and the ``080_pipeline_stages`` epilogue copies all reuse the same
    ``in*_frag_*`` names), so the alias set is collected once over the whole
    body before any rewrite."""
    operand_frags = _collect_operand_frags(body)
    if not operand_frags:
        return body
    alias = {f: f + _BUF1 for f in operand_frags}
    return _rewrite_body(body, alias)


def _collect_operand_frags(body: Body) -> frozenset[str]:
    """Every operand-fragment SSA name loaded by an ``LdmatrixLoad`` inside a
    *peelable* reduce site (static extent ≥ 2). Sites with extent < 2 can't
    double-buffer within the tile (that's the cross-K_o job, M2) and don't
    contribute aliases."""
    names: set[str] = set()
    for site in _reduce_sites(body):
        for s in site.body:
            if isinstance(s, LdmatrixLoad):
                names.add(s.frag)
    return frozenset(names)


def _reduce_sites(body: Body):
    """Yield every peelable reduce ``SerialTile``: a flat body of only
    ``LdmatrixLoad`` + ``MmaSyncPtx`` (≥ 1 of each), static extent ≥ 2."""
    for s in body.iter():
        if isinstance(s, SerialTile) and _is_peelable_reduce(s):
            yield s


def _is_peelable_reduce(s: SerialTile) -> bool:
    if not s.axis.extent.is_static or s.axis.extent.as_static() < 2:
        return False
    has_ldm = has_mma = False
    for c in s.body:
        if isinstance(c, LdmatrixLoad):
            has_ldm = True
        elif isinstance(c, MmaSyncPtx):
            has_mma = True
        else:
            return False  # foreign stmt — not the clean ld+mma reduce shape
    return has_ldm and has_mma


def _rewrite_body(body: Body, alias: dict[str, str]) -> Body:
    """Recursively rewrite ``body``: inject ``_BUF1`` twin decls after operand
    ``RegFragment``s, peel reduce sites, recurse into everything else."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, RegFragment) and s.name in alias:
            out.append(s)
            out.append(RegFragment(name=alias[s.name], role=s.role, shape=s.shape, dtype=s.dtype))
            continue
        if isinstance(s, SerialTile) and _is_peelable_reduce(s):
            out.extend(_peel(s, alias))
            continue
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_rewrite_body(b, alias) for b in nested))
        out.append(s)
    return Body(tuple(out))


def _peel(site: SerialTile, alias: dict[str, str]) -> tuple[Stmt, ...]:
    """Unroll one reduce ``SerialTile`` into the double-buffered schedule::

        ldmatrix step 0          → buf[0]                 # prologue
        for s in 0 .. N-1:
            ldmatrix step s+1    → buf[(s+1) % 2]         # prefetch (if s+1 < N)
            mma      step s       reads buf[s % 2]         # compute
        # the last mma falls out of the loop tail (s = N-1, no prefetch)

    The ``ldmatrix`` of step ``s`` substitutes the K_i axis var → literal ``s``
    (concrete K offset) and renames its destination frag to the buffer for
    ``s % 2``; each ``mma`` renames its ``a``/``b`` operands to the matching
    buffer (``c`` accumulator unchanged — it is not in ``alias``)."""
    n = site.axis.extent.as_static()
    kvar = site.axis.name
    lds = tuple(c for c in site.body if isinstance(c, LdmatrixLoad))
    mmas = tuple(c for c in site.body if isinstance(c, MmaSyncPtx))

    def ld_step(step: int, phase: int) -> tuple[Stmt, ...]:
        rename = (lambda nm: alias.get(nm, nm)) if phase else (lambda nm: nm)
        sigma = Sigma({kvar: Literal(step, "int")})
        return tuple(ld.rewrite(rename, sigma) for ld in lds)

    def mma_step(phase: int) -> tuple[Stmt, ...]:
        # ``a``/``b`` frags live in ``alias`` → buffer-renamed; ``c`` does not →
        # untouched. mma carries no Exprs, so the empty Sigma is a no-op.
        rename = (lambda nm: alias.get(nm, nm)) if phase else (lambda nm: nm)
        sigma = Sigma({})
        return tuple(m.rewrite(rename, sigma) for m in mmas)

    out: list[Stmt] = list(ld_step(0, 0))  # prologue: step 0 → buf0
    for s in range(n):
        if s + 1 < n:
            out.extend(ld_step(s + 1, (s + 1) % 2))  # prefetch next step
        out.extend(mma_step(s % 2))  # compute current step
    return tuple(out)


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
