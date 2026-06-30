"""Knob vocabulary for the move composer — aliased to the legacy tile knobs.

Each composer knob is a *move parameter* (the ``TileMap`` move on a map axis
contributes a thread-tile factor + a register-tile factor; ``TileSerial`` a
K-chunk; ``Tensorize`` a warp/atom geometry). Rather than mint a fresh schema,
the composer **reuses the legacy ``BM/BN/BK/...`` knob objects** (defined
below): the move parameters have the SAME arithmetic role as the
legacy tile dimensions (legacy ``BN`` IS "CTA thread width along N" = the
composer's N thread-tile extent; ``FM/FN`` the register cells; ``WM/WN`` the warp
counts; ``BK/SPLITK/BR/FK/MMA`` identical), so aliasing keeps the tune DB / learned
prior / golden YAMLs / pinned-knob tests (``EMMY_BK=…`` etc.) valid across the
cutover. The greenfield Python names are retained as aliases so the move code
reads in move terms; the stamped ``op.knobs`` key is the legacy string name.

Legacy tier schema: scalar ``{BN,BM,FM,FN,FK,BK,SPLITK,BR}``,
warp ``{WN,WM,FM,FN,BK,SPLITK,MMA}`` — the composer stamps the same per tier. The
register knobs ``FM/FN`` and ``BK`` are tier-shared (a kernel is scalar XOR warp),
so the scalar map-reg and warp-reg aliases point at the one ``FM``/``FN``/``BK``.
"""

from __future__ import annotations

from emmy.compiler.pipeline.knob import Knob, KnobType

# --- The knob SCHEMA (relocated from the deleted ``_enumeration.py``). The move
# composer + tune DB + learned prior + golden YAMLs all key on these legacy
# ``BN/BM/BK/...`` string names; the greenfield ``MAP_*``/``RED_*``/``TC_*`` aliases
# below read in move terms. The enumeration LOGIC that used to live beside these
# (cartesian generator, priority sort) is gone — invalid under the move architecture. ---

_TUNE_AXIS_CHOICES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
_TUNE_WARP_AXIS_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26, 28, 32, 40, 48, 64, 96, 128)


def _mma_features(mma: object) -> dict[str, float]:
    """Learned-prior featurizer for the ``MMA`` knob: expand an atom kind into
    physical cell/dtype properties via ``ATOM_REGISTRY`` (lazy import — avoids an
    import cycle through the tile IR)."""
    from emmy.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get(str(mma))
    if atom is None:
        return {"MMA_tier": 0.0}
    m, n, k = atom.shape
    return {
        "MMA_tier": 1.0,
        "MMA_atom_m": float(m),
        "MMA_atom_n": float(n),
        "MMA_atom_k": float(k),
        "MMA_group_size": float(atom.group_size),
        "MMA_a_bits": float(atom.operand_dtype("a").nbytes * 8),
        "MMA_acc_bits": float(atom.operand_dtype("c").nbytes * 8),
    }


# Scalar-tier THREAD-binding knobs (``off=0`` = warp-tier OFF sentinel).
BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (the N free-axis thread tile)", off=0)
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (the M free-axis thread tile)", off=0)
# The reduce-decomposition knobs ``BK``/``FK``/``SPLITK``/``BR`` are gone — folded into
# the native ``REDUCE@<axis>`` family (``_families``), one ``"s/f/c/t"`` value per reduce
# axis. The candidate menus below stay (the move offers enumerate them); the legacy env
# spellings survive only through ``_knob_legacy`` ingest.
# Warp-tier WARP-binding knobs (``off=0`` = scalar-tier OFF sentinel).
WN = Knob("WN", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA innermost WARP count along the N output axis", off=0)
WM = Knob("WM", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA outer WARP count along the M output axis", off=0)
MMA = Knob(
    "MMA",
    KnobType.STR,
    hints=(),
    help="Warp-tier MMA control: 0 = scalar-only; 1/true = auto-enumerate; an atom kind pins it",
    aliases=("ATOM_KIND",),
    features=_mma_features,
    off="0",
)
# Tier-shared free-axis register knobs (same arithmetic role in both tiers).
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the N (output) axis")
# Staging knob (the ``stage`` move, ``120_stage``): a bitmask over the ranked
# stageable read-sites (``Edge``s) of a reduce kernel — char ``i`` selects ranked
# candidate ``i``. The fork writes the chosen ``Edge``s into ``Schedule.staged``
# (the source of truth ``assemble`` reads); the mask string is the variant
# identity the perf DB / learned prior key on. ``off=""`` = nothing stageable
# (pointwise / no reuse / no K-tower) — an explicit "decided: unused".
STAGE = Knob("STAGE", KnobType.BINMASK, help="Bitmask over ranked stageable read-sites (char i = candidate i)", off="")

# Transport knob (the ``promote_transport`` move, ``130_transport``): a BOOL over a
# fully-staged warp-tier matmul. ``True`` promotes the staged ``Edge``s from
# ``Transport.SYNC`` to ``Transport.TMA`` (``assembly/_slab`` then synthesizes the
# double-buffered ``cp.async.bulk.tensor`` ring + per-source swizzle; ``assembly/020_peel``
# software-pipelines it); ``False`` keeps the SYNC cooperative load. Hints
# ``(True, False)`` so TMA is preferred first on sm_90+; arch-gated + eligibility-gated
# by the fork. ``off=False`` = decided-unused (scalar / ineligible / non-sm90).
TMA = Knob(
    "TMA",
    KnobType.BOOL,
    hints=(True, False),
    help="Promote staged warp-tier matmul read-sites to TMA (cp.async.bulk.tensor). 0 = keep SYNC staging.",
    off=False,
)

# Chain knob (the shared-axis reduce_decomp, ``_build.chain_build``): a BOOL over a streaming
# ``MONOID(SEMIRING)`` flash nest. ``True`` restructures it into the FA-2 shared-score
# form — the P@V output ``d`` rides a register vector ``O[d]`` inside the stream, the
# score is computed once per KV step (the INLINE score edge, shared across ``d``), and
# the twisted carrier splits into a scalar stats carrier + a register-tiled accumulation
# carrier (the two SEMIRING cells the tensor-core tier atomizes). ``False`` keeps the
# scalar streaming nest (the score recomputed per ``d`` block). ``off=False`` = the
# scalar streaming default; greedy picks it until the search-fork integration (Phase 6).
CHAIN = Knob(
    "CHAIN",
    KnobType.BOOL,
    hints=(False, True),
    help="Restructure a streaming flash into the FA-2 shared-score form (register O[d] + INLINE score). 0 = scalar streaming nest.",
    off=False,
)


# Candidate menus, shared by the move ``offers``. Thread-tile extents are the
# per-CTA thread fan-out per axis; register-tile factors are cells-per-thread.
THREAD_CHOICES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
REG_CHOICES: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26, 28, 32, 40, 48, 64, 96, 128)

# Resource ceilings (per CTA). Mirror the legacy gates: ``BM·BN`` ≤ threads,
# ``FM·FN`` ≤ cells-per-thread.
MAX_THREADS_PER_CTA = 1024
MAX_CELLS_PER_THREAD = 128

# Map-tile knobs (the ``TileMap`` move). ``N`` is the innermost free axis, ``M``
# the next-out one. Legacy ``BN``/``BM`` ARE the per-CTA THREAD widths; ``FN``/``FM``
# the register cells per thread.
MAP_N_THREAD = BN
MAP_M_THREAD = BM

# Reduce-decomposition candidate menus (the ``REDUCE@<axis>`` family — the ``TileSerial``
# / cooperative / split-K levers). ``serial`` (legacy ``BK``) is the staged K-chunk,
# ``fold`` (``FK``) the register strip-mine, ``cta`` (``SPLITK``) the cross-CTA split,
# ``coop`` (``BR``) the cooperative-thread partition. The values now live in one
# ``"s/f/c/t"`` ``REDUCE@<axis>`` knob per reduce axis (``_families``); these are the
# per-field menus the move offers enumerate.
BK_CHOICES: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)
FK_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
SPLITK_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
BR_CHOICES: tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1)

# Tensor-core (warp-tier MMA) knobs — the ``Tensorize`` move. ``MMA`` is the atom
# kind ("0" = scalar tier, no tensorize); ``WM``/``WN`` the per-CTA warp counts;
# ``FM``/``FN`` register cells per warp; the K chunk is ``REDUCE@<axis>.serial``.
WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
TC_REG_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
