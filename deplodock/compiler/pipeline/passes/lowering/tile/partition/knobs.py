"""Knob vocabulary for the move composer — aliased to the legacy tile knobs.

Each composer knob is a *move parameter* (the ``TileMap`` move on a map axis
contributes a thread-tile factor + a register-tile factor; ``TileSerial`` a
K-chunk; ``Tensorize`` a warp/atom geometry). Rather than mint a fresh schema,
the composer **reuses the legacy ``BM/BN/BK/...`` knob objects** (defined in
``_enumeration.py``): the move parameters have the SAME arithmetic role as the
legacy tile dimensions (legacy ``BN`` IS "CTA thread width along N" = the
composer's N thread-tile extent; ``FM/FN`` the register cells; ``WM/WN`` the warp
counts; ``BK/SPLITK/BR/FK/MMA`` identical), so aliasing keeps the tune DB / learned
prior / golden YAMLs / pinned-knob tests (``DEPLODOCK_BK=…`` etc.) valid across the
cutover. The greenfield Python names are retained as aliases so the move code
reads in move terms; the stamped ``op.knobs`` key is the legacy string name.

Legacy tier schema (``_enumeration.py``): scalar ``{BN,BM,FM,FN,FK,BK,SPLITK,BR}``,
warp ``{WN,WM,FM,FN,BK,SPLITK,MMA}`` — the composer stamps the same per tier. The
register knobs ``FM/FN`` and ``BK`` are tier-shared (a kernel is scalar XOR warp),
so the scalar map-reg and warp-reg aliases point at the one ``FM``/``FN``/``BK``.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.knob import Knob, KnobType

# --- The knob SCHEMA (relocated from the deleted ``_enumeration.py``). The move
# composer + tune DB + learned prior + golden YAMLs all key on these legacy
# ``BN/BM/BK/...`` string names; the greenfield ``MAP_*``/``RED_*``/``TC_*`` aliases
# below read in move terms. The enumeration LOGIC that used to live beside these
# (cartesian generator, priority sort) is gone — invalid under the move architecture. ---

_TUNE_AXIS_CHOICES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
_TUNE_WARP_AXIS_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26, 28, 32, 40, 48, 64, 96, 128)
_BK_CANDIDATES: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)
_SPLITK_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
_BR_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _mma_features(mma: object) -> dict[str, float]:
    """Learned-prior featurizer for the ``MMA`` knob: expand an atom kind into
    physical cell/dtype properties via ``ATOM_REGISTRY`` (lazy import — avoids an
    import cycle through the tile IR)."""
    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

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
BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)", off=0)
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)", off=0)
BR = Knob("BR", KnobType.INT, hints=_BR_CANDIDATES, help="Cooperative-K thread count (1 = pure serial chunked reduce)", off=0)
# Warp-tier WARP-binding knobs (``off=0`` = scalar-tier OFF sentinel).
WN = Knob("WN", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA innermost WARP count along matmul output N", off=0)
WM = Knob("WM", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA outer WARP count along matmul output M", off=0)
MMA = Knob(
    "MMA",
    KnobType.STR,
    hints=(),
    help="Warp-tier MMA control: 0 = scalar-only; 1/true = auto-enumerate; an atom kind pins it",
    aliases=("ATOM_KIND",),
    features=_mma_features,
    off="0",
)
# Tier-shared knobs (same arithmetic role in both tiers).
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the matmul N (output) axis")
FK = Knob(
    "FK", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread independent accumulators along the reduce (K) axis (1 = single)", off=0
)
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")


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
MAP_N_REG = FN
MAP_M_THREAD = BM
MAP_M_REG = FM

# Reduce-tile knobs (the ``TileSerial`` move): ``BK`` is the staged K-chunk (inner
# serial loop trip count); ``FK`` strip-mines the K axis into independent
# accumulators (``RegisterTile(reduce=True)``); ``SPLITK`` is the cross-CTA split.
BK_CHOICES: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)
FK_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
SPLITK_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
RED_BK = BK
RED_FK = FK
RED_SPLITK = SPLITK

# Cooperative-reduce knob (the ``SplitParallel`` thread binding): ``BR`` threads
# cooperatively reduce one row's K, then a warp-shuffle / tree combine folds the
# partials (emitted by kernel/100_materialize_tile from Accum.axes ∩ ThreadTile).
BR_CHOICES: tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1)
COOP_BR = BR

# Tensor-core (warp-tier MMA) knobs — the ``Tensorize`` move. ``MMA`` is the atom
# kind ("0" = scalar tier, no tensorize); ``WM``/``WN`` the per-CTA warp counts;
# ``FM``/``FN`` register cells per warp; ``BK`` the K chunk in atom-K units.
WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
TC_REG_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
TC_ATOM = MMA
WARP_M = WM
WARP_N = WN
TC_REG_M = FM
TC_REG_N = FN
TC_BK = BK
