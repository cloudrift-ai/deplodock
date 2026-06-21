"""Greenfield knob vocabulary for the move composer.

Each knob is a *move parameter*, not a free-floating tile dimension: the
``TileMap`` move on a map axis contributes a thread-tile factor and a
register-tile factor (the block-tile count is derived from the extent, so it
is not a knob). The legacy ``BM/BN/FM/FN/...`` schema is *not* reused — these
keys become the variant identity in ``op_cache_key`` and the prior feature
vector once the composer is the default (the goldens / prior retrain on the
new vocabulary; see ``plans/melodic-giggling-gem.md``).

Phase 1 declares only the map-tiling knobs (pointwise regime). Reduce /
tensorize / split-K knobs land with their moves in later phases.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.knob import Knob, KnobType

# Candidate menus, shared by the move ``offers``. Thread-tile extents are the
# per-CTA thread fan-out per axis; register-tile factors are cells-per-thread.
THREAD_CHOICES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
REG_CHOICES: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26, 28, 32, 40, 48, 64, 96, 128)

# Resource ceilings (per CTA). Mirror the legacy gates: ``BM·BN`` ≤ threads,
# ``FM·FN`` ≤ cells-per-thread.
MAX_THREADS_PER_CTA = 1024
MAX_CELLS_PER_THREAD = 128

# Map-tile knobs. ``N`` is the innermost free axis, ``M`` the next-out one
# (matching the legacy planner's outer_n / outer_m naming). ``off=1`` is the
# "no tiling on this axis" identity, used when an axis is absent (1-D kernels).
MAP_N_THREAD = Knob("MAP_N_THREAD", KnobType.INT, hints=THREAD_CHOICES, help="N (inner free) thread-tile extent", off=1)
MAP_N_REG = Knob("MAP_N_REG", KnobType.INT, hints=REG_CHOICES, help="N (inner free) register-tile factor", off=1)
MAP_M_THREAD = Knob("MAP_M_THREAD", KnobType.INT, hints=THREAD_CHOICES, help="M (outer free) thread-tile extent", off=1)
MAP_M_REG = Knob("MAP_M_REG", KnobType.INT, hints=REG_CHOICES, help="M (outer free) register-tile factor", off=1)

# Reduce-tile knobs (the ``TileSerial`` move): ``RED_BK`` is the staged K-chunk
# (inner serial loop trip count); ``RED_FK`` strip-mines the K axis into FK
# independent accumulators (``RegisterTile(reduce=True)``). ``off=1`` = no
# chunking / no strip-mine.
BK_CHOICES: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)
FK_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
RED_BK = Knob("RED_BK", KnobType.INT, hints=BK_CHOICES, help="K staged-chunk size (inner serial loop trip count)", off=1)
RED_FK = Knob("RED_FK", KnobType.INT, hints=FK_CHOICES, help="K strip-mine factor (independent accumulators)", off=1)

# Tensor-core (warp-tier MMA) knobs — the ``Tensorize`` move. ``TC_ATOM`` is the
# atom kind ("" = scalar tier, no tensorize); ``WARP_*`` are the per-CTA warp
# counts; ``TC_REG_*`` are register cells per warp; ``TC_BK`` is the K chunk in
# atom-K units. ``off`` values mark the scalar-tier "tensorize declined".
WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
TC_REG_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
TC_ATOM = Knob("TC_ATOM", KnobType.STR, help="tensor-core atom kind ('' = scalar tier)", off="")
WARP_M = Knob("WARP_M", KnobType.INT, hints=WARP_CHOICES, help="warps along M (WM)", off=0)
WARP_N = Knob("WARP_N", KnobType.INT, hints=WARP_CHOICES, help="warps along N (WN)", off=0)
TC_REG_M = Knob("TC_REG_M", KnobType.INT, hints=TC_REG_CHOICES, help="M register cells per warp (FM)", off=0)
TC_REG_N = Knob("TC_REG_N", KnobType.INT, hints=TC_REG_CHOICES, help="N register cells per warp (FN)", off=0)
TC_BK = Knob("TC_BK", KnobType.INT, hints=BK_CHOICES, help="K staged-chunk in atom-K units", off=0)
