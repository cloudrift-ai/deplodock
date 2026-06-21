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
