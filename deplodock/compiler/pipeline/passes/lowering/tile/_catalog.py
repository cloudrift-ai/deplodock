"""The permitted-move catalog — the legal scheduling candidates per node, keyed on ``AxisRole``.

``010_recognize``'s ``_schedule`` emit enumerates these into the scheduling fork: each move is a codec
spelling under the node's axis-named key (``TILE@<k_axis>`` / ``REDUCE@<axis>`` / ``STAGE@<axis>``) that
the existing ``parse`` / ``spell`` grammar, the prior featurizer, and the perf DB already consume — the
move is only the **generator**, not new syntax. This replaces the flat one-candidate ``_tile_specs``
stub (``[""]`` unless pinned) with a bounded, legality-guarded product so an unpinned ``compile`` / ``tune``
explores the tile space, ranked by the learned / analytic prior.

Two invariants keep a cold greedy compile stable and correct:

- **Conservative option-0.** The per-cell / serial pick leads every list, so the emission-order fallback
  (no prior loaded) keeps today's behavior.
- **Legality guards, evaluated up front** so an illegal candidate never emits — the scalar block-thread
  budget (``par_n·par_m ≤ 1024``) here; the warp K-step / symbolic-reduce guards live with their moves
  in ``_schedule`` (``_check_warp_static_k`` etc.). An env pin still wins via ``Knob.narrow`` at the call
  site — the catalog is the *unpinned* candidate set.

Scope today is the **scalar contraction** output tile (the matmul fixture the structural-coverage test
hand-computes). Warp (tensor-core) tiles stay pin-driven — their atom selection is a separate concern —
and the reduce / stage families keep their existing small forks; folding those into the catalog (and the
hierarchical ``build_fork_tree`` levels the multi-node flash warp QK+PV case wants) is the next slice.
"""

from __future__ import annotations

from deplodock.compiler.ir.schedule import TilePlan

# The scalar block-thread budget (CUDA's 1024-thread/CTA hardware limit); a scalar tile launches
# ``par_n·par_m`` threads (one per parallel output cell). Mirrors ``_schedule._MAX_BLOCK_THREADS``.
_MAX_BLOCK_THREADS = 1024

# The scalar register-tile candidate grid: ``(par_n, par_m)`` parallel thread-tile widths ×
# ``(reg_n, reg_m)`` per-thread register sub-tile widths. Bounded and hand-computable — the product the
# structural-coverage test recomputes independently. The parallel widths stay well inside the thread
# budget (``32·16 = 512 ≤ 1024``); the register widths span the square + skewed sub-tiles the prior
# ranks by occupancy / reuse.
_SCALAR_PAR: tuple[tuple[int, int], ...] = ((16, 8), (16, 16), (32, 8), (32, 16))  # (par_n, par_m)
_SCALAR_REG: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (4, 4), (2, 4), (4, 2))  # (reg_n, reg_m)


def scalar_tile_moves() -> list[str]:
    """The scalar-contraction output-tile ``TILE`` codec candidates: per-cell (``""``) first — the
    conservative option-0 — then the register-tile grid (:data:`_SCALAR_PAR` × :data:`_SCALAR_REG`)
    filtered by the ``par_n·par_m ≤ 1024`` thread budget. Each is spelled through :class:`TilePlan`
    so it round-trips the codec grammar exactly."""
    moves = [""]
    for par in _SCALAR_PAR:
        if par[0] * par[1] > _MAX_BLOCK_THREADS:
            continue
        for reg in _SCALAR_REG:
            moves.append(TilePlan(units=par, regs=reg).spell())
    return moves
