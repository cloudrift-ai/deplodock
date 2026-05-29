"""Atom-kind registry — hardware-instruction specs per matmul atom.

An *atom* is the hardware-atomic shape of one matmul-reduce cell. Scalar matmul
isn't represented here (it's the absence of an atom, modelled by
:class:`ScalarTileParams` in ``_enumeration``); only MMA / tensor-core families
register here. Each spec carries the cell shape ``(M, N, K)``, the per-operand
dtype map (``"a"`` / ``"b"`` / ``"c"``; future scaled kinds add ``"a_scale"`` /
``"b_scale"``), the hardware instruction family name (``"wmma"`` / future
``"wgmma"`` / ``"mma_scaled"``), and the group size — the number of threads that
collectively own one cell (32 for WMMA, 128 for wgmma).

The MMA fragment-factorization plan (``plans/mma-fragment-factorization.md``)
threads ``ATOM_KIND`` through the planner; this module is the single source of
truth for "what does kind X mean structurally". The eligibility predicate
:func:`is_atom_eligible` dispatches per kind — see M2 in the plan.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it: this is
a sibling helper, not a pass.

At M1 :data:`ATOM_REGISTRY` is **empty** — the scaffolding lands first
(``AtomTile`` flavor, ``TileParams`` sum-type split, ``Role.ATOM`` wiring) so a
no-op scalar refactor is byte-identical. M2 seeds ``wmma_m16n16k16_f16``; M9
adds bf16 + skewed shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from deplodock.compiler.dtype import DataType


@dataclass(frozen=True)
class AtomSpec:
    """Hardware-instruction spec for one matmul atom kind.

    - ``shape`` is the cell shape ``(M, N, K)`` one instruction realises.
    - ``operand_dtypes`` maps each operand role (``"a"`` / ``"b"`` / ``"c"`` for
      WMMA; scaled kinds extend with ``"a_scale"`` / ``"b_scale"``) to its
      element dtype. The materializer reads this to declare each fragment.
    - ``instruction`` names the hardware instruction family (``"wmma"`` for
      sm_70+ ``wmma::mma_sync``; future ``"wgmma"`` for sm_90+, ``"mma_scaled"``
      for sm_100+ NVFP4/MXFP4). The materializer's per-cell emit branches on
      this — synchronous WMMA vs async wgmma issue/wait, etc.
    - ``group_size`` is the threads-per-cell count (32 for WMMA — one warp;
      128 for wgmma — one warp-group). Used by the warp-tier launch-geometry
      math when computing per-CTA thread count.
    """

    shape: tuple[int, int, int]
    operand_dtypes: Mapping[str, DataType]
    instruction: str
    group_size: int


# Empty at M1. M2 adds ``"wmma_m16n16k16_f16"``; M9 adds bf16 + skewed shapes.
ATOM_REGISTRY: dict[str, AtomSpec] = {}


def atom_spec(kind: str) -> AtomSpec:
    """Resolve ``kind`` to its :class:`AtomSpec`. Raises ``KeyError`` for an
    unregistered kind — there's no "scalar" entry (scalar is the absence of an
    atom, see module docstring)."""
    return ATOM_REGISTRY[kind]


def atom_shape(kind: str) -> tuple[int, int, int]:
    """Cell shape ``(M, N, K)`` of ``kind``."""
    return ATOM_REGISTRY[kind].shape


def atom_group_size(kind: str) -> int:
    """Threads-per-cell of ``kind`` (32 for WMMA, 128 for wgmma)."""
    return ATOM_REGISTRY[kind].group_size
