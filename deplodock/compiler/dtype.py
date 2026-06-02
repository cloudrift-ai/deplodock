"""Generic data-type identity used throughout the compiler.

Holds only generic + numpy information. Backend-specific traits (CUDA C
spelling, cupy dtype, required headers) live in the respective backend
modules (e.g. ``deplodock/compiler/backend/cuda/dtype.py``).

Naming convention: the class is ``DataType``; every argument, variable,
and field that carries one is named ``dtype``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataType:
    """Identity of a tensor element type.

    ``name`` is the canonical token written on ``Tensor.dtype`` and used
    everywhere the graph compares dtypes (e.g. structural keys).

    Two sub-families (see :func:`is_structured`):

    - **Scalar** — a single logical element per value (``F32`` / ``F16`` /
      ``BF16`` / ``I32`` / ``I64``). These are plain ``DataType`` instances.
    - **Structured** (:class:`StructuredType`) — a register-composite value
      with a hardware-specific layout, e.g. the packed vector ``F16x2``
      (``__half2``), as opposed to a plain scalar element.
    """

    name: str
    np: np.dtype
    nbytes: int

    def __str__(self) -> str:
        return self.name

    @property
    def is_structured(self) -> bool:
        """True for register-composite types (packed vectors, fragments)."""
        return False


@dataclass(frozen=True)
class StructuredType(DataType):
    """A register-composite type — a packed vector like ``F16x2`` — as opposed
    to a plain scalar element.

    It occupies a register with a hardware-specific layout (``__half2``); the
    renderer / lowering keys on the concrete subtype rather than treating it as
    a scalar."""

    @property
    def is_structured(self) -> bool:
        return True


F32 = DataType("f32", np.dtype(np.float32), 4)
F16 = DataType("f16", np.dtype(np.float16), 2)
# BFloat16 — same 2-byte footprint as F16 but different exponent / mantissa
# split (8 / 7 vs 5 / 10). Used by WMMA's bf16 atom kinds (M9 of the MMA
# fragment-factorization plan). NumPy has no first-class bf16; we map to
# the closest carrier (uint16 with the bf16 bit-pattern) so Tensor.dtype
# round-trips through serialization. CUDA-side spelling is
# ``__nv_bfloat16`` (see ``backend/cuda/dtype.py``).
BF16 = DataType("bf16", np.dtype(np.uint16), 2)
# Two ``__half`` values packed into a 32-bit register, semantically a
# 2-wide vector of fp16 — the first ``StructuredType``. Same numpy dtype as
# F16 since numpy doesn't distinguish — packing is a CUDA-side storage detail;
# the canonical IR token "f16x2" is what the renderer keys on.
F16x2 = StructuredType("f16x2", np.dtype(np.float16), 4)


# Integer types — appear on ``input_ids`` placeholders from HF whole-model
# traces. The compiler doesn't generate kernels that compute on them today
# (LM-head gather + embedding lookup is index math); they exist so the
# graph can carry the right Tensor.dtype past the placeholder.
I32 = DataType("i32", np.dtype(np.int32), 4)
I64 = DataType("i64", np.dtype(np.int64), 8)


# ===========================================================================
# Atom kinds — the hardware-instruction spec for each tensor-core matmul cell.
# ===========================================================================
#
# An *atom* is the hardware-atomic shape of one matmul-reduce cell, named by an
# ``ATOM_KIND`` string (carried on the ``Mma`` op + the ``TileOp`` knob). Scalar
# matmul isn't an atom (it's the absence of one). The registry lives here so the
# type module is the single source of truth for "what does kind X mean" — the
# MMA lowering (``kernel/005_lower_atom_tile``) reads the shape + operand dtypes,
# and ``ir/tile/ir.py`` reads the shape for launch geometry, all via ``atom_spec``.
#
# Per-kernel *eligibility* (does this LoopOp admit this atom?) is NOT here — it
# depends on the loop/graph/context and stays in the planner
# (``pipeline/passes/lowering/tile/_atom.py``), which imports these specs.


@dataclass(frozen=True)
class AtomSpec:
    """Hardware-instruction spec for one matmul atom kind.

    - ``shape`` is the cell shape ``(M, N, K)`` one instruction realises.
    - ``operand_dtypes`` maps each operand role (``"a"`` / ``"b"`` / ``"c"``;
      scaled kinds extend with ``"a_scale"`` / ``"b_scale"``) to its element
      dtype. The MMA lowering reads it to declare each register array and to
      type the per-operand fragments.
    - ``group_size`` is the threads-per-cell count (32 for the warp-level
      mma.sync atom; 128 for a future wgmma warp-group). Used by the warp-tier
      launch-geometry math when computing per-CTA thread count.

    Today the only family registered is the s16816 ``mma.sync.aligned`` +
    ``ldmatrix`` path; when a second hardware family lands (wgmma, mma_scaled)
    its lowering/gating differences get a discriminator field at that point.
    """

    shape: tuple[int, int, int]
    operand_dtypes: Mapping[str, DataType]
    group_size: int


ATOM_REGISTRY: dict[str, AtomSpec] = {
    # Modern warp-level MMA: ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` (the
    # ``s16816`` cell cuBLAS/CUTLASS use) — the sole tensor-core family. f16 /
    # bf16 operands, f32 accumulate, sm_80+ (the m16n8k16 op is Ampere+).
    # ``kernel/005_lower_atom_tile`` emits the RegFragment/LdmatrixLoad/
    # MmaSyncPtx/RegStore chain. The path has **no gmem-direct load**
    # (ldmatrix is smem→register only).
    "mma_m16n8k16_f16": AtomSpec(
        shape=(16, 8, 16),
        operand_dtypes={"a": F16, "b": F16, "c": F32},
        group_size=32,
    ),
    # bf16 sibling: same s16816 path (bf16 and f16 share the 16-bit fragment
    # layout / ldmatrix.b16, so only the PTX dtype field differs —
    # ``MmaSyncPtx.ab_dtype`` selects the ``dpl_mma_…_bf16`` wrapper). Ampere+.
    "mma_m16n8k16_bf16": AtomSpec(
        shape=(16, 8, 16),
        operand_dtypes={"a": BF16, "b": BF16, "c": F32},
        group_size=32,
    ),
}


# Priority-ordered tuple of MMA atom kinds the planner enumerates: f16 first,
# then bf16 (both Ampere+). The m16n8k16 atom tiles any divisible shape, so it
# covers what the old skewed WMMA kinds (m8n32 / m32n8) did via tiling. Scalar
# is not in this list — it's the absence of an atom, the fallback when no
# mma.sync kind is eligible.
ATOM_KINDS: tuple[str, ...] = (
    "mma_m16n8k16_f16",
    "mma_m16n8k16_bf16",
)


def atom_spec(kind: str) -> AtomSpec:
    """Resolve ``kind`` to its :class:`AtomSpec`. Raises ``KeyError`` for an
    unregistered kind — there's no "scalar" entry (scalar is the absence of an
    atom)."""
    return ATOM_REGISTRY[kind]


def atom_shape(kind: str) -> tuple[int, int, int]:
    """Cell shape ``(M, N, K)`` of ``kind``."""
    return ATOM_REGISTRY[kind].shape


def atom_group_size(kind: str) -> int:
    """Threads-per-cell of ``kind`` (32 for the warp-level mma.sync atom)."""
    return ATOM_REGISTRY[kind].group_size


_BY_NAME: dict[str, DataType] = {dt.name: dt for dt in (F32, F16, BF16, F16x2, I32, I64)}

# Aliases let callers feed PyTorch/numpy-style names without re-canonicalizing
# at every callsite. The canonical name (``F32.name == "f32"``) is what lands
# on ``Tensor.dtype``.
_ALIASES: dict[str, str] = {
    "float32": "f32",
    "float": "f32",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "int32": "i32",
    "int64": "i64",
    "long": "i64",
}


def get(dtype: str | DataType) -> DataType:
    """Resolve a name (canonical or alias) or pass through a ``DataType``."""
    if isinstance(dtype, DataType):
        return dtype
    if dtype in _BY_NAME:
        return _BY_NAME[dtype]
    canonical = _ALIASES.get(dtype)
    if canonical is not None:
        return _BY_NAME[canonical]
    raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(_BY_NAME) + sorted(_ALIASES)}")
