"""Generic data-type identity used throughout the compiler.

Holds only generic + numpy information. Backend-specific traits (CUDA C
spelling, cupy dtype, required headers) live in the respective backend
modules (e.g. ``deplodock/compiler/backend/cuda/dtype.py``).

Naming convention: the class is ``DataType``; every argument, variable,
and field that carries one is named ``dtype``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataType:
    """Identity of a tensor element type.

    ``name`` is the canonical token written on ``Tensor.dtype`` and used
    everywhere the graph compares dtypes (e.g. structural keys).
    """

    name: str
    np: np.dtype
    nbytes: int

    def __str__(self) -> str:
        return self.name


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
# 2-wide vector of fp16. Same numpy dtype as F16 since numpy doesn't
# distinguish — packing is a CUDA-side storage detail; the canonical
# IR token "f16x2" is what the renderer keys on.
F16x2 = DataType("f16x2", np.dtype(np.float16), 4)
# Integer types — appear on ``input_ids`` placeholders from HF whole-model
# traces. The compiler doesn't generate kernels that compute on them today
# (LM-head gather + embedding lookup is index math); they exist so the
# graph can carry the right Tensor.dtype past the placeholder.
I32 = DataType("i32", np.dtype(np.int32), 4)
I64 = DataType("i64", np.dtype(np.int64), 8)


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
