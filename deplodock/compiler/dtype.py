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


_BY_NAME: dict[str, DataType] = {dt.name: dt for dt in (F32, F16)}

# Aliases let callers feed PyTorch/numpy-style names without re-canonicalizing
# at every callsite. The canonical name (``F32.name == "f32"``) is what lands
# on ``Tensor.dtype``.
_ALIASES: dict[str, str] = {
    "float32": "f32",
    "float": "f32",
    "float16": "f16",
    "half": "f16",
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
