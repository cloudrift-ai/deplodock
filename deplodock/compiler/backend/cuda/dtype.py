"""CUDA-backend dtype traits.

Augments :class:`deplodock.compiler.dtype.DataType` with CUDA-specific
information: the C type name used in kernel source, any header that
must be ``#include``'d, and the cupy dtype used for device allocations.

Per-buffer / per-decl dtype lookup helpers (``nbytes_of``) accept the
legacy CUDA C-name spellings (``"float"``, ``"half"``, ...) that older
Kernel IR fields still carry, so all four duplicated ``_DTYPE_BYTES``
tables in the compiler can route through this module.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler import dtype as _dtype
from deplodock.compiler.dtype import DataType

_CUDA_NAME: dict[DataType, str] = {
    _dtype.F32: "float",
    _dtype.F16: "__half",
    _dtype.F16x2: "__half2",
}

# Inverse of _CUDA_NAME for the kernel-internal C-name -> canonical
# lookup. Used by Smem.render to populate ``ctx.buffer_dtypes`` so a
# subsequent Load on the same smem buffer picks the right local C type.
# Unknown C names (e.g. ``"unsigned long long"`` for mbarriers) map to
# None and the caller treats the smem buffer as "not a tensor".
_CANONICAL_FROM_CUDA_NAME: dict[str, str | None] = {v: k.name for k, v in _CUDA_NAME.items()}


def canonical_from_cuda_name(name: str) -> str | None:
    """Inverse of :func:`cuda_name` over the dtypes we model on the graph.
    Returns ``None`` for CUDA C type names that don't correspond to a
    :class:`DataType` (mbarrier slots, alignment helpers, etc.)."""
    return _CANONICAL_FROM_CUDA_NAME.get(name)


_CUDA_INCLUDE: dict[DataType, str | None] = {
    _dtype.F32: None,
    _dtype.F16: "<cuda_fp16.h>",
    _dtype.F16x2: "<cuda_fp16.h>",
}

# Per-op intrinsic spellings and the native-op set used to live here as
# ``INTRINSICS_FP16`` / ``NATIVE_FP16_OPS`` for the Kernel-IR renderer
# to consult on the ``RenderCtx``. They moved to
# :class:`CudaRenderTarget` (``backend/cuda/render_target.py``) — the
# render layer now talks to the target via the :class:`RenderTarget`
# protocol instead of pulling tables from this module.

# Bytes for raw CUDA C type names that appear in Kernel IR today (``Smem.dtype``,
# ``Local.dtype``, ``Literal.dtype`` are ``str`` for now — step 2 of the dtype
# migration converts them to :class:`DataType`). Keep this table in sync with
# the canonical sizes on :class:`DataType`.
_C_NAME_BYTES: dict[str, int] = {
    "float": _dtype.F32.nbytes,
    "double": 8,
    "int": 4,
    "half": _dtype.F16.nbytes,
    "__half": _dtype.F16.nbytes,
    "__half2": _dtype.F16x2.nbytes,
    "__nv_bfloat16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "i32": 4,
    "i64": 8,
    "f64": 8,
    "unsigned long long": 8,
}


def cuda_name(dtype: str | DataType) -> str:
    """C type name for a kernel signature or local decl (``float``, ``__half``)."""
    return _CUDA_NAME[_dtype.get(dtype)]


def cuda_includes(dtypes: Iterable[str | DataType]) -> list[str]:
    """Deduplicated ``#include`` directives required by the given dtypes."""
    seen: dict[str, None] = {}
    for d in dtypes:
        header = _CUDA_INCLUDE.get(_dtype.get(d))
        if header is not None:
            seen.setdefault(header, None)
    return list(seen)


def cupy_dtype(dtype: str | DataType):
    """cupy dtype for device buffer allocation. Lazy-imports cupy."""
    import cupy as cp  # noqa: PLC0415

    return cp.dtype(_dtype.get(dtype).np)


def nbytes_of(dtype: str | DataType) -> int:
    """Bytes per element for any dtype spelling that appears in the compiler:

    canonical names (``"f32"``, ``"f16"``), aliases (``"float32"``,
    ``"float16"``, ``"half"``, ``"float"``), CUDA C type names
    (``"double"``, ``"int"``, ``"unsigned long long"``, ``"bfloat16"``),
    and :class:`DataType` instances.
    """
    if isinstance(dtype, DataType):
        return dtype.nbytes
    if dtype in _C_NAME_BYTES:
        return _C_NAME_BYTES[dtype]
    return _dtype.get(dtype).nbytes
