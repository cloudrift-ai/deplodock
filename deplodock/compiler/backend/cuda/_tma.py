"""TMA descriptor encoding via direct ctypes calls to ``libcuda.so``.

cupy 14.x exposes no binding for ``cuTensorMapEncodeTiled`` (the function
that builds a 128-byte ``CUtensorMap`` descriptor — the host-side handle
TMA load instructions reference). We need it at launch time because the
encoded descriptor embeds the source array's *device pointer*, which is
only known once cupy has allocated the buffer.

This module is a thin shim:

- :func:`encode_tiled` — call ``cuTensorMapEncodeTiled`` and return the
  raw 128-byte descriptor as ``bytes``.
- :func:`descriptor_arg` — wrap those bytes in a ``numpy.ndarray`` view
  so cupy's kernel-launch path passes them by value (matching the
  ``__grid_constant__`` parameter on the kernel side).

Cached at module level so we don't re-resolve the symbol on every launch.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

# CUtensorMap is a 128-byte aligned opaque struct. Layout: 16 × uint64_t.
CU_TENSOR_MAP_SIZE = 128

# Enum values from CUDA driver header `cuda.h` (v12.0+).
# CUtensorMapDataType — ordering tracks the enum decl in `cuda.h`, not the
# misnamed pre-fp16 constant the legacy code used (which mapped FLOAT32 to
# the value 2, actually UINT32; happened to round-trip for fp32 because both
# widths are 4 B but lied about fp16 element width). Real values are:
CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0
CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1
CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2
CU_TENSOR_MAP_DATA_TYPE_INT32 = 3
CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4
CU_TENSOR_MAP_DATA_TYPE_INT64 = 5
CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9
# Map numpy itemsize → reasonable CUtensorMapDataType. Fp16 / bf16 disambiguate
# in the caller (numpy stores both as 2-byte but TMA cares about the float
# encoding for OOB-fill semantics; with ``OOB_FILL_NONE`` the value is
# unused at copy time, so this map is sufficient).
_DTYPE_BY_ITEMSIZE = {
    1: CU_TENSOR_MAP_DATA_TYPE_UINT8,
    2: CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    4: CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    8: CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
}
# CUtensorMapInterleave
CU_TENSOR_MAP_INTERLEAVE_NONE = 0
# CUtensorMapSwizzle
_SWIZZLE = {"NONE": 0, "B32": 1, "B64": 2, "B128": 3}
# CUtensorMapL2promotion
CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
# CUtensorMapFloatOOBfill
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0


@functools.cache
def _libcuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


@functools.cache
def _encode_fn() -> ctypes.CFUNCTYPE:
    fn = _libcuda().cuTensorMapEncodeTiled
    fn.restype = ctypes.c_int  # CUresult
    fn.argtypes = [
        ctypes.c_void_p,  # CUtensorMap*
        ctypes.c_int,  # CUtensorMapDataType
        ctypes.c_uint32,  # rank (cuuint32_t)
        ctypes.c_void_p,  # globalAddress (void*)
        ctypes.POINTER(ctypes.c_uint64),  # globalDim
        ctypes.POINTER(ctypes.c_uint64),  # globalStrides (rank-1 entries, in bytes)
        ctypes.POINTER(ctypes.c_uint32),  # boxDim
        ctypes.POINTER(ctypes.c_uint32),  # elementStrides
        ctypes.c_int,  # interleave
        ctypes.c_int,  # swizzle
        ctypes.c_int,  # l2Promotion
        ctypes.c_int,  # oobFill
    ]
    return fn


def encode_tiled(
    *,
    global_address: int,
    src_shape: tuple[int, ...],
    box_extents: tuple[int, ...],
    elem_size: int = 4,
    swizzle: str = "NONE",
) -> bytes:
    """Build a 128-byte ``CUtensorMap`` descriptor.

    ``global_address`` is the device pointer of the source array (e.g.
    ``cupy.ndarray.data.ptr``). ``src_shape`` is the array's full shape;
    ``box_extents`` is the per-dim TMA box. Both are length == rank.
    ``elem_size`` is bytes per element (4 for fp32). ``swizzle`` is one
    of the keys of :data:`_SWIZZLE` and must match the descriptor mode
    chosen by the lowering pass.

    The returned bytes are the raw descriptor — pass them to a kernel
    launch as a ``numpy`` view (see :func:`descriptor_arg`)."""
    rank = len(src_shape)
    if rank != len(box_extents):
        raise ValueError(f"rank mismatch: src_shape={src_shape!r} vs box_extents={box_extents!r}")
    if rank == 0 or rank > 5:
        raise ValueError(f"TMA rank must be 1..5, got {rank}")
    # Hardware limit: each boxDim must be 1..256. The driver rejects
    # violations with the opaque ``CUresult=1`` (CUDA_ERROR_INVALID_VALUE);
    # name the offending dim instead. The lowering eligibility gate
    # (``050_use_tma._source_eligible``) filters these before codegen, so
    # tripping this means a gate regression upstream.
    for d, b in enumerate(box_extents):
        if not 1 <= int(b) <= 256:
            raise ValueError(f"TMA box dim {d} extent {b} outside the hardware range 1..256 (box_extents={box_extents!r})")

    # CUDA driver descriptor convention: dim[0] is the FASTEST-varying
    # dim (innermost in C/row-major). So we reverse the C-order shapes
    # before handing them to ``cuTensorMapEncodeTiled``.
    #
    # globalStrides has rank-1 entries (innermost stride is implicit
    # ``elem_size``); each is in bytes from one element to the next
    # along that dim.
    rev_shape = tuple(int(d) for d in reversed(src_shape))
    rev_box = tuple(int(b) for b in reversed(box_extents))
    global_dim = (ctypes.c_uint64 * rank)(*rev_shape)
    box_dim = (ctypes.c_uint32 * rank)(*rev_box)
    element_strides = (ctypes.c_uint32 * rank)(*([1] * rank))

    strides_count = max(rank - 1, 0)
    if strides_count > 0:
        # In driver order (innermost-first): stride[i] in bytes for dim i+1.
        # C-contiguous innermost stride is elem_size; stride[0] = innermost
        # extent * elem_size; stride[1] = stride[0] * (next-inner extent), etc.
        byte_strides: list[int] = []
        running = rev_shape[0] * elem_size
        for d in rev_shape[1:]:
            byte_strides.append(running)
            running *= d
        global_strides = (ctypes.c_uint64 * strides_count)(*byte_strides)
        strides_ptr = global_strides
    else:
        strides_ptr = None

    desc_buf = (ctypes.c_uint8 * CU_TENSOR_MAP_SIZE)()
    data_type = _DTYPE_BY_ITEMSIZE.get(elem_size, CU_TENSOR_MAP_DATA_TYPE_FLOAT32)
    res = _encode_fn()(
        ctypes.cast(desc_buf, ctypes.c_void_p),
        data_type,
        rank,
        ctypes.c_void_p(int(global_address)),
        global_dim,
        strides_ptr,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        _SWIZZLE[swizzle],
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if res != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={res}")
    return bytes(desc_buf)


# numpy dtype matching the kernel's CUtensorMap struct (16 × uint64).
# Using a structured dtype with one field forces cupy's launch path to
# pass the bytes by value, mirroring the ``__grid_constant__`` semantics
# of the kernel parameter.
_DESC_DTYPE = np.dtype([("opaque", np.uint64, 16)], align=True)


def descriptor_arg(desc_bytes: bytes) -> np.ndarray:
    """Wrap descriptor bytes as a 0-dim structured array suitable for
    passing to ``cupy.RawKernel`` as a ``__grid_constant__`` parameter."""
    if len(desc_bytes) != CU_TENSOR_MAP_SIZE:
        raise ValueError(f"descriptor must be {CU_TENSOR_MAP_SIZE} bytes, got {len(desc_bytes)}")
    arr = np.zeros((), dtype=_DESC_DTYPE)
    arr["opaque"][:] = np.frombuffer(desc_bytes, dtype=np.uint64)
    return arr
