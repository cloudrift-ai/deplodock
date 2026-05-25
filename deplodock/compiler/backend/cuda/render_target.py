"""CUDA implementation of :class:`RenderTarget`.

Owns every CUDA C-spelling decision the Kernel-IR renderer makes:
type names (``float`` / ``__half``), conversion intrinsics
(``__half2float`` / ``__float2half``), per-dtype op intrinsics
(``expf`` / ``hexp``, ``fmaxf`` / ``__hmax``, ...), and the set of
ops with native fp16 forms.

The Stmt renderer in :mod:`deplodock.compiler.ir.stmt` calls into a
:class:`CudaRenderTarget` instance attached to ``RenderCtx``; the
hardcoded ``__half2``/``__float2half`` strings that used to live next
to ``Load`` / ``Assign`` / ``Write`` are gone.
"""

from __future__ import annotations

from deplodock.compiler import dtype as _dtype

_TYPE_NAME: dict[str, str] = {"f32": "float", "f16": "__half", "f16x2": "__half2"}

# Intrinsic spellings — per dtype.  Keys are abstract op names emitted
# by ``op_to_expr`` (``"exp"``, ``"fmax"``, ``"fabs"``, ...).
_INTRINSIC_F32: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
    "erf": "erff",
}

_INTRINSIC_F16: dict[str, str] = {
    "exp": "hexp",
    "log": "hlog",
    "sqrt": "hsqrt",
    "rsqrt": "hrsqrt",
    "tanh": "htanh",
    "fmax": "__hmax",
    "fmin": "__hmin",
    "fabs": "__habs",
}

# Per-pair (2-wide vector) fp16 intrinsics. Each entry is the cuda_fp16.h
# ``h2`` / ``__h*2`` form. Used when an Init / Accum / Assign carries
# dtype = F16x2 (paired by the ``070_pack_fp16_pairs`` pass).
_INTRINSIC_F16x2: dict[str, str] = {
    "exp": "h2exp",
    "log": "h2log",
    "sqrt": "h2sqrt",
    "rsqrt": "h2rsqrt",
    "tanh": "h2tanh",
    "fmax": "__hmax2",
    "fmin": "__hmin2",
    "fabs": "__habs2",
}

# Abstract op names with a native fp16 form. Binary operators (+, -, *, /)
# work via cuda_fp16.h's operator overloads on ``__half`` and ``__half2``,
# so they don't need an entry in the intrinsic tables but are still
# "native". The same set covers F16 and F16x2 — every op listed has both
# a per-element (``__hmax``, ``hexp``) and per-pair (``__hmax2``,
# ``h2exp``) form.
_NATIVE_FP16_OPS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "tanh",
        "fabs",
        "abs",
        "negative",
        "copy",
        "reciprocal",
        "relu",
        "sigmoid",
    }
)


class CudaRenderTarget:
    """CUDA C / cuda_fp16.h spellings for :class:`RenderTarget`.

    Stateless; safe to share across render calls. The Kernel-IR
    renderer constructs one per ``render_kernelop`` invocation (see
    ``deplodock/compiler/ir/kernel/render.py``).
    """

    def type_name(self, dtype: str) -> str:
        return _TYPE_NAME.get(dtype, "float")

    def literal(self, text: str, dtype: str) -> str:
        # Numeric literals (``0.0f`` / ``1.0f`` / ``-1e+30f``) wrap in
        # ``__float2half`` (scalar) or ``__float2half2_rn`` (pair-
        # broadcast) when the surrounding expression's result dtype is
        # fp16 / fp16x2 so the call composes with ``__half`` / ``__half2``
        # operands. NVRTC folds the call to a constant at compile time.
        if dtype == "f16":
            return f"__float2half({text})"
        if dtype == "f16x2":
            return f"__float2half2_rn({text})"
        return text

    def convert(self, value: str, src_dt: str, dst_dt: str) -> str:
        if src_dt == dst_dt:
            return value
        if dst_dt == "f16" and src_dt == "f32":
            return f"__float2half({value})"
        if dst_dt == "f32" and src_dt == "f16":
            return f"__half2float({value})"
        if dst_dt == "f16x2" and src_dt == "f16":
            # Broadcast scalar __half into both lanes of a __half2.
            return f"__half2half2({value})"
        if dst_dt == "f16x2" and src_dt == "f32":
            return f"__float2half2_rn({value})"
        return value

    def intrinsic(self, op_name: str, result_dt: str) -> str:
        if result_dt == "f16":
            return _INTRINSIC_F16.get(op_name, op_name)
        if result_dt == "f16x2":
            return _INTRINSIC_F16x2.get(op_name, op_name)
        return _INTRINSIC_F32.get(op_name, op_name)

    def has_native_op(self, op_name: str, dtype: str) -> bool:
        if dtype == "f32":
            # f32 has every op natively.
            return True
        if dtype in ("f16", "f16x2"):
            return op_name in _NATIVE_FP16_OPS
        return False

    def vector_type(self, dtype: str, n: int) -> tuple[str, str] | None:
        if dtype == "f32":
            if n in (2, 4):
                return (f"float{n}", "float")
            return None
        if dtype == "f16":
            # n=2 → 4 B (LDS.32) via ``__half2`` (4-byte alignment).
            # n=4 → 8 B (LDS.64) via ``uint2`` punned to 4 ``__half``.
            # n=8 → 16 B (LDS.128) via ``uint4`` punned to 8 ``__half``.
            # The 009b cooperative-reduce permutation guarantees the
            # base address is 16-byte aligned when n=8 (and 8-byte
            # aligned for n=4); for matmul / other shapes the
            # vectorize pass independently checks the affine form.
            if n == 2:
                return ("__half2", "__half")
            if n == 4:
                return ("uint2", "__half")
            if n == 8:
                return ("uint4", "__half")
            return None
        return None


def _ensure_canonical(dtype: str) -> str:
    """Coerce a CUDA C type name back to the canonical dtype token, so
    callers that hold an arbitrary spelling can still ask the target."""
    if dtype in _TYPE_NAME:
        return dtype
    return _dtype.get(dtype).name
