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

_TYPE_NAME: dict[str, str] = {"f32": "float", "f16": "__half"}

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

# Abstract op names with a native fp16 form. Binary operators (+, -, *, /)
# work via cuda_fp16.h's operator overloads on ``__half``, so they don't
# need an entry in ``_INTRINSIC_F16`` but are still "native".
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
        # ``__float2half`` when the surrounding expression's result
        # dtype is fp16, so the call composes with ``__half`` operands.
        # NVRTC folds the call to a half constant at compile time.
        if dtype == "f16":
            return f"__float2half({text})"
        return text

    def convert(self, value: str, src_dt: str, dst_dt: str) -> str:
        if src_dt == dst_dt:
            return value
        if dst_dt == "f16" and src_dt == "f32":
            return f"__float2half({value})"
        if dst_dt == "f32" and src_dt == "f16":
            return f"__half2float({value})"
        return value

    def intrinsic(self, op_name: str, result_dt: str) -> str:
        table = _INTRINSIC_F16 if result_dt == "f16" else _INTRINSIC_F32
        return table.get(op_name, op_name)

    def has_native_op(self, op_name: str, dtype: str) -> bool:
        if dtype != "f16":
            # f32 has every op natively.
            return True
        return op_name in _NATIVE_FP16_OPS

    def vector_type(self, dtype: str, n: int) -> tuple[str, str] | None:
        if dtype == "f32":
            if n in (2, 4):
                return (f"float{n}", "float")
            return None
        if dtype == "f16":
            # Wider fp16 vectors (n=4 → 8 B, n=8 → 16 B) need 4-element /
            # 8-element alignment which we can't statically prove; the
            # ``(4, 2)`` retry in ``render_body`` falls through to n=2.
            if n == 2:
                return ("__half2", "__half")
            return None
        return None


def _ensure_canonical(dtype: str) -> str:
    """Coerce a CUDA C type name back to the canonical dtype token, so
    callers that hold an arbitrary spelling can still ask the target."""
    if dtype in _TYPE_NAME:
        return dtype
    return _dtype.get(dtype).name
