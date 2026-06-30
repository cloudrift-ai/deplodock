"""Render-target abstraction for the Kernel-IR source emitter.

The Kernel-IR ``Stmt.render`` methods produce target-specific source
(today: CUDA / C++). They consult a :class:`RenderTarget` on the
``RenderCtx`` instead of hardcoding C spellings, so the rendering code
stays target-agnostic and the same IR can lower to another C-family
target by passing a different implementation.

Distinct from :mod:`emmy.compiler.target`, which represents the
*hardware* compile target (compute capability — sm_80 / sm_90 / sm_120
— consulted by the tile-IR passes that gate on hardware features).
:class:`RenderTarget` is purely about source-text emission shapes.

Today the only implementation is :class:`CudaRenderTarget` in
``emmy/compiler/backend/cuda/render_target.py``. Add a new
implementation per backend that wants to share the Stmt renderer.
"""

from __future__ import annotations

from typing import Protocol


class RenderTarget(Protocol):
    """Per-target renderer helpers consulted by ``Stmt.render``.

    All methods take canonical dtype tokens (``"f32"``, ``"f16"``, ...);
    the implementation maps them to target-specific C spellings.
    """

    def type_name(self, dtype: str) -> str:
        """C type spelling for an SSA local declaration.

        ``"f32" → "float"``, ``"f16" → "__half"``. Used by ``Load``,
        ``Assign``, ``Accum``, ``Init`` to declare locals in the right
        type.
        """

    def literal(self, text: str, dtype: str) -> str:
        """Wrap a numeric literal text in target-specific dtype-casting.

        ``("0.0f", "f32") → "0.0f"``;
        ``("0.0f", "f16") → "__float2half(0.0f)"``.

        Used by ``Literal.render`` when an enclosing expression's
        result dtype forces literals to compose with non-default-dtype
        operands.
        """

    def convert(self, value: str, src_dt: str, dst_dt: str) -> str:
        """Cast an SSA-name expression across dtypes. Returns ``value``
        unchanged when ``src_dt == dst_dt``. Otherwise wraps with the
        target conversion intrinsic (e.g. ``__half2float(value)``).
        """

    def intrinsic(self, op_name: str, result_dt: str) -> str:
        """Per-dtype intrinsic spelling for an abstract op name.

        Examples: ``("exp", "f32") → "expf"``,
        ``("exp", "f16") → "hexp"``, ``("fmax", "f32") → "fmaxf"``,
        ``("fmax", "f16") → "__hmax"``. Falls back to the op name
        unchanged when no mapping exists.
        """

    def has_native_op(self, op_name: str, dtype: str) -> bool:
        """Whether ``op_name`` has a native form at ``dtype``. Drives the
        ``Assign.render`` dispatch between native and promote-and-demote
        rendering.
        """

    def vector_type(self, dtype: str, n: int) -> tuple[str, str] | None:
        """For an ``n``-wide vector of ``dtype``, return the
        ``(vector_c_type, element_c_type)`` pair, or ``None`` when the
        combination isn't supported.

        ``("f32", 4) → ("float4", "float")``;
        ``("f16", 2) → ("__half2", "__half")``;
        ``("f16", 4) → None`` (would need 8-byte alignment we can't
        statically prove).
        """
