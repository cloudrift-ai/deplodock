"""Flash geometry — the streaming-flash parameters + addressing helpers derived from the
logical gmem buffers, shared by the **enumeration** warp-flash build move (`warp_chain_build`)
and the **assembly** realizer (`_assemble.realize_flash`). Lives in `lowering/` (a sibling of
`_masking` / `_predicates`) so both layers import it without crossing the enumeration↔assembly
boundary. Pure functions over `Buffer` shapes — no pass / dialect dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import BF16, F16, DataType
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal


def static_extent(d) -> int | None:
    """The static extent of a shape entry, or ``None`` for a symbolic dim."""
    if isinstance(d, int):
        return d
    f = getattr(d, "as_static", None)
    return f() if (f is not None and getattr(d, "is_static", True)) else None


def add(*terms) -> Expr:
    """Sum int / Expr terms into one Expr (dropping literal zeros) — flash addressing."""
    out = None
    for t in terms:
        e = Literal(t, "int") if isinstance(t, int) else t
        if isinstance(e, Literal) and e.value == 0:
            continue
        out = e if out is None else BinaryExpr("+", out, e)
    return out if out is not None else Literal(0, "int")


def mul(a, b: int) -> Expr:
    """``a · b`` as an Expr, folding the ``b in {0, 1}`` degenerate cases."""
    return add() if b == 0 else (a if b == 1 else BinaryExpr("*", a if not isinstance(a, int) else Literal(a, "int"), Literal(b, "int")))


@dataclass(frozen=True)
class FlashParams:
    """The streaming-flash shape derived from the logical gmem ``buffers``: the q/k/v/out buffer
    names, ``(B, H, S, D)`` (``S`` is ``None`` for a symbolic ``seq_len``, ``seq_var`` its runtime
    symbol), the GQA ``group`` (q-heads / kv-heads), ``causal``, and the 16-bit operand dtype."""

    q: str
    k: str
    v: str
    out: str
    B: int
    H: int
    S: int | None
    D: int
    group: int
    causal: bool
    seq_var: str | None
    dtype: DataType

    @property
    def symbolic(self) -> bool:
        return self.seq_var is not None

    @property
    def atom_kind(self) -> str:
        return "mma_m16n8k16_bf16" if self.dtype == BF16 else "mma_m16n8k16_f16"


def flash_params(buffers: dict, out: str) -> FlashParams | None:
    """Derive :class:`FlashParams` from the logical gmem ``buffers`` (+ the written ``out``
    buffer), or ``None`` when out of the 16-bit warp scope (a 4th rank-4 input = additive mask,
    non-16-bit dtype, or non-static B/H/D). ``causal`` is detected by a ``*ninf*`` input."""
    ins = [(n, b) for n, b in buffers.items() if len(b.shape) == 4 and n != out]
    if len(ins) != 3:  # an additive mask adds a 4th rank-4 input — out of scope
        return None
    (qn, q), (kn, k), (vn, _v) = ins
    if q.dtype not in (F16, BF16):
        return None
    B, H, D = static_extent(q.shape[0]), static_extent(q.shape[1]), static_extent(q.shape[3])
    S = static_extent(q.shape[2])  # None ⇒ symbolic seq_len
    kvh = static_extent(k.shape[1])
    if kvh in (None, 0) or any(x is None for x in (B, H, D)):
        return None
    seq_var = None if S is not None else next(iter(q.shape[2].expr.free_vars()))
    causal = any("ninf" in n for n in buffers)
    return FlashParams(q=qn, k=kn, v=vn, out=out, B=B, H=H, S=S, D=D, group=H // kvh, causal=causal, seq_var=seq_var, dtype=q.dtype)
