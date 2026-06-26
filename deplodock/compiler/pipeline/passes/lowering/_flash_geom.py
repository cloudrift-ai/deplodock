"""Flash geometry — the streaming-flash parameters + addressing helpers derived from the
logical gmem buffers, shared by the **enumeration** warp-flash build move (`warp_chain_build`)
and the **assembly** realizer (`_assemble.carry_scope_from_graph`). Lives in `lowering/` (a sibling
of `_masking` / `_predicates`) so both layers import it without crossing the enumeration↔assembly
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
    """The streaming-flash **shape + dtype** read off the logical gmem buffers — the generic facts
    any tiled kernel needs, named for **no** attention concept: the output buffer ``out`` (the query
    rows + head dim live on its shape), the head dim ``D`` (the QK^T reduce / P@V output extent), the
    query/kv seq extent ``S`` (``None`` ⇒ symbolic ``seq_len``, ``seq_var`` its runtime symbol), and
    the 16-bit operand ``dtype`` (which ``mma`` atom). The attention-domain facts (which input is q /
    k / v, GQA grouping, causality) are NOT here — they are read structurally off the seed (the σ-tiled
    cells / the carrier / the score ``Select``), never named in a side helper."""

    out: str
    S: int | None
    D: int
    seq_var: str | None
    dtype: DataType

    @property
    def symbolic(self) -> bool:
        return self.seq_var is not None

    @property
    def atom_kind(self) -> str:
        return "mma_m16n8k16_bf16" if self.dtype == BF16 else "mma_m16n8k16_f16"


def flash_params(buffers: dict, out: str) -> FlashParams | None:
    """Derive the :class:`FlashParams` shape/dtype off the logical gmem buffers, or ``None`` when out
    of the 16-bit warp scope (a 4th rank-4 input = additive mask, a non-16-bit operand dtype, or a
    non-static head dim). ``D`` / ``S`` come from the rank-4 **output** shape (the query rows + head
    dim); the operand ``dtype`` from a rank-4 16-bit input. No q/k/v identification, no GQA group, no
    causal flag — those are structural facts read off the seed where they are needed."""
    ob = buffers.get(out)
    if ob is None or len(ob.shape) != 4:
        return None
    ins = [b for n, b in buffers.items() if len(b.shape) == 4 and n != out]
    if len(ins) != 3 or ins[0].dtype not in (F16, BF16):  # a 4th rank-4 input = additive mask — out of scope
        return None
    D = static_extent(ob.shape[3])  # head dim — the QK^T reduce / P@V output extent
    S = static_extent(ob.shape[2])  # query rows; None ⇒ symbolic seq_len
    if D is None:
        return None
    seq_var = None if S is not None else next(iter(ob.shape[2].expr.free_vars()))
    return FlashParams(out=out, S=S, D=D, seq_var=seq_var, dtype=ins[0].dtype)
