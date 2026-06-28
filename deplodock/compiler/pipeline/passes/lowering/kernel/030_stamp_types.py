"""Stamp per-statement dtypes on every Load / Assign / Write.

Runs over a materialized ``KernelOp`` body, after ``010_materialize`` and
before the analytical codegen passes (demote, vectorize) see the IR. After
this pass those passes read dtypes directly off the IR instead of
re-deriving them from the op's ``inputs`` / ``outputs`` side channels.

Stamping rules:

- ``Load(input=B)`` — ``dtype = B``'s buffer dtype, read off the op's
  ``inputs`` / ``outputs`` (matcher-populated graph Tensors). A Load against
  a locally-declared buffer (smem) keeps the render-time fallback.
- ``Assign(op, args)`` — ``dtype = dtype_promote(op.name, [ssa[a] for a in
  args])``, the same rule ``Assign.render`` applies inline. Same-arg squares
  and ``pow`` are promoted to f32 (overflow guard, below).
- ``Write(output, value)`` — ``value_dtype = ssa[value]``. The destination
  buffer dtype stays a render-time concern; only the value side is stamped.
- ``Accum`` / ``Init`` / ``Pack`` / ``Unpack`` are already typed — register
  them in the running ``ssa_dtypes`` so downstream Assigns / Writes pick up
  the right arg dtype.

Idempotent: ``None``-stamped fields are filled, already-stamped ones kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.dtype import get as dtype_get
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Pack, Stmt, Unpack, Write
from deplodock.compiler.ir.stmt.base import dtype_promote
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


@dataclass
class _StampCtx:
    """Per-walk state. ``buf_dtypes`` maps an op I/O buffer name to its
    dtype; ``ssa_dtypes`` accumulates as Stmts are stamped."""

    buf_dtypes: dict[str, DataType]
    ssa_dtypes: dict[str, DataType] = field(default_factory=dict)


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    buf_dtypes = {n: t.dtype for n, t in {**op.inputs, **op.outputs}.items()}
    ctx = _StampCtx(buf_dtypes=buf_dtypes)
    new_body = _stamp_body(op.body, ctx)
    if new_body == op.body:
        raise RuleSkipped("every Load/Assign/Write already stamped")
    return KernelOp(body=new_body, name=op.name, knobs=dict(op.knobs))


def _stamp_body(body: Body, ctx: _StampCtx) -> Body:
    return Body(_stamp_stmt(s, ctx) for s in body)


def _stamp_stmt(s: Stmt, ctx: _StampCtx) -> Stmt:
    if isinstance(s, Load):
        return _stamp_load(s, ctx)
    if isinstance(s, Assign):
        return _stamp_assign(s, ctx)
    if isinstance(s, Write):
        return _stamp_write(s, ctx)
    if isinstance(s, Accum):
        return _stamp_accum(s, ctx)
    if isinstance(s, Init):
        ctx.ssa_dtypes[s.name] = s.dtype or F32
        return s
    if isinstance(s, Pack):
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    if isinstance(s, Unpack):
        ctx.ssa_dtypes[s.low_name] = s.lane_dtype
        ctx.ssa_dtypes[s.high_name] = s.lane_dtype
        return s
    # Block-structured stmts (Tile / Loop / StridedLoop / Cond / Monoid, …):
    # recurse through children via the generic ``nested()`` / ``with_bodies()``
    # protocol — no isinstance ladder.
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(_stamp_body(b, ctx) for b in nested))


def _stamp_accum(s: Accum, ctx: _StampCtx) -> Accum:
    """Freeze the accumulator dtype — the old ``020_place_inits`` policy, applied to the
    ``Accum`` itself (the carrier is the source of truth; ``Loop.render`` derives the seed
    from it). A **selecting** combine (``max`` / ``min``) picks an existing value, so it
    stays in the folded value's dtype — fp16 ``max`` accumulates in fp16. An
    **accumulating** combine (``sum`` / ``prod`` — incl. the matmul fold) builds magnitude,
    so it promotes fp16 to f32 to avoid precision loss / overflow. Already-stamped Accums
    keep their dtype."""
    if s.dtype is not None:
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    value_dt = ctx.ssa_dtypes.get(s.value) or F32
    dt = value_dt if s.op.selecting else F32
    ctx.ssa_dtypes[s.name] = dt
    return replace(s, dtype=dt)


def _stamp_load(s: Load, ctx: _StampCtx) -> Load:
    dt = s.dtype if s.dtype is not None else ctx.buf_dtypes.get(s.input)
    if dt is not None:
        for n in s.names:
            ctx.ssa_dtypes[n] = dt
        if s.dtype is None:
            return Load(names=s.names, input=s.input, index=s.index, dtype=dt)
    return s


def _stamp_assign(s: Assign, ctx: _StampCtx) -> Assign:
    if s.dtype is not None:
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    arg_dtypes = [(ctx.ssa_dtypes.get(a) or F32).name for a in s.args]
    result_dt = dtype_get(dtype_promote(s.op.name, arg_dtypes))
    # Overflow guard: a square (``x * x``) or a ``pow`` of an fp16 value can
    # blow past fp16's 65504 ceiling, giving inf → a garbage reduction
    # (RMSNorm's mean-of-squares). torch computes that reduction in fp32; do
    # the same here. Matmul — ``multiply`` of *distinct* args — keeps its fp16
    # path; only the same-arg square / pow are promoted.
    if result_dt == F16 and _is_overflow_prone(s):
        result_dt = F32
    ctx.ssa_dtypes[s.name] = result_dt
    return replace(s, dtype=result_dt)


def _is_overflow_prone(s: Assign) -> bool:
    """True for elementwise ops that can overflow fp16 from in-range inputs —
    the square ``multiply(a, a)`` and any ``pow``. Distinct-arg ``multiply``
    (matmul) is excluded."""
    if s.op.name == "pow":
        return True
    return s.op.semiring_product and len(s.args) == 2 and s.args[0] == s.args[1]


def _stamp_write(s: Write, ctx: _StampCtx) -> Write:
    if s.value_dtype is not None:
        return s
    # In a multi-value (vector) Write all values share the same SSA dtype by
    # construction (vectorize_stores widens runs of same-dtype values).
    dt = ctx.ssa_dtypes.get(s.values[0])
    if dt is None:
        return s
    return Write(output=s.output, index=s.index, values=s.values, value_dtype=dt)
