"""High-level algebraic op tree — the geometry-free compute layer.

A kernel's compute is a tree of two primitives:

- :class:`Map` — a pointwise op ``out = φ(args)`` (the lift / elementwise).
- :class:`Reduce` — a fold over one axis through a carrier (``Monoid`` + ``Twist``),
  whose **partials are nested ops** (a ``Map``, a ``TensorRef`` load, or another
  ``Reduce``). A contraction (matmul / the SEMIRING) is just ``Reduce(⊕=+)`` over
  ``Map(⊗=·)``; flash is ``Reduce(lse)`` over ``(Map(scale, Reduce(+)∘Map(·)), V)``.

The tree is **geometry-free**: it names the axes it folds and the operands it
reads (``TensorRef`` = buffer + index exprs), but says nothing about threads /
tiling. Geometry is the separate ``Tile(op, placement)`` layer.

:func:`lower` walks the tree and emits ordinary **loop-IR** stmts — the carrier
generates the structure (``Init`` ← identity, the streaming ``Loop`` + the carrier
fold ← ``merge``), the nested ops generate the lift, and the operands' index exprs
come straight from the ``TensorRef``s. So there is no per-kernel builder: a kernel
*is* the lowered tree. (Today ``lower`` targets a single-component carrier — the
contraction / plain reduce; multi-component carriers, e.g. flash's ``(m,d,O)``,
land in the next slice once per-state ``Init`` is plumbed.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt import Assign, Body, Init, Load, Loop
from deplodock.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class TensorRef:
    """An operand read: a buffer plus the index exprs that address it (over the
    iteration axes). The index is the *only* place layout lives — staging / the
    mma fragment read it; nothing is duplicated into the carrier."""

    buf: str
    index: tuple[Expr, ...]


# A partial source: a nested op (Map / Reduce) or a direct operand load (TensorRef).
Source = "Map | Reduce | TensorRef"


@dataclass(frozen=True)
class Map:
    """Pointwise lift: ``out = op(args)``. Each arg is a :class:`TensorRef` (→ a
    ``Load``), a nested op (→ its result), or a bare SSA name already in scope."""

    out: str
    op: ElementwiseImpl
    args: tuple


@dataclass(frozen=True)
class Reduce:
    """Fold over ``axis`` through ``carrier`` (a ``Monoid`` — its ``Twist`` is the
    combine). ``partials`` produce the carrier's ``partial`` contributions (one
    source per ``carrier.partial`` name, in order). ``init_ops`` gives the per-state
    identity-bearing op for the enclosing ``Init`` (one per ``carrier.state``).
    ``out`` is the carried state read after the fold."""

    out: str
    axis: Axis
    carrier: object  # Monoid
    partials: tuple
    init_ops: tuple[ElementwiseImpl, ...]
    dtype: DataType = field(default=F32)


def _lower_source(src, name: str, ssa) -> list[Stmt]:
    """Emit stmts that bind ``name`` to the value produced by ``src`` (a nested
    ``Map`` / ``Reduce`` or a ``TensorRef`` load). ``ssa`` is a fresh-name counter."""
    if isinstance(src, TensorRef):
        return [Load(name=name, input=src.buf, index=src.index)]
    if isinstance(src, Map):
        return _lower_map(src, ssa)
    if isinstance(src, Reduce):
        return lower(src, ssa)
    raise TypeError(f"_lower_source: unsupported partial source {type(src).__name__}")


def _lower_map(m: Map, ssa) -> list[Stmt]:
    """Lower a ``Map`` to ``[<arg binds>…, Assign(out, op, arg_names)]``."""
    stmts: list[Stmt] = []
    arg_names: list[str] = []
    for a in m.args:
        if isinstance(a, str):
            arg_names.append(a)
        elif isinstance(a, TensorRef):
            nm = ssa()
            stmts.append(Load(name=nm, input=a.buf, index=a.index))
            arg_names.append(nm)
        elif isinstance(a, (Map, Reduce)):
            nm = a.out
            stmts += _lower_source(a, nm, ssa)
            arg_names.append(nm)
        else:
            raise TypeError(f"_lower_map: unsupported arg {type(a).__name__}")
    stmts.append(Assign(name=m.out, op=m.op, args=tuple(arg_names)))
    return stmts


def lower(r: Reduce, ssa=None) -> list[Stmt]:
    """Lower a :class:`Reduce` to loop-IR stmts: an ``Init`` per carried state (the
    carrier's identity), then the streaming ``Loop`` whose body computes the
    partials and applies the carrier fold. Pure structure-from-carrier — no
    per-kernel assembly."""
    if ssa is None:
        ssa = _counter()
    out: list[Stmt] = []
    states = r.carrier.state
    for s, op in zip(states, r.init_ops, strict=True):
        out.append(Init(name=s, op=op, dtype=r.dtype))
    body: list[Stmt] = []
    for src, pname in zip(r.partials, r.carrier.partial, strict=True):
        body += _lower_source(src, pname, ssa)
    body.append(r.carrier)  # the Monoid fold (its Twist is the combine)
    out.append(Loop(axis=r.axis, body=Body(tuple(body))))
    return out


def _counter():
    n = [0]

    def fresh() -> str:
        n[0] += 1
        return f"_t{n[0]}"

    return fresh


__all__ = ["TensorRef", "Map", "Reduce", "lower"]
