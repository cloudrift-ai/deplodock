"""The geometry-free compute layer — the lift wrapper and its lowering.

A kernel's compute is a :class:`~deplodock.compiler.ir.tile.structural.Map` (re-exported here) — a
:class:`~deplodock.compiler.ir.stmt.body.Body` of loop-IR stmts holding the per-cell compute. A
reduction is a ``Map`` whose body contains the **annotated reduce ``Loop``** (its
:class:`~deplodock.compiler.ir.axis.AxisRole` + :class:`~deplodock.compiler.ir.stmt.algebra.Carrier`
stamped by recognition) followed by the post-reduce projection; a contraction is a ``Map`` whose
reduce ``Loop`` is ``CONTRACTION`` (the ``⊗`` lift sits in the loop body). The algebra is read
**structurally** off the annotated loop, never a stored node kind — the ``Monoid`` / ``Semiring``
node wrappers are retired.

This module is the thin lowering of that wrapper to loop IR (:func:`lower` — the body verbatim,
the carriers already dissolved into loose folds at recognition) plus the structural reads
(:func:`axis_role` / :func:`reduce_loop`) and the shared contraction-loop builder
(:func:`contraction_loop`)."""

from __future__ import annotations

from deplodock.compiler.ir.axis import AxisRole
from deplodock.compiler.ir.stmt import Assign, Body, Loop, StridedLoop
from deplodock.compiler.ir.stmt.base import Stmt, pretty_body
from deplodock.compiler.ir.tile.structural import Map, Reduction


def reduce_loop(op):
    """The kernel's outermost **annotated** reduce ``Loop`` (its ``carrier`` set by recognition),
    or ``None`` for a pure pointwise / flat-fallback ``Map`` (no annotated reduce). A
    :class:`~deplodock.compiler.ir.tile.structural.Reduction` synthesizes its loop directly; a ``Map``
    is read off the top-level body — the annotated reduce loop is a top-level stmt (a
    single-flat-reduce cell); a nested / multi reduce stays un-annotated (flat fallback) and is
    invisible here, so it materializes on the scalar tier."""
    if isinstance(op, Reduction):
        return op.loop
    if getattr(op, "source", None) is not None:
        return reduce_loop(op.source)  # a Map projecting over a Reduction / Contraction source
    for s in op.body:
        if isinstance(s, (Loop, StridedLoop)) and s.carrier is not None:
            return s
    return None


def axis_role(op) -> AxisRole:
    """The reduce :class:`~deplodock.compiler.ir.axis.AxisRole` of a kernel's outermost reduction,
    read **structurally** off the annotated reduce loop (no stored kind tag): a ``CONTRACTION``
    contraction, a ``TWISTED`` (online-softmax / flash) or ``PLANAR`` (plain ``sum`` / ``max`` /
    ``mean``) reduce, or ``FREE`` for a pure pointwise / flat-fallback ``Map``. This is what the
    schedule / materialize passes dispatch on."""
    rl = reduce_loop(op)
    return rl.role if rl is not None else AxisRole.FREE


def lower(op) -> list[Stmt]:
    """Lower the lift wrapper to loop-IR stmts — the ``Map``'s body verbatim. The carriers are
    already dissolved into loose fold ``Accum``\\ s (and the streaming ``merge`` for a twisted
    carrier) at recognition, and the reduce ``Loop``\\ s carry their role/carrier annotations, so
    one ``lower`` call emits the kernel's per-cell body with nothing left to expand."""
    if isinstance(op, Map):
        prefix = lower(op.source) if op.source is not None else []
        return [*prefix, *op.body]  # the source's reduce/contract loop nest, then the projection body
    if isinstance(op, Reduction):
        return op.lower()
    raise TypeError(f"lower: expected a Map lift wrapper or Reduction, got {type(op).__name__}")


def contraction_loop(lift, fold, operand_bodies, reduce_axis) -> Loop:
    """Build the contraction (matmul) reduce ``Loop`` in the recognizable ``Accum``-in-``Loop``
    form: expand each operand source's stmts (siblings), the ``⊗`` lift ``Assign``
    (``fold.value = lift(operands…)``), and the additive ``fold`` ⊕ (its identity init is the
    ``Loop``'s immediate-``Accum`` prelude — no explicit ``Init``). The loop is stamped
    ``CONTRACTION`` + the degenerate carrier of its additive fold (``fold.as_carrier()``), so the
    K loop carries its combine and the warp / cooperative tiers read the operands structurally off
    the body. Shared by the flash score producer and the scalar register-tile contraction."""
    body: list[Stmt] = []
    names: list[str] = []
    for ob in operand_bodies:
        stmts = list(ob)
        body += stmts
        names.append(stmts[-1].defines()[-1])
    body.append(Assign(name=fold.value, op=lift, args=tuple(names)))
    body.append(fold)
    return Loop(axis=reduce_axis, body=Body(tuple(body)), role=AxisRole.CONTRACTION, carrier=fold.as_carrier())


def pretty(op, indent: str = "") -> list[str]:
    """Structurally pretty-print a kernel op (for dumps) — a
    :class:`~deplodock.compiler.ir.tile.structural.Reduction` as a typed header over its synthesized
    loop nest, the ``Map``'s body (its annotated reduce ``Loop`` + projection), or a bare stmt's own
    pretty."""
    if isinstance(op, Reduction):
        head = f"{indent}Reduction[{op.axis.name}] {op.role.name.lower()}"
        return [head, *pretty_body(Body(op.lower()), indent + "    ")]
    if isinstance(op, Map):
        src = pretty(op.source, indent) if op.source is not None else []
        return [*src, *pretty_body(op.body, indent)]
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = ["Map", "axis_role", "contraction_loop", "lower", "pretty", "reduce_loop"]
