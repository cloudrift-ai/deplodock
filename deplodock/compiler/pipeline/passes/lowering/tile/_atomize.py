"""Atomize — resolve the algebra→hardware-atom binding structurally.

The warp matmul materializer used to ``lower()`` the ``Semiring`` to flat loop-IR and then
re-recognize which operand is the mma ``a`` vs ``b`` (by axis-in-index), whether ``b`` is
transposed, the fold accumulator, and the projection epilogue. Every one of those facts is
already first-class on the ``Semiring`` node (``operands`` / ``fold`` / ``reduce_axis`` /
``out``) and the grid. :func:`semiring_binding` reads them **structurally** — off each
operand's own leaf ``Load`` index, never a flattened loop — and builds an :class:`AtomBinding`
that rides the **schedule** (a sibling of the ``TilePlan`` decision; ``op_cache_key`` digests
``lower(op.op)``, not the schedule, so the perf / prior cache stays byte-identical). The
cooperative reduce needs no binding here — its accumulator dtype + shuffle/tree mechanism are
derived at materialize time (``emit_combine`` off the carrier + ``ReduceStage.combine``).

**Called from ``020_schedule``, not a standalone pass.** The binding is resolved when the warp
option is built (``_warp_option``) — so an atom that **cannot** be bound (e.g. a non-``Load``
operand: a computed-cone / demoted matmul) is rejected at fork construction, alongside
``_check_warp_static_k``, instead of failing several passes later. Leading ``_`` so the pass
loader skips this module.

**Recursion seam (deferred — warp-flash).** Flash is a ``Monoid`` (online-softmax) over a
nested contraction, so a kind-recursive atomize would bind the inner QK^T / PV with the same
:func:`bind_contraction` the root uses — that function is node-addressable for exactly this
reuse. It is **not wired yet** because flash's inner contractions are not structural
``Semiring`` nodes today (``_flash._flash_op`` ``lower()``-s the QK score straight into loop-IR
and carries no per-node ``TilePlan``). Wiring it requires warp-flash to first keep the inner
contractions as ``Semiring`` nodes + attach inner geometry; until then a tree-walk would bind
nothing, so it is intentionally absent."""

from __future__ import annotations

from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile import AtomBinding, Operand
from deplodock.compiler.pipeline.pipeline import LoweringError


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs (the materializer's helper)."""
    return {v for e in index for v in e.free_vars()}


def _operand_leaf(operand) -> Load:
    """The buffer ``Load`` of a one-``Load`` operand ``Map`` — Phase-1's gmem-direct
    contraction operand. A non-``Load`` leaf (a nested reduction / staged-fill prologue) is
    out of scope here (it bails, matching the materializer's gmem-direct guard)."""
    leaf = operand.body[-1] if isinstance(operand, Map) and operand.body else None
    if not isinstance(leaf, Load):
        raise LoweringError("warp tier: a contraction compute prologue isn't supported (gmem-direct, no staging)")
    return leaf


def bind_contraction(semi, m_name: str, n_name: str, epilogue: Body) -> AtomBinding:
    """Resolve the operand→role :class:`AtomBinding` for a contraction ``semi`` whose output is
    indexed by grid axes ``m_name`` / ``n_name``, with projection ``epilogue``.

    **Node-addressable** — it binds any ``Semiring`` node, not just a kernel root — so warp-flash
    can reuse it on flash's nested QK^T / PV contractions (the recursion seam in the module
    docstring). A/B are bound by which output axis each operand's OWN leaf ``Load`` index carries
    (Phase 1: each operand is a one-``Load`` ``Map``); ``b_trans`` from B's last index component."""
    k_name = semi.reduce_axis.name
    leaves = [_operand_leaf(o) for o in semi.operands]
    a_leaf = next((ld for ld in leaves if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in leaves if n_name in _idx_vars(ld.index)), None)
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    b_trans = k_name in b_leaf.index[-1].free_vars()  # B[n,k] (K last) vs canonical B[k,n]
    return AtomBinding(a=Operand(a_leaf, "a"), b=Operand(b_leaf, "b"), b_trans=b_trans, acc=semi.out, epilogue=epilogue)


def semiring_binding(node, grid) -> AtomBinding:
    """The root contraction's :class:`AtomBinding`: extract the ``Semiring`` + output grid +
    projection epilogue (the ``Map`` body, or empty for a bare contraction) and delegate to
    :func:`bind_contraction`. ``node`` is the kernel op (a ``Semiring`` / ``Map``), ``grid`` the
    placement's output axes."""
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    epilogue = node.body if isinstance(node, Map) else Body(())
    return bind_contraction(node.reduce_node, grid[-2].name, grid[-1].name, epilogue)


__all__ = ["bind_contraction", "semiring_binding"]
