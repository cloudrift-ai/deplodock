"""Atomize — resolve the algebra→hardware-atom binding structurally.

The warp matmul materializer needs to know which operand is the mma ``a`` vs ``b`` (by
axis-in-index), whether ``b`` is transposed, the fold accumulator, and the projection epilogue.
:func:`semiring_binding` reads them **structurally** off the lowered ``CONTRACTION`` reduce loop
— the operand ``Load``\\ s indexed over the K axis, the fold ``Accum`` target — and builds an
:class:`AtomBinding` whose facts (the A/B operand ``Load``\\ s, ``acc``, epilogue) are then baked
onto the :class:`~deplodock.compiler.ir.tile.structural.Contraction` structural node at fork-emit
(``_schedule._contraction_node``). Reading off the annotated loop (not an op-tree node) is what let
the ``Semiring`` / ``Monoid`` op-tree node classes retire. The cooperative reduce needs no binding
here — its accumulator dtype + shuffle/tree mechanism are derived at materialize time
(``emit_combine`` off the carrier + ``ReduceStage.combine``).

**Called from ``020_schedule``, not a standalone pass.** The binding is resolved when the tiled
contraction leaf is built (``_warp_option`` / the tiled ``_tile_option``) — so an atom that
**cannot** be bound (e.g. a non-``Load`` operand: a computed-cone / demoted matmul) is rejected at
fork construction, alongside ``_check_warp_static_k``, instead of failing several passes later.
Leading ``_`` so the pass loader skips this module.

**Recursion seam (deferred — warp-flash).** Flash is a ``TWISTED`` kv ``Loop`` (online-softmax)
over a nested ``CONTRACTION`` score ``Loop``, so a recursive atomize would bind the inner QK^T /
PV with the same :func:`bind_contraction` the root uses — that function is loop-addressable for
exactly this reuse. The inner score ``Loop`` IS now a structural ``CONTRACTION`` (built by
``ops.contraction_loop``, the same builder a standalone matmul uses), but it is **not wired yet**:
it carries no per-loop ``TilePlan`` / inner geometry, so a tree-walk would bind nothing. Wiring it
requires warp-flash to first attach that inner geometry; until then it is intentionally absent."""

from __future__ import annotations

from deplodock.compiler.ir.axis import AxisRole
from deplodock.compiler.ir.stmt import Accum, Load, Loop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile import AtomBinding, Operand
from deplodock.compiler.pipeline.pipeline import LoweringError


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs (the materializer's helper)."""
    return {v for e in index for v in e.free_vars()}


def bind_contraction(loop: Loop, m_name: str, n_name: str, epilogue: Body) -> AtomBinding:
    """Resolve the operand→role :class:`AtomBinding` for a ``CONTRACTION`` reduce ``loop`` (the
    lowered ``Accum``-in-``Loop`` form) whose output is indexed by grid axes ``m_name`` /
    ``n_name``, with projection ``epilogue``.

    Reads the facts straight off the loop body — no op-tree node: the contraction operands are the
    ``Load``\\ s in the loop indexed over the reduce (K) axis; A/B are bound by which output axis
    each one's index carries; ``b_trans`` from B's last index component; the fold accumulator is the
    loop body's ``Accum`` target. A clean gmem-direct contraction has plain-``Load`` operands (a
    computed-cone / demoted matmul never reaches CONTRACTION — recognition leaves it a flat reduce),
    so an unbindable body (no m/n-bearing K-load) raises, matching the warp gmem-direct guard."""
    k_name = loop.axis.name
    loads = [s for s in loop.body if isinstance(s, Load) and k_name in _idx_vars(s.index)]
    a_leaf = next((ld for ld in loads if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in loads if n_name in _idx_vars(ld.index)), None)
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    acc = next((s.name for s in loop.body if isinstance(s, Accum)), None)
    if acc is None:
        raise LoweringError("warp tier: contraction loop has no fold accumulator")
    b_trans = k_name in b_leaf.index[-1].free_vars()  # B[n,k] (K last) vs canonical B[k,n]
    return AtomBinding(a=Operand(a_leaf, "a"), b=Operand(b_leaf, "b"), b_trans=b_trans, acc=acc, epilogue=epilogue)


def semiring_binding(node, grid) -> AtomBinding:
    """The root contraction's :class:`AtomBinding`: lower ``node`` to loop-IR, find its
    ``CONTRACTION`` reduce loop, take the projection ``epilogue`` (the stmts after the loop — the
    ``Map`` body, or empty for a bare contraction), and delegate to :func:`bind_contraction`.
    ``node`` is the kernel op, ``grid`` the placement's output axes."""
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    from deplodock.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    stmts = lower(node)
    ridx = next((i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.role is AxisRole.CONTRACTION), None)
    if ridx is None:
        raise LoweringError("warp tier: no contraction loop to bind")
    epilogue = Body(tuple(stmts[ridx + 1 :]))
    return bind_contraction(stmts[ridx], grid[-2].name, grid[-1].name, epilogue)


__all__ = ["bind_contraction", "semiring_binding"]
