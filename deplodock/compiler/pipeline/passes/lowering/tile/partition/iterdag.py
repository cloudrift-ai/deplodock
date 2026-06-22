"""The iteration-DAG view — the body *is* the DAG.

``iter_dag(loop_op)`` is a **derived view** over the ``LoopOp`` body (computed on
demand, exactly like ``Loop.algebra_kind`` — zero serialization, zero
``op_cache_key`` surface, always consistent with the body). It tags every index
axis by role (``PARALLEL`` free axis / ``REDUCE`` contraction axis) and, for a
reduce axis, the carrier whose algebra a decomposition move queries
(``associative`` / ``commutative`` / ``has_identity``). See
``plans/algebra-licensed-decomposition-moves.md`` (phase 2).

This is the one structure the partition consumes. The four regime skeletons
(``PointwiseSkeleton`` / ``MatmulSkeleton`` / ``CoopReduceSkeleton`` /
``FlashSkeleton``) are *projections* of this view — ``walk_nest`` builds the DAG
and reads the skeleton fields off it, so the DAG is proven to carry everything
the skeletons did (one source of truth). Later phases dissolve the skeletons
entirely in favour of ``build_partition(dag)``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Loop, ReduceCarrier, Stmt
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import _split_leading_non_loops


class AxisRole(enum.Enum):
    """An index axis' algebraic role in the iteration DAG."""

    PARALLEL = "parallel"  # a free (map) output axis — no carrier, no recombine
    REDUCE = "reduce"  # a contraction axis — carries an algebra the move queries


@dataclass(frozen=True)
class AxisNode:
    """One index axis of the nest, tagged by role + (for a reduce) its carrier.

    ``parent`` threads the loop-nest containment (the free-axis chain
    outermost→innermost; a nested QK^T reduce points at the streaming KV reduce).
    ``body`` is the statements scoped immediately at this axis. The carrier is the
    algebra the decomposition moves read — ``None`` for a ``PARALLEL`` axis."""

    loop: Loop
    role: AxisRole
    carrier: ReduceCarrier | None
    algebra: AlgebraKind | None
    parent: AxisNode | None
    body: tuple[Stmt, ...]

    @property
    def axis(self) -> Axis:
        return self.loop.axis

    @property
    def symbolic(self) -> bool:
        return not self.loop.axis.extent.is_static

    @property
    def extent(self) -> int:
        """Static extent, or the ``Dim`` hint for a symbolic axis (``0`` if no
        hint) — the size a tile is shaped for."""
        ext = self.loop.axis.extent
        return ext.as_static() if ext.is_static else (ext.hint or 0)


@dataclass(frozen=True)
class IterDag:
    """The derived iteration-DAG view of one ``LoopOp`` body.

    - ``parallel`` — the free-axis chain, outermost-first (the ``PARALLEL``
      nodes; nesting is ``parent``).
    - ``reduce`` — the contraction-axis nodes scoped inside the inner body (the
      ``REDUCE`` nodes; a flash nest's nested QK^T reduce has the streaming KV
      reduce as ``parent``).
    - ``leading`` — non-loop statements before the free chain (per-CTA prologue).
    - ``mid`` — per-outer-axis precompute statements between chain levels that
      ride the inner tile (recomputed per inner element).
    - ``inner_body`` — ``mid + innermost-free-loop body``: the statements the
      reduce axes live in, the tile the σ-rewrite walks.
    """

    parallel: tuple[AxisNode, ...]
    reduce: tuple[AxisNode, ...]
    leading: tuple[Stmt, ...]
    mid: tuple[Stmt, ...]
    inner_body: tuple[Stmt, ...]

    @property
    def algebras(self) -> set[AlgebraKind]:
        """The algebra kinds of the reduce axes (empty for pointwise)."""
        return {n.algebra for n in self.reduce if n.algebra is not None}


def _carrier_of(loop: Loop) -> ReduceCarrier | None:
    """The ``ReduceCarrier`` at the loop's immediate body level (the algebra a
    move queries), or ``None`` for a free loop / a loop that only nests a deeper
    reduce."""
    for s in loop.body:
        if isinstance(s, ReduceCarrier):
            return s
    return None


def _reduce_nodes(stmts: tuple[Stmt, ...], parent: AxisNode | None) -> list[AxisNode]:
    """Walk ``stmts`` recursively for reduce loops, building ``REDUCE`` nodes.

    Recurses through EVERY nested body (matching ``Body.iter_of_type(Loop)``), so
    the node set equals the legacy walk's ``reduce_loops``. ``parent`` tracks the
    nearest enclosing reduce node — a nested contraction (flash's QK^T inside the
    KV stream) points at its enclosing reduce; a reduce under a non-reduce loop
    keeps the outer reduce (or ``None``) as parent."""
    out: list[AxisNode] = []
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            node = AxisNode(
                loop=s,
                role=AxisRole.REDUCE,
                carrier=_carrier_of(s),
                algebra=s.algebra_kind,
                parent=parent,
                body=tuple(s.body),
            )
            out.append(node)
            out.extend(_reduce_nodes(tuple(s.body), node))
        else:
            for nested in s.nested():
                out.extend(_reduce_nodes(tuple(nested), parent))
    return out


def _free_chain_nodes(body: tuple[Stmt, ...]) -> tuple[tuple[AxisNode, ...], tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Build the ``PARALLEL`` chain nodes (outermost-first), plus the leading and
    mid statement tuples — the node form of ``walk._free_chain``."""
    leading, rest = _split_leading_non_loops(body)
    nodes: list[AxisNode] = []
    mid: list[Stmt] = []
    cur = rest
    parent: AxisNode | None = None
    while True:
        cur_lead, cur_rest = _split_leading_non_loops(cur)
        if len(cur_rest) == 1 and isinstance(cur_rest[0], Loop) and not cur_rest[0].is_reduce:
            mid.extend(cur_lead)
            lp = cur_rest[0]
            node = AxisNode(loop=lp, role=AxisRole.PARALLEL, carrier=None, algebra=None, parent=parent, body=tuple(lp.body))
            nodes.append(node)
            parent = node
            cur = tuple(lp.body)
        else:
            break
    return tuple(nodes), tuple(leading), tuple(mid)


def iter_dag(loop_op: LoopOp) -> IterDag:
    """Derive the iteration-DAG view of ``loop_op`` (computed on demand, never
    stored). The free-axis chain + the reduce axes scoped in the inner body, each
    tagged by role and (for a reduce) carrier."""
    body = tuple(loop_op.body)
    parallel, leading, mid = _free_chain_nodes(body)
    inner_body: tuple[Stmt, ...] = ()
    if parallel:
        inner_body = tuple(mid) + parallel[-1].body
    reduce_parent = parallel[-1] if parallel else None
    reduce_nodes = tuple(_reduce_nodes(inner_body, reduce_parent))
    return IterDag(parallel=parallel, reduce=reduce_nodes, leading=leading, mid=mid, inner_body=inner_body)
