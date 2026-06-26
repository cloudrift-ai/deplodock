"""The iteration-DAG view — the body *is* the DAG.

``iter_dag(loop_op)`` is a **derived view** over the ``LoopOp`` body (computed on
demand, exactly like ``Loop.algebra_kind`` — zero serialization, zero
``op_cache_key`` surface, always consistent with the body). It tags every index
axis by role (``PARALLEL`` free axis / ``REDUCE`` contraction axis) and, for a
reduce axis, the carrier whose algebra a decomposition move queries
(``associative`` / ``commutative`` / ``has_identity``).

This is **the one structure the partition consumes** — the typed regime skeletons
are gone (phase 6). ``tree.classify(dag)`` tags the regime off this view and
``tree.build_partition(dag)`` factors its axes; the free-axis / K-info accessors
(``inner_n`` / ``outer_m`` / ``k_node`` / ``k_extent`` /
``k_bound``) replace the skeletons' fields.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Load, Loop, Monoid, ReduceCarrier, Stmt


def _split_leading_non_loops(body: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Split a body into its leading non-loop statements (per-CTA / per-axis
    prologue) and the loop-bearing remainder."""
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


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
    def extent(self) -> int:
        """Static extent, or the ``Dim`` hint for a symbolic axis (``0`` if no
        hint) — the size a tile is shaped for."""
        ext = self.loop.axis.extent
        return ext.as_static() if ext.is_static else (ext.hint or 0)


@dataclass(frozen=True)
class Contraction:
    """A **SEMIRING contraction** read off the DAG — the algebra unit a streaming ``Monoid``
    composes over. A reduce-axis node folding a product of operands into a result edge, producing
    one free **column** output. Algebra-generic, no attention vocabulary: a matmul, flash's QK^T
    (``col`` = the hinge ``kv`` it scores), the embedded P@V (``col`` = the head output ``d``).

    Wraps the reduce :class:`AxisNode`, delegating the node reads its consumers use
    (``loop`` / ``body`` / ``axis`` / ``algebra`` / ``parent`` / ``extent``) so a contraction *is*
    its node from the outside, while adding the composition-level facts (``result`` / ``col``)."""

    node: AxisNode  # the SEMIRING reduce node (the contraction axis; ``algebra`` is SEMIRING)
    result: str  # the SSA name of the contraction's output edge (QK^T: the score)
    col: AxisNode  # the free-output column this contraction produces (QK^T: the hinge ``kv``)

    @property
    def axis(self) -> Axis:
        return self.node.axis

    @property
    def loop(self) -> Loop:
        return self.node.loop

    @property
    def body(self) -> tuple[Stmt, ...]:
        return self.node.body

    @property
    def algebra(self) -> AlgebraKind | None:
        return self.node.algebra

    @property
    def parent(self) -> AxisNode | None:
        return self.node.parent

    @property
    def extent(self) -> int:
        return self.node.extent


@dataclass(frozen=True)
class ContractionChain:
    """The hierarchical **MONOID(SEMIRING)** algebra of a streaming flash, *as a composition* —
    not a flat parse. A ``Monoid`` carrier (the online softmax, folding over the hinge) composed
    over an inner SEMIRING :class:`Contraction` (the QK^T score)::

        S[m,kv] = Σ_e Q[m,e]·K[kv,e]     # inner SEMIRING Contraction: reduce ``e``, col ``kv``
        P[m,kv] = softmax_kv(S)           # the MONOID carrier, over the hinge ``kv``
        O[m,d]  = Σ_kv P[m,kv]·V[kv,d]    # outer SEMIRING (the carrier's twisted P@V): reduce ``kv``

    The chain **invariant is the shared hinge**: the inner contraction's free-output column
    (:attr:`Contraction.col`) IS the carrier's / P@V's reduction axis (:attr:`hinge`) — enforced in
    ``__post_init__``, so a malformed composition is unrepresentable. The outer P@V is not a distinct
    ``Loop``; it lives in the carrier's twisted ``O = O·α + p·v`` accumulation, characterized by
    ``carrier`` + ``hinge``. Flash is therefore *selected structurally* — "a ``Monoid`` whose folded
    inner operand is a SEMIRING ``Contraction``" — never matched as a named shape.

    **Algebra** (this composition) is kept apart from **geometry** (the free-axis roles :attr:`m` /
    :attr:`d` / :attr:`grid`, classified once by :func:`partition_free_axes`) and from the
    **edges** (:attr:`score` / :attr:`out_index`, derived) — geometry and edges are views over the
    composition, never peer fields. A **derived** view (computed on demand by :attr:`IterDag.chain`,
    like :attr:`IterDag.streaming`) — never stored, so it can't drift and doesn't enter
    ``op_cache_key``."""

    carrier: Monoid  # the online-softmax MONOID carrier (its twisted accumulation embeds the outer P@V SEMIRING)
    hinge: AxisNode  # the streaming reduce node — the chain's shared axis (``inner.col is hinge``)
    inner: Contraction  # the QK^T inner SEMIRING contraction, composed
    m: AxisNode  # geometry: the query-row free axis (shared free output of both contractions) — binds THREAD
    d: AxisNode  # geometry: the P@V head-output free axis (the outer SEMIRING's col) — the register vector ``O[d]``
    grid_nodes: tuple[AxisNode, ...]  # geometry: the shared broadcast / batch free axes — bind GRID

    def __post_init__(self) -> None:
        if self.inner.col is not self.hinge:  # the carried-chain invariant: inner's col == the reduced hinge
            raise ValueError("ContractionChain: inner contraction's column must be the hinge (the carried-chain invariant)")

    @property
    def hinge_name(self) -> str:
        return self.hinge.axis.name

    @property
    def inner_name(self) -> str:
        return self.inner.axis.name

    @property
    def score(self) -> str:
        """The INLINE score edge — the inner contraction's result (the carrier's first partial)."""
        return self.inner.result

    @property
    def m_axis(self) -> Axis:
        """The query-row free axis (geometry view) — binds THREAD."""
        return self.m.axis

    @property
    def d_axis(self) -> Axis:
        """The P@V head-output free axis (geometry view) — the register vector ``O[d]``."""
        return self.d.axis

    @property
    def grid(self) -> tuple[Loop, ...]:
        """The shared (batch / head) free loops, bound GRID."""
        return tuple(n.loop for n in self.grid_nodes)

    @property
    def value_load(self) -> Load:
        """The V operand ``Load`` (``carrier.partial[1]``) at the hinge body's top level —
        the outer P@V's value operand, read off the DAG (not the lowered tile block)."""
        return next(s for s in self.hinge.body if isinstance(s, Load) and s.name == self.carrier.partial[1])

    @property
    def out_index(self) -> tuple[Var, Var]:
        """The inner QK^T contraction's score output coords ``(m, kv)`` — the M=query-row /
        N=kv-hinge coordinates the transposed-B A/B classification reads. The score is an
        INLINE register fragment (no ``Write`` to derive them from), so the chain supplies
        them explicitly."""
        return (Var(self.m_axis.name), Var(self.hinge_name))


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
        """The algebra kinds of the reduce axes (empty for a MAP nest)."""
        return {n.algebra for n in self.reduce if n.algebra is not None}

    @property
    def streaming(self) -> bool:
        """The streaming-flash schedule — a tuple ``Monoid`` carrier streaming over a
        *nested* contraction (flash's QK^T reduce inside the KV stream). A **derived**
        structural property (a reduce whose parent is a reduce, with a ``Monoid``
        carrier), computed on demand exactly like ``algebras`` — never stored, so it
        can't drift and doesn't enter ``op_cache_key``. A move that needs the
        distinction (the streaming fork, the knob-pin validator) queries this."""
        nested_reduce = any(n.parent is not None and n.parent.role is AxisRole.REDUCE for n in self.reduce)
        has_monoid = any(isinstance(n.carrier, Monoid) for n in self.reduce)
        return nested_reduce and has_monoid

    @property
    def chain(self) -> ContractionChain | None:
        """The streaming flash's **MONOID(SEMIRING)** composition for this nest, else ``None``.
        Derived on demand and self-validating: the ``Monoid`` carrier over the hinge axis ``kv``,
        the inner SEMIRING :class:`Contraction` (QK^T) nested inside it whose free-output column IS
        the hinge, and the geometry roles (query ``m`` / head ``d`` / batch ``grid``) classified
        generically by :func:`partition_free_axes`. Returns ``None`` unless the nest is a
        well-formed, separable composition — see :class:`ContractionChain`."""
        if not self.streaming:
            return None
        hinge = next((n for n in self.reduce if isinstance(n.carrier, Monoid)), None)
        if hinge is None or not hinge.carrier.partial:
            return None
        inner_node = next(
            (
                n
                for n in self.reduce
                if n.parent is not None and n.parent.axis.name == hinge.axis.name and n.algebra is AlgebraKind.SEMIRING
            ),
            None,
        )
        if inner_node is None:
            return None
        value_load = next((s for s in hinge.body if isinstance(s, Load) and s.name == hinge.carrier.partial[1]), None)
        if value_load is None:
            return None
        # Geometry: the free axes split by the two contraction operands' footprints — query ``m``
        # (inner-only), head ``d`` (outer/V-only), ``grid`` (shared). Role-neutral; the algebra
        # composition below carries no attention vocabulary.
        m_nodes, d_nodes, grid_nodes = partition_free_axes(
            self.parallel, _free_var_footprint(inner_node.body), _free_var_footprint((value_load,))
        )
        if len(m_nodes) != 1 or len(d_nodes) != 1:
            # Not a separable MONOID(SEMIRING): the build moves need exactly one query-row and one
            # head-output free axis. A non-separable nest isn't a buildable chain today — routed as
            # a plain cooperative monoid (the one place the chain concept meets the flash geometry).
            return None
        inner = Contraction(node=inner_node, result=hinge.carrier.partial[0], col=hinge)
        return ContractionChain(carrier=hinge.carrier, hinge=hinge, inner=inner, m=m_nodes[0], d=d_nodes[0], grid_nodes=grid_nodes)

    # --- Free-axis accessors (the tiled output axes). Replace the skeleton's
    # ``inner_n`` / ``outer_m`` fields — the partition consumes
    # these straight off the DAG. ---

    @property
    def inner_n(self) -> AxisNode:
        """The innermost free (PARALLEL) axis — the ``N`` tile axis."""
        return self.parallel[-1]

    @property
    def outer_m(self) -> AxisNode | None:
        """The next-out free axis — the ``M`` tile axis, or ``None`` for a 1-D
        (single free axis) nest."""
        return self.parallel[-2] if len(self.parallel) >= 2 else None

    # --- Reduce-axis (K) accessors. Replace the skeleton's ``k_loop`` / ``k_name``
    # / ``k_extent`` / ``k_bound`` fields. ``k_node`` is the primary reduce. ---

    @property
    def k_node(self) -> AxisNode:
        """The primary reduce (contraction) axis node."""
        return self.reduce[0]

    @property
    def k_extent(self) -> int:
        """The reduce-tiling extent: static K, or the ``Dim`` hint for a symbolic
        (masked) K — uniform via ``AxisNode.extent``."""
        return self.k_node.extent

    @property
    def k_bound(self):
        """The symbolic-K runtime boundary ``Expr`` (``None`` for a static K)."""
        ext = self.k_node.loop.axis.extent
        return None if ext.is_static else ext.expr


def _free_var_footprint(stmts: tuple[Stmt, ...]) -> set[str]:
    """The union of index ``Var`` names every ``Load`` in ``stmts`` references — an operand's
    free-axis footprint (non-Load statements contribute nothing)."""
    return {v for s in stmts if isinstance(s, Load) for e in s.index for v in e.free_vars()}


def partition_free_axes(
    free_nodes: tuple[AxisNode, ...], footprint_a: set[str], footprint_b: set[str]
) -> tuple[tuple[AxisNode, ...], tuple[AxisNode, ...], tuple[AxisNode, ...]]:
    """Split free-axis nodes by two operand footprints (sets of axis names): those whose axis is in
    A's footprint only, in B's only, and the rest (in both, or in neither — the broadcast / batch
    axes a two-operand composition shares). A **role-neutral** classification off the def-use, no
    shape or attention vocabulary — the carried-contraction chain reads ``(a_only, b_only, rest)``
    as ``(query rows m, value output d, grid)``, but any two-operand composition can reuse it.
    Preserves ``free_nodes`` order within each group."""
    a_only: list[AxisNode] = []
    b_only: list[AxisNode] = []
    rest: list[AxisNode] = []
    for n in free_nodes:
        in_a, in_b = n.axis.name in footprint_a, n.axis.name in footprint_b
        (a_only if (in_a and not in_b) else b_only if (in_b and not in_a) else rest).append(n)
    return tuple(a_only), tuple(b_only), tuple(rest)


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
    if not parallel:
        leading_top, rest = _split_leading_non_loops(body)
        if any(isinstance(s, Loop) and s.is_reduce for s in rest):
            # Global reduce (``x[K] → s[1]``): no free output axis, so the body is
            # one top-level reduce loop + Write. Synthesize a degenerate size-1
            # PARALLEL row so the cooperative-reduce regime tiles it as a
            # single-CTA tree reduce — the row var binds nothing in the body (the
            # Write indexes a literal 0), so it costs one grid dim of extent 1.
            from deplodock.compiler.dim import Dim  # noqa: PLC0415

            row = Axis("_grow", Dim(1))
            node = AxisNode(loop=Loop(axis=row, body=rest), role=AxisRole.PARALLEL, carrier=None, algebra=None, parent=None, body=rest)
            reduce_nodes = tuple(_reduce_nodes(rest, node))
            return IterDag(parallel=(node,), reduce=reduce_nodes, leading=leading_top, mid=(), inner_body=rest)
    inner_body: tuple[Stmt, ...] = ()
    if parallel:
        inner_body = tuple(mid) + parallel[-1].body
    reduce_parent = parallel[-1] if parallel else None
    reduce_nodes = tuple(_reduce_nodes(inner_body, reduce_parent))
    return IterDag(parallel=parallel, reduce=reduce_nodes, leading=leading, mid=mid, inner_body=inner_body)
