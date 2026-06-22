"""The block-DAG Tile IR — algorithm (derived views) + Schedule.

A redesign of the Tile phase (``plans/tile-ir-block-dag.md``): staging,
pipelining, warp-specialization, register tiling, split/cooperative-K, and block
placement all become the same kind of operation — a :class:`Schedule` annotation
over an **invariant algorithm**, applied by a single deterministic ``assemble``
step.

Three strata, with a strict rule about where each piece of information lives:

- **Algorithm** — the invariant. A DAG of :class:`Block`\\ s; each block is
  ``name + domain + compute`` where ``compute`` is the scalar Loop-IR body
  (``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` / ``Mma`` /
  ``Monoid``) over logical buffers. The single source of truth; only the
  algebra-/dependency-changing moves touch it.
- **Derived views** — projections of the algorithm, computed on demand, never
  stored: :attr:`Block.reads` / :attr:`Block.writes` (``AccessMap``\\ s read off
  the body's ``Load`` / ``Write`` index exprs), :attr:`Block.carrier`,
  :attr:`Block.atom`, and the :attr:`TileGraph.edges` topology. Because they are
  computed they cannot drift and they do not enter ``op_cache_key`` — the same
  discipline as ``Loop.algebra_kind`` / ``iter_dag``.
- **Schedule** — the variant: every scheduling choice, keyed by block / axis /
  read-site. The scheduling moves edit only this; ``assemble`` applies it to the
  algorithm and emits today's ``KernelOp`` tower.

This module defines the IR + its derived projections. ``assemble`` lives beside
it (``ir/tile/assemble.py``); the move composer (``passes/lowering/tile/
partition``) builds a :class:`TileGraph` + a reference :class:`Schedule`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

from deplodock.compiler.dtype import DataType
from deplodock.compiler.ir.algebra import AlgebraKind, classify_algebra
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.stmt import Body, ReduceCarrier
from deplodock.compiler.ir.stmt.leaves import Load, Mma, Write
from deplodock.compiler.ir.tile.ir import Atom


class Space(enum.Enum):
    GMEM = "gmem"
    SMEM = "smem"  # only ever an assemble artifact (a staged slab); never a stored Buffer
    REG = "reg"


class Binding(enum.Enum):
    GRID = "grid"  # blockIdx        — scope-creating
    SERIAL = "serial"  # for-loop     — scope-creating
    WARP = "warp"  # warp_id          — replication
    THREAD = "thread"  # threadIdx    — replication
    REGISTER = "register"  # unrolled cell — replication
    ATOM = "atom"  # one tensor-core cell — non-addressable (excluded from AccessMap)


class Transport(enum.Enum):
    SYNC = "sync"
    CPASYNC = "cpasync"  # sm_80+
    TMA = "tma"  # sm_90+


class Role(enum.Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"


class AddrKind(enum.Enum):
    AFFINE = "affine"  # source_index[d] = offset[d] + Σ_{i: dims[i]==d} block[i]·Var(axes[i])
    TEMPLATE = "template"  # verbatim coords, domain vars symbolic (collapsed reshape `/`,`%`)


# ---------------------------------------------------------------------------
# AccessMap — the derived index-classification of one Load / Write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccessMap:
    """A DERIVED value: how one ``Load`` / ``Write`` in a body indexes one
    buffer. Produced by classifying the leaf's index ``Expr`` (the legacy
    ``020_stage_inputs._classify``); not stored on blocks. AFFINE carries the
    structure ``assemble`` needs to size slabs, decide TMA box-eligibility, pick
    a swizzle, and clamp."""

    kind: AddrKind
    axes: tuple[str, ...] = ()  # domain axes indexing this buffer (AFFINE)
    dims: tuple[int, ...] = ()  # axes[i] -> source dim
    block: tuple[int, ...] = ()  # per-axis atom-cell stride multiplier
    offset: tuple[Expr, ...] = ()  # per-source-dim CTA-uniform anchor
    template: tuple[Expr, ...] = ()  # TEMPLATE: verbatim source coords
    clamp: tuple[Expr | None, ...] = ()  # per-source-dim safe-read bound (from the gmem Buffer.shape)

    def free_axes(self) -> frozenset[str]:
        """The domain axes this access depends on (drives hoist legality)."""
        if self.kind is AddrKind.AFFINE:
            return frozenset(self.axes)
        out: set[str] = set()
        for e in self.template:
            out |= e.free_vars()
        return frozenset(out)

    @property
    def rank(self) -> int:
        """Number of source dims indexed."""
        if self.kind is AddrKind.AFFINE:
            return len(self.offset)
        return len(self.template)


def _affine_terms(expr: Expr) -> tuple[dict[str, int], Expr] | None:
    """Decompose ``expr`` into ``({var: int_coeff}, const_expr)`` where the
    coefficients are integer literals over distinct vars and ``const_expr`` is
    the variable-free / CTA-uniform remainder. Returns ``None`` when the
    expression is not affine in its vars (``//`` / ``%`` / non-literal product)."""
    if isinstance(expr, Literal):
        return {}, expr
    if isinstance(expr, Var):
        return {expr.name: 1}, Literal(0, "int")
    if isinstance(expr, BinaryExpr):
        op = expr.op
        if op in ("+", "-"):
            lhs = _affine_terms(expr.left)
            rhs = _affine_terms(expr.right)
            if lhs is None or rhs is None:
                return None
            lc, lk = lhs
            rc, rk = rhs
            terms = dict(lc)
            sign = 1 if op == "+" else -1
            for v, c in rc.items():
                terms[v] = terms.get(v, 0) + sign * c
            const = _add(lk, rk) if op == "+" else _add(lk, _mul_const(rk, -1))
            return {v: c for v, c in terms.items() if c != 0}, const
        if op == "*":
            lhs = _affine_terms(expr.left)
            rhs = _affine_terms(expr.right)
            if lhs is None or rhs is None:
                return None
            lc, lk = lhs
            rc, rk = rhs
            # One side must be a pure constant for the product to stay affine.
            if not lc and isinstance(lk, Literal):
                k = int(lk.value)
                return {v: k * c for v, c in rc.items()}, _mul_const(rk, k)
            if not rc and isinstance(rk, Literal):
                k = int(rk.value)
                return {v: k * c for v, c in lc.items()}, _mul_const(lk, k)
            return None
        return None
    return None


def _is_zero(e: Expr) -> bool:
    return isinstance(e, Literal) and e.value == 0


def _add(a: Expr, b: Expr) -> Expr:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    if isinstance(a, Literal) and isinstance(b, Literal):
        return Literal(a.value + b.value, "int")
    return BinaryExpr("+", a, b)


def _mul_const(e: Expr, k: int) -> Expr:
    if k == 0 or _is_zero(e):
        return Literal(0, "int")
    if k == 1:
        return e
    if isinstance(e, Literal):
        return Literal(e.value * k, "int")
    return BinaryExpr("*", e, Literal(k, "int"))


def classify_access(index: tuple[Expr, ...], domain: frozenset[str]) -> AccessMap:
    """Classify one ``Load`` / ``Write`` index tuple against the iteration
    ``domain`` (the set of axis names the block iterates). AFFINE when every
    source dim is affine in the domain axes; TEMPLATE (verbatim, domain vars
    symbolic) otherwise — matching the legacy ``AffineAddressing`` /
    ``TemplateAddressing`` split."""
    axes: list[str] = []
    dims: list[int] = []
    block: list[int] = []
    offset: list[Expr] = []
    for d, e in enumerate(index):
        terms = _affine_terms(e)
        if terms is None:
            return AccessMap(kind=AddrKind.TEMPLATE, template=tuple(index), clamp=(None,) * len(index))
        coeffs, const = terms
        anchor = const
        for v, c in coeffs.items():
            if v in domain:
                axes.append(v)
                dims.append(d)
                block.append(c)
            else:
                # A non-domain (CTA-uniform / outer) var folds into the anchor.
                anchor = _add(anchor, _mul_const(Var(v), c))
        offset.append(anchor)
    return AccessMap(
        kind=AddrKind.AFFINE,
        axes=tuple(axes),
        dims=tuple(dims),
        block=tuple(block),
        offset=tuple(offset),
        clamp=(None,) * len(index),
    )


# ---------------------------------------------------------------------------
# Port / Carrier — derived dataflow + reduce-algebra views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """A DERIVED dataflow endpoint: ``(buffer, AccessMap)`` read off one body
    leaf (``Load`` for a read, ``Write`` for a write)."""

    buffer: str
    access: AccessMap


@dataclass(frozen=True)
class Carrier:
    """A DERIVED view of a folding block's reduce algebra — the legality oracle
    for the reduce-restructuring moves. ``kind`` / traits come from
    ``classify_algebra``; ``mask`` (the symbolic-K identity-fill bound) is read
    off the block's domain. Nothing here is stored: recomputed from the body +
    domain, like ``Loop.algebra_kind``."""

    carrier: ReduceCarrier
    kind: AlgebraKind | None = None  # set by Block.carrier (needs the enclosing loop)
    mask: tuple[str, Expr] | None = None  # (reduce-axis, runtime bound) — symbolic reduce axis

    @property
    def associative(self) -> bool:
        return self.carrier.associative

    @property
    def commutative(self) -> bool:
        return self.carrier.commutative

    @property
    def has_identity(self) -> bool:
        return self.carrier.has_identity


# ---------------------------------------------------------------------------
# Block — a DAG node: the algorithm at one compute site
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A DAG node: the algorithm at one compute site. STORED state is only
    ``name``, ``domain``, ``compute``. Everything else is a projection of
    ``compute`` (+ ``domain``), computed on demand — so it can never drift and
    never enters ``op_cache_key``."""

    name: str
    domain: tuple[Axis, ...]  # iteration axes (extent / real_extent / symbolic) the body references
    compute: Body  # the scalar algorithm over logical buffers — THE source of truth

    def __post_init__(self) -> None:
        if not isinstance(self.compute, Body):
            object.__setattr__(self, "compute", Body.coerce(self.compute))

    @property
    def domain_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.domain)

    @property
    def reads(self) -> tuple[Port, ...]:
        """``Load`` leaves of ``compute`` → ``(buffer, AccessMap)`` (recursing
        through nested reduce loops)."""
        dom = self.domain_names
        return tuple(Port(ld.input, classify_access(ld.index, dom)) for ld in self.compute.iter_of_type(Load))

    @property
    def writes(self) -> tuple[Port, ...]:
        """``Write`` leaves of ``compute`` → ``(buffer, AccessMap)``."""
        dom = self.domain_names
        return tuple(Port(w.output, classify_access(w.index, dom)) for w in self.compute.iter_of_type(Write))

    @property
    def carrier(self) -> Carrier | None:
        """The ``ReduceCarrier`` in ``compute`` (+ derived ``kind`` / ``mask``),
        else ``None``. The reduce axis (and any symbolic ``mask`` bound) is read
        off the enclosing reduce ``Loop`` in the body."""
        from deplodock.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

        for lp in self.compute.iter_of_type(Loop):
            if not lp.is_reduce:
                continue
            inner = [s for s in lp.body if isinstance(s, ReduceCarrier)]
            if not inner:
                continue
            ext = lp.axis.extent
            mask = None if ext.is_static else (lp.axis.name, ext.expr)
            return Carrier(carrier=inner[0], kind=classify_algebra(lp), mask=mask)
        # A carrier directly at the block's top level (already-bracketed body).
        for s in self.compute:
            if isinstance(s, ReduceCarrier):
                return Carrier(carrier=s)
        return None

    @property
    def atom(self) -> Atom | None:
        """The ``Mma``'s atom once atomized, else ``None``."""
        for m in self.compute.iter_of_type(Mma):
            return m.atom
        return None


# ---------------------------------------------------------------------------
# Buffer / Edge — logical value-stores + derived def-use topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Buffer:
    """A LOGICAL value-store: a kernel input/output or an inter-block
    intermediate. SMEM slabs are not Buffers — they are assemble artifacts of a
    ``staged`` annotation. ``pad`` is a schedule property of the slab, not
    here."""

    name: str
    shape: tuple[Expr, ...]
    dtype: DataType
    space: Space = Space.GMEM


@dataclass(frozen=True)
class Edge:
    """A DERIVED value (not stored): one per ``(producer-or-input, consumer,
    buffer)`` from the body's buffer def-use."""

    src: str  # producer block name, or an input Buffer name
    dst: str  # consumer block name
    buffer: str


# ---------------------------------------------------------------------------
# Schedule — the variant: every scheduling choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """The variant — every scheduling choice. The scheduling moves edit only
    this; ``assemble`` applies it to the algorithm. Staging keys are read-sites
    (the derived ``Edge``); a read absent from ``staged`` is gmem-direct."""

    binding: dict[str, Binding] = field(default_factory=dict)  # axis -> hardware role
    scope: dict[str, tuple[str, ...]] = field(default_factory=dict)  # block -> enclosing nest override
    role: dict[str, Role] = field(default_factory=dict)  # block -> producer/consumer (warp-spec)
    launch: dict[str, int] = field(default_factory=dict)  # block -> launch group (one group = one kernel)
    staged: dict[Edge, Transport] = field(default_factory=dict)  # read-site -> SMEM fill transport
    distance: dict[Edge, tuple[tuple[str, int], ...]] = field(default_factory=dict)  # read-site -> retiming offset
    cohort: dict[Edge, int] = field(default_factory=dict)  # read-site -> barrier / pipeline / transport cohort
    ring_depth: dict[Edge, int] = field(default_factory=dict)  # staged read-site -> ring slots; >= max(distance)+1
    pad: dict[Edge, tuple[int, ...]] = field(default_factory=dict)  # staged read-site -> slab bank-conflict pad
    reg_budget: dict[Role, int] = field(default_factory=dict)  # warp-spec register redistribution (SetMaxNReg)
    unroll: dict[str, bool] = field(default_factory=dict)  # SERIAL axis -> #pragma unroll
    grid_swizzle: dict[str, int] = field(default_factory=dict)  # GRID block -> L2 row-group remap

    def with_binding(self, **kw: Binding) -> Schedule:
        return replace(self, binding={**self.binding, **kw})


# ---------------------------------------------------------------------------
# TileGraph — the new Tile IR
# ---------------------------------------------------------------------------


@dataclass
class TileGraph:
    """The new Tile IR. ``assemble(TileGraph) -> KernelOp | Graph[KernelOp]``
    (one kernel per launch group). The edge topology is derived."""

    name: str
    buffers: dict[str, Buffer]  # logical only (inputs / outputs / intermediates)
    blocks: tuple[Block, ...]
    schedule: Schedule

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Buffer def-use across blocks (+ input-source edges), deduplicated.

        For each buffer a block reads, the edge's ``src`` is the (unique) block
        that writes it, or — when no block writes it — the input ``Buffer``
        name itself."""
        writer: dict[str, str] = {}
        for b in self.blocks:
            for p in b.writes:
                writer[p.buffer] = b.name
        out: list[Edge] = []
        seen: set[Edge] = set()
        for b in self.blocks:
            for p in b.reads:
                src = writer.get(p.buffer, p.buffer)
                if src == b.name:
                    continue  # a block reading its own intermediate (accumulator) is not a DAG edge
                e = Edge(src=src, dst=b.name, buffer=p.buffer)
                if e not in seen:
                    seen.add(e)
                    out.append(e)
        return tuple(out)
