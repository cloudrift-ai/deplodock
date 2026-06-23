"""Tile IR — schedule decisions as structural Stmts (wrap-body Stage).

Tile IR sits between Loop IR (math) and Kernel IR (fully-scheduled
kernel form). Its job is to encode the *logical* compute plus the
*scheduling decisions* — without committing to hardware primitives.
Materialization (``passes/lowering/kernel``) consumes Tile IR and
produces Kernel IR.

Pipeline shape::

    Loop IR ──launch_geometry──▶ Tile IR (logical compute, default bindings)
                     ──[strategy passes]──▶ Tile IR (annotated)
                     ──materialize_tile──▶ Kernel IR
                     ──render_kernelop──▶ CUDA source

**StageBundle:** every ``StageBundle`` is a block-structured Stmt whose
``body`` is the *consumer* subtree that uses the staged smem buffers.
The producer (cooperative Load+Write per source) is synthesized at
materialize time from ``StageBundle.sources``. Smem lifetime is
structural (decl-to-end-of-bundle.body, not implicit-end-of-block).

**Sources.** Each bundle carries one or more ``Source`` entries directly;
each Source maps one gmem buffer into one smem slab with its own cache
axes and origin. Multi-source bundles (e.g. A + B in a matmul reduce)
load all behind a single sync boundary. Bundles with genuinely different
consumer scopes nest instead of multi-sourcing.

**Transport policy** (``StagePolicy.SYNC`` / ``BUFFERED`` / ``ASYNC`` /
``TMA``) on the bundle encodes sync cooperative load, ring-buffered sync,
cp.async, or TMA box-copy. ``pipeline_depth > 1`` (ASYNC / TMA) marks a
bundle for temporal pipelining (prologue/main/epilogue), expanded by
``080_pipeline_stages`` before materialization.

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` / ``Cond`` come straight from ``ir.loop`` — buf
names are strings so they're directly renderable.

**Tile launch-geometry.** ``Tile.thread_axes`` / ``Tile.block_axes``:
which output axes are bound to thread coords vs CUDA block coords.
Pointwise has ``thread_axes`` populated and ``block_axes`` empty (one
thread per output element). Cooperative reductions have ``block_axes``
populated and ``thread_axes`` empty; the cooperative thread axis is
synthesized at materialization.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal as _Lit

from deplodock.compiler.dtype import BF16, F16, F32, DataType
from deplodock.compiler.ir.algebra import AlgebraKind, classify_algebra
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.stmt import (
    INDENT,
    Accum,
    Assign,
    Body,
    Cond,
    Load,
    Loop,
    ReduceCarrier,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Write,
    pretty_body,
)

# `render_body` is the per-Stmt body renderer used by the new tile flavors'
# render methods. Local import below to keep top-of-file imports tidy.
from deplodock.compiler.ir.stmt import render_body as _render_body  # noqa: E402
from deplodock.compiler.ir.stmt.base import RenderCtx, _pad
from deplodock.compiler.ir.stmt.blocks import (
    _body_uses_lane_warp,
    _render_grid_axis_decode,
    _render_swizzled_grid_decode,
    _render_thread_axis_decode,
)
from deplodock.compiler.ir.stmt.ir import BodyOp
from deplodock.compiler.ir.stmt.leaves import Mma

# ===========================================================================
# ENUMERATION — the block-DAG Tile IR (algorithm + Schedule)
# ===========================================================================
# The invariant algorithm (a DAG of Blocks) + the Schedule the move composer
# searches. Derived projections (reads/writes/carrier/atom/edges) are computed
# on demand, never stored. ``assemble(TileGraph)`` lowers this to the
# MATERIALIZED tower below. See plans/tile-ir-block-dag.md.


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


class Placement(enum.Enum):
    """Where a DAG edge's producer→consumer value is materialized — the unifying
    edge-placement annotation (``plans/dag-edge-placement-split-as-enumeration.md``).
    DERIVED from the ``Schedule`` (``staged`` + ``launch``), never stored: every
    scheduling choice already lives in those fields, so placement is a projection
    of them the same way ``Block.atom`` / ``TileGraph.edges`` are projections of the
    body — see :meth:`TileGraph.placement`.

    - ``INLINE`` — the value rides registers inside the consumer block (a fused
      cone or a plain gmem-direct read); the default, no annotation.
    - ``SMEM`` — a staged smem slab (``Schedule.staged[edge]``); today's ``stage``
      move.
    - ``GMEM`` — a global intermediate buffer; the producer and consumer live in
      different launch groups (a grid barrier — the ``GMEM`` cut). Says *where the
      buffer lives*, not *how many kernels*: v1 realizes every grid-crossing edge
      as two launches, so a ``GMEM`` edge is exactly a cross-launch-group edge."""

    INLINE = "inline"
    SMEM = "smem"
    GMEM = "gmem"


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

    def pretty(self, indent: str = "") -> list[str]:
        """Readable listing of the non-empty scheduling decisions (for
        ``compile -vv`` / kernel dumps). ``binding`` is rendered on the
        block domain axes by :meth:`TileGraph.pretty`, so it is omitted
        here; every other non-empty field gets one line. Edge-keyed maps
        (``staged`` / ``distance`` / ``cohort`` / ``ring_depth`` / ``pad``)
        key on ``buffer:src->dst``."""

        def edge_key(e: Edge) -> str:
            return f"{e.buffer}:{e.src}->{e.dst}"

        lines: list[str] = []
        if self.scope:
            lines.append(f"{indent}scope: " + ", ".join(f"{b}={'/'.join(s)}" for b, s in self.scope.items()))
        if self.role:
            lines.append(f"{indent}role: " + ", ".join(f"{b}={r.value}" for b, r in self.role.items()))
        if self.launch:
            lines.append(f"{indent}launch: " + ", ".join(f"{b}={g}" for b, g in self.launch.items()))
        if self.staged:
            lines.append(f"{indent}staged: " + ", ".join(f"{edge_key(e)}={t.value}" for e, t in self.staged.items()))
        if self.distance:
            lines.append(f"{indent}distance: " + ", ".join(f"{edge_key(e)}={list(d)}" for e, d in self.distance.items()))
        if self.cohort:
            lines.append(f"{indent}cohort: " + ", ".join(f"{edge_key(e)}={c}" for e, c in self.cohort.items()))
        if self.ring_depth:
            lines.append(f"{indent}ring_depth: " + ", ".join(f"{edge_key(e)}={n}" for e, n in self.ring_depth.items()))
        if self.pad:
            lines.append(f"{indent}pad: " + ", ".join(f"{edge_key(e)}={list(p)}" for e, p in self.pad.items()))
        if self.reg_budget:
            lines.append(f"{indent}reg_budget: " + ", ".join(f"{r.value}={n}" for r, n in self.reg_budget.items()))
        if self.unroll:
            lines.append(f"{indent}unroll: " + ", ".join(f"{a}={v}" for a, v in self.unroll.items()))
        if self.grid_swizzle:
            lines.append(f"{indent}grid_swizzle: " + ", ".join(f"{b}={n}" for b, n in self.grid_swizzle.items()))
        return lines


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

    def placement(self, edge: Edge) -> Placement:
        """The DERIVED :class:`Placement` of one edge — read off ``Schedule.staged``
        + ``Schedule.launch`` (``plans/dag-edge-placement-split-as-enumeration.md``).
        ``stage`` and the ``GMEM`` cut become two values of one query, with ``INLINE``
        the default. A cross-launch-group edge is ``GMEM`` regardless of whether the
        consumer also stages its read of the materialized buffer (the buffer lives in
        gmem either way), so ``GMEM`` is checked first."""
        sched = self.schedule
        block_names = {b.name for b in self.blocks}
        if edge.src in block_names and sched.launch.get(edge.src, edge.src) != sched.launch.get(edge.dst, edge.dst):
            return Placement.GMEM
        if edge in sched.staged:
            return Placement.SMEM
        return Placement.INLINE

    def place_edge(self, edge: Edge, placement: Placement, *, transport: Transport = Transport.SYNC) -> TileGraph:
        """Return a copy of this ``TileGraph`` with ``edge`` placed — the unifying
        edge-placement **move** (``plans/dag-edge-placement-split-as-enumeration.md``).
        The inverse of :meth:`placement`: it writes the ``Schedule`` fields a placement
        implies, so ``stage``/``split``/fuse become one operation over the block-DAG.

        - ``GMEM`` (the cut) — put the producer and consumer in **different** launch
          groups (a grid barrier — the multi-launch ``assemble`` then emits separate
          kernels with a gmem intermediate). Drops any stale staging of the edge.
        - ``SMEM`` (the fused edge) — put both in the **same** launch group and stage
          the edge (``staged[edge] = transport``); the intermediate rides an smem slab
          inside one kernel (the producer fills it, the consumer reads it).
        - ``INLINE`` — same launch group, no staging (the value rides registers).

        Only meaningful for a block→block edge (``edge.src`` is a producer block); an
        input-source edge has no producer to co-place, so only ``SMEM``/``INLINE``
        (its staging) apply and the launch keys are left untouched."""
        sched = self.schedule
        block_names = {b.name for b in self.blocks}
        launch = dict(sched.launch)
        staged = {e: t for e, t in sched.staged.items() if e != edge}
        if edge.src in block_names:
            if placement is Placement.GMEM:
                launch[edge.src] = launch.get(edge.src, 0)
                launch[edge.dst] = launch[edge.src] + 1  # a distinct group → the cut
            else:
                launch[edge.dst] = launch[edge.src] = launch.get(edge.src, 0)  # one kernel
        if placement is Placement.SMEM:
            staged[edge] = transport
        return replace(self, schedule=replace(sched, launch=launch, staged=staged))

    def structural_key(self) -> str:
        """A canonical identity over the algorithm + Schedule, for ``op_cache_key``
        (``plans/tile-ir-block-dag.md``: the key is ``canonical(compute bodies +
        edge topology) + Schedule``). The derived projections never enter it."""
        blocks = tuple((b.name, b.compute.structural_key()) for b in self.blocks)
        binding = tuple(sorted((a, v.value) for a, v in self.schedule.binding.items()))
        edges = tuple(sorted((e.src, e.dst, e.buffer) for e in self.edges))
        return repr((blocks, binding, edges))

    def pretty(self, indent: str = "") -> list[str]:
        """Readable multi-line listing of the block-DAG (for ``compile -vv``
        and kernel dumps): the logical buffers, each block's domain (axis +
        binding) and compute body, the non-empty schedule decisions, and the
        derived def-use edges. The verbose nested-dataclass ``repr`` is what
        this replaces."""
        lines: list[str] = [f"{indent}buffers:"]
        for buf in self.buffers.values():
            shape = ", ".join(d.pretty() if hasattr(d, "pretty") else str(d) for d in buf.shape)
            space = "" if buf.space is Space.GMEM else f" {buf.space.value}"
            lines.append(f"{indent}{INDENT}{buf.name}: {buf.dtype.name}[{shape}]{space}")
        for b in self.blocks:
            dom = ", ".join(_fmt_domain_axis(ax, self.schedule.binding.get(ax.name)) for ax in b.domain)
            lines.append(f"{indent}block {b.name} [{dom}]")
            lines.extend(pretty_body(b.compute, indent + INDENT))
        sched = self.schedule.pretty(indent + INDENT)
        if sched:
            lines.append(f"{indent}schedule:")
            lines.extend(sched)
        edges = self.edges
        if edges:
            lines.append(f"{indent}edges:")
            for e in edges:
                lines.append(f"{indent}{INDENT}{e.buffer}: {e.src} -> {e.dst}")
        return lines


def _fmt_domain_axis(ax: Axis, binding: Binding | None) -> str:
    """``name:extent`` plus ``=binding`` when the axis is bound — the compact
    per-axis label in :meth:`TileGraph.pretty`'s block header."""
    label = f"{ax.name}:{ax.extent}"
    return f"{label}={binding.value}" if binding is not None else label


@dataclass
class TileGraphOp(Op):
    """The node the ENUMERATION passes pass between themselves and hand to
    ASSEMBLY. It carries the **stored algorithm being refined in place** by the F3-b
    incremental body moves (``plans/tile-ir-block-dag.md``): ``000_build`` seeds the
    **logical** (un-tiled) ``TileGraph`` (``_build.seed_graph``), then the tile passes
    rewrite it move by move — the algorithm is a first-class structure refined as the
    search descends, never a function re-derived from a stored knob dict (that is the
    "knob-invariant algorithm" the model calls for).

    - **logical seed → tiled** (``tilegraph`` set throughout) — ``000_build`` emits the
      logical block; ``010_reduce_tile`` applies the reduce-decomposition body move
      (``reduce_decomp``); ``020_thread_tile`` pins the thread knob (no body move);
      ``030_register_tile`` applies the free-axis σ-split body move (``free_tile``),
      after which the algorithm is fully tiled; ``040_seal_scalar_tier`` stamps the
      reduce regime's scalar-tier OFF sentinels; ``050_stage`` annotates
      ``Schedule.staged``. It also carries the derived ``dag`` + regime (``algebra`` /
      ``target_names``) the offer fns read. Each fork pins one more knob group onto
      ``knobs``; the carry-forward ``LoopOp`` knobs ride ``knobs`` automatically (the
      engine merges a predecessor's knobs forward on every rebind).
    - **assembly** consumes the fully-tiled ``tilegraph`` directly
      (``assembly/010_assemble`` → ``assemble_block``): no build there, only the tower
      materialization + slab synthesis from the ``Schedule``.

    ``op_cache_key`` keys on :meth:`structural_key` (the stored ``TileGraph``'s
    canonical algorithm + ``Schedule``) + ``knobs`` — distinct per variant, so the
    search tree never self-parents."""

    name: str = ""
    tilegraph: TileGraph | None = None
    leading: tuple = ()
    # --- enumeration state carried alongside the stored algorithm (untyped to keep
    # the ir layer free of a passes-layer import — set/read by the tile passes) ---
    dag: object = None  # the IterDag the offer fns tile
    algebra: object = None  # AlgebraKind — the regime the passes dispatch on
    target_names: frozenset = frozenset()  # contraction-axis names a reduce move rewrites
    seed_key: str = ""  # the source LoopOp's body structural key
    buffers: dict = field(default_factory=dict)  # logical gmem Buffers (name -> Buffer) from the source LoopOp's I/O

    def structural_key(self) -> str:
        return self.tilegraph.structural_key() if self.tilegraph is not None else self.seed_key

    def pretty_body(self) -> str:
        """Readable rendering of the stored algorithm for ``compile -vv`` /
        kernel dumps — the regime header (``algebra`` / reduce ``targets``),
        the leading hoisted stmts, then the ``TileGraph`` block-DAG. Replaces
        the unreadable nested-dataclass ``repr`` the diff renderer fell back
        to. The caller (``Candidate._format_nodes``) emits the surrounding
        ``<out> = TileGraphOp(<inputs>)`` label, so none is prepended here."""
        lines: list[str] = []
        algebra = getattr(self.algebra, "value", self.algebra)
        head = f"algebra={algebra}" if algebra is not None else "algebra=?"
        if self.target_names:
            head += f"  targets={{{', '.join(sorted(self.target_names))}}}"
        lines.append(head)
        if self.leading:
            lines.append("leading:")
            lines.extend(pretty_body(Body.coerce(self.leading), INDENT))
        if self.tilegraph is not None:
            lines.extend(self.tilegraph.pretty())
        return "\n".join(lines)


# ===========================================================================
# MATERIALIZED — the assemble output: the tower IR
# ===========================================================================
# What ``assemble`` emits and the kernel passes lower to ``KernelOp``: ``TileOp``
# + the typed tile flavors + ``StageBundle`` / ``Source`` / ``WarpSpecialize`` /
# ``AsyncWait`` + ``Atom`` / ``ATOM_REGISTRY``. Slated for removal once
# ``assemble`` emits ``KernelOp`` directly.


SerialKind = _Lit["plain", "stage_inner", "serial_outer", "pipeline"]


# ===========================================================================
# Atom kinds — the hardware-instruction spec for each tensor-core matmul cell.
# ===========================================================================


@dataclass(frozen=True)
class Atom:
    """Hardware-instruction spec for one matmul atom kind — the cell a single
    tensor-core instruction realises (``C[M×N] += A[M×K] · B[K×N]``).

    Carried directly on the :class:`~deplodock.compiler.ir.stmt.Mma` op (and
    keyed by ``name`` in :data:`ATOM_REGISTRY`), so ``kernel/005_lower_atom_tile``
    reads the cell shape + operand dtypes straight off the ``Mma`` — no registry
    lookup at lowering.

    - ``name`` — the ``ATOM_KIND`` (e.g. ``"mma_m16n8k16_f16"``); the registry key.
    - ``shape`` — the cell shape ``(M, N, K)`` one instruction realises.
    - ``operand_dtypes`` — ``(role, dtype)`` pairs (``"a"`` / ``"b"`` / ``"c"``;
      scaled kinds extend with ``"a_scale"`` / ``"b_scale"``). A tuple, not a
      dict, so ``Atom`` stays **hashable** — it rides on a frozen ``Mma`` Stmt.
      Use :meth:`operand_dtype` for role lookup.
    - ``group_size`` — threads-per-cell (32 for the warp-level mma.sync atom;
      128 for a future wgmma warp-group). Drives the warp-tier launch geometry.

    Per-kernel *eligibility* (does a LoopOp admit this atom?) is NOT here — it
    needs the loop / graph / context and lives in the planner
    (``passes/lowering/tile/_atom.py``).
    """

    name: str
    shape: tuple[int, int, int]
    operand_dtypes: tuple[tuple[str, DataType], ...]
    group_size: int

    def operand_dtype(self, role: str) -> DataType:
        """The element dtype of operand ``role`` (``"a"`` / ``"b"`` / ``"c"``)."""
        for r, dt in self.operand_dtypes:
            if r == role:
                return dt
        raise KeyError(f"atom {self.name!r} has no operand role {role!r}")

    def __str__(self) -> str:
        return self.name


# The s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` path is the sole
# tensor-core family — f16 / bf16 operands, f32 accumulate, sm_80+. f16 and
# bf16 share the 16-bit fragment layout (only ``MmaSyncPtx.ab_dtype`` differs);
# scalar matmul is the absence of an atom. ``kernel/005_lower_atom_tile`` emits
# the RegFragment / LdmatrixLoad / MmaSyncPtx / RegStore chain (smem-staged only —
# ldmatrix has no gmem-direct path). Insertion order is the planner's enumeration
# priority (f16 first, then bf16); the launch-geometry / eligibility paths look a
# kind up directly via ``ATOM_REGISTRY[knobs["MMA"]]``.
ATOM_REGISTRY: dict[str, Atom] = {
    "mma_m16n8k16_f16": Atom(
        name="mma_m16n8k16_f16",
        shape=(16, 8, 16),
        operand_dtypes=(("a", F16), ("b", F16), ("c", F32)),
        group_size=32,
    ),
    "mma_m16n8k16_bf16": Atom(
        name="mma_m16n8k16_bf16",
        shape=(16, 8, 16),
        operand_dtypes=(("a", BF16), ("b", BF16), ("c", F32)),
        group_size=32,
    ),
}


# ---------------------------------------------------------------------------
# AsyncWait — explicit wait carrier for pipelined schedules
# ---------------------------------------------------------------------------
#
# Sync-style async / TMA stages (``pipeline_depth == 1``) get an
# implicit wait at their wrap boundary, emitted by ``_emit_stage`` /
# ``emit_tma_stage`` in the materializer. Pipelined stages
# (``pipeline_depth > 1``) need explicit waits at non-default schedule
# positions: ``080_pipeline_stages`` emits ``AsyncWait``
# Stmts between the issue and consume halves of each steady-state K_o
# iteration (and at the epilogue drain). The materializer's
# ``emit_async_wait`` closure lowers them to ``CpAsyncWait(group=keep)``
# for cp.async, or ``MbarrierWait(mbar, phase, slot)`` for TMA.


@dataclass(frozen=True)
class AsyncWait(Stmt):
    """Explicit wait carrier for pipelined async / TMA schedules.

    Sync-style stages (``pipeline_depth == 1``) don't need this — the
    materializer emits an implicit wait at the wrap boundary. Pipelined
    stages do: ``080_pipeline_stages`` peels the steady
    state into issue-now / wait-for-prev / consume-prev, with explicit
    ``AsyncWait`` carrying the schedule:

    - ``keep`` — cp.async ``wait_group`` argument (number of commits to
      leave in flight). ``keep = 1`` in the steady-state body leaves
      the just-issued chunk in flight while waiting for the older one;
      ``keep = 0`` in the epilogue drains every outstanding commit.
    - ``phase`` / ``slot`` — TMA mbarrier-test phase + ring slot for
      the consumer-side ``MbarrierWait``. ``phase = (K_o / bc) % 2``
      tracks how many times the slot has been reused; ``slot = K_o % bc``
      picks the ring slot to wait on.

    The trailing CTA-fence ``Sync`` after the materializer's
    ``MbarrierWait`` / ``CpAsyncWait`` defaults to ``__syncthreads()``.
    Inside a WS consumer subtree it routes to a named ``bar.sync N, M``
    instead — the materializer derives the named-barrier params from
    the enclosing ``WarpSpecialize`` context, not from fields on this
    Stmt. (``__syncthreads()`` is CUDA UB on the warp-divergent
    producer/consumer branch.)
    """

    keep: int = 0
    phase: Expr | None = None
    slot: Expr | None = None

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        if self.phase is not None:
            out = (*out, self.phase)
        if self.slot is not None:
            out = (*out, self.slot)
        return out

    def pretty(self, indent: str = "") -> list[str]:
        extra = ""
        if self.phase is not None:
            extra += f", phase={self.phase.pretty()}"
        if self.slot is not None:
            extra += f", slot={self.slot.pretty()}"
        return [f"{indent}AsyncWait(keep={self.keep}{extra})"]


# ---------------------------------------------------------------------------
# CoopReduce — cooperative per-row reduce prologue (SMEM fused edge, R7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoopReduce(Stmt):
    """A cooperative per-row reduce **prologue** for the SMEM fused edge (the rmsnorm
    producer). All CTA threads stride over the ``cells`` (the M cache axes), each
    reducing the producer's contraction over the **full** row and writing a per-row
    scalar slab (``out_slab``); the matmul's scale-application compute phase then reads
    that slab as a broadcast operand. Emitted as a ``GridTile``-level sibling **before**
    the matmul tower — no second ``ThreadTile``, the cooperative ``StridedLoop`` uses the
    kernel's own thread count. ``100_materialize`` expands it via ``emit_reduce_phase``.

    - ``cells`` — the M cache axes: the cooperative iteration domain + the ``out_slab``
      index. (Materialize flat-decodes a ``tid``-strided index into these.)
    - ``leading`` — per-CTA constants, emitted **once** before the cooperative loop (the
      rmsnorm ``1/H`` reciprocal + ``eps`` loads).
    - ``body`` — the per-cell reduce: a serial reduce loop over the contraction axis + the
      scalar chain (``rsqrt(acc·(1/H)+eps)``) + the ``Write out_slab[cells]``.
    - ``out_slab`` / ``out_dtype`` — the produced per-row smem slab + its element dtype.

    The bodies ARE exposed via :meth:`nested` so ``020_place_inits`` seeds the reduce
    ``Accum`` and ``030_stamp_types`` stamps dtypes; ``010_split_register_axes`` skips it
    (the cooperative fill must not be register-replicated, like ``StageBundle.compute``)."""

    cells: tuple[Axis, ...]
    leading: Body
    body: Body
    out_slab: str
    out_dtype: DataType

    def __post_init__(self) -> None:
        if not isinstance(self.leading, Body):
            object.__setattr__(self, "leading", Body.coerce(self.leading))
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))

    def nested(self) -> tuple[Body, ...]:
        return (self.leading, self.body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        return replace(self, leading=bodies[0], body=bodies[1])

    def external_reads(self) -> tuple[str, ...]:
        return tuple(ld.input for b in (self.leading, self.body) for ld in b.iter_of_type(Load))

    def local_decls(self) -> tuple[str, ...]:
        return (self.out_slab,)

    def pretty(self, indent: str = "") -> list[str]:
        cells = ", ".join(f"{a.name}:{a.extent}" for a in self.cells)
        lines = [f"{indent}coop_reduce[{cells}] -> {self.out_slab}:"]
        lines.extend(pretty_body(self.leading, indent + INDENT))
        lines.extend(pretty_body(self.body, indent + INDENT))
        return lines


# ---------------------------------------------------------------------------
# WarpSpecialize — producer/consumer split for TMA-pipelined kernels
# ---------------------------------------------------------------------------
#
# Tile-IR marker that the materializer (``100_materialize_tile``) lowers
# into the full mbarrier handshake: empty-mbarrier ring (``Smem`` +
# per-slot ``MbarrierInit``), per-K_o ``MbarrierWait`` / ``MbarrierArrive``
# pairs, named ``bar.sync`` consumer fences, ``SetMaxNReg`` register
# budget redistribution, and the producer/consumer ``Cond`` wrapper.
#
# The pass that emits this (``085_warp_specialize``) keeps all Tile-IR
# vocabulary — no ``from deplodock.compiler.ir.kernel.ir import …``.
# Companion to 080's ``AsyncWait``: where ``AsyncWait`` lets the pass
# declare "wait for an async chunk, materializer picks the primitive",
# ``WarpSpecialize`` lets the pass declare "split this ThreadTile body
# into producer and consumer roles, materializer wires the rest".


@dataclass(frozen=True)
class WarpSpecialize(Stmt):
    """Producer/consumer warp split inside a TMA-pipelined kernel.

    Fields:

    - ``producer_body`` — stmts run by producer warp(s) (TMA-issue
      ``StageBundle`` scaffolding inside ``SerialTile(serial_outer)``).
    - ``consumer_body`` — stmts run by consumer warps (``AsyncWait`` +
      reduce loop + output ``Write``). Indices reference the **original**
      thread-axis names directly — no σ-shift. The materializer emits the
      consumer-relative ``threadIdx.x - n_producer_threads`` decode at
      the head of the consumer branch (see :class:`ThreadTile.tid_offset`).
    - ``ring_depth`` — empty-mbarrier slot count (== TMA buffer_count).
    - ``n_producer_threads`` — number of threads in the producer warp(s).
      Today only ``32`` (one producer warp) is emitted; ``SetMaxNReg``
      accounting and the ``Cond(role < n_producer_warps, …)`` predicate
      both derive from this.
    - ``consumer_thread_axes`` — axes describing the consumer-side coord
      structure (the original ``ThreadTile.axes`` for the scalar tier, or the
      ``WarpTile.axes`` WM×WN warp coords for the warp-tier MMA tower — see
      ``consumer_is_warp``). The materializer feeds these into a nested
      ``ThreadTile`` / ``WarpTile`` carrying ``tid_offset=n_producer_threads``
      so consumer threads decode ``threadIdx.x - n_producer_threads`` back into
      these axis names. ``()`` is the legacy / pre-refactor shape, kept for
      back-compat with any caller that doesn't track the axes yet — the new
      materializer arm raises if it's empty when expected.

    The K_o axis the WS pass aligned scheduling against is identified by
    the materializer structurally — the (single) ``SerialTile(serial_outer)``
    in each branch. Carrying the axis *name* would break under
    ``normalize_body``'s canonical rename pass (which renames Axis but
    can't see a plain string field).

    Other quantities (``role`` predicate, ``n_consumer_threads``,
    slot/phase exprs, mbar name) are derivable from these fields plus
    the enclosing ``WarpTile.axes`` (the role axis) and are reconstructed
    at materialize time.

    Nested ``WarpSpecialize`` is rejected at construction.
    """

    producer_body: Body
    consumer_body: Body
    ring_depth: int
    n_producer_threads: int
    consumer_thread_axes: tuple[Axis, ...] = ()
    # When True the consumer axes are *warp*-granularity (the warp-tier MMA
    # tower: ``consumer_thread_axes`` are the WM×WN warp axes, 32 lanes each),
    # so the materializer wraps the consumer body in
    # ``WarpTile(tid_offset=n_producer_threads)`` — ``warp_id = (threadIdx.x -
    # n_producer_threads) / 32`` — rather than a thread-granularity
    # ``ThreadTile``. Default False keeps the scalar (pointwise / cooperative-
    # reduce) WS path unchanged.
    consumer_is_warp: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.producer_body, Body):
            object.__setattr__(self, "producer_body", Body.coerce(self.producer_body))
        if not isinstance(self.consumer_body, Body):
            object.__setattr__(self, "consumer_body", Body.coerce(self.consumer_body))
        if self.ring_depth < 1:
            raise ValueError(f"WarpSpecialize: ring_depth must be >= 1, got {self.ring_depth}")
        if self.n_producer_threads < 1:
            raise ValueError(f"WarpSpecialize: n_producer_threads must be >= 1, got {self.n_producer_threads}")
        if not isinstance(self.consumer_thread_axes, tuple):
            object.__setattr__(self, "consumer_thread_axes", tuple(self.consumer_thread_axes))
        # Nesting check — a WS inside another WS would require the
        # materializer to track a stack of ws_consumer contexts; today
        # 085 guards via the WS knob on the parent TileOp, but assert at
        # the IR level so other rules can't sneak one in.
        for body in (self.producer_body, self.consumer_body):
            for s in body.iter():
                if isinstance(s, WarpSpecialize):
                    raise ValueError("WarpSpecialize cannot nest")

    def nested(self) -> tuple[Body, ...]:
        return (self.producer_body, self.consumer_body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if len(bodies) != 2:
            raise ValueError(f"WarpSpecialize.with_bodies: expected 2 bodies, got {len(bodies)}")
        producer_body, consumer_body = bodies
        return replace(self, producer_body=producer_body, consumer_body=consumer_body)

    def deps(self) -> tuple[str, ...]:
        return ()

    def exprs(self) -> tuple[Expr, ...]:
        return ()

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}warp_specialize(ring={self.ring_depth}, n_prod={self.n_producer_threads}):"
        producer_lines = [f"{indent}{INDENT}producer:", *pretty_body(self.producer_body, indent + INDENT * 2)]
        consumer_lines = [f"{indent}{INDENT}consumer:", *pretty_body(self.consumer_body, indent + INDENT * 2)]
        return [head, *producer_lines, *consumer_lines]


# ---------------------------------------------------------------------------
# Stage primitives: Source + AffineAddressing/TemplateAddressing
# ---------------------------------------------------------------------------

# Bytes per stored element in smem. fp32-only assumption — fp16 paths
# over-count by 2x (soft latent bug; see project_tile_ir_fp32_only memory).
BYTES_PER_ELEM = 4


class SwizzleMode(enum.Enum):
    """TMA shared-memory swizzle pattern.

    Picked by the lowering pass from inner-dim byte stride; consumed by
    the backend's ``cuTensorMapEncodeTiled`` call. Only meaningful when
    ``StageBundle.policy == StagePolicy.TMA``.

    Defined here (above ``Source``) so it can be a ``Source.swizzle`` field
    default. The per-operand swizzle mode lives on the ``Source``, not only on
    the bundle, because A (64 B inner → ``B64``) and B (128 B inner → ``B128``)
    share one StageBundle but need distinct modes.
    """

    NONE = "NONE"
    B32 = "B32"
    B64 = "B64"
    B128 = "B128"


# TMA hardware-swizzle atom widths in bytes, widest-first. The widest atom
# that divides a source's inner-row byte span wins (best conflict spread).
_SWIZZLE_BY_BYTES: tuple[tuple[int, SwizzleMode], ...] = (
    (128, SwizzleMode.B128),
    (64, SwizzleMode.B64),
    (32, SwizzleMode.B32),
)


def pick_swizzle_atom(inner_elems: int, elem_bytes: int) -> tuple[int, SwizzleMode]:
    """Pick the TMA swizzle atom for a source whose collapsed inner-row span
    is ``inner_elems`` elements of ``elem_bytes`` bytes each.

    Returns ``(atom_elems, mode)`` where ``atom_elems`` is the swizzle atom
    width in *elements* (the innermost descriptor-box dim after the box
    reshape) and ``mode`` the matching :class:`SwizzleMode`. The widest atom
    in ``{128, 64, 32}`` B that (a) is ``<=`` the inner span and (b) divides
    it evenly wins — so a 256 B inner (128 fp16 elems) picks ``B128`` (64
    elems, split ``[2, 64]``) and a 64 B inner (32 fp16 elems) picks ``B64``
    (32 elems, no split). Returns ``(inner_elems, NONE)`` when no atom fits
    (the descriptor keeps the unswizzled collapsed box).

    Shared by ``kernel/100_materialize_tile`` (box reshape) and
    ``tile/050_use_tma`` (per-source mode pick) so the two agree on the
    atom width.
    """
    inner_bytes = inner_elems * elem_bytes
    for wb, mode in _SWIZZLE_BY_BYTES:
        we = wb // elem_bytes
        if we >= 1 and we <= inner_elems and inner_elems % we == 0 and inner_bytes % wb == 0:
            return we, mode
    return inner_elems, SwizzleMode.NONE


@dataclass(frozen=True)
class AffineAddressing:
    """Affine slab addressing: each cache axis ``i``'s decoded coord is
    *added* to source dim ``dims[i]``.

    ``source_index[d] = origin[d] + decoded_coord(dims[i] == d)``.

    Common case (matmul, RMSNorm, softmax). Materialize reconstructs
    addresses without symbolic substitution.

    ``block`` is a per-cache-dim structural multiplier. Default ``()``
    means "all-1s, every cache var contributes coef-1 to its source dim"
    — byte-identical to pre-M2 semantics. A non-trivial ``block`` (set
    by ``020_stage_inputs._classify`` when the σ literal coefficient on
    a cache var is > 1, e.g. the MMA ``atom_M`` / ``atom_K`` factor)
    grows the slab and producer iteration range by ``block[i]`` per
    cache dim while keeping the consumer cache vars at their original
    extent. ``affine_decode_per_dim`` (M4) folds ``block[j]`` into the
    composite stride so the per-source-dim index reconstruction matches
    the planner's σ output.
    """

    dims: tuple[int, ...]
    block: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.block:
            return
        if len(self.block) != len(self.dims):
            raise ValueError(f"AffineAddressing.block length {len(self.block)} != dims length {len(self.dims)}")
        for i, b in enumerate(self.block):
            if not isinstance(b, int) or b < 1:
                raise ValueError(f"AffineAddressing.block[{i}] must be int >= 1, got {b!r}")

    def source_index(
        self,
        cache_axes: tuple[Axis, ...],
        coord_for: dict[str, Expr],
        origin: tuple[Expr, ...],
    ) -> tuple[Expr, ...]:
        """Build the per-source-dim index expression ``origin[d] + decoded[d]``
        for the affine reconstruction.

        Thin wrapper around :func:`affine_decode_per_dim` that threads
        ``self.block`` through. Returns a tuple of length ``len(origin)``
        with one Expr per source dim. Source dims not swept by any cache
        axis carry only the origin term. This is the single source of
        truth for ``_stage_expand`` (cooperative producer),
        ``025_unify_sibling_stages._reconstruct_global_index``
        (revert-to-gmem), and M5's ``005_lower_atom_tile`` (MMA fragment
        load): each calls it with the appropriate ``coord_for`` mapping.
        """
        decoded = affine_decode_per_dim(cache_axes, self.dims, coord_for, block=self.block)
        return tuple((origin_d + decoded[d]) if d in decoded else origin_d for d, origin_d in enumerate(origin))


@dataclass(frozen=True)
class TemplateAddressing:
    """Non-affine slab addressing: the consumer Load's original index
    kept verbatim with cache-axis Vars left symbolic. Materialize
    Sigma-substitutes cache-axis Vars → iter-decoded coords.

    Used for collapsed-reshape views (``/``, ``%``) and any case where
    the affine ``origin + decoded`` reconstruction fails. Length ==
    source-buffer rank.
    """

    exprs: tuple[Expr, ...]


@dataclass(frozen=True)
class Source:
    """One gmem operand staged into one smem slab.

    Carries everything needed to materialize the cooperative producer:

    - ``name`` — smem buffer name visible to consumer Loads.
    - ``buf`` — gmem source buffer name (the input).
    - ``cache_axes`` — the smem slab's cache axes (one ``Axis`` per slab
      dim, in slab-layout order); their extents define the slab shape and
      the cooperative-load iteration domain. Which *source* buffer dim
      each cache axis decodes into lives on ``addressing`` (``AffineAddressing.dims``),
      not here — the two used to be bundled in a ``CacheDim`` whose
      ``source_dim`` simply duplicated ``addressing.dims``.
    - ``origin`` — per-source-dim CTA-uniform anchor. The cooperative
      load reads ``buf[origin[d] + cache_var(d)]`` (affine) or
      ``buf[addressing.exprs[d]]`` (template).
    - ``pad`` — per-cache-axis bank-conflict-breaking pad. Empty = no
      pad. Padding affects smem allocation, not the cooperative-load
      iteration extent.
    - ``addressing`` — stored ``AffineAddressing | TemplateAddressing``.
      Affine when every cache axis appears coef-1 (``block=()``) or
      coef-block (e.g. atom-strided MMA) in exactly one source dim;
      template when the consumer's original Load was a collapsed-reshape
      and ``origin + decoded`` can't reconstruct it. Defaults to
      ``AffineAddressing(dims=tuple(range(len(cache_axes))))`` (identity
      cache-axis → source-dim mapping) when omitted, so construction sites
      with the common identity layout can skip it. Pre-M2 this was a derived
      property of ``cache_dims`` + ``template_index``; the refactor
      collapses both addressing-mode payloads into the addressing object
      so ``Source`` stays focused on slab identity / gmem anchor.
    - ``dtype`` — source buffer's element dtype. Stamped by
      ``030_stamp_types`` from ``graph.nodes[buf].output.dtype`` so smem
      allocation (``smem_bytes`` / ``alloc_extents``) and downstream
      materialization can read it off the IR without reaching for the
      matcher-populated graph node. ``None`` keeps legacy fp32-assuming
      behavior for tests that construct Source by hand.
    - ``gmem_extents`` — the gmem buffer's per-source-dim shape (a static
      ``int`` per dim, or the dim's symbolic ``Expr`` — e.g. ``Var('seq_len')``,
      resolved from the runtime kernel arg), stamped by
      ``021_hoist_staged_loads_above_mask`` ONLY when this
      Source's cooperative load is hoisted above a masked-tile boundary
      ``Cond``. A masked output axis tiles past the real extent (e.g. N=256
      tiled at 192 → the second tile spans [192, 384); a symbolic axis tiles
      at its hint), so the cooperative gmem read overruns the buffer for the
      overhang columns. When set, ``_stage_expand.emit_stage`` clamps each
      ``source_index`` dim to ``[0, gmem_extents[d])`` so the producer never
      reads OOB (the overhang slab slots get a clamped duplicate value,
      harmless because the masked output cells they feed are never written).
      ``None`` (the default, clean-divisor tiles) skips the clamp — no perf
      cost on the common path.
    """

    name: str
    buf: str
    cache_axes: tuple[Axis, ...]
    origin: tuple[Expr, ...]
    pad: tuple[int, ...] = ()
    addressing: AffineAddressing | TemplateAddressing | None = None
    dtype: DataType | None = None
    gmem_extents: tuple[int | Expr, ...] | None = None
    # ``kmask`` — ``(source_index dim, bound Expr)`` when this operand is staged
    # for a masked-K (symbolic reduce — SDPA P@V's seq_len) warp matmul. The
    # final K_o tile overruns the runtime extent, and unlike a masked OUTPUT
    # axis (M/N — the overhang feeds a never-written cell, so an edge-clamp is
    # harmless) the reduce overhang feeds the mma ACCUMULATION, so it must be
    # ZERO, not a clamped duplicate. ``_stage_expand.emit_stage`` forces the
    # SYNC transport for a ``kmask`` source (cp.async can't ternary a value) and
    # zero-fills the loaded value where the K coord ``>= bound``. ``None`` (the
    # default) is every static-K / output-only-masked source — no change.
    kmask: tuple[int, object] | None = None
    swizzle: SwizzleMode = SwizzleMode.NONE

    def __post_init__(self) -> None:
        # Default addressing: affine with the identity cache-axis → source-dim
        # mapping. The field is typed Optional only so the default sentinel can
        # be ``None`` (a tuple-of-int default would require a frozen factory
        # for the dims, which is awkward). Frozen-set via object.__setattr__.
        if self.addressing is None:
            object.__setattr__(
                self,
                "addressing",
                AffineAddressing(dims=tuple(range(len(self.cache_axes)))),
            )

    @property
    def alloc_extents(self) -> tuple[int, ...]:
        """Per-cache-axis smem allocation extent: cache extent × block + pad.

        For affine addressing with empty ``block``, returns the bare
        cache extents (pre-M2 behavior). For affine with non-trivial
        ``block`` (M3+ stamps it on atom-strided σ), each extent is
        multiplied by ``block[i]`` so the slab holds the full per-cell
        micro-tile. Pad is added last so a future MMA-friendly swizzle
        could request padded slabs without re-deriving the block math.
        """
        extents = tuple(ax.extent.as_static() for ax in self.cache_axes)
        block: tuple[int, ...] = ()
        if isinstance(self.addressing, AffineAddressing):
            block = self.addressing.block
        if block:
            extents = tuple(e * b for e, b in zip(extents, block, strict=True))
        if not self.pad:
            return extents
        return tuple(e + p for e, p in zip(extents, self.pad, strict=True))

    @property
    def smem_bytes(self) -> int:
        """Bytes of dynamic shared memory this Source allocates (single-slot).

        Uses ``self.dtype.nbytes`` when ``030_stamp_types`` has populated it;
        falls back to the legacy fp32-assuming ``BYTES_PER_ELEM`` constant
        otherwise so handwritten test fixtures without dtype continue to work.
        """
        n = self.dtype.nbytes if self.dtype is not None else BYTES_PER_ELEM
        for e in self.alloc_extents:
            n *= e
        return n

    def with_pad(self, pad: tuple[int, ...]) -> Source:
        return replace(self, pad=pad)


def affine_decode_per_dim(
    cache_axes: tuple[Axis, ...],
    dims: tuple[int, ...],
    coord_for: dict[str, Expr],
    block: tuple[int, ...] = (),
) -> dict[int, Expr]:
    """Reconstruct the per-source-dim coord contribution from a set of
    cache axes that map to those source dims.

    For each source dim ``d``, the axes mapping to ``d`` form a composite
    in most-significant-first order: ``ax_0·(e_1·b_1·e_2·b_2·…·b_0) + … + ax_{k-1}·b_{k-1}``
    where ``e_i`` is ``cache_axes[i].extent`` and ``b_i`` is ``block[i]``
    (defaulting to 1 when ``block=()``). Each axis's coord (an Expr from
    ``coord_for[ax.name]``) is scaled by the product of ``e_j · b_j`` for
    the subsequent cache axes that ALSO map to dim ``d``, times its own
    ``b_i``, then summed per dim.

    Single-axis-per-dim with ``block=()`` collapses to a no-op
    (``stride = 1``, coord added verbatim). Multi-axis-per-dim (matmul
    N-side ``BN_thread × FN_register`` collapse) gets the composite
    stride that mirrors the original ``load.index[d]`` shape.
    Non-trivial ``block`` (e.g. the mma.sync ``(1, atom_M, 1, atom_K)``) folds
    each axis's atom multiplier into its own stride — the slab is sized
    ``extent · block`` per axis, and the per-axis decode reads from
    ``cache_var · block · stride_of_inner_axes`` so the σ output of an
    atom-strided gmem Load round-trips through smem-stage and back.

    The previous shape — ``dict(zip(dims, coord_for))`` — silently
    OVERWROTE the entry when two cache axes shared a dim, keeping only
    the last axis's coord and producing wrong gmem addresses on every
    consumer that reconstructed source indices (``_stage_expand``,
    ``025_unify_sibling_stages._reconstruct_global_index``,
    ``_source_decl_line``). Centralising the math here keeps the
    composite-stride formula consistent across all three sites.
    """
    out: dict[int, Expr] = {}
    use_block = bool(block)
    for i, (ax, d) in enumerate(zip(cache_axes, dims, strict=True)):
        stride = 1
        for j in range(i + 1, len(cache_axes)):
            if dims[j] == d:
                inner_factor = block[j] if use_block else 1
                stride *= cache_axes[j].extent.as_static() * inner_factor
        if use_block:
            stride *= block[i]
        term: Expr = coord_for[ax.name] if stride == 1 else BinaryExpr("*", coord_for[ax.name], Literal(stride, "int"))
        out[d] = term if d not in out else BinaryExpr("+", out[d], term)
    return out


def trivial_stage_body(
    name: str,
    buf: str,
    origin: tuple[Expr, ...],
    axes: tuple[Axis, ...],
    addressing: AffineAddressing | TemplateAddressing,
) -> Body:
    """**Deprecated** — kept for import compatibility during stage-wrap-body refactor.

    Pre-refactor: built the canonical ``Load + Write`` cooperative-load body
    for a Stage. Post-refactor: producer body is reconstructed at materialize
    time from ``Source`` entries; no caller should need this helper. Phase C
    bucket 12 (swizzle split) removes the last reference.
    """
    cache_index = tuple(Var(ax.name) for ax in axes)
    if isinstance(addressing, AffineAddressing):
        coord_for = {ax.name: cache_index[i] for i, ax in enumerate(axes)}
        src_index = addressing.source_index(axes, coord_for, origin)
    else:
        src_index = addressing.exprs
    load_name = f"{name}__src"
    return Body(
        (
            Load(name=load_name, input=buf, index=src_index),
            Write(output=name, index=cache_index, value=load_name),
        )
    )


def _source_pretty(src: Source) -> str:
    """Legacy single-line source description — kept for debugging / dump
    output. New consumer-facing pretty-print uses ``_source_decl_line``
    which formats each source as ``shared name[...] = buf[...]`` at the
    Stage's indent.
    """
    dims = src.addressing.dims if isinstance(src.addressing, AffineAddressing) else tuple(range(len(src.cache_axes)))
    cache = ", ".join(f"{ax.name}:{ax.extent}@{d}" for ax, d in zip(src.cache_axes, dims, strict=True))
    origin = ", ".join(e.pretty() for e in src.origin)
    pad = f" pad=({', '.join(str(p) for p in src.pad)})" if src.pad and any(src.pad) else ""
    tpl = ""
    if isinstance(src.addressing, TemplateAddressing):
        tpl = " template=[" + ", ".join(e.pretty() for e in src.addressing.exprs) + "]"
    return f"{src.name}<-{src.buf}(origin=({origin}), slab=({cache})){pad}{tpl}"


def _source_decl_line(src: Source) -> str:
    """Render one ``Source`` as ``shared <name>[<cache_axes>] = <buf>[<source_index>];``.

    Cache axes show their extents (``a5:64, a3:16``). The source index
    prefers the literal ``TemplateAddressing.exprs`` when set (preserves
    explicit stride math like ``a3*16 + a6``); otherwise reconstructs
    from ``origin + decoded`` per affine addressing semantics.

    Trailing ``pad`` and stage-flavor suffixes are NOT appended here — the
    Stage subclasses prepend / postfix those at the call site.
    """
    cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in src.cache_axes)
    if isinstance(src.addressing, TemplateAddressing):
        idx = ", ".join(e.pretty() for e in src.addressing.exprs)
    else:
        # Composite-stride decode via ``AffineAddressing.source_index``:
        # for multi-axis-per-source-dim, the i-th axis carries the
        # product of subsequent same-dim ``extent · block`` as its
        # stride. Single-axis-per-dim with ``block=()`` collapses to
        # stride 1 = bare ``ax.name``.
        coord_for = {ax.name: Var(ax.name) for ax in src.cache_axes}
        full_index = src.addressing.source_index(src.cache_axes, coord_for, src.origin)
        idx = ", ".join(e.pretty() for e in full_index)
    pad = f" pad=({', '.join(str(p) for p in src.pad)})" if src.pad and any(src.pad) else ""
    return f"shared {src.name}[{cache}] = {src.buf}[{idx}]{pad}"


# ---------------------------------------------------------------------------
# Tile-flavor pretty-print helper — bracket-on-right style
# ---------------------------------------------------------------------------
#
# Every tile flavor renders its axes as Python-style ``for X in 0..N:``
# lines (one per axis, progressively indented like a regular loop nest),
# with a vertical-pipe-and-corner bracket on the right margin grouping
# the axes belonging to one tile. The tile's label (``GridTile``,
# ``serial_outer``, ``reduce stage_inner``, etc.) sits on the closing
# corner.
#
# Example::
#
#     for a0 in 0..8:                  │
#         for a1 in 0..1:              └ GridTile
#             for a2 in 0..16:         │
#                 for a3 in 0..16:     └ ThreadTile
#                     for a4 in 0..64: └ reduce stage_inner
#                         <body>
#
# Single-axis tiles render as one line with ``└ <label>`` directly.


_BRACKET_PAD = 2  # spaces between for-text and the right-margin bracket


def _render_tile_bracket(
    indent: str,
    for_lines: list[str],
    label: str,
    body: Body,
) -> list[str]:
    """Render a tile flavor's ``for`` lines with a right-margin bracket
    grouping them, then recurse into the body at the post-innermost indent.

    ``for_lines`` are the bare ``for ... :`` strings (one per axis),
    rendered without indentation; this helper adds progressive indent.
    ``indent`` is the indent prefix for the outermost ``for``. ``label``
    is the tile-flavor annotation (``"GridTile"``, ``"reduce stage_inner"``,
    etc.) that lands on the closing corner.
    """
    # Each successive for-line indents one more level (Python loop-nest
    # convention). Compute the absolute indent per line.
    lines: list[tuple[str, str]] = []  # (indent_prefix, for_text)
    cur = indent
    for text in for_lines:
        lines.append((cur, text))
        cur = cur + INDENT
    # Right-margin column: pad to the longest (indent+text) of this group.
    max_w = max(len(ind) + len(text) for ind, text in lines)
    margin_col = max_w + _BRACKET_PAD

    out: list[str] = []
    for i, (ind, text) in enumerate(lines):
        line = ind + text
        pad = " " * (margin_col - len(line))
        is_last = i == len(lines) - 1
        bracket = f"└ {label}" if is_last else "│"
        out.append(line + pad + bracket)
    # Body lives inside the innermost ``for``, at one more indent level.
    out.extend(pretty_body(body, cur))
    return out


# ---------------------------------------------------------------------------
# Tile flavors — typed parallel / serial scoping wrappers
# ---------------------------------------------------------------------------
#
# Each tile flavor's *type* encodes its binding decision (block-grid /
# threadIdx / warp-id / register / serial / strided). Together with the
# wrap-body ``Stage`` family, these are the only block-structured Stmts
# allowed inside a ``TileOp.body`` post-``001_launch_geometry``. ``Loop`` /
# ``StridedLoop`` / ``Tile`` survive in Loop IR (``LoopOp.body``) and as
# transient inputs to ``001_launch_geometry``, but downstream Tile-IR
# passes and Tile→Kernel materialization only see the new flavors.
#
# Shape contract (mirrors ``Stage``'s wrap-body):
#
# - ``ParallelTile`` subclasses (``GridTile`` / ``ThreadTile`` /
#   ``WarpTile`` / ``RegisterTile``) carry ``axes: tuple[Axis, ...]`` +
#   ``body: Body``. The body executes once per coord tuple; coords are
#   implicit from the binding (``blockIdx`` / ``threadIdx`` /
#   ``threadIdx.x / 32`` / per-thread register cell). ``ThreadTile`` and
#   ``WarpTile`` are mutually exclusive inside one ``TileOp.body``
#   (``TileOp.__post_init__`` rejects mixes) — both bind ``threadIdx``.
# - ``SerialTileBase`` subclasses (``SerialTile`` / ``StridedTile``)
#   carry ``axis: Axis`` + ``body: Body`` and run sequentially. Reduce
#   semantics are derived: ``is_reduce`` iff the body contains ``Accum``.


@dataclass(frozen=True)
class ParallelTile(Stmt):
    """Abstract base for tile flavors that bind a parallel axis tuple.

    Subclasses pick a parallel coord (``blockIdx`` / ``threadIdx`` /
    register file) for the body to be executed under. Coord decode happens
    at materialize time; the tile itself only carries the axes + body.
    """

    axes: tuple[Axis, ...]
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return replace(self, body=body)

    def binds_axes(self) -> frozenset[str]:
        return frozenset(ax.name for ax in self.axes)

    def deps(self) -> tuple[str, ...]:
        return ()

    def _pretty_axes(self) -> str:
        return ", ".join(f"{ax.name}:{ax.extent}" for ax in self.axes) or "-"

    def _pretty_label(self) -> str:
        """Right-margin bracket label. Subclasses override to append
        flavor-specific metadata if any."""
        return type(self).__name__.lower().replace("tile", "")

    def pretty(self, indent: str = "") -> list[str]:
        if not self.axes:
            # Degenerate empty-axis tile (shouldn't normally happen) — just
            # render the label as a one-line header so the body still nests.
            head = f"{indent}{self._pretty_label()}"
            return [head, *pretty_body(self.body, indent + INDENT)]
        for_lines = [f"for {ax.name} in 0..{ax.extent}" for ax in self.axes]
        return _render_tile_bracket(indent, for_lines, self._pretty_label(), self.body)


@dataclass(frozen=True)
class GridTile(ParallelTile):
    """CTA-grid parallel tile. Axes lift to ``blockIdx`` (row-major).

    Replaces ``Tile`` with ``BIND_BLOCK`` axes. Split-K is derived at
    codegen time from ``escape_analysis.atomic_axes`` (Write index vs
    enclosing block axes) — no per-tile metadata required.

    ``swizzle_group_m`` selects an L2-friendly CTA-ID remap for matmul-shape
    grids (axes ending in ``(M_b, N_b)``): consecutive CTAs walk down M
    in groups of ``swizzle_group_m`` before stepping N, so a row-group of
    CTAs shares A's row tile in L2 (Triton/CUTLASS/cuBLAS convention).
    ``swizzle_group_m == 1`` (the default) keeps the row-major decode and
    is a structural no-op; the swizzled path is stamped by
    ``tile/025_swizzle_blocks.py`` on matmul-shape GridTiles. The field
    feeds ``_pretty_label`` so the structural digest tracks it.
    """

    swizzle_group_m: int = 1

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return GridTile(axes=self.axes, body=body, swizzle_group_m=self.swizzle_group_m)

    def _pretty_label(self) -> str:
        if self.swizzle_group_m == 1:
            return "grid"
        return f"grid swizzle_M={self.swizzle_group_m}"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit ``blockIdx.x`` axis decode + body. The inner ``ThreadTile``
        renders its threadIdx decode under ``ctx.inside_grid_tile=True``,
        so no per-CTA bounds guard is needed at this level."""
        if self.swizzle_group_m != 1:
            out = list(_render_swizzled_grid_decode(self.axes, "blockIdx.x", self.swizzle_group_m, ctx))
        else:
            out = list(_render_grid_axis_decode(self.axes, "blockIdx.x", ctx))
        inner_ctx = replace(ctx, inside_grid_tile=True)
        out.extend(_render_body(self.body, inner_ctx))
        return out


@dataclass(frozen=True)
class ThreadTile(ParallelTile):
    """Thread-parallel tile. Axes lift to ``threadIdx`` (row-major flatten).

    Replaces ``Tile`` with ``BIND_THREAD`` axes. Cooperative-K
    cooperativity is derived at materialize / render time from
    ``Accum.axes ∩ ThreadTile.axes`` — see
    ``ir/tile/escape_analysis.py``.

    ``tid_offset`` (default ``0``) shifts the linear thread index the
    cooperative-form decode is computed against — the per-axis decls
    use ``(threadIdx.x - tid_offset)`` instead of plain ``threadIdx.x``.
    Non-zero values are emitted by the warp-specialize materializer to
    drop a ``ThreadTile(consumer_thread_axes, tid_offset=n_producer_threads, …)``
    inside the consumer ``Cond.else_body``, so the original consumer-side
    thread axes decode against a consumer-relative tid in ``[0,
    n_consumer_threads)``. The field carries no semantic meaning outside
    that materializer-emitted nesting; planner-emitted ``ThreadTile``s
    keep the default ``0``.
    """

    tid_offset: int = 0

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return ThreadTile(axes=self.axes, body=body, tid_offset=self.tid_offset)

    def _pretty_label(self) -> str:
        if self.tid_offset:
            return f"thread offset={self.tid_offset}"
        return "thread"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Two render forms picked by ``ctx.inside_grid_tile``.

        - **Cooperative** (inside ``GridTile``): emit ``threadIdx.x``
          axis decode (optionally offset by ``tid_offset`` — used by the
          warp-specialize consumer arm) + optional ``lane`` / ``warp``
          helper decls + body. No extra brace level — the surrounding
          ``__global__`` provides one.
        - **Standalone** (pointwise — no enclosing ``GridTile``): flatten
          all axes into a linear ``tid``; bounds-guard against the product
          of extents.
        """
        pad = _pad(ctx.indent)
        if ctx.inside_grid_tile:
            idx_expr = "threadIdx.x" if self.tid_offset == 0 else f"(threadIdx.x - {self.tid_offset})"
            out = list(_render_grid_axis_decode(self.axes, idx_expr, ctx))
            if _body_uses_lane_warp(self.body):
                out.append(f"{pad}int lane = threadIdx.x & 31;")
                out.append(f"{pad}int warp = threadIdx.x >> 5;")
            out.extend(_render_body(self.body, ctx))
            return out

        if self.tid_offset:
            raise NotImplementedError("standalone ThreadTile with non-zero tid_offset not supported")
        inner = ctx.child()
        n_threads = 1
        for ax in self.axes:
            n_threads *= ax.extent.as_static()
        out = [
            f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
            f"{pad}if (tid < {n_threads}) {{",
        ]
        out.extend(_render_thread_axis_decode(self.axes, inner))
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class RegisterTile(ParallelTile):
    """Per-thread register-cell tile. Body replicated F× per axis by
    ``010_split_register_axes``.

    Replaces ``Loop(role=REGISTER)``. The ``axes`` tuple carries one or
    more register axes (typically M_r / N_r for matmul); the planner
    chooses the extents (``FM`` / ``FN`` knobs). After the
    ``010_split_register_axes`` pass runs, every ``RegisterTile`` is
    consumed: the body is fully unrolled, SSA names get per-cell
    suffixes, and the ``RegisterTile`` wrapper disappears.

    ``reduce`` marks the **reduce-axis** (K_f) register tile emitted by the
    partition planner for the ``FK`` multiple-accumulator optimization
    (see ``plans/fk-register-tile-reductions.md``). It strip-mines the K
    (reduce) axis so each of the ``FK`` cells owns an independent
    accumulator. When the wrapped body contains ``Accum``s,
    ``010_split_register_axes`` replicates each into ``acc_0 .. acc_{FK-1}``
    and then appends a cross-accumulator tree-fold collapsing them back into
    the original accumulator name after the enclosing K serial loops close;
    a K_f tile wrapping a non-reduce (post-pointwise) body carries the flag
    too but folds nothing (its replicas are independent FK-unrolled writes).
    The flag also routes knob stamping (``FK`` vs ``FM``/``FN``). The default
    ``reduce=False`` is the output-cell (FM/FN) tile — replicate-and-drop,
    no fold.
    """

    reduce: bool = False

    def _pretty_label(self) -> str:
        return "register reduce" if self.reduce else "register"

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            "RegisterTile must be consumed by 006a_register_tile_planned before render — "
            f"reached render with axes={tuple(ax.name for ax in self.axes)!r}"
        )


@dataclass(frozen=True)
class AtomTile(ParallelTile):
    """Hardware-atomic-cell tile — one coord = one MMA fragment cell.

    Marker for the per-cell hardware-atomic extent on a matmul-reduce kernel
    (see ``plans/mma-fragment-factorization.md``). The axes carry the cell
    shape (e.g. ``(M=16, N=8)``); the body inside is the per-cell compute that
    ``011_lower_atom_cell`` turns into an ``Mma`` + the lowering replaces with
    the fragment chain.

    ``atom`` is the :class:`Atom` this cell realises, carried **structurally**
    on the tile — its presence (and this field) is the "this kernel factorizes
    through tensor cores" signal, and ``011_lower_atom_cell`` reads ``.atom``
    off it to build the ``Mma`` (no ``ATOM_KIND`` knob lookup; the knob is the
    tuning shadow of the same choice, not the semantic source). Scalar matmul
    kernels never emit an ``AtomTile``.

    Render is intentionally unimplemented: every ``AtomTile`` must be
    consumed before kernel render, mirroring ``RegisterTile``'s contract.
    """

    atom: Atom

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            "AtomTile must be consumed by the MMA materializer "
            "(kernel/005_lower_atom_tile) before render — an AtomTile here "
            "usually means tile/011_lower_atom_cell could not tag the cell "
            "(operand A/B classification failed) — "
            f"reached render with axes={tuple(ax.name for ax in self.axes)!r}"
        )


@dataclass(frozen=True)
class WarpTile(ParallelTile):
    """Warp-parallel tile — one coord tuple = one warp (32 lanes).

    The body executes once per warp coord; the 32 lanes inside the warp
    execute it collectively (cooperative MMA, lane-aware shuffles, etc.).
    Materialization rules / consumers — MMA fragment factorization
    (``plans/mma-fragment-factorization.md``), warp-specialized TMA
    pipelining refactor — emit ``WarpTile`` *inside* an outer
    ``GridTile`` to bind warps to a CTA. The cooperative form is the
    only one supported in v1; a standalone top-level ``WarpTile``
    (pointwise-style "one warp per output cell") has no consumer today.

    Rendering binds ``warp_id = threadIdx.x / 32`` (the row-major decode
    over the warp axes uses ``warp_id`` as the linear index), and
    unconditionally exposes ``lane = threadIdx.x & 31`` — the body
    presumes a lane is available (that's the entire reason a warp coord
    exists). Launch-bounds wiring (``_launch_bounds_for`` /
    ``_launch_geometry``) multiplies the warp-axis product by 32.

    Mutual exclusion with ``ThreadTile`` inside one ``TileOp.body`` is
    enforced by ``TileOp.__post_init__`` — both bind ``threadIdx`` and
    mixing would re-bind the same coord at two scopes.

    ``tid_offset`` (default ``0``, must be a multiple of 32) shifts the warp
    decode: ``warp_id = (threadIdx.x - tid_offset) / 32``. Used by the
    warp-specialized consumer branch — ``085_warp_specialize`` wraps the
    warp-tier MMA consumer in ``WarpTile(tid_offset = n_producer_threads)`` so
    its warps decode against the *consumer* warp range ``[n_producer_warps,
    total_warps)``. ``lane`` is unaffected (the offset is whole warps, so
    ``(threadIdx.x - 32k) & 31 == threadIdx.x & 31``).
    """

    tid_offset: int = 0

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return WarpTile(axes=self.axes, body=body, tid_offset=self.tid_offset)

    def _pretty_label(self) -> str:
        return "warp" if not self.tid_offset else f"warp offset={self.tid_offset}"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Cooperative form (inside ``GridTile``): ``warp_id`` decl + row-
        major warp-axis decode against ``warp_id`` + unconditional ``lane``
        decl + body. Standalone (no enclosing ``GridTile``) is not
        supported in v1 — pointwise kernels use ``ThreadTile`` and a
        top-level ``WarpTile`` has no consumer yet.
        """
        if not ctx.inside_grid_tile:
            raise NotImplementedError("WarpTile outside GridTile not supported in v1")
        pad = _pad(ctx.indent)
        tid = "threadIdx.x" if self.tid_offset == 0 else f"(threadIdx.x - {self.tid_offset})"
        out: list[str] = [f"{pad}int warp_id = {tid} / 32;"]
        out.extend(_render_grid_axis_decode(self.axes, "warp_id", ctx))
        out.append(f"{pad}int lane = threadIdx.x & 31;")
        out.extend(_render_body(self.body, ctx))
        return out


@dataclass(frozen=True)
class SerialTileBase(Stmt):
    """Abstract base for serial-iteration tile flavors. One axis, one body."""

    axis: Axis
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

    def deps(self) -> tuple[str, ...]:
        return ()

    @property
    def is_reduce(self) -> bool:
        """A serial tile is a reduce iff its immediate body contains a
        ``ReduceCarrier`` — an ``Accum`` or its tensor-core fused form ``Mma``
        (which accumulates ``c += a @ b``)."""
        return any(isinstance(s, ReduceCarrier) for s in self.body)


@dataclass(frozen=True)
class SerialTile(SerialTileBase):
    """Sequential iteration over ``axis``. Replaces ``Loop``.

    ``kind`` carries the planner's structural intent:

    - ``"plain"``: ordinary serial loop (no special role).
    - ``"serial_outer"``: outer chunked-K loop driving slab refresh
      (today's ``Role.SERIAL_OUTER``). Targeted by ``040_use_ring_buffers``
      / ``015_pipeline_k_outer``.
    - ``"stage_inner"``: inner reduce loop inside a ``Stage``'s wrapped
      body (today's ``Role.STAGE_INNER``). Slab-axis marker for
      ``020_stage_inputs``.
    - ``"pipeline"``: serial outer loop marked for temporal pipelining
      by ``015_pipeline_k_outer``.

    ``unroll=True`` annotates the loop for ``#pragma unroll`` at render
    time. Set by ``090_mark_unroll``; has no effect on iteration semantics.
    """

    kind: SerialKind = "plain"
    unroll: bool = False

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return SerialTile(axis=self.axis, body=body, kind=self.kind, unroll=self.unroll)

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Per-Loop accumulator-init prelude (same as ``Loop.render``) +
        ``for (int axis = 0; axis < extent; axis++) { body }``."""
        from deplodock.compiler.dtype import F32 as _F32  # noqa: PLC0415

        pad = _pad(ctx.indent)
        out: list[str] = []
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
                ctx.ssa_dtypes[s.name] = (s.dtype or _F32).name
        var = self.axis.name
        # ``Dim.__str__`` returns the bare value (literal for static, symbolic
        # name for ``Dim('seq_len')``) so both static and symbolic SerialTile
        # extents render correctly. A COMPOSITE extent (the ceil-div K_o bound
        # of a masked-K warp tile, ``(seq_len+31)//32``) must go through the C
        # expr renderer so ``//`` becomes ``/`` — ``str`` would leak the Python
        # spelling and nvcc would read it as a line comment.
        ext = self.axis.extent
        if ext.is_static or isinstance(ext.expr, Var):
            extent = str(ext)
        else:
            extent = f"({ext.expr.render(ctx)})"
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class StridedTile(SerialTileBase):
    """Strided serial iteration: ``for (axis = start; axis < extent; axis += step)``.

    Replaces ``StridedLoop``. Cooperative thread-stride iteration when a
    surrounding ``ThreadTile`` axis covers the stride (typical
    ``start = Var('tid'), step = BLOCK_SIZE``). Reduce semantics derive
    from body content like ``SerialTile``.
    """

    start: Expr = field(default_factory=lambda: Literal(0, "int"))
    step: Expr = field(default_factory=lambda: Literal(1, "int"))
    unroll: bool = False

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return StridedTile(axis=self.axis, body=body, start=self.start, step=self.step, unroll=self.unroll)

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = (self.start,)
        if isinstance(self.step, Expr):
            out = (*out, self.step)
        return out

    def pretty(self, indent: str = "") -> list[str]:
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}for {self.axis.name} in {start}..{self.axis.extent}:{step}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``for (int axis = start; axis < extent; axis += step)`` with the
        same per-Loop accumulator-init prelude as ``SerialTile.render``."""
        from deplodock.compiler.dtype import F32 as _F32  # noqa: PLC0415

        pad = _pad(ctx.indent)
        out: list[str] = []
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
                ctx.ssa_dtypes[s.name] = (s.dtype or _F32).name
        var = self.axis.name
        start_str = self.start.render(ctx)
        step_str = self.step.render(ctx) if isinstance(self.step, Expr) else str(self.step)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {self.axis.extent.as_static()}; {var} += {step_str}) {{")
        inner = ctx.child()
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


# ---------------------------------------------------------------------------
# StageBundle — single-policy cooperative-staging unit
# ---------------------------------------------------------------------------
#
# Replaces the wrap-body Stage hierarchy. The intermediate ``Stage``
# grouping layer is gone: a ``StageBundle`` holds its ``sources`` directly
# (the gmem transport operands), the consumer ``body``, an optional
# ``compute`` phase, and the staging policy — sync transport,
# ring-buffered, cp.async, TMA. A bundle is single-policy: every source
# shares the same transport. K_o-dependency partitioning produces separate
# bundles with different policies rather than mixed bundles.
#
# Mapping from old hierarchy:
#   Stage(sources, body)                                 → StageBundle(sources, body, SYNC)
#   BufferedStage(sources, body, buffer_count, phase)    → StageBundle(sources, body, BUFFERED, buffer_count, phase)
#   AsyncBufferedStage(sources, body, ..., depth)        → StageBundle(sources, body, ASYNC, ..., pipeline_depth)
#   TmaBufferedStage(sources, body, ..., depth)          → StageBundle(sources, body, TMA, ..., pipeline_depth)
#   ComputeStage(sources, body, compute, ...)            → StageBundle(sources, body, compute=..., SYNC, ...)
#
# The hoisted-invariant cooperative compute (sibling-smem → own-smem
# producer template) is a distinct ``StageBundle.compute`` phase; the
# transport sources are homogeneous (multi-source loads behind one
# barrier; per-source TMA descriptors are emitted directly).


class StagePolicy(enum.Enum):
    """Transport policy for a ``StageBundle`` — applied uniformly to every
    source.

    - ``SYNC``      — cooperative ``Load + Write`` + ``__syncthreads``
                      barrier. No ring-buffering.
    - ``BUFFERED``  — sync transport into ``buffer_count >= 2`` rotating
                      slabs selected by ``phase``. Drops the leading
                      pre-load sync since consecutive iterations write
                      to different physical slabs.
    - ``ASYNC``     — ring-buffered ``cp.async``; requires sm_80+.
                      Implicit wait at wrap boundary when
                      ``pipeline_depth == 1``; explicit
                      ``AsyncWait`` peeling when ``> 1``.
    - ``TMA``       — ring-buffered ``cp.async.bulk.tensor`` (TMA box
                      copy); requires sm_90+. Mbarrier-based completion.
    """

    SYNC = "sync"
    BUFFERED = "buffered"
    ASYNC = "async"
    TMA = "tma"


@dataclass(frozen=True)
class StageBundle(Stmt):
    """Single-policy cooperative-staging unit: a homogeneous group of gmem
    transport ``sources`` (loaded behind one barrier) wrapping one consumer
    ``body``, plus an optional hoisted-invariant ``compute`` phase.

    The bundle owns the consumer scope (``body``) and the staging policy
    (``policy`` plus policy-specific fields ``buffer_count`` / ``phase`` /
    ``pipeline_depth``). Per-source TMA swizzle lives on each ``Source``.

    ``compute`` (optional) is a *self-describing* cooperative compute body:
    ``Load``s reading already-staged sibling slabs (the transport
    ``sources``), ``Assign``s applying a transform, and a single ``Write``
    into a freshly derived smem slab. It is emitted as a distinct bundle
    *phase* — after the transport sources, before the consumer body — so the
    transport sources stay homogeneous and the slab name / loop domain /
    dtype are recovered at materialize from the body itself (the
    ``Write.output`` name, the cone sources' ``cache_axes``, the ``Write``
    value dtype). Set by ``030_hoist_invariant_compute``.

    Invariants:

    - ``sources`` is non-empty.
    - All sources share the bundle's policy; mixed-policy bundles are
      illegal. K_o-dep partition (040_use_ring_buffers) emits separate
      bundles with different policies rather than mixing.
    - Source ORDER == issue order. Passes looking for a specific source
      MUST locate it by name, never by position.
    - ``buffer_count >= 2`` requires ``phase``; allowed only for
      ``BUFFERED`` / ``ASYNC`` / ``TMA`` policies. ``SYNC`` has
      ``buffer_count == 1``, ``phase is None``.
    - ``pipeline_depth > 1`` allowed only for ``ASYNC`` / ``TMA``.
    - ``TMA`` policy: every Source must have empty ``pad`` (TMA writes
      contiguous rows; bank-pad would misalign).

    Iteration / recursion:

    - ``nested()`` returns ``(self.compute or Body(()), self.body)`` — the
      compute phase (a plain ``Body``) and the consumer body. An absent
      compute phase is exposed as ``Body(())`` so the arity stays fixed;
      ``with_bodies`` collapses an empty leading body back to ``None``.
      Sources carry no nested Stmt bodies, so they aren't exposed here.
    """

    sources: tuple[Source, ...]
    body: Body
    compute: Body | None = None
    policy: StagePolicy = StagePolicy.SYNC
    buffer_count: int = 1
    phase: Expr | None = None
    pipeline_depth: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))
        if self.compute is not None and not isinstance(self.compute, Body):
            object.__setattr__(self, "compute", Body.coerce(self.compute))
        if not self.sources:
            raise ValueError("StageBundle: requires at least one Source")
        if self.policy == StagePolicy.SYNC:
            if self.buffer_count != 1:
                raise ValueError(f"StageBundle SYNC: buffer_count must be 1, got {self.buffer_count}")
            if self.phase is not None:
                raise ValueError("StageBundle SYNC: phase must be None")
            if self.pipeline_depth != 1:
                raise ValueError(f"StageBundle SYNC: pipeline_depth must be 1, got {self.pipeline_depth}")
        else:
            if self.buffer_count < 1:
                raise ValueError(f"StageBundle: buffer_count must be >= 1, got {self.buffer_count}")
            if self.buffer_count >= 2 and self.phase is None:
                raise ValueError(f"StageBundle {self.policy.value}: phase required when buffer_count >= 2")
        if self.pipeline_depth != 1 and self.policy not in (StagePolicy.ASYNC, StagePolicy.TMA):
            raise ValueError(f"StageBundle: pipeline_depth > 1 requires ASYNC or TMA policy, got {self.policy.value}")
        if self.policy == StagePolicy.TMA:
            for src in self.sources:
                if src.pad and any(src.pad):
                    raise ValueError(f"StageBundle TMA: source {src.name!r} pad must be empty, got {src.pad!r}")

    def nested(self) -> tuple[Body, ...]:
        return (self.compute if self.compute is not None else Body(()), self.body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if len(bodies) != 2:
            raise ValueError(f"StageBundle.with_bodies: expected 2 bodies (compute, body), got {len(bodies)}")
        compute_body, body = bodies
        compute = compute_body if compute_body else None
        return replace(self, compute=compute, body=body)

    def deps(self) -> tuple[str, ...]:
        return ()

    def external_reads(self) -> tuple[str, ...]:
        return tuple(s.buf for s in self.sources)

    def local_decls(self) -> tuple[str, ...]:
        out: tuple[str, ...] = tuple(s.name for s in self.sources)
        # The compute phase fills a fresh smem slab named by its Write — a
        # kernel-local buffer, not a signature input/output. Declaring it
        # here keeps ``BodyOp._derive_io_names`` from promoting it to a
        # kernel arg (the materializer pre-emits its ``Smem`` decl).
        if self.compute is not None:
            out = (*out, *(s.output for s in self.compute if isinstance(s, Write)))
        return out

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        for s in self.sources:
            out = (*out, *s.origin)
            if isinstance(s.addressing, TemplateAddressing):
                out = (*out, *s.addressing.exprs)
        if self.phase is not None:
            out = (*out, self.phase)
        return out

    @property
    def smem_bytes(self) -> int:
        per_slab = sum(s.smem_bytes for s in self.sources)
        return per_slab * max(self.buffer_count, 1)

    def _policy_label(self) -> str:
        """Render a compact ``policy[buffer_count@phase depth=N]`` label used
        in the bundle header line."""
        if self.policy == StagePolicy.SYNC:
            return "sync"
        parts: list[str] = [f"{self.policy.value}[{self.buffer_count}"]
        if self.phase is not None:
            parts.append(f"@{self.phase.pretty()}")
        if self.pipeline_depth > 1:
            parts.append(f" depth={self.pipeline_depth}")
        parts.append("]")
        return "".join(parts)

    def pretty(self, indent: str = "") -> list[str]:
        """Render as ``bundle <policy>:`` header with per-source decl lines,
        the optional cooperative compute phase, and the consumer body
        indented beneath."""
        inner = indent + INDENT
        out: list[str] = [f"{indent}bundle {self._policy_label()}:"]
        for s in self.sources:
            out.append(f"{inner}{_source_decl_line(s)}")
        if self.compute is not None:
            # The compute body names its inputs (sibling cone slabs) and
            # its output slab via a final Write; the cache-axis loop nest
            # is synthesized at materialize, so render it as a flat
            # ``cooperative:`` block of the body stmts.
            out.append(f"{inner}cooperative:")
            out.extend(pretty_body(self.compute, inner + INDENT))
        out.extend(pretty_body(self.body, inner))
        return out


# ---------------------------------------------------------------------------
# Source-aware traversal
# ---------------------------------------------------------------------------

# A ``map_staged`` visitor: given a stmt and the in-scope ``Source`` table,
# return a replacement tuple (splice in, no descent into that stmt) or ``None``
# (descend structurally). The handler is the only thing a caller writes; the
# StageBundle / WarpSpecialize / nested descent — and the Source threading that
# makes the in-scope table available — is shared.
StagedHandler = Callable[["Stmt", dict[str, "Source"]], "tuple[Stmt, ...] | None"]


def map_staged(body: Body, handler: StagedHandler, *, sources: dict[str, Source] | None = None) -> Body:
    """Rewrite ``body`` while threading the in-scope ``Source`` table.

    For each stmt, call ``handler(stmt, sources)``: a returned tuple splices in
    (no descent into that stmt); ``None`` means descend structurally. Descending
    a :class:`StageBundle` adds its stage ``Source``s to the table; a
    :class:`WarpSpecialize` adds the producer's, then rewrites only the consumer
    body — so a consumer access always sees the slabs its producer staged. Other
    block stmts recurse with the table unchanged.

    The one source-aware traversal lowering passes share instead of re-rolling
    the descent + threading by hand. Used both as a transformer (handler returns
    rewrites) and, with an always-``None`` collecting handler, as a visitor (the
    rebuilt body is discarded; the handler accumulates via closure)."""
    scope = sources if sources is not None else {}
    out: list[Stmt] = []
    for s in body:
        repl = handler(s, scope)
        if repl is not None:
            out.extend(repl)
            continue
        if isinstance(s, StageBundle):
            inner = dict(scope)
            for src in s.sources:
                inner[src.name] = src
            compute = map_staged(s.compute, handler, sources=inner) if s.compute is not None else Body(())
            out.append(s.with_bodies((compute, map_staged(s.body, handler, sources=inner))))
            continue
        if isinstance(s, WarpSpecialize):
            inner = dict(scope)
            for st in s.producer_body.iter():
                if isinstance(st, StageBundle):
                    for src in st.sources:
                        inner[src.name] = src
            out.append(s.with_bodies((s.producer_body, map_staged(s.consumer_body, handler, sources=inner))))
            continue
        if s.nested():
            out.append(s.with_bodies(tuple(map_staged(sub, handler, sources=scope) for sub in s.nested())))
            continue
        out.append(s)
    return Body(out)


# ---------------------------------------------------------------------------
# Top-level: TileOp
# ---------------------------------------------------------------------------


@dataclass
class TileOp(BodyOp):
    """One GPU kernel as a Tile IR program — pre-materialization.

    :class:`BodyOp` subclass parallel to ``LoopOp``: lives as a graph
    node, carries a body of Tile IR statements plus a kernel name.
    Materialization turns a ``TileOp`` into a ``KernelOp``.
    """

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.stmt import normalize_body

        coerced = Body.coerce(self.body)
        normalized = normalize_body(coerced, hoist=False)
        self.body = normalized if isinstance(normalized, Body) else Body(normalized)
        n_tiles = sum(1 for s in self.body if isinstance(s, (GridTile, ThreadTile, WarpTile)))
        if n_tiles > 1:
            raise ValueError(f"TileOp.body must contain at most one outer GridTile/ThreadTile/WarpTile, got {n_tiles}")
        # ThreadTile and WarpTile both bind threadIdx; mixing them inside one
        # body re-binds the same coord at two scopes. The outer-tile check
        # above already catches them at top level; this catches the
        # cooperative form (GridTile wrapping a ThreadTile and a WarpTile
        # sibling, or one of them nesting inside the other).
        has_thread = any(isinstance(s, ThreadTile) for s in self.body.iter())
        has_warp = any(isinstance(s, WarpTile) for s in self.body.iter())
        if has_thread and has_warp:
            raise ValueError("TileOp.body cannot contain both a ThreadTile and a WarpTile (both bind threadIdx)")
        self._seed_io_placeholders()

    def _launch_geometry(self) -> tuple[tuple[Axis, ...], tuple[Axis, ...]]:
        """``(block_axes, thread_axes)`` for the outermost tile flavor.

        Returns ``((), ())`` if no ``GridTile``/``ThreadTile`` is present
        (e.g. a degenerate body, or a warp-cooperative body whose inner
        tile is a ``WarpTile`` rather than a ``ThreadTile`` — see
        :meth:`_warp_axes`). For ``GridTile`` wrapping a ``ThreadTile``,
        the block axes come from the GridTile and thread axes from the
        inner ThreadTile. For a standalone ``ThreadTile`` (pointwise), the
        block set is empty.
        """
        for s in self.body:
            if isinstance(s, GridTile):
                block_axes = s.axes
                for child in s.body:
                    if isinstance(child, ThreadTile):
                        return block_axes, child.axes
                return block_axes, ()
            if isinstance(s, ThreadTile):
                return (), s.axes
        return (), ()

    def _warp_axes(self) -> tuple[Axis, ...]:
        """Inner-``WarpTile`` axes, or ``()`` if no ``WarpTile`` is present.

        Companion accessor to :meth:`_launch_geometry`. Warp-cooperative
        bodies (``GridTile > WarpTile > …``, today emitted only by the
        MMA-fragment-factorization consumer plan) bind 32 lanes per warp
        coord — callers that compute per-CTA thread budgets must add
        ``prod(_warp_axes().extent) * 32`` on top of the (typically empty)
        thread-axes product. ``ThreadTile`` and ``WarpTile`` are mutually
        exclusive inside one body (``__post_init__`` rejects mixes), so
        either this returns ``()`` or :meth:`_launch_geometry`'s
        ``thread_axes`` does.
        """
        for s in self.body:
            if isinstance(s, GridTile):
                for child in s.body:
                    if isinstance(child, WarpTile):
                        return child.axes
                return ()
            if isinstance(s, WarpTile):
                return s.axes
        return ()

    def validate(self, ctx) -> bool:
        """Reject post-register-tile variants whose launch geometry would
        exceed device limits (threads-per-CTA and dynamic smem).

        Pre-register-tile TileOps skip the THREAD check; the smem check
        runs whenever Stages are present.
        """
        from math import prod  # noqa: PLC0415

        # Dedupe by Source name: pipelining
        # (080_pipeline_stages) replicates an
        # ``AsyncBufferedStage`` for prologue + steady-state issue, both
        # writing into the same smem buffer (same name, same allocation).
        # Counting them independently would double-charge the budget and
        # silently reject pipelined variants on smem-tight kernels.
        per_source: dict[str, int] = {}
        for s in self.body.iter():
            if isinstance(s, StageBundle):
                for src in s.sources:
                    # Single-slot per-source slab — the ring ``buffer_count``
                    # factor is intentionally NOT applied here (this budget
                    # gate counts the un-multiplied footprint; the old
                    # per-Stage walk did the same since ``Stage`` carried no
                    # ``buffer_count``).
                    per_source[src.name] = src.smem_bytes
        staged = sum(per_source.values())
        if staged > ctx.max_dynamic_smem:
            return False

        if "FM" not in self.knobs:
            return True
        _, thread_axes = self._launch_geometry()
        if not thread_axes:
            return True
        threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in thread_axes)
        return threads <= ctx.max_threads_per_cta


# ---------------------------------------------------------------------------
# Cooperative thread-block size — number of threads per CUDA block when a
# Tile uses BIND_THREAD axes from a cooperative strategy.
# ---------------------------------------------------------------------------

from deplodock.compiler.tuning import cooperative_block_size as _coop_block_size  # noqa: E402

BLOCK_SIZE = _coop_block_size()


__all__ = [
    "Atom",
    "ATOM_REGISTRY",
    # Shared expressions (re-exported for convenience)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (re-exported)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    "StridedLoop",
    # Tile-IR statements — typed tile flavor hierarchy
    "ParallelTile",
    "GridTile",
    "ThreadTile",
    "RegisterTile",
    "WarpTile",
    "AtomTile",
    "SerialTileBase",
    "SerialTile",
    "StridedTile",
    "SerialKind",
    "StageBundle",
    "StagePolicy",
    "SwizzleMode",
    "AffineAddressing",
    "TemplateAddressing",
    "Source",
    "AsyncWait",
    # Source-aware traversal (transformer / visitor over staged bodies)
    "map_staged",
    "StagedHandler",
    "trivial_stage_body",  # deprecated stub during refactor
    "BYTES_PER_ELEM",
    "Stmt",
    # Top-level
    "TileOp",
    # Scheduling constants
    "BLOCK_SIZE",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]

# Register Tile-IR stmts with the shared rewrite/simplify dispatch.
from deplodock.compiler.ir.tile import passes as _passes  # noqa: E402, F401
