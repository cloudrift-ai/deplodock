"""Tile schedule type system — how a kernel's axes bind to the hardware.

Split out of :mod:`.ir`. The whole layer's thesis is that the **schedule is separate
from the combine**: the combine (the ⊕) lives in the op tree
(:mod:`deplodock.compiler.ir.stmt.algebra` + :mod:`.structural`), and the schedule — which axes
are parallel, how the reduce axis partitions across hardware levels — is the **codec value
types** defined here (:class:`ReducePlan` / :class:`TilePlan` / :class:`Stage` / :class:`WarpSpec` +
:class:`Placement`). They ride on the structural nodes (a :class:`~.structural.Contraction`'s
``tile``, a :class:`~.structural.Reduction`'s ``reduce``) and on the thin root
:class:`~.ir.TileOp` fields (``place`` / ``workers`` + the residual reduce/tier/stage) — there is no
longer a ``Kernel`` / ``TileSchedule`` wrapper (both collapsed into ``TileOp``).

A reduction's only freedom is **how the reduce axis is partitioned across hardware
levels** (:class:`ReducePlan`); the combine *mechanism* at each level is **derived** from
the level (:meth:`ReduceStage.combine`), and the combine *algebra* rides the carrier (the
``Twist``). So the same op + the same materializer extend across kernel kinds — only the
carrier and the partition change.

The schedule is **flat and kind-free**. The old per-kind ``MapSchedule`` / ``MonoidSchedule`` /
``SemiringSchedule`` zoo (paired in ``MapKernel`` / ``MonoidKernel`` / ``SemiringKernel``) is gone, as
are the ``Monoid`` / ``Semiring`` op-tree node kinds: a kernel's structure is read from its annotated
reduce loop's :class:`~deplodock.compiler.ir.axis.AxisRole` (``ops.axis_role``), not a Python type, so a
pointwise kernel, a ``PLANAR`` / ``TWISTED`` reduce, and a ``CONTRACTION`` contraction all schedule
through the same codec value types and use the subset their axes admit. Warp specialization is
**orthogonal**, not a second schedule kind: it rides an optional ``workers: WarpSpec | None`` root
field (``None`` = uniform SIMT) — a role→warp-count split *over* the fixed pipeline.

Every codec here — ``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec`` —
routes its ``parse`` / ``spell`` through the shared schema engine (:mod:`.codec`), declaring a
``Schema`` of typed ``Field``\\s and keeping only its own semantics. ``WarpSpec`` materialization
(the producer/consumer warp emission) is reserved this cut (``# TODO(warp-spec)``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.ir.atom import SCALAR_ATOM, Atom, AtomKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.tile.codec import Emit, Field, FieldKind, Schema, decode, encode, field_default
from deplodock.compiler.ir.tile.role import ROLE_REGISTRY, RoleKind


class Level(enum.Enum):
    """One hardware level the reduce axis can be partitioned across, coarse→fine."""

    GRID = "grid"  # across CTAs (split-K) — emitted by 030_split, never the in-kernel walk
    BLOCK = "block"  # cooperative threads within a CTA (warp shuffle / smem tree)
    REG = "reg"  # ILP register-fold accumulators
    SERIAL = "serial"  # the per-thread serial remainder (never spelled — derived)


class Fold(enum.Enum):
    """The per-level combine *mechanism* — derived from the :class:`Level`, not tuned."""

    SERIAL = "serial"  # no cross-unit combine (the serial / reg remainder)
    REG = "reg"  # register tree (ILP) — TODO(reg)
    SHFL = "shfl"  # lane-level ``__shfl_xor_sync`` butterfly
    SMEM = "smem"  # cross-warp / block-wide smem tree-halve
    ATOMIC = "atomic"  # cross-CTA ``atomicAdd`` finalize — TODO(cta), 030_split only


@dataclass(frozen=True)
class ReduceStage:
    """One level's **tuned** partition: a ``width`` of partials at a hardware ``level``.

    The combine *mechanism* is **derived** (:meth:`combine`), not stored — the level
    implies the fold, and a BLOCK width derives warp-shuffle vs hierarchical-smem from the
    warp size. ``width`` is power-of-two for BLOCK (the butterfly / tree reorder).

    ``finalize`` is meaningful only at ``GRID``: how ``030_split`` realizes the cross-CTA
    combine — ``"atomic"`` (the partial kernel ``atomicAdd``\\ s into the output, one kernel,
    additive carriers only) or ``"kernel"`` (a deferred sibling combine kernel over a
    workspace, the only legal arm for a twisted carrier). The ``g<n>[a|k]`` codec letter."""

    level: Level
    width: int = 1
    finalize: str = "kernel"  # GRID only: "atomic" | "kernel" (the g<n> finalize letter)

    def combine(self, *, warp_size: int, segmented: bool = False) -> tuple[Fold, ...]:
        """The derived per-level combine fold(s), fine→coarse within this stage.

        - ``SERIAL`` / ``REG`` → ``()`` (no cross-unit combine; REG-fold is TODO(reg)).
        - ``GRID`` → ``(ATOMIC,)`` (the split-K finalize — emitted by ``030_split``).
        - ``BLOCK`` → the intra-CTA hierarchy: a lone ``SHFL`` when ``segmented`` (the
          per-row segmented butterfly for strided-cooperative rows) or ``width ≤ warp``
          (one warp); ``(SHFL, SMEM)`` when ``width`` is a clean warp multiple (lanes then
          the cross-warp tree); else a standalone ``(SMEM,)`` block tree. Power-of-two
          ``width`` required."""
        if self.level in (Level.SERIAL, Level.REG):
            return ()
        if self.level is Level.GRID:
            return (Fold.ATOMIC,)
        # BLOCK.
        w = self.width
        if w & (w - 1):
            raise ValueError(f"BLOCK reduce width must be a power of two, got {w}")
        if segmented or w <= warp_size:
            return (Fold.SHFL,)
        if w % warp_size == 0:
            return (Fold.SHFL, Fold.SMEM)
        return (Fold.SMEM,)


#: The ``REDUCE`` codec schema (decoded / encoded by :mod:`codec`): ``g<n>[a|k]`` (GRID cross-CTA
#: split + finalize letter), ``b<n>`` (BLOCK cooperative threads), ``r<n>`` (REG ILP fold).
_REDUCE_SCHEMA = Schema(
    "REDUCE",
    (
        Field("g", FieldKind.TUPLE, suffix=(("a", "atomic"), ("k", "kernel")), suffix_default="kernel"),
        Field("b", FieldKind.TUPLE),
        Field("r", FieldKind.TUPLE),
    ),
    expect="expect g<n> / b<n> / r<n>",
)


@dataclass(frozen=True)
class ReducePlan:
    """The kernel's single reduce partition — the **tuned widths only**, coarse→fine.

    There is one reduce carrier per kernel (1:1 and singular — the carrier owns the axis),
    so the plan holds no axis; the per-thread ``serial`` remainder is derived by the
    materializer as ``ceil(extent / parallel)``. ``stages=()`` is the scalar serial fold
    (today's one-thread-per-cell tier)."""

    stages: tuple[ReduceStage, ...] = ()

    @classmethod
    def of(cls, *, cta: int = 1, coop: int = 1, reg: int = 1, finalize: str = "kernel") -> ReducePlan:
        """Build a plan from per-level widths (1 = absent). Order is coarse→fine:
        GRID (cta) → BLOCK (coop) → REG (reg). ``finalize`` rides the GRID stage."""
        stages: list[ReduceStage] = []
        if cta > 1:
            stages.append(ReduceStage(Level.GRID, cta, finalize=finalize))
        if coop > 1:
            stages.append(ReduceStage(Level.BLOCK, coop))
        if reg > 1:
            stages.append(ReduceStage(Level.REG, reg))
        return cls(tuple(stages))

    @classmethod
    def parse(cls, spec: str | None) -> ReducePlan:
        """Decode the ``REDUCE`` knob codec (the schedule's single reduce-partition knob,
        decided in ``020_schedule``) into a plan: ``/``-separated level-named tokens,
        coarse→fine — ``g<n>[a|k]`` (GRID cross-CTA split + finalize letter), ``b<n>``
        (BLOCK cooperative threads), ``r<n>`` (REG ILP fold). Empty / ``None`` = the scalar
        serial fold. (The ``serial`` remainder is never spelled — it's derived as
        ``ceil(extent / parallel)``.) Ser/de routes through :mod:`codec` (``_REDUCE_SCHEMA``)."""
        v = decode(_REDUCE_SCHEMA, spec)
        width, finalize = v["g"]
        return cls.of(cta=width, coop=v["b"], reg=v["r"], finalize=finalize)

    def spell(self) -> str:
        """The ``REDUCE`` codec string for this plan (inverse of :meth:`parse`); ``""`` for
        the scalar serial fold. A GRID stage re-emits its finalize letter (``g<n>a`` /
        ``g<n>k``)."""
        return encode(_REDUCE_SCHEMA, {"g": (self.cta, self.finalize), "b": self.coop, "r": self.reg})

    @property
    def parallel(self) -> int:
        """The total parallel degree = ∏ stage widths (the lane/CTA fan-out the serial
        remainder divides into)."""
        p = 1
        for s in self.stages:
            p *= s.width
        return p

    @property
    def needs_split(self) -> bool:
        """True iff any stage is a cross-CTA GRID split (``030_split`` territory)."""
        return any(s.level is Level.GRID for s in self.stages)

    def _width(self, level: Level) -> int:
        for s in self.stages:
            if s.level is level:
                return s.width
        return 1

    @property
    def coop(self) -> int:
        """The BLOCK (cooperative-thread) width, or 1 if no BLOCK stage."""
        return self._width(Level.BLOCK)

    @property
    def cta(self) -> int:
        """The GRID (cross-CTA split) width, or 1 if no GRID stage."""
        return self._width(Level.GRID)

    @property
    def finalize(self) -> str:
        """The GRID stage's cross-CTA finalize — ``"atomic"`` | ``"kernel"`` (``"kernel"``
        if no GRID stage; the value is only meaningful when :attr:`needs_split`)."""
        for s in self.stages:
            if s.level is Level.GRID:
                return s.finalize
        return "kernel"

    @property
    def reg(self) -> int:
        """The REG (ILP fold) width, or 1 if no REG stage."""
        return self._width(Level.REG)

    @property
    def block_stage(self) -> ReduceStage | None:
        """The single BLOCK :class:`ReduceStage`, or ``None`` (scalar serial)."""
        for s in self.stages:
            if s.level is Level.BLOCK:
                return s
        return None


#: The scalar form of the ``TILE`` codec: ``n<N>[x<M>]`` (parallel thread-tile, n-then-m) and
#: ``f<fn>[x<fm>]`` (register sub-tile) — no atom token.
_TILE_SCALAR_SCHEMA = Schema(
    "TILE",
    (Field("n", FieldKind.TUPLE, arity=2), Field("f", FieldKind.TUPLE, arity=2)),
    expect="expect n<N>[x<M>] / f<fn>[x<fm>]",
)

#: The warp form of the ``TILE`` codec: ``a:<atom>`` (required, the registered atom), ``w<WM>x<WN>``
#: (warps m-then-n, always both dims), ``f<FM>x<FN>`` (register sub-tile), ``k<bk>`` (K-chunk,
#: omitted at 1). ``a``/``w``/``f`` always spell; ``k`` only when > 1.
_WARP_SCHEMA = Schema(
    "WARP",
    (
        Field("a", FieldKind.NAME, required=True, emit=Emit.ALWAYS),
        Field("w", FieldKind.TUPLE, arity=2, emit=Emit.ALWAYS, suppress_trailing=False),
        Field("f", FieldKind.TUPLE, arity=2, emit=Emit.ALWAYS, suppress_trailing=False),
        Field("k", FieldKind.TUPLE),
    ),
    expect="expect a:<atom> / w<WM>x<WN> / f<FM>x<FN> / k<bk>",
)


def is_warp_codec(spec: str | None) -> bool:
    """True iff a ``TILE`` codec value names a tensor-core atom (an ``a:<atom>`` token) — the **warp**
    form, vs the scalar register sub-tile (``n../f..``). This is the single string-side discriminator
    for the unified output-tile knob: a contraction's output tile is *either* the scalar fragment *or*
    the warp mma tile, never both, and the value self-describes which. Empty / ``None`` (the per-cell
    scalar baseline) is not warp."""
    return bool(spec) and any(t.strip().startswith("a:") for t in spec.split("/"))


@dataclass(frozen=True)
class TilePlan:
    """The contraction's output tile — **one descriptor for both tiers**, discriminated by
    :attr:`atom`: a tensor-core :class:`AtomKind` (the warp mma tile) or the scalar
    :class:`~deplodock.compiler.ir.atom.ScalarAtom` (the register sub-tile, the ``Scalar``
    fragment). Each tiled output axis splits into a **unit** width (warps for mma / parallel threads
    for scalar — :attr:`units`) and a **register** width (atom sub-cells / register cells per unit —
    :attr:`regs`); ``bk`` chunks the K (contraction) axis (mma only). The all-``1`` scalar tile (a
    scalar ``atom``, ``units``/``regs`` ``(1, 1)``) is the per-cell tier — one thread per output cell.

    ``units`` / ``regs`` are stored in each tier's **native codec order** — warp ``(WM, WN)`` /
    ``(FM, FN)`` (m-then-n), scalar ``(par_n, par_m)`` / ``(reg_n, reg_m)`` (n-then-m, the
    featurizer's inner/coalesced ``n`` vs outer ``m``); the :attr:`units_m` / :attr:`units_n` /
    :attr:`reg_m` / :attr:`reg_n` accessors normalize that order. Spelled by the unified ``TILE``
    knob — the warp form ``a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>`` or the scalar ``n<N>x<M>/f<fn>x<fm>``
    (no atom token); :func:`is_warp_codec` discriminates string-side, :attr:`is_warp` on the object.
    Decided in ``020_schedule``."""

    atom: Atom = SCALAR_ATOM
    units: tuple[int, int] = (1, 1)  # warp (WM, WN) m-then-n / scalar (par_n, par_m) n-then-m
    regs: tuple[int, int] = (1, 1)  # warp (FM, FN) m-then-n / scalar (reg_n, reg_m) n-then-m
    bk: int = 1  # K-chunk per inner mma step, in atom_k units (mma only; 1 for scalar)

    @classmethod
    def parse(cls, spec: str | None) -> TilePlan:
        """Decode the unified ``TILE`` knob into a tile: the warp form (``a:<atom>/w../f../k..``) →
        an mma tile, or the scalar form (``n../f..``, no atom token) → a scalar register sub-tile
        (``atom`` defaults to the scalar atom). Empty / ``None`` = the per-cell tier. Ser/de routes
        through :mod:`codec`."""
        if is_warp_codec(spec):
            v = decode(_WARP_SCHEMA, spec)
            return cls(atom=v["a"], units=v["w"], regs=v["f"], bk=v["k"])
        v = decode(_TILE_SCALAR_SCHEMA, spec)
        return cls(units=v["n"], regs=v["f"])

    def spell(self) -> str:
        """The ``TILE`` codec string for this tile (inverse of :meth:`parse`); ``""`` for the
        per-cell tier. The warp form when :attr:`atom` is a tensor-core atom, else the scalar form."""
        if self.is_warp:
            return encode(_WARP_SCHEMA, {"a": self.atom, "w": self.units, "f": self.regs, "k": self.bk})
        return encode(_TILE_SCALAR_SCHEMA, {"n": self.units, "f": self.regs})

    @property
    def is_warp(self) -> bool:
        """True iff this is the tensor-core (mma) tile — :attr:`atom` is a real :class:`AtomKind`."""
        return isinstance(self.atom, AtomKind)

    @property
    def is_tiled(self) -> bool:
        """True iff this materializes a tile: a warp tile always does; a scalar tile only when some
        unit / register width > 1 (else it is the per-cell tier — one thread per output cell)."""
        return self.is_warp or any(v > 1 for v in (*self.units, *self.regs))

    @property
    def units_m(self) -> int:
        """Units on the outer (m) output axis — ``WM`` (warp) / ``par_m`` (scalar)."""
        return self.units[0] if self.is_warp else self.units[1]

    @property
    def units_n(self) -> int:
        """Units on the inner (n) output axis — ``WN`` (warp) / ``par_n`` (scalar)."""
        return self.units[1] if self.is_warp else self.units[0]

    @property
    def reg_m(self) -> int:
        """Register sub-cells on the outer (m) axis — ``FM`` (warp) / ``reg_m`` (scalar)."""
        return self.regs[0] if self.is_warp else self.regs[1]

    @property
    def reg_n(self) -> int:
        """Register sub-cells on the inner (n) axis — ``FN`` (warp) / ``reg_n`` (scalar)."""
        return self.regs[1] if self.is_warp else self.regs[0]

    @property
    def block_threads(self) -> int:
        """The per-CTA thread count = ``units_m · units_n · atom.lanes`` (mma: ``WM·WN·32``;
        scalar: ``par_n·par_m·1``)."""
        return self.units[0] * self.units[1] * self.atom.lanes

    @property
    def tile_m(self) -> int:
        """The per-CTA output rows = ``units_m · reg_m · atom_m``."""
        return self.units_m * self.reg_m * self.atom.atom_m

    @property
    def tile_n(self) -> int:
        """The per-CTA output cols = ``units_n · reg_n · atom_n``."""
        return self.units_n * self.reg_n * self.atom.atom_n


@dataclass(frozen=True)
class Placement:
    """Kind-neutral free-axis → grid binding (the parallel output axes and their grid
    mapping). ``010_recognize`` builds an UNMAPPED placement (just ``free``);
    ``020_schedule`` maps every free axis onto ``grid`` (the per-cell tier)."""

    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()

    @property
    def is_mapped(self) -> bool:
        """True once the free axes are bound (``grid`` set) — or there were none to bind
        (a scalar-output kernel materializes on an empty grid)."""
        return bool(self.grid) or not self.free

    def on_grid(self) -> Placement:
        """The scalar-tier mapping: bind every free axis onto the thread grid."""
        return Placement(free=self.free, grid=self.free)


# --------------------------------------------------------------------------- #
# The operand-transport + warp-split descriptors. ``Stage`` (cp.async / TMA) is built and
# materialized; ``WarpSpec`` (the WSPEC worker split) is pin-only this cut — its codec + schedule
# field land, but its producer/consumer codegen is reserved (``# TODO(warp-spec)`` in
# lowering/kernel/010_materialize).
# --------------------------------------------------------------------------- #


#: The codec transport token (``cp``) vs the canonical stored value (``cp.async``).
_TRANSPORT_CODEC = {"sync": "sync", "cp": "cp.async", "tma": "tma"}
_TRANSPORT_SPELL = {v: k for k, v in _TRANSPORT_CODEC.items()}

#: The ``STAGE`` codec schema: ``d<depth>`` and the transport always spell; ``ring`` only when set;
#: ``p<reg_depth>`` only at ≥ 2. The transport choices derive from ``_TRANSPORT_CODEC`` (one source).
_STAGE_SCHEMA = Schema(
    "STAGE",
    (
        Field("d", FieldKind.TUPLE, emit=Emit.ALWAYS),
        Field("transport", FieldKind.CHOICE, choices=tuple(_TRANSPORT_CODEC.items()), default="sync", emit=Emit.ALWAYS),
        Field("ring", FieldKind.FLAG, emit=Emit.TRUE),
        Field("p", FieldKind.TUPLE),
    ),
    expect="expect d<depth> / sync|cp|tma / ring / p<reg_depth>",
)


@dataclass(frozen=True)
class Stage:
    """One operand-transport pipeline over the serial reduce loop — one ``Stage`` per reduce
    loop (a reduce ``Loop`` ⇒ one reduce axis ⇒ one pipeline). The schedule's
    operand-staging knob, decided in ``020_schedule`` and materialized in
    ``010_materialize``.

    A constructed ``Stage`` means staging is **on** (the reused gmem operands ride a shared-
    memory slab); ``schedule.stage is None`` is the register / gmem-direct baseline (no
    slab). Spelled by the ``STAGE`` codec ``d<depth>/sync|cp|tma[/ring][/p<reg_depth>]``
    (decided in ``020_schedule``). ``smem`` (the staged-operand buffer names) is **derived
    during lowering** — the materializer stages the reused gmem reads — not spelled by the
    codec; an empty ``smem`` means "stage every reused operand".

    The pipeline has two buffering levels down the memory hierarchy, each with its own depth:
    ``depth`` is the **gmem→smem** ring (the cp.async / TMA prefetch over the serial reduce
    loop), ``reg_depth`` is the **smem→register** double-buffer (the ldmatrix ping-pong over
    the inner atom-K steps, breaking the WAR hazard on the operand fragments). They are
    orthogonal — ``d3/cp/p2`` is a 3-deep gmem ring feeding a 2-deep register ping-pong.
    ``reg_depth = 1`` (the default) is the "optional register" OFF point (no inner prefetch).
    The slab K-*granularity* (how much K is resident) is ``TilePlan.bk``, NOT a third depth
    here — granularity and buffer depth are kept distinct."""

    depth: int = 1  # gmem→smem ring depth over the reduce loop (1 = single buffer, no prefetch)
    transport: str = "sync"  # sync | cp.async | tma (the gmem→smem producer)
    smem: tuple[str, ...] = ()  # operands staged through smem (derived; not in the codec)
    ring: bool = False  # ring buffer vs static double-buffer
    reg_depth: int = 1  # smem→register double-buffer depth (1 = no inner ldmatrix prefetch)

    def __post_init__(self) -> None:
        if self.transport not in _TRANSPORT_SPELL:
            raise ValueError(f"bad Stage transport {self.transport!r} (expect sync | cp.async | tma)")
        if self.depth < 1:
            raise ValueError(f"Stage depth must be ≥ 1, got {self.depth}")
        if self.ring and self.depth < 2:
            raise ValueError("a ring buffer needs depth ≥ 2 (nothing to cycle at depth 1)")
        if self.reg_depth < 1:
            raise ValueError(f"Stage reg_depth must be ≥ 1, got {self.reg_depth}")

    @classmethod
    def parse(cls, spec: str | None) -> Stage:
        """Decode the ``STAGE`` knob codec into a stage: ``/``-separated tokens —
        ``d<depth>`` (gmem→smem ring depth), ``sync`` | ``cp`` | ``tma`` (the transport), an
        optional ``ring`` flag, and an optional ``p<reg_depth>`` (smem→register double-buffer
        depth). Empty / ``None`` = the depth-1 ``sync`` default (the caller maps an empty
        ``STAGE`` to ``stage=None``, the gmem-direct baseline — ``parse`` is only reached on a
        non-empty spec). ``smem`` is filled in later by the scheduler. Ser/de routes through
        :mod:`codec`; ``cls(...)`` then runs :meth:`__post_init__` (the depth / ring semantics)."""
        v = decode(_STAGE_SCHEMA, spec)
        return cls(depth=v["d"], transport=v["transport"], ring=v["ring"], reg_depth=v["p"])

    def spell(self) -> str:
        """The ``STAGE`` codec string for this stage (inverse of :meth:`parse`). ``smem`` is
        derived, so it is not spelled; ``reg_depth`` is spelled only when ≥ 2 (the ``p1``
        default is omitted, so an unstaged-register config round-trips byte-identical)."""
        return encode(
            _STAGE_SCHEMA,
            {"d": self.depth, "transport": self.transport, "ring": self.ring, "p": self.reg_depth},
        )

    @property
    def is_async(self) -> bool:
        """True for the asynchronous-copy transports (``cp.async`` / ``tma``) — the ones
        that issue a commit/wait or mbarrier handshake rather than a plain ``__syncthreads``."""
        return self.transport in ("cp.async", "tma")


#: The ``WSPEC`` codec schema — one GROUP field per registered :class:`RoleKind` (token =
#: warp-count value, params = the role's per-role param schema). Built from ``ROLE_REGISTRY`` so a
#: new role needs no edit here. The COMPUTE role is implicit (``TilePlan.units``), never a field.
_WSPEC_SCHEMA = Schema(
    "WSPEC",
    tuple(Field(r.token, FieldKind.GROUP, params=r.params) for r in ROLE_REGISTRY.values()),
    expect="expect <role><warps>[:<param>,...]",
)


@dataclass(frozen=True)
class RoleAlloc:
    """One warp-specialized band: a registered :class:`RoleKind`, its dedicated warp count, and its
    per-role param values (canonical-sorted, default-valued params dropped so they don't re-spell)."""

    role: RoleKind
    warps: int = 1
    params: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class WarpSpec:
    """The worker-mapping pin — a role→warp-count allocation over the fixed pipeline, ORTHOGONAL to
    it (``reduce`` / ``tile`` / ``stage``): it adds no pipeline parameter, only the warp split. The
    COMPUTE (mma-consumer) role is implicit (sized by ``TilePlan.units``, never listed); each
    :class:`RoleAlloc` is a band of dedicated warps split off the uniform pipeline, so the CTA
    launches ``TilePlan.block_threads + 32·aux_warps`` threads. ``workers is None`` on the schedule
    is uniform SIMT (every warp does every role's work, software-pipelined in-warp); a constructed
    ``WarpSpec`` means specialization is on. Spelled by the ``WSPEC`` codec ``<token><np>[:<param>,
    ...]`` per role (``p2`` / ``p2:q8`` / ``p2:q8/s1``), decided in ``020_schedule``.

    **Reserved this cut**: the schedule field + the codec land (pin-only), but ``010_materialize``
    does not yet emit producer/consumer warps (``# TODO(warp-spec)``)."""

    roles: tuple[RoleAlloc, ...] = ()

    @classmethod
    def parse(cls, spec: str | None) -> WarpSpec:
        """Decode the ``WSPEC`` codec into a role allocation (``""`` / ``None`` = no roles). Ser/de
        routes through :mod:`codec` (``_WSPEC_SCHEMA``)."""
        v = decode(_WSPEC_SCHEMA, spec)
        allocs: list[RoleAlloc] = []
        for token, role in ROLE_REGISTRY.items():
            group = v[token]
            if group is None:  # role absent from the codec
                continue
            params = tuple(sorted((p.token, group[p.token]) for p in role.params if group[p.token] != field_default(p)))
            allocs.append(RoleAlloc(role=role, warps=group[""], params=params))
        return cls(tuple(allocs))

    def spell(self) -> str:
        """The ``WSPEC`` codec string for this allocation (inverse of :meth:`parse`); ``""`` when
        there are no roles (uniform SIMT)."""
        values: dict[str, object] = {token: None for token in ROLE_REGISTRY}
        for a in self.roles:
            values[a.role.token] = {"": a.warps, **dict(a.params)}
        return encode(_WSPEC_SCHEMA, values)

    @property
    def aux_warps(self) -> int:
        """Total dedicated (non-COMPUTE) warps the split adds on top of the ``TilePlan`` warp grid."""
        return sum(a.warps for a in self.roles)

    def is_legal(self, sched: object) -> bool:
        """True iff every allocated role is meaningful for ``sched`` (a producer needs a ``stage``).
        A pin failing this degrades to uniform (``workers=None``) — the same pin-validity rule the
        other codecs follow."""
        return all(a.role.legal(sched) for a in self.roles)


__all__ = [
    "Fold",
    "Level",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "RoleAlloc",
    "Stage",
    "TilePlan",
    "WarpSpec",
]
