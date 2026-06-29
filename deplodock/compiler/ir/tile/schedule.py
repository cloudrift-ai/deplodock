"""Tile schedule type system — how a kernel's axes bind to the hardware.

Split out of :mod:`.ir`. The whole layer's thesis is that the **schedule is separate
from the combine**: the combine (the ⊕) lives in the op tree
(:mod:`deplodock.compiler.ir.stmt.algebra`), and the schedule — which axes are parallel,
how the reduce axis partitions across hardware levels — lives here, on a typed
``*Schedule`` paired with the op-tree node in a ``*Kernel``.

A reduction's only freedom is **how the reduce axis is partitioned across hardware
levels** (:class:`ReducePlan`); the combine *mechanism* at each level is **derived** from
the level (:meth:`ReduceStage.combine`), and the combine *algebra* rides the carrier (the
``Twist``). So the same op + the same materializer extend across kernel kinds — only the
carrier and the partition change.

The schedule is **flat, typed by the outermost algebra kind**: a ``Map`` →
:class:`MapSchedule`, a reduction ``Monoid`` → :class:`MonoidSchedule`, a contraction
``Semiring`` → :class:`SemiringSchedule` (flash — a ``Monoid`` over a nested partial
``Semiring`` — is a :class:`MonoidSchedule`; the op tree nests, the schedule does not).
The schedule of a reducing kind is **either** that kind's uniform (SIMT) schedule **or**
:class:`WarpSpec` (the warp-role pipeline, reserved). The pairing in the ``*Kernel``
types makes a ``Monoid``-with-``MapSchedule`` mismatch unrepresentable.

This module builds the **uniform** arm (``ReducePlan`` / ``TilePlan`` / ``WarpTile`` /
``Stage``); ``Channel`` / ``WarpSpec`` (warp specialization) are reserved slots (``# TODO``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.tile.atom import AtomKind
from deplodock.compiler.ir.tile.binding import AtomBinding
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


#: The scalar ``TILE`` codec schema: ``n<N>[x<M>]`` (parallel thread-tile, n-then-m) and
#: ``f<fn>[x<fm>]`` (register sub-tile). The warp fragment uses ``_WARP_SCHEMA`` instead, selected
#: string-side by :func:`is_warp_codec` before either class's ``parse`` runs.
_TILE_SCALAR_SCHEMA = Schema(
    "TILE",
    (Field("n", FieldKind.TUPLE, arity=2), Field("f", FieldKind.TUPLE, arity=2)),
    expect="expect n<N>[x<M>] / f<fn>[x<fm>]",
)


@dataclass(frozen=True)
class TilePlan:
    """The free-axis output tile — the **tuned widths only** for the (≤2) tiled output
    axes, the scalar (``Scalar`` fragment) counterpart of :class:`ReducePlan`.

    Each tiled free axis splits into a **parallel** width (threads per axis — the
    ``n``/``m`` slots) and a **register** width (per-thread register sub-cells — the ``f``
    slots): one thread owns a ``reg_m × reg_n`` block of output cells, reusing each loaded
    operand across the block (the arithmetic-intensity lever for scalar matmul). All-``1``
    is the per-cell tier (one thread per output cell), exactly as ``ReducePlan()`` is the
    serial fold. The two axes follow the featurizer's canonical ``n`` (inner / coalesced)
    vs ``m`` (outer) labelling (:func:`knob._free_slots`)."""

    par_n: int = 1
    reg_n: int = 1
    par_m: int = 1
    reg_m: int = 1

    @classmethod
    def parse(cls, spec: str | None) -> TilePlan:
        """Decode the ``TILE`` knob codec (the schedule's free-axis output-tile knob,
        decided in ``020_schedule``) into a plan: ``/``-separated tokens — ``n<N>[x<M>]``
        (the parallel thread-tile, ``N`` threads on the inner axis, optional ``M`` on the
        outer) and ``f<fn>[x<fm>]`` (the register sub-tile, ``fn`` cells on the inner axis,
        optional ``fm`` on the outer). The ``x`` is a plain dimension separator (n-then-m
        order). Empty / ``None`` = the per-cell tier. Ser/de routes through :mod:`codec`."""
        v = decode(_TILE_SCALAR_SCHEMA, spec)
        par_n, par_m = v["n"]
        reg_n, reg_m = v["f"]
        return cls(par_n=par_n, reg_n=reg_n, par_m=par_m, reg_m=reg_m)

    def spell(self) -> str:
        """The ``TILE`` codec string for this plan (inverse of :meth:`parse`); ``""`` for
        the per-cell tier."""
        return encode(_TILE_SCALAR_SCHEMA, {"n": (self.par_n, self.par_m), "f": (self.reg_n, self.reg_m)})

    @property
    def is_tiled(self) -> bool:
        """True iff this plan tiles the output (any parallel or register width > 1)."""
        return self.par_n > 1 or self.par_m > 1 or self.reg_n > 1 or self.reg_m > 1

    @property
    def reg(self) -> tuple[int, int]:
        """The register sub-tile ``(reg_m, reg_n)`` — outer (m) then inner (n) cells/thread."""
        return (self.reg_m, self.reg_n)

    @property
    def cells(self) -> int:
        """Register cells per thread = ``reg_m · reg_n``."""
        return self.reg_m * self.reg_n

    @property
    def slots(self) -> tuple[int, int, int, int]:
        """The featurizer's canonical ``(par_n, reg_n, par_m, reg_m)`` free-split tuple."""
        return (self.par_n, self.reg_n, self.par_m, self.reg_m)


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
# Reserved slots — the tensor-core tile, operand pipelining, warp specialization.
# Defined so the type system is complete (the schedule fields reference them); not
# constructed by this cut.
# --------------------------------------------------------------------------- #


#: The warp ``TILE`` codec schema: ``a:<atom>`` (required, the registered atom), ``w<WM>x<WN>``
#: (warps m-then-n, always both dims), ``f<FM>x<FN>`` (register sub-tile), ``k<bk>`` (K-chunk,
#: omitted at 1). ``a``/``w``/``f`` always spell; ``k`` only when > 1 — the old ``spell`` policy.
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


@dataclass(frozen=True)
class WarpTile:
    """The tensor-core mma tile — a **shared** descriptor on both :class:`MonoidSchedule`
    (flash's inner QK/PV — ``# TODO(warp-flash)``) and :class:`SemiringSchedule` (matmul,
    built). A contraction is mma-tiled onto the one mapping whether it is the top node or
    nested; never a nested/second schedule.

    Each warp owns a ``reg`` (``FM × FN``) block of ``atom`` cells; the CTA runs ``WM × WN``
    warps. So the per-CTA output tile is ``tile_m × tile_n`` (:attr:`tile_m` / :attr:`tile_n`)
    and the CTA launches :attr:`block_threads` ``= WM·WN·32`` threads. ``bk`` chunks the K
    (contraction) axis ``bk`` atom-cells per inner mma step. Spelled by the warp form of the
    unified ``TILE`` knob — ``a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>`` (an ``a:<atom>`` token
    selects this :class:`WarpTile` over the scalar :class:`TilePlan`; see :func:`is_warp_codec`),
    decided in ``020_schedule``."""

    atom: AtomKind
    warps: tuple[int, int] = (1, 1)  # (WM, WN) — warps per CTA, m then n
    reg: tuple[int, int] = (1, 1)  # (FM, FN) — atom sub-tiles per warp, m then n
    bk: int = 1  # K-chunk per inner mma step, in atom_k units

    @classmethod
    def parse(cls, spec: str) -> WarpTile:
        """Decode the ``WARP`` codec into a tile: ``/``-separated tokens —
        ``a:<atom>`` (the registered atom kind), ``w<WM>x<WN>`` (warps, m then n),
        ``f<FM>x<FN>`` (register sub-tile, m then n), ``k<bk>`` (K-chunk). The ``x`` is a plain
        dimension separator. The atom token is mandatory; the rest default to ``1``. Ser/de
        routes through :mod:`codec`."""
        v = decode(_WARP_SCHEMA, spec)
        return cls(atom=v["a"], warps=v["w"], reg=v["f"], bk=v["k"])

    def spell(self) -> str:
        """The ``WARP`` codec string for this tile (inverse of :meth:`parse`)."""
        return encode(_WARP_SCHEMA, {"a": self.atom, "w": self.warps, "f": self.reg, "k": self.bk})

    @property
    def tile_m(self) -> int:
        """The per-CTA output rows = ``WM · FM · atom_m``."""
        return self.warps[0] * self.reg[0] * self.atom.atom_m

    @property
    def tile_n(self) -> int:
        """The per-CTA output cols = ``WN · FN · atom_n``."""
        return self.warps[1] * self.reg[1] * self.atom.atom_n

    @property
    def block_threads(self) -> int:
        """The per-CTA thread count = ``WM · WN · 32`` (32 lanes per warp)."""
        return self.warps[0] * self.warps[1] * 32


def is_warp_codec(spec: str | None) -> bool:
    """True iff a ``TILE`` codec value spells the **warp** fragment (a :class:`WarpTile`) rather
    than the scalar :class:`TilePlan` — i.e. it carries an ``a:<atom>`` token naming a tensor-core
    atom. This is the single discriminator for the unified output-fragment knob: a contraction's
    output tile is *either* the scalar register sub-tile (``n../f..``) *or* the warp mma tile
    (``a:.../w../f../k..``), never both, and the value self-describes which. Empty / ``None`` (the
    per-cell scalar baseline) is not warp."""
    return bool(spec) and any(t.strip().startswith("a:") for t in spec.split("/"))


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
    loop (a ``Monoid`` / ``Semiring`` ⇒ one reduce axis ⇒ one pipeline). The schedule's
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
    The slab K-*granularity* (how much K is resident) is ``WarpTile.bk``, NOT a third depth
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
#: new role needs no edit here. The COMPUTE role is implicit (``WarpTile.warps``), never a field.
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
    COMPUTE (mma-consumer) role is implicit (sized by ``WarpTile.warps``, never listed); each
    :class:`RoleAlloc` is a band of dedicated warps split off the uniform pipeline, so the CTA
    launches ``WarpTile.block_threads + 32·aux_warps`` threads. ``workers is None`` on the schedule
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
        """Total dedicated (non-COMPUTE) warps the split adds on top of the ``WarpTile`` grid."""
        return sum(a.warps for a in self.roles)

    def is_legal(self, sched: object) -> bool:
        """True iff every allocated role is meaningful for ``sched`` (a producer needs a ``stage``).
        A pin failing this degrades to uniform (``workers=None``) — the same pin-validity rule the
        other codecs follow."""
        return all(a.role.legal(sched) for a in self.roles)


# --------------------------------------------------------------------------- #
# The three uniform (SIMT) schedules — one thread / block / warp mapping per kind.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MapSchedule:
    """A pointwise kernel's schedule — just the free-axis grid binding (no reduce, no
    warp-spec: pointwise never warp-specializes)."""

    place: Placement


@dataclass(frozen=True)
class MonoidSchedule:
    """A reduction (``Monoid``) kernel's schedule.

    Three **orthogonal** reduce-axis fields (kept distinct so they don't become a
    grab-bag): ``reduce`` (the :class:`ReducePlan` partition), ``warp_tile`` (the mma
    operand tile — flash's inner QK/PV, ``# TODO(warp-flash)``), ``stage`` (operand
    transport — the :class:`Stage` smem pipeline, ``None`` = gmem-direct). ``block`` are
    free axes resident in the CTA alongside the cooperative lanes (strided-cooperative
    rows)."""

    place: Placement
    block: tuple[Axis, ...] = ()
    reduce: ReducePlan = field(default_factory=ReducePlan)
    warp_tile: WarpTile | None = None  # TODO(warp-flash)
    stage: Stage | None = None  # None = gmem-direct (no smem slab)
    workers: WarpSpec | None = None  # None = uniform SIMT; else the producer/compute warp split (WSPEC)


@dataclass(frozen=True)
class SemiringSchedule:
    """A contraction (``Semiring``) kernel's schedule — the same orthogonal reduce-axis
    fields as :class:`MonoidSchedule` (``warp_tile`` is the matmul's mma tile), plus the
    free-axis output :class:`TilePlan` (``tile`` — the scalar register sub-tile, the
    ``Scalar`` fragment of the matmul, decided by the ``TILE`` knob in ``020_schedule``).
    ``stage`` is the operand smem pipeline (``None`` = gmem-direct)."""

    place: Placement
    block: tuple[Axis, ...] = ()
    reduce: ReducePlan = field(default_factory=ReducePlan)
    tile: TilePlan = field(default_factory=TilePlan)
    warp_tile: WarpTile | None = None  # TODO(warp-flash)
    stage: Stage | None = None  # None = gmem-direct (no smem slab)
    workers: WarpSpec | None = None  # None = uniform SIMT; else the producer/compute warp split (WSPEC)
    bind: AtomBinding | None = None  # the operand→role binding, filled by 020_schedule after
    # the warp tile is chosen (None until then / on a scalar-tile or split-partial contraction)


# --------------------------------------------------------------------------- #
# The op + schedule pairs — one uniform ``*Schedule`` per kind. Warp specialization
# is an orthogonal ``workers: WarpSpec | None`` field ON that schedule (None = uniform
# SIMT), NOT a union arm — it adds a warp split over the fixed pipeline, not a replacement.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MapKernel:
    """A pointwise kernel: a ``Map`` op + its :class:`MapSchedule` (no warp-spec — pointwise
    never warp-specializes)."""

    op: object  # a Map (pure pointwise; the kernel root)
    schedule: MapSchedule


@dataclass(frozen=True)
class MonoidKernel:
    """A reduction kernel: a ``Monoid`` op (or a projection ``Map`` *over* one) + its
    :class:`MonoidSchedule` (whose ``workers`` field carries any warp split)."""

    op: object  # a Monoid, or a Map(source=Monoid) projection
    schedule: MonoidSchedule


@dataclass(frozen=True)
class SemiringKernel:
    """A contraction kernel: a ``Semiring`` op (or a projection ``Map`` *over* one) + its
    :class:`SemiringSchedule` (whose ``workers`` field carries any warp split)."""

    op: object  # a Semiring, or a Map(source=Semiring) projection
    schedule: SemiringSchedule


#: A scheduled kernel — keyed by the op kind (no ``classify_algebra`` tag). The pairing
#: makes a kind/schedule mismatch unrepresentable.
Kernel = MapKernel | MonoidKernel | SemiringKernel


def kernel_for(node, place: Placement) -> Kernel:
    """Wrap a lifted op-tree ``node`` + its :class:`Placement` in the matching ``*Kernel``,
    keyed by the (peeled) op kind — a bare reduction or a projection ``Map`` over one is a
    ``Monoid`` / ``Semiring`` kernel; anything else is a ``MapKernel``. The peel itself is
    ``node.reduce_node`` (see :mod:`deplodock.compiler.ir.stmt.algebra`)."""
    from deplodock.compiler.ir.stmt.algebra import Monoid, Semiring  # noqa: PLC0415

    inner = node.reduce_node
    if isinstance(inner, Monoid):
        return MonoidKernel(op=node, schedule=MonoidSchedule(place=place))
    if isinstance(inner, Semiring):
        return SemiringKernel(op=node, schedule=SemiringSchedule(place=place))
    return MapKernel(op=node, schedule=MapSchedule(place=place))


__all__ = [
    "Fold",
    "Kernel",
    "Level",
    "MapKernel",
    "MapSchedule",
    "MonoidKernel",
    "MonoidSchedule",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "RoleAlloc",
    "SemiringKernel",
    "SemiringSchedule",
    "Stage",
    "TilePlan",
    "WarpSpec",
    "WarpTile",
    "kernel_for",
]
