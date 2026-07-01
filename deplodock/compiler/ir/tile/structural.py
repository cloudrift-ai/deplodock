"""Structural tile-IR nodes — the scheduled map/reduce skeleton, lifted to ``ir/tile/``.

These are the **post-schedule** structural nodes a ``TileOp`` is built from: each holds its own
scheduling parameters (the contraction's :class:`~.schedule.TilePlan`) plus its algebra params, with
the derived geometry exposed as ``@property`` (so ``structural_key`` digests only the compact param
fields and the ``--ir`` dumps stay readable). The kernel materializer reads the schedule straight off
the node — it never re-recognizes structure the tile IR already holds.

:class:`Contraction` is the first of these (moved here from ``ir/kernel/ir.py`` — it was always a
tile-level scheduling node, not a hardware primitive). The expander (``lowering/kernel/_factor``)
synthesizes the actual ``mma.sync`` / scalar register-tile loop per atom; the node stores only the
operands + the projection epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.expr import Expr, Literal
from deplodock.compiler.ir.schedule import ReducePlan, TilePlan
from deplodock.compiler.ir.stmt import INDENT, Accum, Body, Carrier, Load, Loop, RenderCtx, Stmt, pretty_body

if TYPE_CHECKING:
    from deplodock.compiler.ir.atom import Atom


def _ext_expr(axis: Axis) -> Expr:
    """The axis extent as an ``Expr`` — a literal int (static) or the symbolic ``Dim`` expr."""
    return Literal(axis.extent.as_static(), "int") if axis.extent.is_static else axis.extent.expr


def _overhangs(axis: Axis, tile: int) -> bool:
    """True iff a ``tile``-wide CTA block overhangs ``axis`` (symbolic or non-divisible extent) —
    so its tail must be masked."""
    if tile <= 1:
        return False
    return not (axis.extent.is_static and axis.extent.as_static() % tile == 0)


@dataclass(frozen=True)
class Reduction:
    """A scheduled ``PLANAR`` / ``TWISTED`` reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/stmt/algebra``). It splits the reduce's **algebra** (the loop-carried
    :class:`~deplodock.compiler.ir.stmt.algebra.Carrier` — degenerate ``id`` for a plain
    ``sum`` / ``max`` / ``mean``, twisted ``exp`` for online-softmax / flash) from its **structure**
    (the reduce ``axis`` + the per-element ``partial`` it folds). The fold ``Loop`` is **synthesized on
    demand** (:attr:`loop`), never stored — so the same node tiles under any
    :class:`~deplodock.compiler.ir.schedule.ReducePlan` (the reduce partition rides the node's
    ``reduce`` field, read via ``ops.reduce_plan``).

    It holds **no projection**: a bare reduce (``sum`` / ``max``) is the kernel root (its grid ``Write``
    is glue); a reduce with a post-fold sweep (softmax / RMSNorm) is the ``source`` of a wrapping
    :class:`~deplodock.compiler.ir.tile.structural.Map` whose body IS that projection. It is NOT a ``Stmt``
    — like ``Map`` it is an op-tree node a :class:`~deplodock.compiler.ir.tile.ir.TileOp` holds;
    :func:`ops.lower` flattens it to the synthesized loop (``[loop]``), so ``op_cache_key`` and the
    ``_reduce`` expander stay byte-identical to the bare-loop form.

    The **scheduling param** is the ``reduce`` partition (:class:`ReducePlan` — GRID split / BLOCK coop
    / REG ILP), stamped onto the node by ``020_schedule`` (its decided value lives **here** on the node
    — read via ``ops.reduce_plan``). ``lower`` ignores it (it's metadata the materializer / ``030_split``
    read), so it leaves ``op_cache_key`` byte-identical."""

    carrier: Carrier  # the loop-carried ⊕ algebra (degenerate id / twisted exp)
    axis: Axis  # the reduce axis
    partial: Body = field(default_factory=Body)  # the per-cell fold body (the reduce Loop's body)
    role: AxisRole = AxisRole.PLANAR  # PLANAR (plain) or TWISTED (online-softmax / flash)
    unroll: bool = False
    reduce: ReducePlan = field(default_factory=ReducePlan)  # the reduce partition (schedule slice), stamped by 020_schedule
    # An OPTIONAL nested structural node the per-element ``partial`` folds over — the ``Reduction ⊃
    # Contraction`` composition. Flash is ``Reduction(role=TWISTED, source=Contraction(QK))``: the
    # streaming KV reduce whose per-step partial is a nested ``Σ Q·K`` contraction; a split-K matmul is
    # ``Reduction(source=Contraction, reduce=g<n>)``. ``None`` for a bare reduce (``sum`` / ``max`` /
    # softmax's PLANAR row reduce), whose ``partial`` is plain loop-IR stmts. :attr:`loop` splices the
    # source's lowered loop nest ahead of the ``partial`` inside the synthesized reduce ``Loop``.
    source: Reduction | Contraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.partial, Body):
            object.__setattr__(self, "partial", Body.coerce(self.partial))

    @classmethod
    def from_loop(cls, loop: Loop) -> Reduction:
        """Build a :class:`Reduction` from an already-annotated reduce ``Loop`` (its ``carrier`` /
        ``role`` / ``axis`` / body) — the recognize-side constructor. :attr:`loop` reconstructs the
        exact same ``Loop`` (any post-fold projection rides the wrapping ``Map``, not here)."""
        return cls(carrier=loop.carrier, axis=loop.axis, partial=loop.body, role=loop.role, unroll=loop.unroll)

    @property
    def loop(self) -> Loop:
        """The synthesized annotated reduce ``Loop`` — reconstructed from the params. With no
        :attr:`source` it is byte-identical to the loop :meth:`from_loop` captured; a ``source``
        (the ``Reduction ⊃ Contraction`` composition) splices the source's lowered loop nest ahead
        of the ``partial`` inside the loop body (so flash's kv loop holds the nested ``Σ Q·K``
        contraction loop, exactly the loop-in-body form the scalar tier expands)."""
        body = self.partial if self.source is None else Body((*self.source.lower(), *self.partial))
        return Loop(axis=self.axis, body=body, unroll=self.unroll, role=self.role, carrier=self.carrier)

    @property
    def out(self) -> str:
        """The bound output name — the carrier state's primary component (a bare reduce's grid
        ``Write`` is glue; a projected reduce's output name lives on the wrapping ``Map``)."""
        return self.carrier.out

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — just the synthesized reduce
        ``Loop`` (a wrapping ``Map`` appends its projection)."""
        return [self.loop]


@dataclass(frozen=True)
class Contraction(Stmt):
    """A contraction **before** atom factorization — built **recognize-side** at fork-emit
    (``_schedule._contraction_node`` resolves the operand→role binding via ``_atomize.semiring_binding``
    and stamps the resolved ``tile``), then expanded in ``010_materialize`` (``_factor.factorize``).
    :func:`ops.lower` / ``ops.reduce_loop`` flatten it back to the synthesized mul-add ``CONTRACTION``
    loop nest (:attr:`loop`), the same generation ``_factor._synth_reduce`` register-tiles. **ONE
    flat node** that cleanly splits the **algebra params** (what to contract) from the **schedule**
    (how to tile it): the params are the tiled output ``axes`` ``(m, n)``, the contraction ``k_axis``,
    the leading batch ``lead_axes``, the structured A/B operand ``Load``\\ s, the fold accumulator
    ``acc``, and the projection ``epilogue`` (which carries the output ``Write``); the schedule is the
    one ``tile`` field — a resolved :class:`~deplodock.compiler.ir.schedule.TilePlan` carrying the
    leaf ``atom`` (tensor-core :class:`AtomKind` or the ``1×1`` :class:`ScalarAtom`), the per-CTA
    **UNIT** grid + per-unit **REGISTER** sub-tile, and the K-chunk. Keeping the schedule a single
    swappable field is what lets the same operand/acc params be tiled by a different ``TilePlan`` (the
    flash inner QK/PV reuse).

    The contraction itself is **never stored** — both tiers *synthesize* it from the operands:
    ``_factor.reduce_codegen`` lowers the mma atom into ``ldmatrix`` + ``mma.sync`` and the scalar atom into a
    ``for k: acc += a*b`` register-tiled loop — then run the ``epilogue`` (``acc`` is the SSA name the
    synthesized reduce produces and the epilogue consumes). The operand buffers ride
    :meth:`external_reads`; the epilogue is the only nested ``Body``. ``_factor.factorize`` reads the
    **derived** geometry below (``tile_m`` / ``mask_m`` / ``m_b`` / ``m_uvar`` / ``block_threads`` /
    ``b_trans`` / …) straight off the node; it's ``@property``, so it stays out of the fields
    ``structural_key`` digests (the node IS keyed as an intermediate ``KernelOp``). The atom selects
    the codegen — there is no separate ``Leaf`` / per-atom subclass."""

    axes: tuple[Axis, Axis]  # the tiled output (m_axis, n_axis) — params
    k_axis: Axis  # the contraction axis — params
    a_load: Load  # params
    b_load: Load  # params
    acc: str  # params
    tile: TilePlan  # the schedule: leaf atom + unit/register widths + K-chunk (the only schedule field)
    lead_axes: tuple[Axis, ...] = ()  # params
    epilogue: Body = field(default_factory=Body)  # params

    def __post_init__(self) -> None:
        if not isinstance(self.epilogue, Body):
            object.__setattr__(self, "epilogue", Body(self.epilogue))

    @property
    def out(self) -> str:
        """The bound output name — the fold accumulator (a bare contraction's grid ``Write`` stores
        ``acc`` at the cell; a fused-epilogue contraction carries its own ``Write`` in ``epilogue``)."""
        return self.acc

    @property
    def loop(self) -> Loop:
        """The synthesized ``CONTRACTION`` reduce ``Loop`` — the canonical ``for k: v = a*b; acc += v``
        mul-add form (built by the shared ``ops.contraction_loop``, the same generation
        ``_factor._synth_reduce`` register-tiles). Lets :func:`ops.lower` / ``ops.reduce_loop`` flatten
        the node back to the loop nest; the node never stores the loop."""
        from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ops import contraction_loop  # noqa: PLC0415 — avoid an import cycle

        return contraction_loop(
            lift=ElementwiseImpl("multiply"),
            fold=Accum(name=self.acc, value=f"{self.acc}__v", op=ElementwiseImpl("add"), axes=(self.k_axis.name,)),
            operand_bodies=([self.b_load], [self.a_load]),  # B[k, n], A[m, k] — keep B-then-A load reuse
            reduce_axis=self.k_axis,
        )

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body — the synthesized reduce ``Loop`` followed by the projection
        ``epilogue`` (a fused-epilogue contraction's own stmts, empty for a bare one). The materializer
        expands the node through ``_factor.factorize`` instead; this is the structural-key / dump path."""
        return [self.loop, *self.epilogue]

    # ---- params: the (m, n) output axes unpacked ---------- #
    @property
    def m_axis(self) -> Axis:
        return self.axes[0]

    @property
    def n_axis(self) -> Axis:
        return self.axes[1]

    # ---- schedule: the leaf atom + unit/register widths, read off the ``tile`` (m-then-n,
    # normalized by ``TilePlan``'s accessors) ---------- #
    @property
    def atom(self) -> Atom:
        return self.tile.atom

    @property
    def units_m(self) -> int:
        return self.tile.units_m

    @property
    def units_n(self) -> int:
        return self.tile.units_n

    @property
    def reg_m(self) -> int:
        return self.tile.reg_m

    @property
    def reg_n(self) -> int:
        return self.tile.reg_n

    # ---- derived geometry (read by ``_factor.factorize``) — the per-CTA tile dims come straight off
    # the ``tile``; the masks / extents combine the ``tile`` widths with the output axes (params) ---- #
    @property
    def tile_m(self) -> int:
        return self.tile.tile_m

    @property
    def tile_n(self) -> int:
        return self.tile.tile_n

    @property
    def mask_m(self) -> bool:
        return _overhangs(self.m_axis, self.tile_m)

    @property
    def mask_n(self) -> bool:
        return _overhangs(self.n_axis, self.tile_n)

    @property
    def m_ext(self) -> Expr:
        return _ext_expr(self.m_axis)

    @property
    def n_ext(self) -> Expr:
        return _ext_expr(self.n_axis)

    @property
    def block_threads(self) -> int | None:
        bt = self.tile.block_threads
        return bt if bt > 1 else None  # None ⇒ the scalar default block size

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical B[k, n] — read off the
        binding load, the same test ``_atomize`` made when it bound the operand."""
        return self.k_axis.name in self.b_load.index[-1].free_vars()

    # The bound GRID-block / UNIT axis names — the original m/n names live in the operand indices,
    # so the bound axes take a fresh ``_b`` (block) / ``_u`` (unit) suffix.
    @property
    def m_b(self) -> str:
        return self.m_axis.name + "_b"

    @property
    def n_b(self) -> str:
        return self.n_axis.name + "_b"

    @property
    def m_uvar(self) -> str:
        return self.m_axis.name + "_u"

    @property
    def n_uvar(self) -> str:
        return self.n_axis.name + "_u"

    # ---- the kernel-stmt protocol (the epilogue is the only nested Body; the operand buffers are
    # external reads; the synthesized reduce produces ``acc``) ---------- #
    def nested(self) -> tuple[Body, ...]:
        return (self.epilogue,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (epilogue,) = bodies
        return replace(self, epilogue=epilogue)

    def defines(self) -> tuple[str, ...]:
        return (self.acc,)

    def external_reads(self) -> tuple[str, ...]:
        return (self.a_load.input, self.b_load.input)

    def pretty(self, indent: str = "") -> list[str]:
        t = " trans" if self.b_trans else ""
        ops = f"{self.a_load.input} @ {self.b_load.input}{t} -> {self.acc} ({self.atom.name})"
        return [f"{indent}Contraction [{self.m_axis.name}, {self.n_axis.name}] {ops}", *pretty_body(self.epilogue, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Contraction must be expanded by 010_materialize before render")


# ``Body.structural_key()`` dispatches :func:`deplodock.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register ``Contraction``'s handler here, with the node.
from deplodock.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Contraction, rename, sigma, axis_fn):
    # Route the operand Loads + accumulator + epilogue through the generic rewrite (SSA / Expr / axis
    # canonicalization); map the skeleton axes; pass the ``tile`` schedule through unchanged.
    # ``b_trans`` is derived from ``b_load`` (a property), so the rewritten load carries it.
    return Contraction(
        axes=tuple(axis_fn(a) for a in s.axes),
        k_axis=axis_fn(s.k_axis),
        a_load=_rewrite(s.a_load, rename, sigma, axis_fn),
        b_load=_rewrite(s.b_load, rename, sigma, axis_fn),
        acc=rename(s.acc),
        tile=s.tile,
        lead_axes=tuple(axis_fn(a) for a in s.lead_axes),
        epilogue=Body(tuple(_rewrite(c, rename, sigma, axis_fn) for c in s.epilogue)),
    )


@dataclass(frozen=True)
class Map:
    """A pointwise lift / projection wrapper around a :class:`Body` of loop-IR stmts, optionally over
    a reduction / contraction ``source``.

    ``body`` is the per-cell pointwise / projection compute: operand ``Load``\\ s, the lift
    ``Assign``\\ s, and — at the kernel root — the output ``Write``. ``source`` is the structural node
    it projects over (a :class:`Reduction` / :class:`Contraction` — ``project ∘ reduce``) or ``None``
    for a pure pointwise map. A pure pointwise cell is a ``Map`` of plain stmts (``source=None``);
    softmax / RMSNorm is a ``Map`` whose ``body`` is the post-fold sweep over a ``Reduction`` source.
    (A not-yet-migrated reduce / contraction may still sit *inside* ``body`` as an annotated ``Loop``
    — that legacy form lowers the same way via :func:`ops.lower`.) ``out`` is the bound output name
    (the body's last def, or the source's carried state for an empty-body wrap). It HAS a Body, not IS
    one."""

    body: Body = field(default_factory=Body)
    source: Reduction | Contraction | None = None  # the project∘reduce source, or None (pure pointwise)

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))

    @property
    def out(self) -> str:
        """The bound output name. With no projection body it is the ``source``'s carried state; when
        the body's last stmt is an annotated reduce ``Loop`` (a legacy bare reduction whose grid-cell
        ``Write`` is glue), the carried state's primary component (``loop.carrier.out``); otherwise the
        last defining stmt's name (a pointwise lift / a post-reduce projection)."""
        if len(self.body) == 0 and self.source is not None:
            return self.source.out
        last = self.body[-1]
        carrier = getattr(last, "carrier", None)
        if carrier is not None:
            return carrier.out
        return last.defines()[-1]


__all__ = ["Contraction", "Map", "Reduction"]
