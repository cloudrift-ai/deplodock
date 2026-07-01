"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine* — the :class:`Map` / :class:`Reduction` / :class:`Contraction`
nodes defined **in this module**, alongside ``ir/stmt/algebra``) directly,
plus a thin set of **root-global schedule fields** — the free-axis → grid
:class:`~.schedule.Placement` (``place``) and the warp split (``workers``). The
per-node schedule slices ride the structural nodes themselves (a
:class:`Contraction`'s ``tile``, a :class:`Reduction`'s
``reduce``); the residual root fields (``reduce`` / ``tier`` / ``stage``)
hold the schedule for the not-yet-nodified forms (a non-tiled
contraction's split-K, the pin-only ``STAGE`` / ``WSPEC``; flash is now a
``Map(source=Reduction(source=Contraction))`` node tree, so its partition rides
the node). There is no per-kind kernel/schedule type: the algebra is read
structurally off the axes' :class:`~deplodock.compiler.ir.axis.AxisRole`
(``ops.axis_role``), so MAP / MONOID / SEMIRING all ride the same ``TileOp``.

The combine lives entirely in the ``op`` wrapper (the :class:`Map` /
:class:`Reduction` / :class:`Contraction` nodes here + ``ir/stmt/algebra``): a
node whose per-cell loop nest carries the role (``AxisRole``) + the decoupled
``Carrier`` (the ⊕ algebra). The algebra is **not stored as a node kind**; the
role/carrier are read off the node / annotated loop where a pass needs them
(``ops.axis_role`` / ``ops.reduce_loop``). ``lower(op)`` flattens the structural
tree back to the loop nest.

The structural nodes are the **post-schedule** skeleton a ``TileOp`` is built from:
each holds its own scheduling parameters (the contraction's :class:`~.schedule.TilePlan`)
plus its algebra params, with the derived geometry exposed as ``@property`` (so
``structural_key`` digests only the compact param fields and the ``--ir`` dumps stay
readable). The kernel materializer reads the schedule straight off the node — it never
re-recognizes structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr, Literal
from deplodock.compiler.ir.schedule import Placement, ReducePlan, Stage, TilePlan, WarpSpec
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


def _flatten_nodes(body: Body) -> tuple[Stmt, ...]:
    """Flatten any nested :class:`Contraction` that sits as a stmt in ``body`` to its own lowered
    loop nest (``Contraction`` is a ``Stmt``, so it can ride inside a reduce ``partial`` — the flash
    PV ``Σ_j P·V``); plain stmts pass through. One recursion rule for a reduce whose per-step partial
    composes a contraction, mirroring the ``Reduction.source`` splice."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Contraction):
            out.extend(s.lower())
        else:
            out.append(s)
    return tuple(out)


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
    :class:`Map` whose body IS that projection. It is NOT a ``Stmt``
    — like ``Map`` it is an op-tree node a :class:`TileOp` holds;
    :func:`ops.lower` flattens it to the synthesized loop (``[loop]``), so ``op_cache_key`` and the
    ``_factor._factorize_reduce`` expander stay byte-identical to the bare-loop form.

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
        :attr:`source` and a plain ``partial`` it is byte-identical to the loop :meth:`from_loop`
        captured; a ``source`` (the ``Reduction ⊃ Contraction`` composition) splices the source's
        lowered loop nest ahead of the ``partial`` inside the loop body (so flash's kv loop holds the
        nested ``Σ Q·K`` contraction loop, exactly the loop-in-body form the scalar tier expands). A
        **nested structural node inside the ``partial``** — the flash PV ``Contraction`` (``Σ_j P·V``)
        that folds the block into the carrier — is flattened to its own loop nest in place, the same
        recursion the ``source`` splice does: one structural rule for a reduce whose per-step partial
        composes a contraction."""
        prefix = self.source.lower() if self.source is not None else ()
        body = Body((*prefix, *_flatten_nodes(self.partial)))
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
class Side:
    """One tiled output axis of a contraction — the outer ``m`` or inner ``n`` — paired with its
    derived per-CTA tile geometry. The two ride as a ``(m, n)`` pair (:attr:`Contraction.mn`)
    mirroring the schedule's ``(m, n)`` tuples (``TilePlan.units`` / ``regs``), so the factorizer
    threads one object per axis instead of a dozen loose ``*_m`` / ``*_n`` args. The tile width,
    unit / register counts, and bound block/unit var names are stamped by :meth:`Contraction._side`;
    the ``mask`` / ``ext`` are derived from the axis + width."""

    axis: Axis  # the output axis (a param)
    tile: int  # the per-CTA width = units · reg · atom_dim
    units: int  # parallel units on this axis (warps for mma / threads for scalar)
    reg: int  # register sub-cells per unit on this axis
    block: str  # the bound grid-block var (the axis name + ``_b``)
    unit: str  # the bound unit var (the axis name + ``_u``)

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def mask(self) -> bool:
        """True iff a ``tile``-wide CTA block overhangs the axis (its tail must be masked)."""
        return _overhangs(self.axis, self.tile)

    @property
    def ext(self) -> Expr:
        """The axis extent as an ``Expr`` — a static literal or the symbolic ``Dim`` expr."""
        return _ext_expr(self.axis)


@dataclass(frozen=True)
class Contraction(Stmt):
    """A contraction **before** atom factorization — built **recognize-side** at fork-emit
    (``_schedule._contraction_node`` resolves the operand→role binding via ``_atomize.semiring_binding``
    and stamps the resolved ``tile``), then expanded in ``010_materialize`` (``_factor.factorize``).
    :func:`ops.lower` / ``ops.reduce_loop`` flatten it back to the synthesized mul-add ``CONTRACTION``
    loop nest (:attr:`loop`), the same ``for k: v = a·b; acc += v`` form ``_factor._ScalarOps.reduce``
    register-tiles through the shared ``_contract_kloop`` skeleton. **ONE
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
    **derived** geometry below (the ``(m, n)`` :class:`Side` pair — ``tile`` / ``mask`` / ``block`` /
    ``unit`` per axis — plus ``block_threads`` / ``b_trans``) straight off the node; it's ``@property``, so it stays out of the fields
    ``structural_key`` digests (the node IS keyed as an intermediate ``KernelOp``). The atom selects
    the codegen — there is no separate ``Leaf`` / per-atom subclass."""

    axes: tuple[Axis, Axis]  # the tiled output (m_axis, n_axis) — params
    k_axis: Axis  # the contraction axis — params
    a_operand: Load | Body  # A: a gmem ``Load`` OR a computed register-resident ``Body`` (flash PV's ``P = exp(S − M)``) — params
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
    def a_body(self) -> tuple[Stmt, ...]:
        """The A operand's producing stmts — a singleton gmem ``Load``, or a computed body's stmts
        (a **register-resident** A: flash PV's ``P = exp(S − M)``, produced from an in-register score,
        not a gmem address). The last stmt's def is the operand value ``contraction_loop`` multiplies."""
        return (self.a_operand,) if isinstance(self.a_operand, Load) else tuple(self.a_operand)

    @property
    def a_computed(self) -> bool:
        """True when A is a computed register-resident operand (a ``Body``), not a gmem ``Load`` — the
        mma tier reads it as a fragment, the scalar tier as the value."""
        return not isinstance(self.a_operand, Load)

    @property
    def a_name(self) -> str:
        """The A operand's bound SSA name (its producing body's last def)."""
        return self.a_body[-1].defines()[-1]

    @property
    def loop(self) -> Loop:
        """The synthesized ``CONTRACTION`` reduce ``Loop`` — the canonical ``for k: v = a*b; acc += v``
        mul-add form (built by the shared ``ops.contraction_loop``, the same fold ``_factor``'s scalar
        contraction tier register-tiles). Lets :func:`ops.lower` / ``ops.reduce_loop`` flatten the node
        back to the loop nest; the node never stores the loop."""
        from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ops import contraction_loop  # noqa: PLC0415 — avoid an import cycle

        return contraction_loop(
            lift=ElementwiseImpl("multiply"),
            fold=Accum(name=self.acc, value=f"{self.acc}__v", op=ElementwiseImpl("add"), axes=(self.k_axis.name,)),
            operand_bodies=([self.b_load], self.a_body),  # B[k, n], A[m, k] (or A's computed body) — keep B-then-A load reuse
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

    @property
    def atom(self) -> Atom:
        return self.tile.atom

    # ---- the (m, n) output sides: each axis paired with its derived per-CTA tile geometry (width /
    # unit / register counts, read off the ``tile``) + the bound block/unit var names (the original
    # m/n names live in the operand indices, so the bound axes take a fresh ``_b`` / ``_u`` suffix).
    # ``_factor`` threads these as the ``(m, n)`` pair, mirroring the schedule's tuples. ---------- #
    @property
    def m(self) -> Side:
        return self._side(0)

    @property
    def n(self) -> Side:
        return self._side(1)

    @property
    def mn(self) -> tuple[Side, Side]:
        """The ``(m, n)`` output sides — the per-axis geometry the factorizer tiles through."""
        return (self.m, self.n)

    def _side(self, i: int) -> Side:
        t, ax = self.tile, self.axes[i]
        tile, units, reg = (t.tile_m, t.units_m, t.reg_m) if i == 0 else (t.tile_n, t.units_n, t.reg_n)
        return Side(axis=ax, tile=tile, units=units, reg=reg, block=ax.name + "_b", unit=ax.name + "_u")

    @property
    def block_threads(self) -> int | None:
        bt = self.tile.block_threads
        return bt if bt > 1 else None  # None ⇒ the scalar default block size

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical B[k, n] — read off the
        binding load, the same test ``_atomize`` made when it bound the operand."""
        return self.k_axis.name in self.b_load.index[-1].free_vars()

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
        a_inputs = tuple(s.input for s in self.a_body if isinstance(s, Load))
        return (*a_inputs, self.b_load.input)

    def pretty(self, indent: str = "") -> list[str]:
        t = " trans" if self.b_trans else ""
        a_src = self.a_operand.input if isinstance(self.a_operand, Load) else self.a_name
        ops = f"{a_src} @ {self.b_load.input}{t} -> {self.acc} ({self.atom.name})"
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
    a_operand = (
        _rewrite(s.a_operand, rename, sigma, axis_fn)
        if isinstance(s.a_operand, Load)
        else Body(tuple(_rewrite(c, rename, sigma, axis_fn) for c in s.a_operand))
    )
    return Contraction(
        axes=tuple(axis_fn(a) for a in s.axes),
        k_axis=axis_fn(s.k_axis),
        a_operand=a_operand,
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
    A **contraction** rides its annotated ``CONTRACTION`` ``Loop`` in ``body`` (recognition emits the
    flat ``Map``): a *tiled* / warp / split-K contraction is nodified to a :class:`Contraction` by
    ``_schedule`` before materialize, but a *scalar per-cell* contraction keeps the flat ``Map`` all the
    way down — both lower identically via :func:`ops.lower`. ``out`` is the bound output name (the
    body's last def, the reduce loop's ``carrier.out`` for that per-cell contraction, or the source's
    carried state for an empty-body wrap). It HAS a Body, not IS one."""

    body: Body = field(default_factory=Body)
    source: Reduction | Contraction | None = None  # the project∘reduce source, or None (pure pointwise)

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))

    @property
    def out(self) -> str:
        """The bound output name. With no projection body it is the ``source``'s carried state; when
        the body's last stmt is an annotated reduce ``Loop`` — a **scalar per-cell contraction** that
        rides the flat ``Map`` through materialize (only tiled / warp / split-K contractions nodify to
        a :class:`Contraction`), its grid-cell ``Write`` synthesized as store glue — the carried
        state's primary component (``loop.carrier.out``); otherwise the last defining stmt's name (a
        pointwise lift / a post-reduce projection)."""
        if len(self.body) == 0 and self.source is not None:
            return self.source.out
        last = self.body[-1]
        carrier = getattr(last, "carrier", None)
        if carrier is not None:
            return carrier.out
        return last.defines()[-1]


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Map` /
    :class:`Reduction` / :class:`Contraction`, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``lower(op)``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.
    - ``reduce`` — the reduce-axis partition (:class:`~.schedule.ReducePlan`) for a not-yet-nodified
      reduce (a non-tiled contraction's split-K); a ``Reduction`` node (a plain reduce, softmax, **or
      flash** — now a ``Map(source=Reduction)``) carries its own partition (read via
      ``ops.reduce_plan``, which falls back here).
    - ``tier`` — the output fragment (:class:`~.schedule.TilePlan`) for a non-tiled / split-partial
      contraction; a tiled contraction rides its ``tile`` on the ``Contraction`` node. ``None`` = per-cell.
    - ``stage`` — the operand smem pipeline (:class:`~.schedule.Stage`); ``None`` = gmem-direct (pin-only).

    The contraction operand→role binding is not a ``TileOp`` field — a tiled contraction carries its
    A/B operands / accumulator / epilogue on its ``Contraction`` node (``op``), the single source of
    truth; ``_schedule._contraction_node`` resolves them via ``_atomize.semiring_binding``."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    reduce: ReducePlan = field(default_factory=ReducePlan)
    tier: TilePlan | None = None
    stage: Stage | None = None
    workers: WarpSpec | None = None

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from deplodock.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        return "\n".join(pretty(self.op, "    "))


__all__ = ["Contraction", "Map", "Reduction", "TileOp"]
