"""The algebraic vocabulary — the lift, the carrier, and the contraction view.

The algebra of a kernel is **in the body**: the lift is a :class:`Map` (pointwise
``⊗``), the fold ``⊕`` is the carrier (an ``Accum`` scalar fold, or a :class:`Monoid`
+ :class:`Twist`), and a contraction (matmul) is the structural ``reduce(⊕) ∘ map(⊗)``
recognized on demand as a :class:`Semiring`. There is no stored algebra tag to keep in
sync — passes read the structure directly. These primitives are consolidated here
(rather than scattered across the leaf vocabulary and a top-level ``ir/algebra``) so
the algebra lives in one place:

- :class:`Map` — the pointwise lift, a typed :class:`Body` of stmts.
- :class:`Twist` / :class:`Monoid` — the loop-carried associative combine (the fold ⊕).
- :class:`Semiring` — the structural contraction (the matmul) as a node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, render_merge_program
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.carrier import (
    Channel,
    exp_combine_states,
    exp_merge,
    id_accums,
    id_combine_states,
    id_merge,
)
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load


@dataclass(frozen=True)
class Map:
    """A pointwise lift over an optional nested :data:`AlgebraNode` ``source``.

    - ``source`` — the nested computation this Map maps over (a ``Monoid`` / ``Semiring`` /
      ``Map``), lowered BEFORE the body, whose stmts read its result; ``None`` = a pure
      pointwise lift. ``Map(source=reduce, body=…)`` is ``project ∘ reduce`` — flash's
      ``O/l`` over the ``(m, l, O)`` ``Monoid``, a fused epilogue over a ``Semiring``.
    - ``body`` — the pointwise :class:`Body` of loop-IR stmts (operand ``Load``\\ s, the
      lift ``Assign``\\ s, an optional masking ``Select``, and — at the kernel root — the
      output ``Write``) that binds a value name as its last defining stmt.

    Used as a carrier partial / contraction operand (``out`` = the body's last bound name)
    or as the kernel root (last stmt = the ``Write``). It HAS a Body (composition), not
    IS one."""

    source: AlgebraNode | None = None
    body: Body = field(default_factory=Body)

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))

    @property
    def out(self) -> str:
        """The bound output name — the body's last defining stmt's name (the projected
        value / the partial this Map supplies)."""
        return self.body[-1].defines()[-1]

    @property
    def reduce_node(self) -> Monoid | Semiring | None:
        """The nested reduction this Map projects over (``project ∘ reduce``) — its
        ``source`` when that is a ``Monoid`` / ``Semiring``, else ``None`` (a pure or flat
        pointwise Map reduces over nothing at this level)."""
        return self.source if isinstance(self.source, (Monoid, Semiring)) else None


@dataclass(frozen=True)
class Semiring:
    """A semiring contraction ``reduce(⊕) ∘ map(⊗)`` — the structural matmul — as a
    **first-class** algebra node. Its ``operands`` are themselves algebra nodes
    (:class:`Map` / :class:`Monoid` / :class:`Semiring`), so a contraction reads a
    buffer (a ``Map`` of one ``Load``) or another reduction's result. Lowered by
    ``ir.tile.ops.lower`` to the recognizable ``Accum``-in-``Loop`` form the matmul
    tier reads; recognized from that form on demand by :meth:`match`.

    - ``lift`` — the ⊗ product op; distributes over ⊕ and has a multiplicative identity.
    - ``fold`` — the ⊕ additive carrier (an ``Accum``: identity 0, assoc + comm).
    - ``operands`` — the contracted inputs, each an :data:`AlgebraNode` (≥ 2).
    - ``reduce_axis`` — the contracted (K) :class:`Axis` (extent-bearing — it sizes the
      lowered ``Loop``; the fold ``Accum``'s ``axes`` is names-only, a different layer).
    """

    lift: ElementwiseImpl
    fold: Accum
    operands: tuple[AlgebraNode, ...]
    reduce_axis: Axis

    @property
    def out(self) -> str:
        """The bound result name — the contraction's accumulator (``fold.name``). Derived,
        not stored: we always know what a semiring accumulates (and a future mma-fragment
        output, which is not a single SSA name, won't fit a stored ``str``)."""
        return self.fold.name

    @property
    def is_additive(self) -> bool:
        """The ``(×, +)`` semiring the tensor-core mma implements — the gate for the
        mma atom (a tropical / min-plus contraction is still a semiring, still tiles,
        but has no hardware atom)."""
        return self.lift.name == "multiply" and self.fold.op.reduce_canon == "add"

    @property
    def reduce_node(self) -> Semiring:
        """This contraction IS the reduction (identity for the projection-peeling query)."""
        return self

    @staticmethod
    def match(loop) -> Semiring | None:
        """Recognize a semiring contraction on a loop-IR ``loop``, or ``None``.
        Duck-typed on ``.is_reduce`` / ``.axis`` / ``.body`` so it serves a Loop-IR
        ``Loop`` / ``StridedLoop`` alike (the caller restricts the type).

        A reduce loop is a contraction iff its single ``Accum`` fold's partial is
        produced by a lift that **distributes over** the fold op, contracting ≥ 2
        distinct operand buffers over the reduce axis. The operands are reconstructed
        as one-``Load`` :class:`Map` nodes (the buffer read)."""
        if not getattr(loop, "is_reduce", False):
            return None
        body = loop.body
        accs = [s for s in body if isinstance(s, Accum)]
        if len(accs) != 1:
            return None
        fold = accs[0]
        lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
        if lift is None or not lift.op.distributes_over(fold.op):
            return None
        k = loop.axis.name
        loads = [ld for ld in body if isinstance(ld, Load) and k in {v for e in ld.index for v in e.free_vars()}]
        if len({ld.input for ld in loads}) < 2:
            return None
        return Semiring(lift=lift.op, fold=fold, operands=tuple(Map(body=[ld]) for ld in loads), reduce_axis=loop.axis)

    def as_monoid(self) -> Monoid:
        """This contraction AS the degenerate ``Monoid`` it is — the carrier-algebra fact that a
        **SEMIRING is a MONOID with a ⊗ lift**. State = the additive ``fold`` accumulator; the single
        ``partial`` source is a ``Map`` that loads the operands and computes their ``lift`` ⊗ product;
        an ``id``-family ``Twist`` whose channel is the ``fold`` ⊕. Lowering it reproduces
        :func:`~deplodock.compiler.ir.tile.ops._lower_semiring` exactly, so a contraction flows through
        the **same** carrier-generic reduce machinery (``Monoid.render`` / the cooperative ``_reduce``
        tier) as any monoid reduce — no contraction special case. (The mirror of
        :meth:`~deplodock.compiler.ir.stmt.leaves.Accum.as_monoid` for a scalar fold, but the ``⊗``
        product is the partial rather than a sibling ``value``.)"""
        lift = Assign(name=self.fold.value, op=self.lift, args=tuple(o.out for o in self.operands))
        product = Map(body=Body((*(s for o in self.operands for s in o.body), lift)))
        return Monoid(
            state=State(names=(self.fold.name,)),
            partial=(product,),
            twist=Twist(family="id", channels=(Channel(fold=self.fold.op, term=self.fold.value, dtype=self.fold.dtype),)),
            axis=self.reduce_axis,
        )


def _rename_assign_args(a: Assign, sub: dict[str, str]) -> Assign:
    return Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)


@dataclass(frozen=True)
class State:
    """The carried state of a :class:`Monoid` — the internal-state SSA ``names``. The state
    is the *carrier* a :class:`Twist` operates on: the twist's ``merge`` folds a partial into
    :attr:`names`, its ``combine_states`` merges this state with a second one named by
    :attr:`other`. The seed (neutral element) is NOT stored here — a carrier dissolves into
    its fold ``Accum``\\ s (:meth:`Monoid.dissolve`), and each fold's seed is its
    ``op.identity`` (so there's one source of truth for the seed: the fold, not a
    separately-authored identity that could drift)."""

    names: tuple[str, ...]

    @property
    def other(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine — ``"<n>__o"``
        per component, the right operand ``combine_states`` reads."""
        return tuple(f"{n}__o" for n in self.names)


@dataclass(frozen=True)
class Twist:
    """The ψ-conjugated combine of a :class:`Monoid` — the part that VARIES while the monoid
    algebra (carried state + neutral element) stays the same.

    Transport of structure (the article): a monoid ``(·, e)`` conjugated by a bijection ψ gives
    the twisted combine ``x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))``. The twist is carried here as DATA, in one
    of two modes:

    - **SPEC mode** (``family`` set, with ``channels``): the combine is generated on demand by
      :class:`Monoid` from ``(family, channels, state)`` — the inverse-ψ generator builds the
      naive combine and a per-family stabilizer recovers the numerically-stable form (see
      ``ir/stmt/carrier.py``). ``"exp"`` is online-softmax / flash's max-rescale LSE; ``"id"`` is
      the degenerate identity twist (a plain reduce). The stored program fields stay empty.
    - **BOUND mode** (``family is None``): the programs are stored verbatim — used by the one-shot
      ``Monoid.as_state_merge`` finalize, whose ``merge`` IS its ``combine_states``.

    A ``Monoid`` reads ``monoid.merge`` / ``.combine_states`` / ``.state_b`` (the mode-dispatching
    accessors), never these fields directly.
    """

    family: str | None = None
    channels: tuple[Channel, ...] = ()
    merge: tuple[Stmt, ...] = ()
    combine_states: tuple[Assign, ...] = ()
    state_b: tuple[str, ...] = ()


@dataclass(frozen=True)
class Carrier:
    """The carrier **algebra** of a reduce — its carried :class:`State` plus the ψ-conjugated
    :class:`Twist` combine — decoupled from any op-tree node or loop position.

    This is the pure-algebra half of the (retiring) :class:`Monoid`: it knows how to derive the
    streaming fold (:attr:`merge`), the cross-partition fold (:attr:`combine_states`), the
    degenerate ``Accum`` form (:meth:`as_accums`), and the one-shot cross-partition combine
    (:meth:`as_state_merge`) from ``(state, twist)`` — exactly as ``Monoid`` did — but it is NOT a
    ``Stmt`` and carries no ``partial`` / ``axis``. A reduce ``Loop`` carries one
    (``loop.carrier``) so the cooperative / cross-CTA materializers read the combine off the loop,
    not a node. ``Monoid`` delegates every algebra method here during the transition."""

    state: State
    twist: Twist

    @property
    def out(self) -> str:
        """The bound output name — the primary carried state component."""
        return self.state.names[0]

    @property
    def state_b(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine — derived
        (``"<n>__o"``) in spec mode, the stored value in bound mode."""
        tw = self.twist
        return self.state.other if tw.family is not None else tw.state_b

    @cached_property
    def merge(self) -> tuple[Stmt, ...]:
        """The streaming single-element fold program — generated from the channel spec in spec
        mode, the stored program in bound mode."""
        tw = self.twist
        if tw.family == "exp":
            return exp_merge(self.state.names, tw.channels, key=self.state.names[0])
        if tw.family == "id":
            return id_merge(self.state.names, tw.channels)
        return tw.merge

    @cached_property
    def combine_states(self) -> tuple[Assign, ...]:
        """The cross-partition state⊕state fold program."""
        tw = self.twist
        if tw.family == "exp":
            return exp_combine_states(self.state.names, self.state_b, key=self.state_b[0])
        if tw.family == "id":
            return id_combine_states(self.state.names, self.state_b, tw.channels)
        return tw.combine_states

    def partial_names(self) -> tuple[str, ...]:
        """The bound name of each partial the merge folds in (its external reads)."""
        return _merge_reads(self.merge, self.state.names)

    def dissolve(self) -> list[Stmt]:
        """The loose fold stmts this carrier lowers to — bare ``Accum``\\ s for a degenerate
        carrier (:meth:`as_accums`), else the streaming ``merge``."""
        accums = self.as_accums()
        return list(accums) if accums is not None else list(self.merge)

    def as_accums(self) -> list[Accum] | None:
        """If this is a **degenerate** carrier (the identity twist — each state component folded
        by its own self-op, no rescale temps), the equivalent ``Accum`` folds; else ``None``."""
        if self.twist.family == "id":  # the degenerate family IS the bare folds
            return id_accums(self.state.names, self.twist.channels)
        if self.twist.family is not None:  # a twisted family (exp) is never degenerate
            return None
        merge = self.merge
        names = set(self.state.names)
        if len(merge) != len(self.state.names):
            return None
        accums: list[Accum] = []
        for a in merge:
            if a.name not in names or len(a.args) != 2 or a.args[0] != a.name:
                return None
            accums.append(Accum(name=a.name, value=a.args[1], op=a.op, dtype=a.dtype))
        return accums

    def as_state_merge(self, other: tuple[str, ...]) -> StateMerge:
        """A one-shot :class:`StateMerge` stmt that folds this carrier's ``state`` with a second
        fully-reduced state named ``other`` (the cross-partition combine's right operand). The
        merge program IS ``combine_states`` with ``state_b`` renamed to ``other``, so the
        cooperative-tree / cross-CTA reduce renders it through the same machinery as a streaming
        step."""
        if self.twist.family == "exp":
            merged = exp_combine_states(self.state.names, other, key=other[0])
        else:
            sub = dict(zip(self.state_b, other, strict=True))
            merged = tuple(_rename_assign_args(a, sub) for a in self.combine_states)
        return StateMerge(state=self.state, merge=merged, state_b=other)


@dataclass(frozen=True)
class StateMerge(Stmt):
    """The cross-partition state⊕state combine, as a **renderable** loop-IR stmt (its right
    operand is a second fully-reduced state named :attr:`state_b`). Emitted by
    :meth:`Carrier.as_state_merge` for the REG tree / cooperative-tree / cross-CTA finalize; it
    renders the ψ-rescale state reassignment via ``render_merge_program`` (the same path
    ``Monoid.render`` took for a bound-mode state-merge). Unlike ``Accum`` it is not a fold
    carrier — it sits in a combine region, not a streaming fold loop, so it never makes its
    enclosing loop ``is_reduce``."""

    state: State
    merge: tuple[Stmt, ...]
    state_b: tuple[str, ...]

    def deps(self) -> tuple[str, ...]:
        return self.state_b

    def defines(self) -> tuple[str, ...]:
        return self.state.names

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}({', '.join(self.state.names)}) <- combine_states({', '.join(self.state_b)})"]
        for a in self.merge:
            lines += a.pretty(indent + "    ")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        return render_merge_program(self.merge, self.state.names, ctx)


@dataclass(frozen=True)
class Monoid(Stmt):
    """A loop-carried **monoid** combine — a general associative reduce over
    internal state, the tuple-valued sibling of ``Accum`` (scalar fold) and
    ``Mma`` (tensor-core fragment fold).

    A monoid is *(identity element, associative binary operation, carried state)*;
    ``Monoid`` makes all three explicit so the streaming-reduce machinery isn't
    tied to any one recurrence:

    - ``state`` — the carried :class:`State`: the internal-state SSA ``names``
      (read-and-written across the reduce axis — the carried read is implicit, like
      ``Accum.name`` / ``Mma.c`` — and the defs visible after the loop). The seed (neutral
      element) per component is its fold's ``op.identity``: the carrier :meth:`dissolve`\\ s
      into its fold ``Accum``\\ s at lowering, and a schedule realization seeds those (the
      serial reduce's ``Loop.render`` declares before the loop; cooperative / cross-CTA
      seeds each partial — same fold, different placement).
    - ``partial`` — this iteration's contribution sources, one per partial slot: each an
      :data:`AlgebraNode` (the self-contained op-tree form, e.g. flash's score is a
      ``Map``, a nested contraction a ``Semiring``) or a bound ``str`` name (the loop-IR
      carrier form). The bound name (``partial_names``) is the right operand the merge
      folds into the state. A self-contained reduction also carries ``axis`` (its ``out``
      is derived); ``ir.tile.ops.lower`` expands the nodes into the loop.
    - ``twist`` — the :class:`Twist`: the ψ-conjugated combine, **as data**, the one part that
      varies (degenerate / online-softmax / flash / a future fragment realization) while the
      algebra above stays the same. In SPEC mode it is a name-free ``(family, channels)`` from
      which the ``merge`` (streaming fold) and ``combine_states`` (cross-partition fold)
      programs are generated; read them off the mode-dispatching accessors
      ``monoid.merge`` / ``.combine_states`` / ``.state_b`` (each ``Assign`` targeting a
      ``state`` name is a state update; statement ORDER is load-bearing; the second operand of
      ``combine_states`` is named by ``state_b``, ``"<s>__o"``).
    - ``axis`` — the reduce :class:`Axis` of a self-contained op-tree carrier (``None`` on
      a loop-IR carrier, whose axis is the enclosing ``Loop``'s). A monoid is associative
      with identity by construction; commutativity is unused (split/reorder legality is a
      future cooperative-tier concern).

    The **φ projection** (``project`` in ``project ∘ reduce ∘ map``) is no longer a carrier
    field — it is a :class:`Map` *over* this Monoid: ``Map(source=monoid, body=[O/l])``
    lowers the reduce then the post-loop pointwise project. Flash's ``O_i / l_i`` is that
    Map; a plain reduce / matmul needs none (its ``out`` is the state itself).

    The whole operation lives **inside this carrier**, not as loose body
    statements, so the gates that reject online algorithms (``accums_independent``,
    ``classify_fragment_epilogue``) never see the cross-state coupling, and the
    partition planner places one carrier.

    Example — flash attention's **online softmax** (the log-sum-exp monoid): state
    ``(m, l, O)`` = running row-max / denominator / output, partial ``(s, v)`` =
    this key's score + value, identity ``(−inf, 0, 0)``, and the merge::

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        l = l·alpha + p;     O = O·alpha + p·v;         m = m_new   (last)

    associative + commutative — which is what makes split-KV (flash-decoding) and
    cooperative-combine legal. The flash instance is built by
    ``lowering/tile/_flash.flash_combine``.
    """

    state: State
    # The partial-contribution sources, each an :data:`AlgebraNode` — the self-contained
    # op-tree form. EMPTY ``()`` on a loop-IR carrier: there the partials are sibling
    # stmts in the enclosing ``Loop`` and their names are read off the twist's ``merge``
    # (``partial_names`` / ``deps``), so the carrier needs no source nodes.
    partial: tuple[AlgebraNode, ...]
    twist: Twist
    # Self-contained reduction axis (op-tree node). ``None`` = a loop-IR carrier stmt that
    # sits inside an existing ``Loop`` (axis from the enclosing loop, partials are
    # siblings). Set by the op-tree builders; ``ir.tile.ops.lower`` reads it to emit the
    # ``Loop`` and clears ``partial`` on the in-loop carrier it leaves.
    axis: Axis | None = None

    @cached_property
    def carrier(self) -> Carrier:
        """The decoupled algebra payload (:class:`Carrier`) — ``(state, twist)``. The op-tree
        ``Monoid`` delegates every algebra query here; a reduce ``Loop`` stores one directly
        (``loop.carrier``). Cached — ``Monoid`` is frozen with a ``__dict__`` (never ``slots``)."""
        return Carrier(self.state, self.twist)

    @property
    def out(self) -> str:
        """The bound output name — the primary carried state component."""
        return self.carrier.out

    @property
    def reduce_node(self) -> Monoid:
        """This carrier IS the reduction (identity for the projection-peeling query)."""
        return self

    @property
    def state_b(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine."""
        return self.carrier.state_b

    @property
    def merge(self) -> tuple[Stmt, ...]:
        """The streaming single-element fold program (delegated to :attr:`carrier`)."""
        return self.carrier.merge

    @property
    def combine_states(self) -> tuple[Assign, ...]:
        """The cross-partition state⊕state fold program (delegated to :attr:`carrier`)."""
        return self.carrier.combine_states

    def partial_names(self) -> tuple[str, ...]:
        """The bound name of each partial the merge folds in (the merge's external reads)."""
        return self.carrier.partial_names()

    def dissolve(self) -> list[Stmt]:
        """The loose fold stmts this carrier lowers to — bare ``Accum``\\ s for a degenerate
        carrier, else the streaming ``merge`` (delegated to :attr:`carrier`). The ``Monoid`` stmt
        is a recognition-time grouping; once the schedule is realized it dissolves into these
        folds (seeding via the fold ``Accum``\\ s' ``op.identity`` in ``Loop.render``), so no
        ``Monoid`` stmt reaches a rendered loop body."""
        return self.carrier.dissolve()

    def as_accums(self) -> list[Accum] | None:
        """The degenerate-carrier ``Accum`` folds, or ``None`` (delegated to :attr:`carrier`)."""
        return self.carrier.as_accums()

    def as_state_merge(self, other: tuple[str, ...]) -> Monoid:
        """Return a one-shot ``Monoid`` that merges this carrier's ``state`` with
        a second fully-reduced state named ``other`` (the cross-partition
        combine's right operand). The returned carrier's ``merge`` IS
        ``combine_states`` with ``state_b`` renamed to ``other`` — so the
        cooperative-tree / cross-CTA reduce renders the state-merge through the same
        machinery as a streaming step (``other`` is read off that merge as its partials).
        """
        if self.twist.family == "exp":
            # Regenerate (not rename) so the finalize temps key on ``other[0]`` — distinct
            # REG-tier folds get distinct temps, no manual uniquify needed.
            merged = exp_combine_states(self.state.names, other, key=other[0])
        else:
            sub = dict(zip(self.state_b, other, strict=True))
            merged = tuple(_rename_assign_args(a, sub) for a in self.combine_states)
        return Monoid(
            state=self.state,  # the State (names + identity) carries over unchanged
            partial=(),  # a loop-IR carrier — its partials (``other``) are read off ``merged``
            twist=Twist(merge=merged, combine_states=merged, state_b=other),  # BOUND mode (no family)
        )

    def deps(self) -> tuple[str, ...]:
        # Mirror ``Accum`` / ``Mma``: the carried-state read is implicit
        # (loop-carried), so only this iteration's partial contribution is listed —
        # keeps sibling-def analyses from treating the state as a same-scope read.
        return self.partial_names()

    def defines(self) -> tuple[str, ...]:
        return self.state.names

    def pretty(self, indent: str = "") -> list[str]:
        # Summary header (carried state <- combine(partials)) followed by the stored
        # ``merge`` program with real names — the state-update reassignments print as
        # ordinary lines (the dump is text, not SSA-constrained), so the carrier's
        # actual fold is visible in the loop-IR dump instead of an opaque one-liner.
        state = ", ".join(self.state.names)
        partial = ", ".join(self.partial_names())
        lines = [f"{indent}({state}) <- combine({partial})"]
        for a in self.merge:
            lines += a.pretty(indent + "    ")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit the merge program in fp32. A streaming carrier never renders here — it
        :meth:`dissolve`\\ s into loose fold ``Accum``\\ s at lowering. This is reached only
        by the cross-partition **state-merge** (:meth:`as_state_merge`), rendered at the
        enclosing scope (its state already declared + seeded by the partition reduces). Each
        ``Assign``/``Accum`` targeting a ``state`` name is a reassignment of the carried
        value; every other ``Assign`` declares a local temp. The builder orders the program
        so each old-state read precedes that state's update (flash reads the old ``m`` for
        ``alpha`` before ``m = m_new``)."""
        return render_merge_program(self.merge, self.state.names, ctx)


# The three algebra node kinds — the compute tree's vocabulary. A partial / operand
# source is one of these. Defined after the classes; annotations are strings (``from
# __future__ import annotations``) so the forward reference in the fields above resolves.
AlgebraNode = Map | Monoid | Semiring


def _partial_name(p: AlgebraNode) -> str:
    """The SSA name an algebra-node source binds — the value a carrier folds / a
    contraction reads. Every node kind exposes it as ``out`` (``Map`` = its body's last
    def, ``Monoid`` = primary state, ``Semiring`` = the fold accumulator)."""
    if isinstance(p, (Map, Monoid, Semiring)):
        return p.out
    raise TypeError(f"_partial_name: unsupported source {type(p).__name__}")


def _stmt_reads(a: Stmt) -> tuple[str, ...]:
    """The arg reads of one merge-program stmt. An ``Assign`` reads its ``args``; an
    ``Accum`` reads its folded ``value`` and (when redirected) its rescaled ``base`` — its
    carried ``name`` is the loop-carried state, not a same-program read."""
    if isinstance(a, Accum):
        return (a.base, a.value) if a.base is not None and a.base != a.name else (a.value,)
    return a.args


def _merge_reads(merge: tuple[Stmt, ...], state_names: tuple[str, ...]) -> tuple[str, ...]:
    """The external read names of a merge program — args read but neither carried state
    nor a temp defined within the program — in first-use order. These are the partials the
    merge folds into the state (the source of truth for ``Monoid.partial_names``). The
    program is a mix of ``Assign`` temps/rescales and ``Accum`` folds (a twisted carrier's
    streaming merge); both expose their reads via :func:`_stmt_reads` and their def via
    ``name``."""
    state, defined, seen, reads = set(state_names), set(), set(), []
    for a in merge:
        for arg in _stmt_reads(a):
            if arg not in state and arg not in defined and arg not in seen:
                seen.add(arg)
                reads.append(arg)
        defined.add(a.name)
    return tuple(reads)


__all__ = ["AlgebraNode", "Carrier", "Map", "Semiring", "State", "StateMerge", "Twist", "Monoid"]
