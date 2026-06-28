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

from dataclasses import dataclass, field, replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, render_merge_program
from deplodock.compiler.ir.stmt.body import Body
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


def _rename_assign_args(a: Assign, sub: dict[str, str]) -> Assign:
    return Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)


@dataclass(frozen=True)
class State:
    """The carried state of a :class:`Monoid` — the internal-state SSA ``names``. The state
    is the *carrier* a :class:`Twist` operates on: the twist's ``merge`` folds a partial into
    :attr:`names`, its ``combine_states`` merges this state with a second one named by
    :attr:`other`. The seed (neutral element) is NOT stored here — it's the fold's
    ``op.identity``, read via :meth:`Monoid.seed_identities` (so there's one source of truth
    for the seed: the fold, not a separately-authored identity that could drift)."""

    names: tuple[str, ...]

    @property
    def other(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine — ``"<n>__o"``
        per component, the right operand ``combine_states`` reads."""
        return tuple(f"{n}__o" for n in self.names)


@dataclass(frozen=True)
class Twist:
    """The ψ-conjugated combine of a :class:`Monoid` — the part that VARIES while
    the monoid algebra (carried state + neutral element) stays the same.

    Transport of structure (the article): a monoid ``(·, e)`` conjugated by a
    bijection ψ gives the twisted combine ``x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))``. The
    monoid is shared; ψ is the twist, and ψ lives entirely in the combine programs
    carried here **as data** — a plain reduction's identity twist
    (``Twist.degenerate``: componentwise ``state_i = op_i(state_i, partial_i)``),
    online softmax's max-rescale, a future mma-fragment realization — all the same
    monoid, differing only in these programs:

    - ``merge`` — fold one partial into the state (the streaming reduce step). A mix of
      ``Assign`` temps/rescales and ``Accum`` folds: a twisted carrier's ψ rescale is a
      preceding ``Assign`` (``lm = l·alpha``) and the fold itself a ``base``-``Accum``
      (``l = lm + p``), so each state component carries its own seed (``op.identity``) and
      ``Loop.render`` seeds it — no explicit ``Init``. A degenerate twist's merge is all
      ``Assign`` (the identity-twist form ``Twist.degenerate`` builds).
    - ``combine_states`` — merge two fully-reduced states (the cross-partition step),
      reading the second operand named by ``state_b``. All ``Assign`` (rendered by the
      cross-thread combine via ``render_merge_program``; unchanged by the streaming
      ``Accum`` form).
    """

    merge: tuple[Stmt, ...]
    combine_states: tuple[Assign, ...] = ()
    state_b: tuple[str, ...] = ()

    @staticmethod
    def degenerate(state: tuple[str, ...], partial: tuple[str, ...], ops: tuple[ElementwiseImpl, ...], dtype=None) -> Twist:
        """The identity twist: componentwise ``state_i = op_i(state_i, partial_i)``
        — a plain reduction is a monoid with no rescale (ψ = id). ``combine_states``
        / ``state_b`` are left empty; :class:`Monoid` auto-derives them (the additive
        carrier's partial lifts to a state)."""
        merge = tuple(Assign(name=s, op=op, args=(s, p), dtype=dtype) for s, p, op in zip(state, partial, ops, strict=True))
        return Twist(merge=merge)


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
      element) per component is the fold's ``op.identity``, read via :meth:`seed_identities`
      (the uniform carrier interface a schedule realization places — serial loop today,
      cooperative / cross-CTA later).
    - ``partial`` — this iteration's contribution sources, one per partial slot: each an
      :data:`AlgebraNode` (the self-contained op-tree form, e.g. flash's score is a
      ``Map``, a nested contraction a ``Semiring``) or a bound ``str`` name (the loop-IR
      carrier form). The bound name (``partial_names``) is the right operand the merge
      folds into the state. A self-contained reduction also carries ``axis`` (its ``out``
      is derived); ``ir.tile.ops.lower`` expands the nodes into the loop.
    - ``twist`` — the :class:`Twist`: the ψ-conjugated combine, **as data**, the
      one part that varies (degenerate / scalar / fragment) while the algebra above
      stays the same. It carries the ``merge`` program (fold a partial into the
      state — each ``Assign`` targeting a ``state`` name is a state update,
      ``name = …;``, every other a local fp32 temp; statement ORDER is
      load-bearing) and the ``combine_states`` program (merge two fully-reduced
      partition states, reading the second operand ``state_b`` — the form the
      cross-partition combine needs). Read the combine off the twist directly
      (``monoid.twist.merge`` / ``.combine_states`` / ``.state_b``).
      ``__post_init__`` completes the twist
      against this state: defaults ``state_b`` to ``"<s>__o"`` and, for an
      **additive** carrier whose partial lifts to a state, auto-derives
      ``combine_states`` from ``merge``; an asymmetric monoid (flash's LSE) authors
      both on the twist.
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

    @property
    def out(self) -> str:
        """The bound output name — the primary carried state (the φ projection, if any, is
        a :class:`Map` *over* this Monoid, not a field here). Derived, not stored: we
        always know what a monoid accumulates (and an mma-fragment output won't fit a
        stored ``str``). Seeds at lowering ride on the carrier's fold ``Accum``\\ s
        (``op.identity``), derived by ``Loop.render`` — no explicit ``Init``."""
        return self.state.names[0]

    def partial_names(self) -> tuple[str, ...]:
        """The bound name of each partial the merge folds in — the twist's ``merge``
        external reads (args that are neither carried state nor merge-internal temps), in
        first-use order. Derived from the merge (its source of truth), so it holds whether
        ``partial`` carries op-tree source nodes or is the empty loop-IR carrier."""
        return _merge_reads(self.twist.merge, self.state.names)

    def __post_init__(self) -> None:
        # Complete the twist's cross-partition surface against this monoid's state.
        # ``object.__setattr__`` — the dataclass is frozen.
        tw = self.twist
        # Default the second-operand state names ("<s>__o", i.e. ``state.other``) so a
        # twist built with only ``merge`` still has a complete cross-partition surface.
        state_b = tw.state_b or self.state.other
        combine_states = tw.combine_states
        # Auto-derive ``combine_states`` from ``merge`` for an additive carrier
        # (the partial lifts to a state, one component each): substitute the
        # partial reads with the second-operand state names. An asymmetric monoid
        # (different partial / state arity, e.g. flash LSE) must author it.
        partial_names = self.partial_names()
        if not combine_states and len(partial_names) == len(self.state.names):
            sub = dict(zip(partial_names, state_b, strict=True))
            combine_states = tuple(_rename_assign_args(a, sub) for a in tw.merge)
        if state_b != tw.state_b or combine_states != tw.combine_states:
            object.__setattr__(self, "twist", replace(tw, state_b=state_b, combine_states=combine_states))

    def carried_names(self) -> tuple[str, ...]:
        """The carried state names — the uniform carrier interface (matching
        :meth:`Accum.carried_names`) a schedule realization reads alongside
        :meth:`seed_identities`."""
        return self.state.names

    def seed_identities(self) -> tuple[float, ...]:
        """The seed (neutral element) for each carried state component — the ``op.identity``
        of the merge fold that writes it (an ``Accum`` for a twisted carrier, the
        identity-twist ``Assign`` for a degenerate one). The fold is the single source of
        truth for the seed (matching :meth:`Accum.seed_identities`); a schedule realization
        reads this to place the seed (the serial reduce declares it before the loop; a future
        cooperative / cross-CTA realization seeds each partial). Raises if a component's fold
        carries no identity (e.g. a ``copy``-spelled state-merge) — that program is a combine,
        not a streaming fold, and is never a seed source."""
        folds = {s.name: s.op for s in self.twist.merge if s.name in set(self.state.names)}
        out: list[float] = []
        for n in self.state.names:
            op = folds.get(n)
            if op is None or op.identity is None:
                raise ValueError(f"Monoid carrier {n!r}: no identity-bearing fold to seed from")
            out.append(op.identity)
        return tuple(out)

    def as_accums(self) -> list[Accum] | None:
        """If this is a **degenerate** carrier (the identity twist — each state
        component folded by its own self-op, ``state_i = op_i(state_i, partial_i)``,
        no rescale temps), return the equivalent list of ``Accum`` folds; else
        ``None``. A plain reduce (``sum`` / ``max`` / ``min``) is degenerate; a
        twisted carrier (online softmax / flash) is not (its merge reads sibling
        components + rescale temps). Lets ``ir.tile.ops.lower`` emit a degenerate
        reduce as bare ``Accum``\\ s — seed derived from ``op.identity`` by
        ``Loop.render``, no explicit ``Init`` — instead of a ``Monoid`` carrier."""
        merge = self.twist.merge
        names = set(self.state.names)
        if len(merge) != len(self.state.names):
            return None
        accums: list[Accum] = []
        for a in merge:
            # identity-twist shape: ``state = op(state, partial)`` — target is a state
            # component, left operand is that same component, one partial right operand.
            if a.name not in names or len(a.args) != 2 or a.args[0] != a.name:
                return None
            accums.append(Accum(name=a.name, value=a.args[1], op=a.op, dtype=a.dtype))
        return accums

    def as_state_merge(self, other: tuple[str, ...]) -> Monoid:
        """Return a one-shot ``Monoid`` that merges this carrier's ``state`` with
        a second fully-reduced state named ``other`` (the cross-partition
        combine's right operand). The returned carrier's ``merge`` IS
        ``combine_states`` with ``state_b`` renamed to ``other`` — so the
        cooperative-tree / cross-CTA reduce renders the state-merge through the same
        machinery as a streaming step (``other`` is read off that merge as its partials).
        """
        sub = dict(zip(self.twist.state_b, other, strict=True))
        merged = tuple(_rename_assign_args(a, sub) for a in self.twist.combine_states)
        return Monoid(
            state=self.state,  # the State (names + identity) carries over unchanged
            partial=(),  # a loop-IR carrier — its partials (``other``) are read off ``merged``
            twist=Twist(merge=merged, combine_states=merged, state_b=other),
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
        for a in self.twist.merge:
            lines += a.pretty(indent + "    ")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit the merge program in fp32 (the flat-``Map`` fallback, where the carrier
        renders standalone). Each ``Assign``/``Accum`` targeting a ``state`` name is a
        reassignment of the carried value (declared + seeded by ``Loop.render`` from
        ``state.identity``); every other ``Assign`` declares a local temp. The builder
        orders the program so each old-state read precedes that state's update (e.g. flash
        reads the old ``m`` for ``alpha`` before ``m = m_new``)."""
        return render_merge_program(self.twist.merge, self.state.names, ctx)


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


__all__ = ["AlgebraNode", "Map", "Semiring", "State", "Twist", "Monoid"]
