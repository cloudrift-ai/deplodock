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
- :class:`Operand` / :class:`Semiring` — the structural contraction view.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, render_merge_program
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load


class Map(Body):
    """A pointwise body — a typed :class:`Body` (a sequence of loop-IR stmts: operand
    ``Load``\\ s, the lift ``Assign``\\ s, an optional masking ``Select``, and the output
    ``Write`` at the kernel root) that binds a value name as its last defining stmt. It
    has no fields of its own: it IS its stmts, and so carries Body's analysis helpers.
    Used as a carrier partial (supplying that partial's stmts, last-binding the partial
    name) or as the kernel root (last stmt = the ``Write``)."""


@dataclass(frozen=True)
class Operand:
    """One ⊗ input of a contraction — a buffer read plus its index exprs. The
    index is what the scheduler reads to derive operand reuse (which free axis the
    operand is invariant in) and layout (for smem staging / the mma fragment)."""

    buf: str
    index: tuple[Expr, ...]


@dataclass(frozen=True)
class Semiring:
    """Structural view of a semiring contraction in a reduce loop —
    ``reduce(⊕) ∘ map(⊗)``. Built by :meth:`match` from the loop body, never
    stored (the body is the source of truth).

    - ``fold`` — the ⊕ monoid carrier (an additive ``Accum``: identity 0, assoc + comm).
    - ``lift`` — the ⊗ product op; distributes over ⊕ and has a multiplicative identity.
    - ``operands`` — the contracted inputs (≥ 2 distinct buffers).
    - ``reduce_axis`` — the contracted (K) axis.
    """

    fold: Accum
    lift: ElementwiseImpl
    operands: tuple[Operand, ...]
    reduce_axis: Axis

    @property
    def is_additive(self) -> bool:
        """The ``(×, +)`` semiring the tensor-core mma implements — the gate for the
        mma atom (a tropical / min-plus contraction is still a semiring, still tiles,
        but has no hardware atom)."""
        return self.lift.name == "multiply" and self.fold.op.reduce_canon == "add"

    @staticmethod
    def match(loop) -> Semiring | None:
        """Recognize a semiring contraction on ``loop``, or ``None``. Duck-typed on
        ``.is_reduce`` / ``.axis`` / ``.body`` so it serves a Loop-IR ``Loop`` /
        ``StridedLoop`` alike (the caller restricts the type).

        A reduce loop is a contraction iff its single ``Accum`` fold's partial is
        produced by a lift that **distributes over** the fold op, contracting ≥ 2
        distinct operands over the reduce axis."""
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
        operands = tuple(
            Operand(ld.input, ld.index) for ld in body if isinstance(ld, Load) and k in {v for e in ld.index for v in e.free_vars()}
        )
        if len({o.buf for o in operands}) < 2:
            return None
        return Semiring(fold=fold, lift=lift.op, operands=operands, reduce_axis=loop.axis)


def _rename_assign_args(a: Assign, sub: dict[str, str]) -> Assign:
    return Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)


@dataclass(frozen=True)
class State:
    """The carried state of a :class:`Monoid` — the internal-state SSA ``names`` and
    their per-component ``identity`` (the monoid's neutral element, one :class:`Expr`
    each; empty = unseeded). The state is the *carrier* a :class:`Twist` operates on:
    the twist's ``merge`` folds a partial into :attr:`names`, its ``combine_states``
    merges this state with a second one named by :attr:`other`. ``identity`` seeds the
    enclosing ``Init`` at the carrier's scope (split-KV / cooperative-combine reductions
    read it to seed their partial accumulators)."""

    names: tuple[str, ...]
    identity: tuple[Expr, ...] = ()

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

    - ``merge`` — fold one partial into the state (the streaming reduce step);
    - ``combine_states`` — merge two fully-reduced states (the cross-partition
      step), reading the second operand named by ``state_b``.
    """

    merge: tuple[Assign, ...]
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
      ``Accum.name`` / ``Mma.c`` — and the defs visible after the loop) plus their
      per-component ``identity`` (the monoid's neutral element, one ``Expr`` each; the
      enclosing ``Init`` seeds it at the carrier's scope, and split-KV /
      cooperative-combine reductions read it to seed their partial accumulators).
    - ``partial`` — this iteration's contribution SSA names (the right operand the
      step folds into the state).
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
    - ``commutative`` — whether the operation also commutes (split-KV legality);
      ``associative`` / ``has_identity`` are ``True`` by construction (it *is* a
      monoid). ``axes`` are the reduction axes, threaded through ``rewrite``.
    - ``finalize`` — the carrier's **φ projection**: the post-reduction program that
      maps the *final* carried state to the kernel's output value (the article's
      ``project`` in ``project ∘ reduce ∘ map``), emitted by ``lower(Reduce)`` AFTER
      the streaming loop. As data — a tuple of ``Assign``\\ s reading the state. Empty
      = identity (the state itself is the output, e.g. a plain reduce / matmul). Flash
      authors ``O_i / l_i`` (normalize the streamed output by the denominator). Named
      ``finalize`` is the φ of ``project ∘ reduce ∘ map`` — the post-reduction map to
      the output value (distinct from any cross-partition / cooperative realization,
      which the carrier's ``combine_states`` data describes).

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
    partial: tuple[str, ...]
    twist: Twist
    commutative: bool = True
    axes: tuple[str, ...] = ()
    finalize: tuple[Assign, ...] = ()  # φ: final state → output value (post-loop); empty = identity

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
        if not combine_states and len(self.partial) == len(self.state.names):
            sub = dict(zip(self.partial, state_b, strict=True))
            combine_states = tuple(_rename_assign_args(a, sub) for a in tw.merge)
        if state_b != tw.state_b or combine_states != tw.combine_states:
            object.__setattr__(self, "twist", replace(tw, state_b=state_b, combine_states=combine_states))

    def as_state_merge(self, other: tuple[str, ...]) -> Monoid:
        """Return a one-shot ``Monoid`` that merges this carrier's ``state`` with
        a second fully-reduced state named ``other`` (the cross-partition
        combine's right operand). The returned carrier's ``merge`` IS
        ``combine_states`` with ``state_b`` renamed to ``other`` and its
        ``partial`` set to ``other`` — so the cooperative-tree / cross-CTA reduce
        renders the state-merge through the same machinery as a streaming step.
        """
        sub = dict(zip(self.twist.state_b, other, strict=True))
        merged = tuple(_rename_assign_args(a, sub) for a in self.twist.combine_states)
        return Monoid(
            state=self.state,  # the State (names + identity) carries over unchanged
            partial=other,
            twist=Twist(merge=merged, combine_states=merged, state_b=other),
            commutative=self.commutative,
            axes=self.axes,
        )

    def deps(self) -> tuple[str, ...]:
        # Mirror ``Accum`` / ``Mma``: the carried-state read is implicit
        # (loop-carried), so only this iteration's partial contribution is listed —
        # keeps sibling-def analyses from treating the state as a same-scope read.
        return self.partial

    def defines(self) -> tuple[str, ...]:
        return self.state.names

    def pretty(self, indent: str = "") -> list[str]:
        state = ", ".join(self.state.names)
        partial = ", ".join(self.partial)
        return [f"{indent}({state}) <- combine({partial})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit the merge program in fp32: each ``Assign`` targeting a ``state``
        name is a reassignment of the carried value (already declared by an
        enclosing ``Init``); every other ``Assign`` declares a local temp. The
        builder orders the program so each old-state read precedes that state's
        update (e.g. flash reads the old ``m`` for ``alpha`` before ``m = m_new``).
        """
        return render_merge_program(self.twist.merge, self.state.names, ctx)


__all__ = ["Map", "Operand", "Semiring", "State", "Twist", "Monoid"]
