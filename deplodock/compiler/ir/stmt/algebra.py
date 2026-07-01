"""The carrier algebra — the decoupled reduce carrier.

The algebra of a kernel is **in the body**: the fold ``⊕`` is a :class:`Carrier` (state + a
ψ-conjugated :class:`Twist`) that rides on the reduce ``Loop`` it folds through (``loop.carrier``).
A contraction (matmul) is just a reduce ``Loop`` whose ``AxisRole`` is ``CONTRACTION`` — the ``⊗``
lift sits in the loop body, recognized structurally on demand, never stored as a node kind. The
old ``Monoid`` / ``Semiring`` op-tree node *wrappers* are retired; passes read the structure
(the annotated loop + its carrier) directly. These primitives are consolidated here so the
algebra lives in one place:

- :class:`State` / :class:`Twist` / :class:`Carrier` — the loop-carried associative combine (⊕).
- :class:`StateMerge` — the renderable cross-partition state⊕state combine a carrier emits.

The lift / projection wrapper itself — :class:`~deplodock.compiler.ir.tile.ir.Map` — is a
*tile-IR* op-tree node (it carries the kernel's schedule), so it lives in ``ir/tile/ir.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, render_merge_program
from deplodock.compiler.ir.stmt.carrier import (
    Channel,  # noqa: F401 — re-exported for the carrier builders / tests
    exp_combine_states,
    exp_merge,
    id_accums,
    id_combine_states,
    id_merge,
)
from deplodock.compiler.ir.stmt.leaves import Accum, Assign


def _rename_assign_args(a: Assign, sub: dict[str, str]) -> Assign:
    return Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)


@dataclass(frozen=True)
class State:
    """The carried state of a reduce :class:`Carrier` — the internal-state SSA ``names``. The
    state is the *carrier* a :class:`Twist` operates on: the twist's ``merge`` folds a partial
    into :attr:`names`, its ``combine_states`` merges this state with a second one named by
    :attr:`other`. The seed (neutral element) is NOT stored here — a carrier dissolves into its
    fold ``Accum``\\ s (:meth:`Carrier.as_accums`), and each fold's seed is its ``op.identity``
    (so there's one source of truth for the seed: the fold, not a separately-authored identity
    that could drift)."""

    names: tuple[str, ...]

    @property
    def other(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine — ``"<n>__o"``
        per component, the right operand ``combine_states`` reads."""
        return tuple(f"{n}__o" for n in self.names)


@dataclass(frozen=True)
class Twist:
    """The ψ-conjugated combine of a reduce carrier — the part that VARIES while the carrier
    algebra (carried state + neutral element) stays the same.

    Transport of structure (the article): a monoid ``(·, e)`` conjugated by a bijection ψ gives
    the twisted combine ``x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))``. The twist is carried here as DATA, in one
    of two modes:

    - **SPEC mode** (``family`` set, with ``channels``): the combine is generated on demand by
      :class:`Carrier` from ``(family, channels, state)`` — the inverse-ψ generator builds the
      naive combine and a per-family stabilizer recovers the numerically-stable form (see
      ``ir/stmt/carrier.py``). ``"exp"`` is online-softmax / flash's max-rescale LSE; ``"id"`` is
      the degenerate identity twist (a plain reduce). The stored program fields stay empty.
    - **BOUND mode** (``family is None``): the programs are stored verbatim — used by the one-shot
      ``Carrier.as_state_merge`` finalize, whose ``merge`` IS its ``combine_states``.

    A :class:`Carrier` reads ``carrier.merge`` / ``.combine_states`` / ``.state_b`` (the
    mode-dispatching accessors), never these fields directly.
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

    It knows how to derive the streaming fold (:attr:`merge`), the cross-partition fold
    (:attr:`combine_states`), the degenerate ``Accum`` form (:meth:`as_accums`), and the
    one-shot cross-partition combine (:meth:`as_state_merge`) from ``(state, twist)``. It is NOT
    a ``Stmt`` and carries no ``partial`` / ``axis``: a reduce ``Loop`` carries one
    (``loop.carrier``) so the streaming / cooperative / cross-CTA materializers read the combine
    off the loop, not a node. A *degenerate* carrier (the ``id`` twist) is a plain
    ``sum`` / ``max`` / ``mean`` reduce; a *twisted* one (``exp``) is online-softmax / flash. A
    contraction's carrier is the degenerate carrier of its additive fold (the ``⊗`` lift sits in
    the loop body)."""

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
    renders the ψ-rescale state reassignment via ``render_merge_program``. Unlike ``Accum`` it is
    not a fold carrier — it sits in a combine region, not a streaming fold loop, so it never
    makes its enclosing loop ``is_reduce``."""

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
    merge folds into the state. The program is a mix of ``Assign`` temps/rescales and ``Accum``
    folds (a twisted carrier's streaming merge); both expose their reads via :func:`_stmt_reads`
    and their def via ``name``."""
    state, defined, seen, reads = set(state_names), set(), set(), []
    for a in merge:
        for arg in _stmt_reads(a):
            if arg not in state and arg not in defined and arg not in seen:
                seen.add(arg)
                reads.append(arg)
        defined.add(a.name)
    return tuple(reads)


__all__ = ["Carrier", "State", "StateMerge", "Twist"]
