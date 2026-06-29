"""Carrier-algebra generator + per-family stabilizer (transport of structure).

A streaming reduce carrier (online softmax, flash attention) is a **twisted monoid**: a
base monoid ``(max, +, +, …)`` conjugated by a bijection ψ. Rather than hand-author the
numerically-stable combine programs, we *generate* them:

1. **Generate** the naive symmetric combine ``ψ ∘ base_combine ∘ (ψ⁻¹ × ψ⁻¹)`` — associativity
   is inherited from the base monoid for free ("association via inverse ψ"). The naive form
   carries overflowing ``exp(m)`` factors.
2. **Stabilize** (per family) by algebraic rewriting — distribute the ψ-rescale over the base
   add, fuse exponentials (``e^a·e^b → e^{a+b}``), fold multiplicative identities, DCE the dead
   overflowing temps, CSE shared rescales. For the exp/LSE family this lands on the stable form
   where every surviving ``exp`` has a provably ``≤ 0`` argument.
3. **Certify** stability structurally: every ``exp`` arg is ``x − max(…, x, …)``.

``combine_states`` (state⊕state, the cross-partition fold) and ``merge`` (the streaming
single-element fold) are both derived from one channel spec — ``merge`` is ``combine_states``
with the second operand replaced by the per-element injection, its final per-channel writes
retagged to seed-riding ``base``-``Accum``\\ s.

Scope: the **exp/LSE family** (covers attention + online softmax — they differ only in channel
count). The generation (1) is family-agnostic; only the stabilizer (2) is per-family.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.leaves import Accum, Assign


class UnstableCarrierError(ValueError):
    """A generated combine failed the stability certificate — an ``exp`` whose argument is not
    provably ``≤ 0``. Raised instead of silently emitting overflow-prone code."""


# --------------------------------------------------------------------------------------------
# Channel spec — the carrier's per-component algebra (the part that VARIES across carriers).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One carried component. ``fold`` is the base-monoid ⊕ across the reduce axis (``maximum``
    for the pivot, ``add`` for an accumulator). ``term`` is the per-element injected value (the
    score name for the pivot, ``1.0`` for the denominator, the value name for an expectation).
    ``lift`` is the ⊗ combining the softmax weight with ``term`` (``None`` = pivot/denominator;
    ``multiply`` = expectation) — at the scalar tier ⊗ is the implicit scalar multiply already
    in the generation; ``lift`` is carried so a future fragment realizer can lower ⊗ to a
    contraction (mma). See the tile-lowering ARCHITECTURE notes on tier independence."""

    fold: ElementwiseImpl
    term: str | float
    lift: ElementwiseImpl | None = None


# --------------------------------------------------------------------------------------------
# Term — a tiny symbolic algebra used ONLY to generate + stabilize the combine programs.
# (Distinct from ir.expr.Expr, which is for index/predicate codegen.)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _T:
    op: str  # 'leaf' | 'lit' | 'exp' | 'neg' | 'maximum' | 'add' | 'multiply' | 'subtract'
    a: tuple  # leaf -> (name,), lit -> (value,), else tuple[_T, ...]


def _leaf(n: str) -> _T:
    return _T("leaf", (n,))


def _lit(v: float) -> _T:
    return _T("lit", (v,))


def _term(t: str | float) -> _T:
    return _lit(t) if isinstance(t, (int, float)) else _leaf(t)


# Term-op -> ElementwiseImpl name (the only place the spelling is chosen).
_OP = {"exp": "exp", "neg": "negative", "maximum": "maximum", "add": "add", "multiply": "multiply", "subtract": "subtract"}


def _flatten_mul(t: _T) -> list[_T]:
    if t.op == "multiply":
        return _flatten_mul(t.a[0]) + _flatten_mul(t.a[1])
    return [t]


def _fold_exponents(args: list[_T]) -> _T:
    """Combine a product of exponentials' exponents into one: ``[a, neg(b)] → a − b``."""
    acc = args[0]
    if acc.op == "neg":  # leading negation (our carriers never do this, but stay total)
        acc = _T("subtract", (_lit(0.0), acc.a[0]))
    for x in args[1:]:
        acc = _T("subtract", (acc, x.a[0])) if x.op == "neg" else _T("add", (acc, x))
    return acc


def _combine_exps(factors: list[_T]) -> _T:
    """A product with no ``add`` factors: drop multiplicative-identity ``1.0``, fuse all ``exp``
    factors into one ``exp(Σ exponents)``, rebuild the product."""
    nonexp = [f for f in factors if f.op != "exp" and not (f.op == "lit" and f.a[0] == 1.0)]
    exps = [f for f in factors if f.op == "exp"]
    out = list(nonexp)
    if exps:
        out.append(_T("exp", (_fold_exponents([e.a[0] for e in exps]),)))
    if not out:
        return _lit(1.0)
    acc = out[0]
    for f in out[1:]:
        acc = _T("multiply", (acc, f))
    return acc


def _expand_product(factors: list[_T]) -> _T:
    """Normalize a product into a sum of products: distribute over the first ``add`` factor,
    else fuse the exponentials. Recurses until no ``add`` factor remains."""
    flat: list[_T] = []
    for f in factors:
        flat += _flatten_mul(f)
    for i, f in enumerate(flat):
        if f.op == "add":
            rest = flat[:i] + flat[i + 1 :]
            return _T("add", (_expand_product([f.a[0]] + rest), _expand_product([f.a[1]] + rest)))
    return _combine_exps(flat)


def _simplify(t: _T) -> _T:
    """Algebraic stabilization of a generated term: bottom-up, expanding products of sums and
    fusing exponentials so the overflowing ``exp(m)`` cancels against the ``exp(−M)`` rescale."""
    if t.op in ("leaf", "lit"):
        return t
    if t.op == "exp":
        return _T("exp", (_simplify(t.a[0]),))
    if t.op == "neg":
        return _T("neg", (_simplify(t.a[0]),))
    if t.op in ("maximum", "subtract", "add"):
        return _T(t.op, (_simplify(t.a[0]), _simplify(t.a[1])))
    if t.op == "multiply":
        return _expand_product([_simplify(t.a[0]), _simplify(t.a[1])])
    raise AssertionError(f"_simplify: unexpected term op {t.op!r}")


def _reads(t: _T, name: str) -> bool:
    if t.op == "leaf":
        return t.a[0] == name
    if t.op == "lit":
        return False
    return any(_reads(x, name) for x in t.a)


# --------------------------------------------------------------------------------------------
# Generation: naive ψ ∘ base_combine ∘ (ψ⁻¹ × ψ⁻¹) for the exp/LSE family.
# --------------------------------------------------------------------------------------------


def _gen_outputs(state: tuple[str, ...], b0: _T, b_rest: list[_T]) -> list[_T]:
    """Per-component naive→simplified output term. ``state`` = operand A (carried) names; ``b0`` =
    operand B pivot term; ``b_rest`` = operand B per-accumulator terms (state names for
    combine_states, injected values for merge). Channel 0 is the max pivot."""
    m_a = _leaf(state[0])
    M = _T("maximum", (m_a, b0))
    outs = [M]  # pivot
    for i in range(1, len(state)):
        a_i = _leaf(state[i])
        b_i = b_rest[i - 1]
        lifted_a = _T("multiply", (a_i, _T("exp", (m_a,))))  # ψ⁻¹: a_i · e^{m_a}
        lifted_b = _T("multiply", (b_i, _T("exp", (b0,))))  # ψ⁻¹: b_i · e^{b0}
        base_sum = _T("add", (lifted_a, lifted_b))
        proj = _T("multiply", (base_sum, _T("exp", (_T("neg", (M,)),))))  # ψ: · e^{−M}
        outs.append(_simplify(proj))
    outs[0] = _simplify(outs[0])
    return outs


# --------------------------------------------------------------------------------------------
# Emission: lower the simplified terms to an Assign/Accum program with CSE.
# --------------------------------------------------------------------------------------------


def _emit(outs: list[_T], state: tuple[str, ...], key: str, *, merge: bool) -> tuple[Stmt, ...]:
    """Emit the combine program. ``merge`` mode retags each channel's final write into a
    seed-riding ``base``-``Accum`` (streaming fold); else plain ``Assign`` reassignment
    (state⊕state). Temps are namespaced on ``key`` so distinct folds never collide."""
    memo: dict[_T, str] = {}
    body: list[Stmt] = []
    n = [0]

    def fresh() -> str:
        name = f"{key}__t{n[0]}"
        n[0] += 1
        return name

    def realize(t: _T) -> str:
        if t.op == "leaf":
            return t.a[0]
        if t.op == "lit":
            raise AssertionError(f"literal {t.a[0]} survived stabilization")
        if t in memo:
            return memo[t]
        args = tuple(realize(x) for x in t.a)
        name = fresh()
        body.append(Assign(name, _OP[t.op], args))
        memo[t] = name
        return name

    writes: list[Stmt] = []
    # Accumulator channels.
    for sname, out in list(zip(state[1:], outs[1:], strict=True)):
        assert out.op == "add", f"accumulator channel must reduce to a sum, got {out.op}"
        p, q = out.a
        if not merge:
            writes.append(Assign(sname, "add", (realize(p), realize(q))))
            continue
        base_t, val_t = (p, q) if _reads(p, sname) else (q, p)
        base = realize(base_t)  # the rescaled old carried state (e.g. l·alpha)
        val = realize(val_t)  # this element's contribution (e.g. p or p·v)
        writes.append(Accum(name=sname, value=val, op="add", base=base, dtype=F32))
    # Pivot (channel 0).
    pivot = outs[0]
    assert pivot.op == "maximum"
    if not merge:
        writes.append(Assign(state[0], "copy", (realize(pivot),)))
    else:
        realize(pivot)  # the max temp the rescales read
        a0, b0 = pivot.a
        other = b0 if a0.op == "leaf" and a0.a[0] == state[0] else a0
        writes.append(Accum(name=state[0], value=realize(other), op="maximum", dtype=F32))
    return tuple(body + writes)


# --------------------------------------------------------------------------------------------
# Stability certificate.
# --------------------------------------------------------------------------------------------


def _max_operands(name: str, defs: dict[str, Assign]) -> set[str] | None:
    """The (recursively flattened) operand names of a ``maximum`` defining ``name``, or ``None``
    if ``name`` is not a maximum temp."""
    d = defs.get(name)
    if d is None or d.op.name != "maximum":
        return None
    ops: set[str] = set()
    for a in d.args:
        nested = _max_operands(a, defs)
        ops |= nested if nested is not None else {a}
    return ops


def _certify(prog: tuple[Stmt, ...]) -> None:
    """Every ``exp`` argument must be ``subtract(x, R)`` with ``R`` a ``maximum`` whose operand
    set contains ``x`` — so ``arg = x − max(…, x, …) ≤ 0`` and ``exp(arg) ≤ 1``."""
    defs = {s.name: s for s in prog if isinstance(s, Assign)}
    for s in prog:
        if not (isinstance(s, Assign) and s.op.name == "exp"):
            continue
        arg = defs.get(s.args[0])
        if arg is None or arg.op.name != "subtract":
            raise UnstableCarrierError(f"exp arg {s.args[0]!r} is not a subtract — cannot prove ≤ 0")
        x, r = arg.args
        ops = _max_operands(r, defs)
        if ops is None or x not in ops:
            raise UnstableCarrierError(f"exp({x} − {r}): {r!r} is not a max over a set containing {x!r}")


# --------------------------------------------------------------------------------------------
# Public: build the exp-family carrier.
# --------------------------------------------------------------------------------------------


def exp_combine_states(state: tuple[str, ...], state_b: tuple[str, ...], *, key: str | None = None) -> tuple[Assign, ...]:
    """The cross-partition state⊕state combine for an exp-family carrier of arity ``len(state)``.
    Temps namespaced on ``key`` (defaults to ``state_b[0]`` so distinct REG-tier folds — which
    rename ``state_b`` — never collide)."""
    outs = _gen_outputs(state, _leaf(state_b[0]), [_leaf(n) for n in state_b[1:]])
    prog = _emit(outs, state, key or state_b[0], merge=False)
    _certify(prog)
    return prog  # type: ignore[return-value]


def exp_merge(state: tuple[str, ...], channels: tuple[Channel, ...], *, key: str | None = None) -> tuple[Stmt, ...]:
    """The streaming single-element fold for an exp-family carrier. The injection singleton is
    the channels' ``term``s (pivot ← score, denom ← 1, expectation ← value)."""
    score = channels[0].term
    assert isinstance(score, str), "pivot term (the score) must be an SSA name"
    outs = _gen_outputs(state, _leaf(score), [_term(c.term) for c in channels[1:]])
    prog = _emit(outs, state, key or state[0], merge=True)
    _certify(prog)
    return prog


# Channel constructors mirroring the expectation-semiring vocabulary.
_ADD = ElementwiseImpl("add")
_MUL = ElementwiseImpl("multiply")
_MAX = ElementwiseImpl("maximum")


def pivot(score: str) -> Channel:
    return Channel(fold=_MAX, term=score)


def denom() -> Channel:
    return Channel(fold=_ADD, term=1.0)


def expect(value: str) -> Channel:
    return Channel(fold=_ADD, term=value, lift=_MUL)


def exp_channels(score: str, accumulators: list[Channel]) -> tuple[Channel, ...]:
    """The full exp-family channel tuple over a carrier (pivot first). ``accumulators`` are the
    non-pivot channels (``denom()`` / ``expect(v)``)."""
    return (pivot(score), *accumulators)
