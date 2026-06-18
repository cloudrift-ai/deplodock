"""Tests for symbolic-divisor modulo folding in ``BinaryExpr.simplify``.

A collapsed reshape composed with a matmul read emits a delinearized index like
``((i * stride + feat) // stride) % seq_len`` for the seq coordinate. When
``seq_len`` is a compile-time constant the whole thing constant-folds to ``i``;
when ``seq_len`` is symbolic the literal-only ``_div_mod_decompose`` cannot fold
the outer ``% seq_len``, leaving a runtime integer modulo in the gmem index
(the slow masked-tile repack producers).

``SimplifyCtx.bounds`` carries a symbolic exclusive upper bound per loop axis
(``i < seq_len``), and ``_mod_below_divisor`` uses it to fold ``i % seq_len →
i``. ``axis.extend_simplify_ctx`` registers those bounds (and the
non-negativity the inner ``// stride`` fold needs) for both static and symbolic
axes. These tests pin the fold and its safety guards.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis, extend_simplify_ctx
from deplodock.compiler.ir.expr import BinaryExpr, Interval, Literal, SimplifyCtx, Var


def _flat_seq_coord(stride: int) -> BinaryExpr:
    """``((a0 * stride + a2) // stride) % seq_len`` — the delinearized seq
    coordinate the o_proj / P@V repack producers carry (stride = collapsed
    feature width: 2048 for o_proj, 1024 for P@V)."""
    a0, a2, sl = Var("a0"), Var("a2"), Var("seq_len")
    flat = BinaryExpr("+", BinaryExpr("*", a0, Literal(stride, "int")), a2)
    return BinaryExpr("%", BinaryExpr("//", flat, Literal(stride, "int")), sl)


def _loop_ctx(stride: int) -> SimplifyCtx:
    """Ctx as the loop simplifier builds it: seq row ``a0`` over symbolic
    ``seq_len``, feature ``a2`` over static ``stride``."""
    ctx = SimplifyCtx.empty()
    ctx = extend_simplify_ctx(ctx, Axis("a0", "seq_len"))
    ctx = extend_simplify_ctx(ctx, Axis("a2", stride))
    return ctx


def test_bare_var_mod_symbolic_extent_folds():
    """``a0 % seq_len → a0`` when ``a0``'s loop extent is exactly ``seq_len``."""
    a0, sl = Var("a0"), Var("seq_len")
    ctx = extend_simplify_ctx(SimplifyCtx.empty(), Axis("a0", "seq_len"))
    assert BinaryExpr("%", a0, sl).simplify(ctx) == a0


def test_collapsed_seq_coord_folds_to_row():
    """The full delinearized seq coordinate collapses to the row var ``a0``,
    removing both the inner ``// stride`` and the outer ``% seq_len``."""
    for stride in (1024, 2048):
        folded = _flat_seq_coord(stride).simplify(_loop_ctx(stride))
        assert folded == Var("a0"), f"stride={stride}: got {folded.pretty()}"


def test_collapsed_seq_coord_numerically_equivalent():
    """Folded form must agree with the original for every valid (a0, a2) at a
    concrete seq_len — the fold is a strength-reduction, not a change of value."""
    stride = 2048
    orig = _flat_seq_coord(stride)
    folded = orig.simplify(_loop_ctx(stride))
    for seq_len in (7, 64, 512):
        for a0 in range(seq_len):
            for a2 in (0, 1, stride // 2, stride - 1):
                env = {"a0": a0, "a2": a2, "seq_len": seq_len}
                assert orig.eval(env) == folded.eval(env) == a0


def test_no_fold_without_bound():
    """A var with no registered exclusive bound keeps its ``% seq_len`` — the
    fold must not fire on un-bounded operands."""
    b, sl = Var("b"), Var("seq_len")
    ctx = SimplifyCtx({"b": Interval(0, 1 << 30)})  # range but no bounds entry
    assert BinaryExpr("%", b, sl).simplify(ctx) == BinaryExpr("%", b, sl)


def test_no_fold_on_bound_mismatch():
    """``a0 % other`` does NOT fold when the divisor differs from a0's extent."""
    a0 = Var("a0")
    ctx = extend_simplify_ctx(SimplifyCtx.empty(), Axis("a0", "seq_len"))
    e = BinaryExpr("%", a0, Var("other"))
    assert e.simplify(ctx) == e


def test_symbolic_axis_gets_nonneg_range():
    """A symbolic-extent axis must register ``lo == 0`` (the non-negativity the
    ``(i*c + …)//c → i`` div fold relies on) plus its extent as the bound."""
    ctx = extend_simplify_ctx(SimplifyCtx.empty(), Axis("a0", "seq_len"))
    assert ctx.ranges["a0"].lo == 0
    assert ctx.bounds["a0"] == Var("seq_len")


def test_segmented_k_collapses_oproj_delinearized_read():
    """The segmented-K mechanism: a C-aligned split of the flat matmul K plus
    range-aware simplify collapses the o_proj attn-out delinearized read to a
    clean ``[head, seq, within]`` access — no div/mod, no producer needed.

    Real o_proj A-operand index (flat K = a2, over physical [16, seq, 128]):
        head = (a0*2048 + a2)/128 % 16 ,  seq = (a0*2048 + a2)/2048 % seq_len ,
        dim  = (a0*2048 + a2) % 128
    Split a2 = a2_o*128 + a2_i (a2_o = head 0..15, a2_i = within 0..127); with
    axis ranges/bounds the three coords fold to a2_o / a0 / a2_i."""
    from deplodock.compiler.ir.sigma import Sigma

    a0, a2 = Var("a0"), Var("a2")
    flat = BinaryExpr("+", BinaryExpr("*", a0, Literal(2048, "int")), a2)
    head = BinaryExpr("%", BinaryExpr("//", flat, Literal(128, "int")), Literal(16, "int"))
    seq = BinaryExpr("%", BinaryExpr("//", flat, Literal(2048, "int")), Var("seq_len"))
    dim = BinaryExpr("%", flat, Literal(128, "int"))

    seg = Sigma({"a2": BinaryExpr("+", BinaryExpr("*", Var("a2_o"), Literal(128, "int")), Var("a2_i"))})
    ctx = SimplifyCtx.empty()
    ctx = extend_simplify_ctx(ctx, Axis("a0", "seq_len"))
    ctx = extend_simplify_ctx(ctx, Axis("a2_o", 16))
    ctx = extend_simplify_ctx(ctx, Axis("a2_i", 128))

    assert seg.apply(head).simplify(ctx) == Var("a2_o")   # head, no div/mod
    assert seg.apply(seq).simplify(ctx) == Var("a0")       # seq, no % seq_len
    assert seg.apply(dim).simplify(ctx) == Var("a2_i")     # contiguous inner K


def test_static_axis_unchanged_behavior():
    """Static axis still gets a precise interval; a literal divisor folds via the
    existing ``_div_mod_decompose`` path (unaffected by the symbolic addition)."""
    ctx = extend_simplify_ctx(SimplifyCtx.empty(), Axis("a", 32))
    assert ctx.ranges["a"] == Interval(0, 31)
    # i % 32 with i in [0,32) folds to i (literal path).
    assert BinaryExpr("%", Var("a"), Literal(32, "int")).simplify(ctx) == Var("a")
