"""The fragment-tier flash-softmax realizer (``assembly/_frag_softmax.py``) — CPU-only,
the structural oracle for "generate the m16n8 softmax phases from the carrier algebra".

Asserts that ``realize_fragment_softmax`` maps the ``flash_combine`` ``Monoid``'s ``merge``
program onto the right fragment ops + row-distributed scalars (the analog of the cooperative
path's ``emit_combine``), without any GPU compile.
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel.ir import FRAG, ROW, UNIFORM, FragmentApply, FragmentCausalMask, FragmentRowReduce, Reassign
from deplodock.compiler.ir.stmt import Assign, Init
from deplodock.compiler.pipeline.passes.loop.recognize._flash import flash_combine
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._frag_softmax import (
    FragGeom,
    realize_fragment_softmax,
    realize_score_mask,
)


def _geom(nd: int = 4) -> FragGeom:
    return FragGeom(
        atom_m=16,
        atom_n=8,
        score_frags=("Sf0_frag", "Sf1_frag"),
        prob_frags=("Pf0", "Pf1"),
        accum_frags=tuple(f"Of{n}" for n in range(nd)),
    )


def _carrier():
    return flash_combine("m_i", "l_i", "O_i", "s", "v")


def test_init_seeds_the_fold_identities():
    """Two stats states × 2 rows: m is a max-fold (-inf identity), l an add-fold (0)."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    inits = [s for s in fs.init if isinstance(s, Init)]
    assert len(inits) == 4, [s.pretty()[0] for s in fs.init]
    by_op = {s.name: s.op.name for s in inits}
    assert by_op == {"m_i0": "maximum", "m_i1": "maximum", "l_i0": "add", "l_i1": "add"}


def test_merge_emits_two_rowreduces_then_the_exp_map():
    """rowmax over the score frags (max), the P = exp(S - m) map, rowsum over the prob frags
    (add) — the fragment-distributed online-softmax, recovered from the scalar program. The exp
    map is now two generic ``FragmentApply``s per N-atom (subtract then exp), not the former fused
    ``FragmentExp``."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    reduces = [s for s in fs.merge if isinstance(s, FragmentRowReduce)]
    applies = [s for s in fs.merge if isinstance(s, FragmentApply)]

    assert len(reduces) == 2, [s.pretty()[0] for s in fs.merge]
    rowmax, rowsum = reduces
    # First fold reduces the SCORE fragments by max; second reduces the PROB fragments by add.
    assert rowmax.op.name == "maximum" and rowmax.frags == ("Sf0_frag", "Sf1_frag")
    assert rowsum.op.name == "add" and rowsum.frags == ("Pf0", "Pf1")

    # Per N-atom: subtract(score, m_new) then exp(...) -> prob. The exp reads a FRAG, subtracts a
    # per-row (ROW) scalar; the exp result lands in the geometry's prob fragments.
    subs = [a for a in applies if a.op.name == "subtract"]
    exps = [a for a in applies if a.op.name == "exp"]
    assert len(subs) == 2 and len(exps) == 2
    assert [a.args[0] for a in subs] == ["Sf0_frag", "Sf1_frag"]  # FRAG = the score
    assert all(a.kinds == (FRAG, ROW) for a in subs)
    assert {a.args[1] for a in subs} == {subs[0].args[1]}  # same per-row new-max for every N-atom
    assert [e.out for e in exps] == ["Pf0", "Pf1"] and all(e.kinds == (FRAG,) for e in exps)


def test_state_is_reassigned_in_merge_update_is_empty():
    """m and l are loop-carried: reassigned in-place at the end of merge (after every read of
    their old value), so the update phase is empty."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    assert fs.update == ()
    reassigned = {s.name for s in fs.merge if isinstance(s, Reassign)}
    assert reassigned == {"m_i0", "m_i1", "l_i0", "l_i1"}


def test_rescale_scales_every_accumulator_by_alpha():
    """The twist O *= alpha is one in-place ``FragmentApply`` (multiply) per D-atom; the per-row
    (ROW) alpha is shared across accumulators (the rescale is the accum carrier's only realized
    step — p·v + O += … are the P@V Mma)."""
    for nd in (2, 4, 8):
        fs = realize_fragment_softmax(_carrier(), geom=_geom(nd))
        scales = [s for s in fs.rescale if isinstance(s, FragmentApply)]
        assert len(scales) == nd
        assert all(s.op.name == "multiply" and s.in_place and s.kinds == (FRAG, ROW) for s in scales)
        assert [s.out for s in scales] == [f"Of{n}" for n in range(nd)]
        assert [s.args[0] for s in scales] == [f"Of{n}" for n in range(nd)]  # in-place: out is the first FRAG arg
        assert {s.args[1] for s in scales} == {scales[0].args[1]}  # one alpha, broadcast to every accumulator


def test_epilogue_normalizes_every_accumulator_by_the_denominator():
    """O /= l per D-atom — the add-fold state (l) is the denominator; an in-place ``FragmentApply``
    divide by the per-row denom (the former FragmentScale-by-1/l)."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom(nd=4))
    scales = [s for s in fs.epilogue if isinstance(s, FragmentApply)]
    assert len(scales) == 4
    assert all(s.op.name == "divide" and s.in_place and s.kinds == (FRAG, ROW) and s.args[1] == ("l_i0", "l_i1") for s in scales)


def test_scalar_updates_are_row_distributed():
    """Every per-row scalar stat is two scalars (rows g / g+8) — the 0/1 suffixing."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    scalar_names = {s.name for s in fs.merge if isinstance(s, Assign)}
    # the subtract (m - m_new) and the exp (alpha) both appear suffixed 0 and 1
    assert any(n.endswith("0") for n in scalar_names) and any(n.endswith("1") for n in scalar_names)
    assert all(n.endswith("0") or n.endswith("1") for n in scalar_names), scalar_names


def test_score_mask_is_one_causal_mask_per_score_frag():
    fs_geom = _geom()
    from deplodock.compiler.ir.expr import Literal, Var

    masks = realize_score_mask(fs_geom, q_row_base=Var("qb"), kv_col_bases=(Literal(0, "int"), Literal(8, "int")))
    assert [m.frag for m in masks] == ["Sf0_frag", "Sf1_frag"]
    assert all(isinstance(m, FragmentCausalMask) for m in masks)


# --- FragmentApply: the one generic pointwise node (subsumes the former FragmentExp/Scale) ---


def test_fragment_apply_renders_frag_row_and_uniform_args():
    """A FRAG arg indexes per element; a ROW arg broadcasts by row (0/1 suffix); a UNIFORM arg is
    verbatim; ``in_place`` reassigns ``out`` (no ``float`` decl). The three arg kinds are what let
    FragmentApply express every former FragmentExp / FragmentScale shape."""
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.stmt.base import RenderCtx

    mul_row = FragmentApply(out="z", op=ElementwiseImpl("multiply"), args=("x", ("a0", "a1")), kinds=(FRAG, ROW))
    src = "\n".join(mul_row.render(RenderCtx(indent=1)))
    assert "float z[4];" in src
    assert "z[0] = x[0] * a0;" in src and "z[2] = x[2] * a1;" in src
    assert mul_row.deps() == ("x", "a0", "a1")  # ROW arg exposes both per-row scalars (rename-safe)

    scale = FragmentApply(out="O", op=ElementwiseImpl("multiply"), args=("O", "0.25f"), kinds=(FRAG, UNIFORM), in_place=True)
    src2 = "\n".join(scale.render(RenderCtx(indent=1)))
    assert "float O[4];" not in src2  # in-place: no declaration
    assert "O[0] = O[0] * 0.25f;" in src2 and "O[3] = O[3] * 0.25f;" in src2


def test_classify_emits_frag_apply_for_a_generic_op():
    """The op cap is gone: a fragment-producing op the softmax vocabulary never had (here ``tanh``)
    classifies as a generic ``frag_apply`` step instead of raising NotImplementedError."""
    from deplodock.compiler.ir.stmt import Assign
    from deplodock.compiler.ir.stmt.carrier_algebra import classify_merge_program

    steps, frag = classify_merge_program((Assign("p", "tanh", ("s",)),), "s", state_names=())
    assert "p" in frag
    assert [(st.role, st.op.name) for st in steps] == [("frag_apply", "tanh")]


def _carrier_with_generic_op():
    """A twisted ``(m, l, O)`` carrier whose STATS merge applies a generic fragment op (``relu``) to
    the score before the fold — to exercise the realizer's ``frag_apply`` branch with a real
    carrier (the algebra is contrived, only the op vocabulary matters)."""
    from deplodock.compiler.ir.expr import Literal
    from deplodock.compiler.ir.stmt import Assign, Monoid

    merge = (
        Assign("sq", "relu", ("s",)),  # a generic fragment op (not exp / not a fold)
        Assign("mx", "maximum", ("m_i", "sq")),  # rowmax over the relu'd score
        Assign("ds", "subtract", ("sq", "mx")),
        Assign("p", "exp", ("ds",)),  # fused exp -> prob
        Assign("l_i", "add", ("l_i", "p")),  # denom (add-fold state)
        Assign("pv", "multiply", ("p", "v")),
        Assign("O_i", "add", ("O_i", "pv")),  # value-dependent accumulation
        Assign("m_i", "copy", ("mx",)),  # state, last
    )
    return Monoid(
        state=("m_i", "l_i", "O_i"),
        partial=("s", "v"),
        merge=merge,
        identity=(Literal(-1e30), Literal(0.0), Literal(0.0)),
        commutative=True,
        axes=("kv",),
    )


def test_realizer_emits_frag_apply_for_a_generic_carrier_op():
    """The realizer turns a generic stats-merge fragment op (``relu`` of the score, one per N-atom)
    into a ``FragmentApply`` — carrier-vocabulary generality reaching the tensor-core tier, the
    same realizer that handles softmax's exp/fold."""
    fs = realize_fragment_softmax(_carrier_with_generic_op(), geom=_geom())
    relus = [s for s in fs.merge if isinstance(s, FragmentApply) and s.op.name == "relu"]
    assert [r.out for r in relus] == ["sq_0", "sq_1"]  # one per N-atom (the 2 score frags)
    assert [r.args for r in relus] == [("Sf0_frag",), ("Sf1_frag",)]
    assert all(r.kinds == (FRAG,) for r in relus)
