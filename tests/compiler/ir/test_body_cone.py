"""Unit tests for ``Body.backward_cone`` / ``forward_cone`` / ``defs_die_at``.

The cone API is the shared dataflow substrate behind the rules that used to
hand-roll slicing (``_split_demoted``, ``021_hoist_staged_loads_above_mask``)
— see ``passes/ARCHITECTURE.md``. Construction never fails: unresolved names
are ``external_reads``, and every eligibility judgment stays in the rule.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Select, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import SelectBranch


def _ld(name: str, src: str, *idx: str) -> Load:
    return Load(name=name, input=src, index=tuple(Var(v) for v in idx))


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _cell() -> Body:
    """A demoted-matmul cell: ``p = cone(x) * w`` with a plain B load."""
    return Body(
        [
            _ld("a0", "x", "m", "k"),
            _asn("a1", "negative", "a0"),
            _ld("b", "w", "k", "n"),
            _asn("p", "multiply", "a1", "b"),
            Accum(name="acc", value="p"),
        ]
    )


# --- backward_cone ----------------------------------------------------


def test_backward_cone_members_in_body_order():
    body = _cell()
    cone = body.backward_cone(["a1"])
    assert [s.defines()[0] for s in cone.members] == ["a0", "a1"]


def test_backward_cone_externals_carry_axes_not_members():
    cone = _cell().backward_cone(["a1"])
    # The cone reads its index axes from the enclosing scope; the resolved
    # SSA chain (a0) is internal, the untouched B side absent entirely.
    assert cone.external_reads == frozenset({"m", "k"})


def test_backward_cone_root_not_defined_here_is_external():
    """A root from another scope level yields no members and surfaces as an
    external read — the chaining signal, not an error."""
    cone = _cell().backward_cone(["stat"])
    assert cone.members == ()
    assert cone.external_reads == frozenset({"stat"})


def test_backward_cone_wrapper_member_joins_whole():
    """A reduce Loop resolves through its exposed Accum and joins as one
    member; its own loop axis is internal, the row axis it reads is not."""
    stat_loop = Loop(axis=Axis("r", 32), body=(_ld("s0", "x", "m", "r"), Accum(name="stat", value="s0", op="maximum")))
    body = Body([stat_loop, _asn("e", "exp", "stat")])
    cone = body.backward_cone(["e"])
    assert cone.members == (stat_loop, body[1])
    assert "r" not in cone.external_reads
    assert "m" in cone.external_reads


def test_backward_cone_select_predicate_axis_is_read():
    """An axis reaching the cone only through a Select predicate counts as
    read — materializing without it would drop a real dependence (the
    rotate-half rotary shape)."""
    body = Body(
        [
            _ld("a0", "x", "k"),
            _ld("a1", "x", "m"),
            Select(name="v", branches=(SelectBranch(value="a0", select=Var("n").lt(Var("m"))), SelectBranch(value="a1", select=Var("m")))),
        ]
    )
    cone = body.backward_cone(["v"])
    assert "n" in cone.external_reads


# --- forward_cone -----------------------------------------------------


def test_forward_cone_taints_transitive_readers():
    body = _cell()
    cone = body.forward_cone([body[0]])  # seed: the unsafe Load a0
    assert [s.defines()[0] for s in cone.members] == ["a0", "a1", "p", "acc"]
    # b is read by a member but produced outside the cone.
    assert "b" in cone.external_reads


def test_forward_cone_skips_independent_stmts():
    body = _cell()
    cone = body.forward_cone([body[2]])  # seed: the B-side Load
    assert [s.defines()[0] for s in cone.members] == ["b", "p", "acc"]


# --- defs_die_at ------------------------------------------------------


def test_defs_die_at_allows_designated_consumer():
    body = _cell()
    cone = body.backward_cone(["a1"])
    assert body.defs_die_at(cone.members, roots=["a1"], allowed=[body[3]]) is True


def test_defs_die_at_rejects_escaping_read():
    body = Body([*_cell(), Write(output="leak", index=(Var("m"),), values=("a1",))])
    cone = body.backward_cone(["a1"])
    assert body.defs_die_at(cone.members, roots=["a1"], allowed=[body[3]]) is False


def test_defs_die_at_rejects_consumer_reading_past_root():
    """The allowed consumer may read only the roots — a multiply that also
    reads a cone-internal temp blocks the cut."""
    body = Body(
        [
            _ld("a0", "x", "m", "k"),
            _asn("a1", "negative", "a0"),
            _ld("b", "w", "k", "n"),
            _asn("p", "multiply", "a1", "a0"),
            Accum(name="acc", value="p"),
        ]
    )
    cone = body.backward_cone(["a1"])
    assert body.defs_die_at(cone.members, roots=["a1"], allowed=[body[3]]) is False


def test_defs_die_at_member_subtree_reads_are_internal():
    """A wrapper member's internal uses of moved names don't count as
    escapes — the whole subtree moves."""
    body = Body(
        [
            _asn("c0", "negative", "z"),
            Loop(axis=Axis("k", 8), body=(_ld("c1", "x", "k"), _asn("c2", "multiply", "c1", "c0"), Accum(name="acc", value="c2"))),
            _asn("out", "exp", "acc"),
        ]
    )
    cone = body.backward_cone(["acc"])
    assert body[0] in cone.members and body[1] in cone.members
    assert body.defs_die_at(cone.members, roots=["acc"], allowed=[body[2]]) is True
