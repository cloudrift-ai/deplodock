"""Unit tests for ``Body.deps_closure`` / ``depends_on`` / ``independent``.

Exercises the dataflow-dependency primitives on small hand-built
bodies — softmax-style reduces, sibling cross-loop reads, hoisted
reciprocals — that mirror the patterns the SDPA-reduce kernel
gates 013/015 care about.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop
from emmy.compiler.ir.stmt.body import Body


def _ld(name: str, src: str, *idx: str) -> Load:
    return Load(name=name, input=src, index=tuple(Var(v) for v in idx))


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _acc(name: str, value: str, op: str = "add") -> Accum:
    return Accum(name=name, value=value, op=op)


# --- closure shape ---------------------------------------------------


def test_load_closure_includes_index_axis_vars():
    body = Body([_ld("v", "x", "k")])
    closure = body.deps_closure
    assert closure["v"] == frozenset({"k"})


def test_assign_closure_unions_args():
    body = Body(
        [
            _ld("a", "x", "i"),
            _ld("b", "y", "j"),
            _asn("c", "add", "a", "b"),
        ]
    )
    closure = body.deps_closure
    assert closure["c"] == frozenset({"a", "b", "i", "j"})


def test_accum_outside_form_subtracts_loop_axis():
    """``acc <- add(in0)`` inside ``Loop k`` exposes ``acc`` to the outer
    scope with ``k`` subtracted (reduced result no longer varies with k)."""
    inner = (_ld("in0", "x", "k"), _acc("acc", "in0", "add"))
    body = Body([Loop(axis=Axis("k", 32), body=inner)])
    closure = body.deps_closure
    assert closure["in0"] == frozenset({"k"})
    assert closure["acc"] == frozenset()  # k subtracted


def test_accum_outer_axis_kept():
    """An outer free axis remains in ``acc``'s closure even when its
    immediate reduce axis is subtracted."""
    inner = (
        Loop(
            axis=Axis("k", 32),
            body=(_ld("in0", "x", "a", "k"), _acc("acc", "in0", "add")),
        ),
    )
    body = Body([Loop(axis=Axis("a", 8), body=inner)])
    closure = body.deps_closure
    assert closure["acc"] == frozenset({"a"})


# --- depends_on ------------------------------------------------------


def test_depends_on_axis_one_way():
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("v", "k") is True
    assert body.depends_on("k", "v") is False  # k has no closure entry


def test_depends_on_set_args():
    body = Body(
        [
            _ld("a", "x", "i"),
            _ld("b", "y", "j"),
            _asn("c", "add", "a", "b"),
        ]
    )
    assert body.depends_on("c", {"i"}) is True
    assert body.depends_on("c", {"i", "j"}) is True
    assert body.depends_on("c", {"z"}) is False
    assert body.depends_on({"a", "b"}, "i") is True
    assert body.depends_on({"a"}, "j") is False


def test_depends_on_self_in_b():
    """A name listed in both ``a`` and ``b`` short-circuits true:
    ``depends_on(x, {x, ...})`` reads as 'is x related to anything in b?'"""
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("v", {"v", "z"}) is True


def test_depends_on_external_name_empty_closure():
    """External names (Tile-input buffers, names from enclosing scopes)
    aren't keys in the closure — depends_on treats them as having empty
    deps. Symmetric: querying with an external as ``a`` returns False."""
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("ext", "k") is False


def test_depends_on_empty_set():
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on(set(), "k") is False
    assert body.depends_on("v", set()) is False


# --- independent -----------------------------------------------------


def test_independent_unrelated():
    body = Body(
        [
            _ld("a", "x", "i"),
            _ld("b", "y", "j"),
        ]
    )
    assert body.independent("a", "b") is True
    assert body.independent("a", "j") is True
    assert body.independent("a", "i") is False  # a depends on i


def test_independent_symmetric():
    """If a depends on b OR b depends on a, they're not independent."""
    body = Body(
        [
            _ld("a", "x", "k"),
            _asn("b", "exp", "a"),  # b reads a
        ]
    )
    assert body.independent("a", "b") is False
    assert body.independent("b", "a") is False  # symmetric


# --- SDPA-reduce-style sibling-reduce pattern ------------------------


def test_sibling_reduce_invariance():
    """Mirror SDPA-reduce: first reduce produces ``acc0``, second reduce
    reads ``acc0`` cross-loop. ``acc0`` is loop-invariant w.r.t. the
    second reduce's axis, so depends_on(acc0, k2) should be False."""
    body = Body(
        [
            Loop(
                axis=Axis("k1", 32),
                body=(_ld("in0", "x", "k1"), _acc("acc0", "in0", "max")),
            ),
            Loop(
                axis=Axis("k2", 32),
                body=(
                    _ld("in1", "x", "k2"),
                    _asn("v0", "subtract", "in1", "acc0"),  # cross-loop read of acc0
                    _asn("v1", "exp", "v0"),
                    _acc("acc1", "v1", "add"),
                ),
            ),
        ]
    )
    closure = body.deps_closure
    # acc0 outside-form: closure[in0]={k1} minus {k1} = {} → invariant
    assert closure["acc0"] == frozenset()
    assert body.depends_on("acc0", "k2") is False
    # v0 inside the second loop reads in1 (which has k2) and acc0 (no k2).
    assert body.depends_on("v0", "k2") is True
    assert body.depends_on("v0", "k1") is False
