"""The atom layer's ``atomize_cell`` move in isolation — the fixed contract.

``plans/tensor-core-streaming-flash-mma.md`` Phase 0 factored the ``atomize`` body
edit out of ``_build.warp_build``'s matmul-staging geometry into ``_atom`` so it is
an independently-testable unit, callable with operands of **any provenance** (it
names A / B by SSA value). These tests pin that contract — a canonical matmul cell
``[Load, Load, Assign(mul), Accum]`` fuses to one ``Mma`` with the right A / B / C
SSA names, transposed-B flag, and atom spec; the move walks a ``SerialTile`` reduce
tower to the cell; a non-cell body is returned untouched. Pure structural shapes, no
planner / GPU (the PTX the ``Mma`` lowers to is covered by the matmul / transposed-B
e2e suites). The contract must hold before the Phase-2 flash nest calls the unit on
its register-fragment inner contractions.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Mma, Write
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, SerialTile
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell

_ATOM = ATOM_REGISTRY["mma_m16n8k16_f16"]


def _cell(b_index) -> tuple:
    """The canonical matmul cell ``[Load a, Load b, p = a*b, acc += p]`` with B
    indexed ``b_index`` (``(Var k, Var j)`` canonical K-major / ``(Var j, Var k)``
    transposed N×K)."""
    return (
        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
        Load(name="b_v", input="b", index=b_index),
        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
        Accum(name="acc", value="p"),
    )


def _only_mma(stmts) -> Mma:
    mmas = [s for s in Body.coerce(stmts).iter_of_type(Mma)]
    assert len(mmas) == 1, f"expected exactly one Mma, got {mmas}"
    return mmas[0]


def test_canonical_cell_fuses_to_mma():
    """B indexed ``(k, j)`` (K in the first dim) → ``Mma(c=acc, a=a_v, b=b_v)``,
    ``b_trans=False``, the atom spec carried, the operand Loads kept plain."""
    out = atomize_cell(_cell((Var("k"), Var("j"))), atom=_ATOM, k_name="k", write=None)
    mma = _only_mma(out)
    assert (mma.c, mma.a, mma.b) == ("acc", "a_v", "b_v")
    assert mma.b_trans is False
    assert mma.atom is _ATOM
    # The fuse replaces the mul + Accum with the Mma but keeps the operand Loads.
    assert [s.name for s in out if isinstance(s, Load)] == ["a_v", "b_v"]
    assert not any(isinstance(s, Assign) for s in out)
    assert not any(isinstance(s, Accum) for s in out)


def test_transposed_b_cell_sets_b_trans():
    """B indexed ``(j, k)`` (Q@K^T — both operands carry K last) needs the Write's
    M / N coordinates to disambiguate A from B; the N×K operand sets ``b_trans``."""
    write = Write(output="c", index=(Var("i"), Var("j")), value="acc")
    out = atomize_cell(_cell((Var("j"), Var("k"))), atom=_ATOM, k_name="k", write=write)
    mma = _only_mma(out)
    assert (mma.c, mma.a, mma.b) == ("acc", "a_v", "b_v")
    assert mma.b_trans is True


def test_walks_serial_tile_reduce_tower_to_cell():
    """Called with ``k_name=None`` (as ``warp_build`` does), the move recurses
    through a ``SerialTile`` reduce tower and reads the K name off the reduce axis."""
    cell = _cell((Var("k"), Var("j")))
    tower = (SerialTile(axis=Axis("k", 16), body=Body(cell)),)
    out = atomize_cell(tower, atom=_ATOM, k_name=None, write=Write(output="c", index=(Var("i"), Var("j")), value="acc"))
    mma = _only_mma(out)
    assert (mma.c, mma.a, mma.b) == ("acc", "a_v", "b_v")


def test_non_cell_body_returned_untouched():
    """A body that is not the canonical cell (two Accums) fuses nothing — no Mma,
    the statements pass through unchanged."""
    body = (
        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
        Assign(name="p", op=ElementwiseImpl("copy"), args=("a_v",)),
        Accum(name="acc", value="p"),
    )
    out = atomize_cell(body, atom=_ATOM, k_name="k", write=None)
    assert not any(isinstance(s, Mma) for s in out)
    assert out == body
