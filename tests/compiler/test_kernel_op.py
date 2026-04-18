"""Tests for the structural LoopOp IR with SSA body (Assign/Update/Write/Select)."""

import pytest

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import (
    Assign,
    Axis,
    LocalBuffer,
    LoopOp,
    Port,
    Select,
    SelectBranch,
    Update,
    Write,
    flat_body_to_nested,
    flatten_body,
)
from deplodock.compiler.ir.tensor import ElementwiseOp


def _loop(*, axes=(), inputs=(), locals=(), body=()):
    """Build a LoopOp from a flat body + axes hint.

    Test-local shim: LoopOp.axes is now a property over the body's Loop tree,
    so the constructor no longer accepts axes=. Fixtures that still think in
    terms of (axes, flat_body) use this helper to get the nested body form
    the IR requires.
    """
    return LoopOp(inputs=inputs, locals=locals, body=flat_body_to_nested(axes, body))


# ---------------------------------------------------------------------------
# Axis / Port
# ---------------------------------------------------------------------------


def test_axis_free():
    a = Axis("a0", 8, "free")
    assert a.name == "a0"
    assert a.extent == 8
    assert a.kind == "free"


def test_axis_reduce():
    a = Axis("a1", 4, "reduce")
    assert a.kind == "reduce"


def test_port_default_is_empty_index():
    assert Port().index == ()


def test_port_with_index():
    p = Port(index=(Var("a0"), Var("a1")))
    assert len(p.index) == 2


def test_port_broadcast_literal():
    p = Port(index=(Literal(0, "int"), Var("a0")))
    assert isinstance(p.index[0], Literal)


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("$0", "$1"))
    assert a.name == "out"
    assert a.op.fn == "add"
    assert a.args == ("$0", "$1")


# ---------------------------------------------------------------------------
# LoopOp construction
# ---------------------------------------------------------------------------


def _pointwise_axes(n: int) -> tuple[Axis, ...]:
    return tuple(Axis(f"a{i}", 4, "free") for i in range(n))


def test_kernel_pointwise():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("z", ElementwiseOp("add"), args=("$0", "$1")),
            Write(output=0, index=(Var("a0"),), value="z"),
        ),
    )
    assert len(flatten_body(k.body)) == 2


def test_kernel_reduce():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p,),
        locals=(LocalBuffer(name="s", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Update(target="s", value="$0"),
            Write(output=0, index=(Var("a0"),), value="s"),
        ),
    )
    assert any(isinstance(lb.combine, ElementwiseOp) for lb in k.locals)


def test_kernel_matmul():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        locals=(LocalBuffer(name="dot", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Update(target="dot", value="mul"),
            Write(output=0, index=(Var("a0"),), value="dot"),
        ),
    )
    assert len(flatten_body(k.body)) == 3


def test_kernel_matmul_bias():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p_mk = Port(index=(Var("a0"), Var("a1")))
    p_bias = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p_mk, p_mk, p_bias),
        locals=(LocalBuffer(name="dot", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Update(target="dot", value="mul"),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
            Write(output=0, index=(Var("a0"),), value="out"),
        ),
    )
    assert len(flatten_body(k.body)) == 4


def test_kernel_softmax_two_accumulators():
    # Under nested SSA scoping, per-segment SSA names (sub, ex) are scoped to
    # their own reduce Loop — the old flat fixture relied on laxness. Here we
    # just verify a kernel with two accumulators + two Updates builds fine.
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p,),
        locals=(
            LocalBuffer(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
            LocalBuffer(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
        ),
        body=(
            Update(target="mx", value="$0"),
            Update(target="sm", value="$0"),
            Write(output=0, index=(Var("a0"),), value="mx"),
        ),
    )
    assert any(lb.name == "mx" for lb in k.locals)
    assert any(lb.name == "sm" for lb in k.locals)


def test_kernel_unary_chain():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("neg", ElementwiseOp("neg"), args=("$0",)),
            Assign("exp", ElementwiseOp("exp"), args=("neg",)),
            Write(output=0, index=(Var("a0"),), value="exp"),
        ),
    )
    assert len(flatten_body(k.body)) == 3


def test_kernel_scatter_output_via_select():
    """Select replaces Mux for coord-predicated dispatch on output."""
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("z", ElementwiseOp("add"), args=("$0", "$1")),
            Select(
                name="v",
                branches=(
                    SelectBranch(value="$0", select=Var("c1")),
                    SelectBranch(value="$1", select=Var("c2")),
                ),
            ),
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )
    assert isinstance(flatten_body(k.body)[1], Select)


# ---------------------------------------------------------------------------
# Validator — SSA / accumulator / v1 pins
# ---------------------------------------------------------------------------


def test_rejects_duplicate_axis_name():
    # Two same-named nested Loops form a shadow; the validator rejects.
    with pytest.raises(ValueError, match="shadows"):
        _loop(axes=(Axis("a0", 4, "free"), Axis("a0", 8, "free")))


def test_ssa_rejects_undefined_arg():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="not defined"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(Assign("y", ElementwiseOp("exp"), args=("z",)),),
        )


def test_ssa_rejects_duplicate_name():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="already defined"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("y", ElementwiseOp("exp"), args=("$0",)),
                Assign("y", ElementwiseOp("neg"), args=("y",)),
            ),
        )


def test_ssa_rejects_forward_reference():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="not defined"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("a", ElementwiseOp("add"), args=("$0", "b")),
                Assign("b", ElementwiseOp("exp"), args=("$0",)),
            ),
        )


def test_ssa_allows_input_name_reuse_in_multiple_args():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("a", ElementwiseOp("exp"), args=("$0",)),
            Assign("b", ElementwiseOp("neg"), args=("$0",)),
            Assign("c", ElementwiseOp("add"), args=("a", "b")),
        ),
    )
    assert len(flatten_body(k.body)) == 3


def test_rejects_update_without_matching_local():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="does not name a LocalBuffer"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_accumulator_without_reduce_axis():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="no reduce axis"):
        _loop(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_reduce_axis_without_accumulator():
    from deplodock.compiler.ir.loop import Loop

    a0 = Axis("a0", 4, "free")
    a1 = Axis("a1", 8, "reduce")
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="no accumulator"):
        LoopOp(
            inputs=(p,),
            body=(
                Loop(
                    axis=a0,
                    body=(Loop(axis=a1, body=(Assign("a", ElementwiseOp("exp"), args=("$0",)),)),),
                ),
            ),
        )


def test_rejects_shaped_localbuffer_in_v1():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="shape=\\(\\)"):
        _loop(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0), shape=(4,)),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_nonthread_scope_in_v1():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="scope='thread'"):
        _loop(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0), scope="block"),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_nondense_output_indices():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="dense"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("z", ElementwiseOp("neg"), args=("$0",)),
                Write(output=2, index=(Var("a0"),), value="z"),
            ),
        )


def test_rejects_unused_accumulator():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="never Updated"):
        _loop(
            axes=axes,
            inputs=(p,),
            locals=(
                LocalBuffer(name="a", combine=ElementwiseOp("add"), init=Literal(0.0)),
                LocalBuffer(name="b", combine=ElementwiseOp("add"), init=Literal(0.0)),
            ),
            body=(Update(target="a", value="$0"),),
        )


# ---------------------------------------------------------------------------
# Loop Stmt — nested body form
# ---------------------------------------------------------------------------


def test_loop_stmt_basic():
    from deplodock.compiler.ir.loop import Loop

    # Pointwise kernel with body wrapped in a free Loop.
    loop = _loop(
        axes=(Axis("a0", 8, "free"),),
        inputs=(Port(index=(Var("a0"),)),),
        body=(
            Loop(
                axis=Axis("a0", 8, "free"),
                body=(
                    Assign("v", ElementwiseOp("neg"), args=("$0",)),
                    Write(output=0, index=(Var("a0"),), value="v"),
                ),
            ),
        ),
    )
    assert isinstance(loop.body[0], Loop)
    assert loop.body[0].axis.name == "a0"


def test_loop_stmt_reduce_kernel():
    from deplodock.compiler.ir.loop import Loop

    # Reduce kernel — free a0 outer, reduce k inner.
    loop = _loop(
        axes=(Axis("a0", 4, "free"), Axis("k", 8, "reduce")),
        inputs=(Port(index=(Var("a0"), Var("k"))),),
        locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Loop(
                axis=Axis("a0", 4, "free"),
                body=(
                    Loop(
                        axis=Axis("k", 8, "reduce"),
                        body=(Update(target="acc", value="$0"),),
                    ),
                    Write(output=0, index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )
    # Validator accepts the nested form.
    from deplodock.compiler.ir.loop import iter_loops

    loops = iter_loops(loop.body)
    assert len(loops) == 2
    assert loops[0].axis.name == "a0"
    assert loops[1].axis.name == "k"


def test_loop_stmt_softmax_sibling_reduces():
    from deplodock.compiler.ir.loop import Loop

    # Softmax shape: two sibling reduce Loops over the same axis "k", inside
    # a single outer free iteration. Sibling same-name is legal.
    loop = _loop(
        axes=(Axis("a0", 4, "free"), Axis("a1", 8, "free"), Axis("k", 8, "reduce")),
        inputs=(Port(index=(Var("a0"), Var("a1"))), Port(index=(Var("a0"), Var("k"))), Port(index=(Var("a0"), Var("k")))),
        locals=(
            LocalBuffer(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
            LocalBuffer(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
        ),
        body=(
            Loop(
                axis=Axis("a0", 4, "free"),
                body=(
                    Loop(
                        axis=Axis("a1", 8, "free"),
                        body=(
                            Loop(axis=Axis("k", 8, "reduce"), body=(Update(target="mx", value="$1"),)),
                            Loop(axis=Axis("k", 8, "reduce"), body=(Update(target="sm", value="$2"),)),
                            Assign("v", ElementwiseOp("add"), args=("mx", "sm")),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="v"),
                        ),
                    ),
                ),
            ),
        ),
    )
    from deplodock.compiler.ir.loop import iter_loops

    reduce_loops = [L for L in iter_loops(loop.body) if L.axis.kind == "reduce"]
    assert len(reduce_loops) == 2
    assert all(L.axis.name == "k" for L in reduce_loops)


def test_loop_axis_nested_shadow_rejected():
    """Nested Loop inside another Loop with the same axis name is ambiguous."""
    from deplodock.compiler.ir.loop import Loop

    with pytest.raises(ValueError, match="shadows"):
        _loop(
            axes=(Axis("a0", 4, "free"), Axis("k", 8, "reduce")),
            inputs=(Port(index=(Var("a0"), Var("k"))),),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
            body=(
                Loop(
                    axis=Axis("k", 8, "reduce"),  # outer reduce k
                    body=(
                        Loop(
                            axis=Axis("k", 8, "reduce"),  # nested reduce also named k — shadow
                            body=(Update(target="acc", value="$0"),),
                        ),
                    ),
                ),
                Write(output=0, index=(Var("a0"),), value="acc"),
            ),
        )


def test_loop_axis_sibling_same_name_allowed():
    """Sibling Loops with the same axis name are fine — that's softmax's two K-loops."""
    from deplodock.compiler.ir.loop import Loop

    # Two reduce Loops over the same axis, sibling (not nested).
    _loop(
        axes=(Axis("a0", 4, "free"), Axis("k", 8, "reduce")),
        inputs=(Port(index=(Var("a0"), Var("k"))), Port(index=(Var("a0"), Var("k")))),
        locals=(
            LocalBuffer(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
            LocalBuffer(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
        ),
        body=(
            Loop(axis=Axis("k", 8, "reduce"), body=(Update(target="mx", value="$0"),)),
            Loop(axis=Axis("k", 8, "reduce"), body=(Update(target="sm", value="$1"),)),
            Assign("v", ElementwiseOp("add"), args=("mx", "sm")),
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )


def test_loop_ssa_scoping():
    """Assign defined inside Loop body is invisible outside — caller Assign must fail."""
    from deplodock.compiler.ir.loop import Loop

    # A Loop body defines "v"; a sibling statement tries to reference it.
    with pytest.raises(ValueError, match="not defined"):
        _loop(
            axes=(Axis("a0", 4, "free"),),
            inputs=(Port(index=(Var("a0"),)),),
            body=(
                Loop(
                    axis=Axis("a0_inner", 4, "free"),
                    body=(Assign("v", ElementwiseOp("neg"), args=("$0",)),),
                ),
                # Reference 'v' outside the Loop — should fail.
                Write(output=0, index=(Var("a0"),), value="v"),
            ),
        )


# ---------------------------------------------------------------------------
# Shim: flat → nested
# ---------------------------------------------------------------------------


def test_flat_body_to_nested_pointwise():
    from deplodock.compiler.ir.loop import Loop

    axes = (Axis("a0", 8, "free"),)
    flat = (
        Assign("v", ElementwiseOp("neg"), args=("$0",)),
        Write(output=0, index=(Var("a0"),), value="v"),
    )
    nested = flat_body_to_nested(axes, flat)
    assert len(nested) == 1
    assert isinstance(nested[0], Loop) and nested[0].axis.name == "a0"
    inner = nested[0].body
    assert len(inner) == 2
    assert isinstance(inner[0], Assign)
    assert isinstance(inner[1], Write)


def test_flat_body_to_nested_reduce_splits_at_update():
    from deplodock.compiler.ir.loop import Loop

    axes = (Axis("a0", 4, "free"), Axis("k", 8, "reduce"))
    flat = (
        Update(target="acc", value="$0"),
        Write(output=0, index=(Var("a0"),), value="acc"),
    )
    nested = flat_body_to_nested(axes, flat)
    assert len(nested) == 1 and isinstance(nested[0], Loop) and nested[0].axis.name == "a0"
    inner = nested[0].body
    assert len(inner) == 2
    assert isinstance(inner[0], Loop) and inner[0].axis.name == "k"
    assert isinstance(inner[0].body[0], Update)
    assert isinstance(inner[1], Write)


def test_flat_body_to_nested_idempotent():
    from deplodock.compiler.ir.loop import Loop

    already_nested = (
        Loop(
            axis=Axis("a0", 8, "free"),
            body=(
                Assign("v", ElementwiseOp("neg"), args=("$0",)),
                Write(output=0, index=(Var("a0"),), value="v"),
            ),
        ),
    )
    assert flat_body_to_nested((Axis("a0", 8, "free"),), already_nested) == already_nested


def test_flat_body_to_nested_softmax_two_updates():
    from deplodock.compiler.ir.loop import Loop

    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "free"), Axis("k", 8, "reduce"))
    flat = (
        Update(target="mx", value="$1"),
        Assign("v_mx", ElementwiseOp("neg"), args=("mx",)),
        Update(target="sm", value="v_mx"),
        Write(output=0, index=(Var("a0"), Var("a1")), value="sm"),
    )
    nested = flat_body_to_nested(axes, flat)
    # Structure: Loop(a0, [Loop(a1, [Loop(k, [Update mx]), Loop(k, [Assign, Update sm]), Write])])
    assert len(nested) == 1 and nested[0].axis.name == "a0"
    inner_a1 = nested[0].body[0]
    assert isinstance(inner_a1, Loop) and inner_a1.axis.name == "a1"
    inner_stmts = inner_a1.body
    reduce_loops = [s for s in inner_stmts if isinstance(s, Loop)]
    assert len(reduce_loops) == 2
    assert all(L.axis.name == "k" for L in reduce_loops)
    writes = [s for s in inner_stmts if isinstance(s, Write)]
    assert len(writes) == 1


# ---------------------------------------------------------------------------
# Phase 2: consumers handle nested bodies
# ---------------------------------------------------------------------------


def _flat_reduce_kernel() -> LoopOp:
    """Flat-body reduce kernel for parity comparison."""
    return _loop(
        axes=(Axis("a0", 4, "free"), Axis("k", 8, "reduce")),
        inputs=(Port(index=(Var("a0"), Var("k"))),),
        locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Update(target="acc", value="$0"),
            Write(output=0, index=(Var("a0"),), value="acc"),
        ),
    )


def _nested_reduce_kernel() -> LoopOp:
    """Same kernel as _flat_reduce_kernel, but body is nested via explicit Loops."""
    from deplodock.compiler.ir.loop import Loop

    return LoopOp(
        inputs=(Port(index=(Var("a0"), Var("k"))),),
        locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Loop(
                axis=Axis("a0", 4, "free"),
                body=(
                    Loop(axis=Axis("k", 8, "reduce"), body=(Update(target="acc", value="$0"),)),
                    Write(output=0, index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )


def test_analyze_kernel_handles_nested_body():
    """analyze_kernel should produce the same plan for flat and nested forms."""
    from deplodock.compiler.ir.loop_plan import analyze_kernel

    flat = _flat_reduce_kernel()
    nested = _nested_reduce_kernel()
    shapes = {"$0": (4, 8)}
    out_shape = (4,)
    plan_flat = analyze_kernel(flat, shapes, out_shape)
    plan_nested = analyze_kernel(nested, shapes, out_shape)
    # Both should produce a Loop step (reduce) and a trailing Write.
    assert len(plan_flat.steps) == len(plan_nested.steps)
    assert len(plan_flat.trailing_writes) == len(plan_nested.trailing_writes)


def test_execute_loop_op_handles_nested_body():
    """Numpy interpreter should produce the same output for flat and nested forms."""
    import numpy as np

    from deplodock.compiler.backend.loop.backend import execute_loop_op

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)

    flat = _flat_reduce_kernel()
    nested = _nested_reduce_kernel()
    out_flat = execute_loop_op(flat, [x], (4,))
    out_nested = execute_loop_op(nested, [x], (4,))
    np.testing.assert_allclose(out_flat, out_nested, rtol=1e-5)
