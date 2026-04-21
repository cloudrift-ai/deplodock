"""Tests for the structural LoopOp IR with SSA body (Assign/Accum/Write/Select)."""

import pytest

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Select,
    SelectBranch,
    Write,
    flat_body_to_nested,
    flatten_body,
)
from deplodock.compiler.ir.tensor_ir import ElementwiseOp


def Port(index=()):
    """Back-compat test shim — Port is gone at the IR level, but fixtures
    still spell reads as ``Port(index=...)``. The shim just returns the
    index tuple for ``_loop`` to synthesize body-form Loads from."""
    return tuple(index)


def _loop(*, axes=(), inputs=(), body=()):
    """Build a LoopOp from a flat body + axes hint.

    Test-local shim: ``LoopOp.axes`` is a property over the body's Loop
    tree, so ``axes=`` feeds ``flat_body_to_nested`` to produce the
    nested form. When ``inputs=(...)`` is provided (legacy Port-tuples),
    each ``$N`` ref is rewritten to a freshly-named Load that is inserted
    just before the referencing statement. Load cache resets at each
    ``Accum`` and at each nested ``Loop`` boundary so sibling reduce
    sweeps (softmax pattern) get their own scope-local Loads.
    """
    if not inputs:
        return LoopOp(body=flat_body_to_nested(axes, body))

    index_of = {i: tuple(p) for i, p in enumerate(inputs)}
    fresh_counter = [0]

    def process(stmts):
        """Rewrite ``$N`` refs in ``stmts`` and insert Loads at first use."""
        local_loads: dict[int, str] = {}
        extra_loads: list = []

        def rewrite_arg(a: str) -> str:
            if not a.startswith("$"):
                return a
            try:
                src = int(a[1:])
            except ValueError:
                return a
            cached = local_loads.get(src)
            if cached is not None:
                return cached
            fresh_counter[0] += 1
            name = f"in{src}_{fresh_counter[0]}"
            local_loads[src] = name
            extra_loads.append(Load(name=name, source=src, index=index_of[src]))
            return name

        result: list = []
        for stmt in stmts:
            if isinstance(stmt, Assign):
                stmt = Assign(stmt.name, stmt.op, tuple(rewrite_arg(a) for a in stmt.args))
            elif isinstance(stmt, Accum):
                stmt = Accum(stmt.name, rewrite_arg(stmt.value), stmt.op)
            elif isinstance(stmt, Write):
                stmt = Write(stmt.output, stmt.index, rewrite_arg(stmt.value))
            elif isinstance(stmt, Select):
                stmt = Select(
                    stmt.name,
                    tuple(SelectBranch(rewrite_arg(b.value), b.select) for b in stmt.branches),
                )
            elif isinstance(stmt, Loop):
                # Nested Loop: its body is a fresh scope — recursive call has
                # its own ``local_loads`` / ``extra_loads`` so Loads land
                # inside the Loop body.
                stmt = Loop(axis=stmt.axis, body=process(stmt.body))
            result.extend(extra_loads)
            extra_loads = []
            result.append(stmt)
            if isinstance(stmt, Accum):
                local_loads = {}

        return tuple(result)

    return LoopOp(body=flat_body_to_nested(axes, process(body)))


# ---------------------------------------------------------------------------
# Axis / Port
# ---------------------------------------------------------------------------


def test_axis_construction():
    a = Axis("a0", 8)
    assert a.name == "a0"
    assert a.extent == 8


# ---------------------------------------------------------------------------
# Body-form Load / Accum (new IR: inputs + accumulators as statements)
# ---------------------------------------------------------------------------


def test_load_stmt_binding():
    """Load introduces an SSA name; LoopOp.loads collects them in pre-order."""
    k = LoopOp(
        body=(
            Loop(
                axis=Axis("a0", 4),
                body=(
                    Load("x_val", source=0, index=(Var("a0"),)),
                    Assign("y", ElementwiseOp("neg"), ("x_val",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    loads = k.loads
    # rename_ssa_sequential canonicalizes Load names to in0, in1, ...
    assert len(loads) == 1 and loads[0].name == "in0" and loads[0].source == 0
    assert k.num_inputs == 1


def test_load_stmt_multiple_sources():
    """num_inputs derives from the max source index across all Loads."""
    k = LoopOp(
        body=(
            Loop(
                axis=Axis("a0", 4),
                body=(
                    Load("a", source=0, index=(Var("a0"),)),
                    Load("b", source=2, index=(Var("a0"),)),  # non-contiguous source idx is fine
                    Assign("y", ElementwiseOp("add"), ("a", "b")),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    assert k.num_inputs == 3


def test_update_synthesizes_accum_decl():
    """An ``Accum`` in a reduce Loop implicitly declares its accumulator.
    ``LoopOp.accums`` synthesizes the declaration info (name, combine
    op, identity value) from the Accum. No explicit decl stmt is needed."""
    k = LoopOp(
        body=(
            Loop(
                axis=Axis("row", 4),
                body=(
                    Loop(
                        axis=Axis("k", 8),
                        body=(
                            Load("x_val", source=0, index=(Var("row"), Var("k"))),
                            Accum(name="acc", value="x_val", op=ElementwiseOp("add")),
                        ),
                    ),
                    Write(output=0, index=(Var("row"),), value="acc"),
                ),
            ),
        ),
    )
    # rename_ssa_sequential canonicalizes Accum names to acc0, acc1, ...
    assert len(k.accums) == 1 and k.accums[0].name == "acc0"
    assert k.accums[0].op.fn == "add"
    assert isinstance(k.accums[0].init, Literal) and k.accums[0].init.value == 0.0


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("a", "b"))
    assert a.name == "out"
    assert a.op.fn == "add"
    assert a.args == ("a", "b")


# ---------------------------------------------------------------------------
# LoopOp construction
# ---------------------------------------------------------------------------


def _pointwise_axes(n: int) -> tuple[Axis, ...]:
    return tuple(Axis(f"a{i}", 4) for i in range(n))


def test_kernel_pointwise():
    axes = (Axis("a0", 4),)
    p = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("z", ElementwiseOp("add"), args=("$0", "$1")),
            Write(output=0, index=(Var("a0"),), value="z"),
        ),
    )
    assert len(flatten_body(k.body)) == 4  # 2 synthesized Loads + Assign + Write


def test_kernel_reduce():
    axes = (Axis("a0", 4), Axis("a1", 8))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p,),
        body=(
            Accum(name="s", value="$0"),
            Write(output=0, index=(Var("a0"),), value="s"),
        ),
    )
    assert any(isinstance(lb.op, ElementwiseOp) for lb in k.accums)


def test_kernel_matmul():
    axes = (Axis("a0", 4), Axis("a1", 8))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Accum(name="dot", value="mul"),
            Write(output=0, index=(Var("a0"),), value="dot"),
        ),
    )
    # 2 synthesized Loads + Assign + Accum + Write — accumulator info is carried on Accum.op now.
    assert len(flatten_body(k.body)) == 5


def test_kernel_matmul_bias():
    axes = (Axis("a0", 4), Axis("a1", 8))
    p_mk = Port(index=(Var("a0"), Var("a1")))
    p_bias = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p_mk, p_mk, p_bias),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Accum(name="dot", value="mul"),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
            Write(output=0, index=(Var("a0"),), value="out"),
        ),
    )
    # 3 synthesized Loads + Assign + Accum + Assign + Write (no decl stmt — Accum carries op).
    assert len(flatten_body(k.body)) == 7


def test_kernel_softmax_two_accumulators():
    # Under nested SSA scoping, per-segment SSA names (sub, ex) are scoped to
    # their own reduce Loop — the old flat fixture relied on laxness. Here we
    # just verify a kernel with two accumulators + two Updates builds fine.
    axes = (Axis("a0", 4), Axis("a1", 8))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p,),
        body=(
            Accum(name="mx", value="$0", op=ElementwiseOp("max")),
            Accum(name="sm", value="$0"),
            Write(output=0, index=(Var("a0"),), value="mx"),
        ),
    )
    # rename_ssa_sequential renames Accums to acc0, acc1 in definition order.
    assert any(lb.name == "acc0" for lb in k.accums)
    assert any(lb.name == "acc1" for lb in k.accums)


def test_kernel_unary_chain():
    axes = (Axis("a0", 4),)
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
    # 1 synthesized Load + 2 Assigns + Write.
    assert len(flatten_body(k.body)) == 4


def test_kernel_scatter_output_via_select():
    """Select replaces Mux for coord-predicated dispatch on output."""
    axes = (Axis("a0", 4),)
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
    assert any(isinstance(s, Select) for s in flatten_body(k.body))


# ---------------------------------------------------------------------------
# Validator — SSA / accumulator / v1 pins
# ---------------------------------------------------------------------------


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
    # 1 synthesized Load + 3 Assigns.
    assert len(flatten_body(k.body)) == 4


def test_update_op_conflict_rejected():
    """Multiple Updates to the same target must share the same op — mixing
    ``max`` and ``add`` on one accumulator is semantically ambiguous."""
    axes = (Axis("a0", 4),)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="conflicts with"):
        _loop(
            axes=axes,
            inputs=(p,),
            body=(
                Accum(name="acc", value="$0", op=ElementwiseOp("max")),
                Accum(name="acc", value="$0", op=ElementwiseOp("add")),
            ),
        )


# ---------------------------------------------------------------------------
# Loop Stmt — nested body form
# ---------------------------------------------------------------------------


def test_loop_stmt_basic():
    from deplodock.compiler.ir.loop import Loop

    # Pointwise kernel with body wrapped in a free Loop.
    loop = _loop(
        axes=(Axis("a0", 8),),
        inputs=(Port(index=(Var("a0"),)),),
        body=(
            Loop(
                axis=Axis("a0", 8),
                body=(
                    Assign("v", ElementwiseOp("neg"), args=("$0",)),
                    Write(output=0, index=(Var("a0"),), value="v"),
                ),
            ),
        ),
    )
    assert isinstance(loop.body[0], Loop)
    assert loop.body[0].axis.name == "a0"


def test_loop_axis_sibling_same_name_allowed():
    """Sibling Loops with the same axis name are fine — that's softmax's two K-loops."""
    from deplodock.compiler.ir.loop import Loop

    # Two reduce Loops over the same axis, sibling (not nested).
    _loop(
        axes=(Axis("a0", 4), Axis("k", 8)),
        inputs=(Port(index=(Var("a0"), Var("k"))), Port(index=(Var("a0"), Var("k")))),
        body=(
            Loop(axis=Axis("k", 8), body=(Accum(name="mx", value="$0", op=ElementwiseOp("max")),)),
            Loop(axis=Axis("k", 8), body=(Accum(name="sm", value="$1"),)),
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
            axes=(Axis("a0", 4),),
            inputs=(Port(index=(Var("a0"),)),),
            body=(
                Loop(
                    axis=Axis("a0_inner", 4),
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

    axes = (Axis("a0", 8),)
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

    axes = (Axis("a0", 4), Axis("k", 8))
    flat = (
        Accum(name="acc", value="$0"),
        Write(output=0, index=(Var("a0"),), value="acc"),
    )
    nested = flat_body_to_nested(axes, flat)
    assert len(nested) == 1 and isinstance(nested[0], Loop) and nested[0].axis.name == "a0"
    inner = nested[0].body
    assert len(inner) == 2
    assert isinstance(inner[0], Loop) and inner[0].axis.name == "k"
    assert isinstance(inner[0].body[0], Accum)
    assert isinstance(inner[1], Write)


def test_flat_body_to_nested_idempotent():
    from deplodock.compiler.ir.loop import Loop

    already_nested = (
        Loop(
            axis=Axis("a0", 8),
            body=(
                Assign("v", ElementwiseOp("neg"), args=("$0",)),
                Write(output=0, index=(Var("a0"),), value="v"),
            ),
        ),
    )
    assert flat_body_to_nested((Axis("a0", 8),), already_nested) == already_nested


def test_flat_body_to_nested_softmax_two_updates():
    from deplodock.compiler.ir.loop import Loop

    axes = (Axis("a0", 4), Axis("a1", 8), Axis("k", 8))
    flat = (
        Accum(name="mx", value="$1"),
        Assign("v_mx", ElementwiseOp("neg"), args=("mx",)),
        Accum(name="sm", value="v_mx"),
        Write(output=0, index=(Var("a0"), Var("a1")), value="sm"),
    )
    nested = flat_body_to_nested(axes, flat)
    # Structure: Loop(a0, [Loop(a1, [Loop(k, [Accum mx]), Loop(k, [Assign, Accum sm]), Write])])
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
        axes=(Axis("a0", 4), Axis("k", 8)),
        inputs=(Port(index=(Var("a0"), Var("k"))),),
        body=(
            Accum(name="acc", value="$0"),
            Write(output=0, index=(Var("a0"),), value="acc"),
        ),
    )


def _nested_reduce_kernel() -> LoopOp:
    """Same kernel as _flat_reduce_kernel, but body is nested via explicit Loops."""
    from deplodock.compiler.ir.loop import Loop

    return LoopOp(
        body=(
            Loop(
                axis=Axis("a0", 4),
                body=(
                    Loop(
                        axis=Axis("k", 8),
                        body=(
                            Load(name="x_k", source=0, index=(Var("a0"), Var("k"))),
                            Accum(name="acc", value="x_k", op=ElementwiseOp("add")),
                        ),
                    ),
                    Write(output=0, index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )


def test_analyze_kernel_handles_nested_body():
    """analyze_kernel should produce the same plan for flat and nested forms."""
    from deplodock.compiler.ir.loop.plan import analyze_kernel

    flat = _flat_reduce_kernel()
    nested = _nested_reduce_kernel()
    out_shape = (4,)
    plan_flat = analyze_kernel(flat, out_shape)
    plan_nested = analyze_kernel(nested, out_shape)
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
