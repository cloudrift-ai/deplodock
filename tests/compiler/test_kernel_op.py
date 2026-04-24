"""Tests for the structural LoopOp IR with SSA body (Assign/Accum/Write/Select)."""

import pytest

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
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
)
from deplodock.compiler.ir.tensor.ir import ElementwiseOp


def Port(index=()):
    """Back-compat test shim — Port is gone at the IR level, but fixtures
    still spell reads as ``Port(index=...)``. The shim just returns the
    index tuple for ``_loop`` to synthesize body-form Loads from."""
    return tuple(index)


def _nest(axes, body):
    """Wrap a flat SSA body into nested Loop-block form.

    Idempotent: if body already contains a Loop, returns it unchanged.
    Reduce axes are inferred as axes with extent > 1 that don't appear
    in any Write index (when the body contains an Accum).
    """
    if any(isinstance(s, Loop) for s in body):
        return tuple(body)

    has_update = any(isinstance(s, Accum) for s in body)
    reduce_axis_names: frozenset[str] = frozenset()

    if has_update:
        write_axis_names: set[str] = set()

        def _walk(e):
            if isinstance(e, Var):
                write_axis_names.add(e.name)
            elif isinstance(e, BinaryExpr):
                _walk(e.left)
                _walk(e.right)

        for s in body:
            if isinstance(s, Write):
                for e in s.index:
                    _walk(e)
        reduce_axis_names = frozenset(a.name for a in axes if a.name not in write_axis_names and int(a.extent) > 1)

    free_axes = [a for a in axes if a.name not in reduce_axis_names]

    if not has_update:
        wrapped: tuple = tuple(body)
        for a in reversed(free_axes):
            wrapped = (Loop(axis=a, body=wrapped),)
        return wrapped

    reduce_axes = [a for a in axes if a.name in reduce_axis_names]
    if len(reduce_axes) != 1:
        return tuple(body)
    reduce_axis = reduce_axes[0]

    segments: list[list] = []
    current: list = []
    for stmt in body:
        current.append(stmt)
        if isinstance(stmt, Accum):
            segments.append(current)
            current = []
    tail = current

    nested: list = []
    for seg in segments:
        nested.append(Loop(axis=reduce_axis, body=tuple(seg)))
    nested.extend(tail)

    wrapped = tuple(nested)
    for a in reversed(free_axes):
        wrapped = (Loop(axis=a, body=wrapped),)
    return wrapped


def _loop(*, axes=(), inputs=(), body=()):
    """Build a LoopOp from a flat body + axes hint.

    Test-local shim: ``LoopOp.axes`` is a property over the body's Loop
    tree, so ``axes=`` feeds ``_nest`` to produce the nested form. When
    ``inputs=(...)`` is provided (legacy Port-tuples), each ``$N`` ref
    is rewritten to a freshly-named Load that is inserted just before
    the referencing statement. Load cache resets at each ``Accum`` and
    at each nested ``Loop`` boundary so sibling reduce sweeps (softmax
    pattern) get their own scope-local Loads.
    """
    if not inputs:
        return LoopOp(body=_nest(axes, body))

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

    return LoopOp(body=_nest(axes, process(body)))


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
                    Assign("y", ElementwiseOp("negative"), ("x_val",)),
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
                            Accum(name="acc", value="x_val", op="add"),
                        ),
                    ),
                    Write(output=0, index=(Var("row"),), value="acc"),
                ),
            ),
        ),
    )
    # rename_ssa_sequential canonicalizes Accum names to acc0, acc1, ...
    assert len(k.accums) == 1 and k.accums[0].name == "acc0"
    assert k.accums[0].op.name == "add"
    assert isinstance(k.accums[0].init, Literal) and k.accums[0].init.value == 0.0


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("a", "b"))
    assert a.name == "out"
    assert a.op.name == "add"
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
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4  # 2 synthesized Loads + Assign + Write


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
    assert any(isinstance(lb.op, ElementwiseImpl) for lb in k.accums)


def test_kernel_matmul():
    axes = (Axis("a0", 4), Axis("a1", 8))
    p = Port(index=(Var("a0"), Var("a1")))
    k = _loop(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("multiply", ElementwiseOp("multiply"), args=("$0", "$1")),
            Accum(name="dot", value="multiply"),
            Write(output=0, index=(Var("a0"),), value="dot"),
        ),
    )
    # 2 synthesized Loads + Assign + Accum + Write — accumulator info is carried on Accum.op now.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 5


def test_kernel_matmul_bias():
    axes = (Axis("a0", 4), Axis("a1", 8))
    p_mk = Port(index=(Var("a0"), Var("a1")))
    p_bias = Port(index=(Var("a0"),))
    k = _loop(
        axes=axes,
        inputs=(p_mk, p_mk, p_bias),
        body=(
            Assign("multiply", ElementwiseOp("multiply"), args=("$0", "$1")),
            Accum(name="dot", value="multiply"),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
            Write(output=0, index=(Var("a0"),), value="out"),
        ),
    )
    # 3 synthesized Loads + Assign + Accum + Assign + Write (no decl stmt — Accum carries op).
    assert sum(1 for s in k if not isinstance(s, Loop)) == 7


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
            Accum(name="mx", value="$0", op="maximum"),
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
            Assign("negative", ElementwiseOp("negative"), args=("$0",)),
            Assign("exp", ElementwiseOp("exp"), args=("negative",)),
            Write(output=0, index=(Var("a0"),), value="exp"),
        ),
    )
    # 1 synthesized Load + 2 Assigns + Write.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4


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
    assert any(isinstance(s, Select) for s in k)


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
                Assign("y", ElementwiseOp("negative"), args=("y",)),
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
            Assign("b", ElementwiseOp("negative"), args=("$0",)),
            Assign("c", ElementwiseOp("add"), args=("a", "b")),
        ),
    )
    # 1 synthesized Load + 3 Assigns.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4


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
                Accum(name="acc", value="$0", op="maximum"),
                Accum(name="acc", value="$0", op="add"),
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
                    Assign("v", ElementwiseOp("negative"), args=("$0",)),
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
            Loop(axis=Axis("k", 8), body=(Accum(name="mx", value="$0", op="maximum"),)),
            Loop(axis=Axis("k", 8), body=(Accum(name="sm", value="$1"),)),
            Assign("v", ElementwiseOp("add"), args=("mx", "sm")),
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )


def test_loop_ssa_scoping():
    """Assign defined inside Loop body is invisible outside — caller Assign must fail.

    The inner Assign binds to the inner axis (via ``$0``'s index) so LICM
    can't hoist it out; the outer reference is a true scope violation.
    """
    from deplodock.compiler.ir.loop import Loop

    with pytest.raises(ValueError, match="not defined"):
        _loop(
            axes=(Axis("a0", 4), Axis("a0_inner", 4)),
            inputs=(Port(index=(Var("a0_inner"),)),),
            body=(
                Loop(
                    axis=Axis("a0_inner", 4),
                    body=(Assign("v", ElementwiseOp("negative"), args=("$0",)),),
                ),
                # Reference 'v' outside the inner Loop — should fail.
                Write(output=0, index=(Var("a0"),), value="v"),
            ),
        )


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
                            Accum(name="acc", value="x_k", op="add"),
                        ),
                    ),
                    Write(output=0, index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )


def test_execute_loop_op_handles_nested_body():
    """Numpy interpreter should produce the same output for flat and nested forms."""
    import numpy as np

    from deplodock.compiler.ir.loop.interpret import execute_loop_op

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)

    flat = _flat_reduce_kernel()
    nested = _nested_reduce_kernel()
    out_flat = execute_loop_op(flat, [x], (4,))
    out_nested = execute_loop_op(nested, [x], (4,))
    np.testing.assert_allclose(out_flat, out_nested, rtol=1e-5)
