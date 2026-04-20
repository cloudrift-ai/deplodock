"""Unit tests for ``rules/fusion/_merge_core.py``.

Exercises the σ solver and full ``merge_loop_ops`` across the shapes today's
fusion grammar handles: direct-axis elementwise chains, offset-slice reads,
reductions feeding elementwise, and legality rejects (non-affine writer,
multi-reduce, free-axis leakage).
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinOp, Literal, Var
from deplodock.compiler.ir.loop_ir import Accum, Assign, Axis, Load, Loop, LoopOp, Select, SelectBranch, Write, flat_body_to_nested
from deplodock.compiler.ir.tensor_ir import ElementwiseOp
from deplodock.compiler.rules.fusion._merge_core import _bind_axis, _solve_sigma, merge_loop_ops


def Port(index=()):
    """Back-compat test shim — Port is gone at the IR level, but fixtures
    still spell reads as ``Port(index=...)``. The shim returns the index
    tuple for ``_loop`` to synthesize body-form Loads from."""
    return tuple(index)


def _loop(*, axes=(), inputs=(), body=()):
    """Build a LoopOp from a flat body + axes hint.

    Test-local shim: ``LoopOp.axes`` is a property over the body's Loop
    tree, so ``axes=`` feeds ``flat_body_to_nested`` to produce the
    nested form. When ``inputs=(...)`` is provided (legacy Port-tuples),
    each ``$N`` ref is rewritten to a freshly-named Load inserted just
    before the referencing stmt. Load cache resets at each ``Accum`` and
    at each nested ``Loop`` so sibling reduce sweeps get scope-local Loads.
    """
    if not inputs:
        return LoopOp(body=flat_body_to_nested(axes, body))

    index_of = {i: tuple(p) for i, p in enumerate(inputs)}
    fresh_counter = [0]

    def process(stmts):
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
                stmt = Loop(axis=stmt.axis, body=process(stmt.body))
            result.extend(extra_loads)
            extra_loads = []
            result.append(stmt)
            if isinstance(stmt, Accum):
                local_loads = {}

        return tuple(result)

    return LoopOp(body=flat_body_to_nested(axes, process(body)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def test_solver_direct_axis():
    sigma = _solve_sigma((Var("a"),), (Var("b"),), {"a"})
    assert sigma == {"a": Var("b")}


def test_solver_multi_dim():
    sigma = _solve_sigma((Var("i"), Var("j")), (Var("p"), Var("q")), {"i", "j"})
    assert sigma == {"i": Var("p"), "j": Var("q")}


def test_solver_literal_broadcast():
    """A Literal in the writer index pins to that value; no binding."""
    sigma = _solve_sigma((Literal(0, "int"),), (Var("b"),), set())
    assert sigma == {}


def test_solver_offset_plus():
    # writer = a + 5 ; reader = b  →  a = b - 5
    sigma = _solve_sigma(
        (BinOp("+", Var("a"), Literal(5, "int")),),
        (Var("b"),),
        {"a"},
    )
    assert sigma == {"a": BinOp("-", Var("b"), Literal(5, "int"))}


def test_solver_offset_minus():
    # writer = a - 5 ; reader = b  →  a = b + 5
    sigma = _solve_sigma(
        (BinOp("-", Var("a"), Literal(5, "int")),),
        (Var("b"),),
        {"a"},
    )
    assert sigma == {"a": BinOp("+", Var("b"), Literal(5, "int"))}


def test_solver_rejects_non_affine():
    # writer = a * 2 — unsupported form
    sigma = _solve_sigma((BinOp("*", Var("a"), Literal(2, "int")),), (Var("b"),), {"a"})
    assert sigma is None


def test_solver_rejects_length_mismatch():
    sigma = _solve_sigma((Var("a"),), (Var("b"), Var("c")), {"a"})
    assert sigma is None


def test_solver_rejects_conflicting_binding():
    # Same producer axis `a` bound to two different reader expressions.
    sigma = _solve_sigma((Var("a"), Var("a")), (Var("b"), Var("c")), {"a"})
    assert sigma is None


def test_bind_axis_ignores_non_producer_var():
    # A Var not in producer_axes isn't a valid writer anchor — unsupported.
    assert _bind_axis(Var("other"), Var("b"), {"a"}) is None


# ---------------------------------------------------------------------------
# Full merge — elementwise chain
# ---------------------------------------------------------------------------


def _simple_elementwise(fn: str, axis_name: str = "a0", extent: int = 8) -> LoopOp:
    return _loop(
        axes=(Axis(name=axis_name, extent=extent),),
        inputs=(Port(index=(Var(axis_name),)),),
        body=(
            Assign(name="v", op=ElementwiseOp(fn), args=("$0",)),
            Write(output=0, index=(Var(axis_name),), value="v"),
        ),
    )


def test_merge_elementwise_chain():
    producer = _simple_elementwise("neg")
    consumer = _simple_elementwise("exp")
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is not None
    # Merged iteration space = consumer axes (producer's bound a0).
    assert len(merged.axes) == 1
    assert merged.axes[0].name == "a0"
    # Consumer had 1 source (the producer); producer had 1 source (external).
    # After merge the consumer source is consumed, so only producer's source
    # survives → 1 distinct source.
    assert merged.num_inputs == 1
    # Body: producer Assign(neg) → bridge Assign(copy) → consumer Assign(exp) → Write.
    from deplodock.compiler.ir.loop_ir import flatten_body

    flat = flatten_body(merged.body)
    assign_fns = [s.op.fn for s in flat if isinstance(s, Assign)]
    assert assign_fns == ["neg", "copy", "exp"]
    writes = [s for s in flat if isinstance(s, Write)]
    assert len(writes) == 1


def test_merge_elementwise_ssa_rename_on_collision():
    # Both producer and consumer use SSA name "v" — collision must be renamed.
    producer = _simple_elementwise("neg")  # uses "v"
    consumer = _simple_elementwise("exp")  # also uses "v"
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is not None
    ssa_names = [s.name for s in merged.body if isinstance(s, Assign)]
    assert len(ssa_names) == len(set(ssa_names)), f"duplicate SSA names: {ssa_names}"


# ---------------------------------------------------------------------------
# Merge with offset slice
# ---------------------------------------------------------------------------


def test_merge_with_offset_read():
    """Consumer reads producer at `a0 + 5` — σ folds the offset into producer ports."""
    producer = _simple_elementwise("neg", axis_name="a0", extent=16)
    consumer = _loop(
        axes=(Axis(name="c0", extent=3),),
        inputs=(Port(index=(BinOp("+", Var("c0"), Literal(5, "int")),)),),
        body=(
            Assign(name="e", op=ElementwiseOp("exp"), args=("$0",)),
            Write(output=0, index=(Var("c0"),), value="e"),
        ),
    )
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is not None
    # Merged iterates over consumer's c0 (extent 3).
    assert merged.axes[0].name == "c0"
    assert merged.axes[0].extent == 3
    # Producer's Load originally indexed (Var("a0"),); σ["a0"] = Var("c0") + 5,
    # so the merged kernel's Load should read at (c0 + 5,).
    merged_loads = merged.loads
    assert len(merged_loads) == 1
    assert len(merged_loads[0].index) == 1
    idx = merged_loads[0].index[0]
    assert isinstance(idx, BinOp) and idx.op == "+"
    assert isinstance(idx.left, Var) and idx.left.name == "c0"
    assert isinstance(idx.right, Literal) and idx.right.value == 5


# ---------------------------------------------------------------------------
# Merge with reduction
# ---------------------------------------------------------------------------


def _reduce_sum_over_axis1(outer: int = 4, inner: int = 8) -> LoopOp:
    """Kernel: out[i] = sum_j (a[i, j] * b[i, j]). One input port (external buf)."""
    return _loop(
        axes=(
            Axis(name="i", extent=outer),
            Axis(name="j", extent=inner),
        ),
        inputs=(Port(index=(Var("i"), Var("j"))),),
        body=(
            Assign(name="m", op=ElementwiseOp("neg"), args=("$0",)),
            Accum(name="acc", value="m"),
            Write(output=0, index=(Var("i"),), value="acc"),
        ),
    )


def test_merge_reduce_then_elementwise():
    """Producer reduces over j; consumer adds a scalar bias per row."""
    producer = _reduce_sum_over_axis1()
    consumer = _loop(
        axes=(Axis(name="i", extent=4),),
        inputs=(
            Port(index=(Var("i"),)),  # from producer
            Port(index=(Var("i"),)),  # external bias
        ),
        body=(
            Assign(name="y", op=ElementwiseOp("add"), args=("$0", "$1")),
            Write(output=0, index=(Var("i"),), value="y"),
        ),
    )
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is not None
    # Merged axes: consumer's i + producer's reduce axis j.
    names = [a.name for a in merged.axes]
    assert "i" in names and "j" in names
    reduce_axes = [a for a in merged.axes if a.name in merged.reduce_axis_names]
    assert len(reduce_axes) == 1 and reduce_axes[0].name == "j"
    # Locals merged: the acc from the producer survives.
    local_names = [lb.name for lb in merged.accums]
    assert "acc" in local_names
    # Body must contain the Accum, then the bridge-copy, then the consumer's add.
    from deplodock.compiler.ir.loop_ir import flatten_body

    flat = flatten_body(merged.body)
    fns = [s.op.fn for s in flat if isinstance(s, Assign)]
    assert fns == ["neg", "copy", "add"]
    assert any(isinstance(s, Accum) for s in flat)


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------


def test_merge_reduce_with_singleton_batch_dim():
    """Reduction output with a singleton batch dim still merges into a consumer.

    Reproduces the RMSNorm failure: producer reduces shape ``(1, N, D) -> (1, N, 1)``.
    Its write index at the batch dim is ``Literal(0)`` (not ``Var(a0)``) because
    the dim extent is 1. Without the singleton-free-axis collapse in merge_core,
    ``a0`` looks unbound and the merge refuses.
    """
    # Producer: sum-of-squares over the last dim, with a singleton batch axis.
    producer = _loop(
        axes=(
            Axis(name="a0", extent=1),
            Axis(name="a1", extent=32),
            Axis(name="a2", extent=2048),
        ),
        inputs=(Port(index=(Literal(0, "int"), Var("a1"), Var("a2"))),),
        body=(
            Assign(name="sq", op=ElementwiseOp("mul"), args=("$0", "$0")),
            Accum(name="acc", value="sq"),
            Write(output=0, index=(Literal(0, "int"), Var("a1"), Literal(0, "int")), value="acc"),
        ),
    )
    # Consumer: trivial elementwise over the reduction result (1, 32, 1).
    consumer = _loop(
        axes=(
            Axis(name="a0", extent=1),
            Axis(name="a1", extent=32),
            Axis(name="a2", extent=1),
        ),
        inputs=(Port(index=(Literal(0, "int"), Var("a1"), Literal(0, "int"))),),
        body=(
            Assign(name="v", op=ElementwiseOp("rsqrt"), args=("$0",)),
            Write(output=0, index=(Var("a0"), Var("a1"), Var("a2")), value="v"),
        ),
    )
    merged = merge_loop_ops(producer, 0, consumer, 0)
    assert merged is not None, "merge must succeed with singleton batch dim"
    reduce_axes = [a for a in merged.axes if a.name in merged.reduce_axis_names]
    assert len(reduce_axes) == 1 and reduce_axes[0].extent == 2048


def test_merge_alias_unifies_reduce_axes():
    """Two reductions over the same data axis collapse to one reduce axis when
    the caller passes an alias, turning a would-be 2-reduce merge into 1."""
    # Producer reduces x along a1 (extent 8).
    producer = _loop(
        axes=(Axis(name="a0", extent=4), Axis(name="a1", extent=8)),
        inputs=(Port(index=(Var("a0"), Var("a1"))),),
        body=(
            Accum(name="acc_p", value="$0", op=ElementwiseOp("max")),
            Write(output=0, index=(Var("a0"),), value="acc_p"),
        ),
    )
    # Consumer also reduces x along b1 (extent 8), reads producer's output at [b0],
    # adds them.
    consumer = _loop(
        axes=(Axis(name="b0", extent=4), Axis(name="b1", extent=8)),
        inputs=(
            Port(index=(Var("b0"),)),  # from producer
            Port(index=(Var("b0"), Var("b1"))),  # same external buffer, different reduce axis
        ),
        body=(
            Accum(name="acc_c", value="$1"),
            Assign(name="v", op=ElementwiseOp("add"), args=("$0", "acc_c")),
            Write(output=0, index=(Var("b0"),), value="v"),
        ),
    )
    # With alias a1 → b1: producer's reduce axis unifies with consumer's,
    # so the merged kernel has a single reduce axis.
    merged = merge_loop_ops(producer, 0, consumer, 0, axis_aliases={"a1": "b1"})
    assert merged is not None
    reduce_axes = [a for a in merged.axes if a.name in merged.reduce_axis_names]
    assert len(reduce_axes) == 1, f"expected 1 reduce axis, got {[a.name for a in reduce_axes]}"
    assert reduce_axes[0].name == "b1", "producer's a1 should be aliased away"
    # Both accumulators survive: producer's acc_p (max) and consumer's acc_c (add).
    combine_fns = {lb.combine.fn for lb in merged.accums if lb.combine is not None}
    assert combine_fns == {"max", "add"}


def test_merge_alias_ignores_already_bound():
    """axis_aliases entries are no-ops for axes already bound by σ."""
    producer = _loop(
        axes=(Axis(name="a0", extent=8),),
        inputs=(Port(index=(Var("a0"),)),),
        body=(
            Assign(name="v", op=ElementwiseOp("neg"), args=("$0",)),
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )
    consumer = _loop(
        axes=(Axis(name="b0", extent=8),),
        inputs=(Port(index=(Var("b0"),)),),
        body=(
            Assign(name="v", op=ElementwiseOp("exp"), args=("$0",)),
            Write(output=0, index=(Var("b0"),), value="v"),
        ),
    )
    # a0 is already bound by σ (writer[0] = reader[0]). The alias should be silently ignored.
    merged = merge_loop_ops(producer, 0, consumer, 0, axis_aliases={"a0": "b0"})
    assert merged is not None


def test_merge_rejects_free_axis_leak():
    """Producer's writer leaves a free axis unbound in σ — unsafe to replicate."""
    # Producer writes at index (Var("a0"), Literal(0)); the a1 axis is free but doesn't
    # appear in the writer, so a consumer reading at (Var("c0"), Var("c1")) can't bind a1.
    producer = _loop(
        axes=(
            Axis(name="a0", extent=4),
            Axis(name="a1", extent=8),
        ),
        inputs=(Port(index=(Var("a0"), Var("a1"))),),
        body=(
            Assign(name="v", op=ElementwiseOp("neg"), args=("$0",)),
            # Writer only has a0 — a1 is unused downstream.
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )
    consumer = _loop(
        axes=(Axis(name="c0", extent=4),),
        inputs=(Port(index=(Var("c0"),)),),
        body=(
            Assign(name="e", op=ElementwiseOp("exp"), args=("$0",)),
            Write(output=0, index=(Var("c0"),), value="e"),
        ),
    )
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    # a1 is free and unbound → merge refuses.
    assert merged is None


def test_merge_rejects_non_affine_writer():
    """Writer `a * 2` — solver refuses."""
    producer = _loop(
        axes=(Axis(name="a", extent=4),),
        inputs=(Port(index=(Var("a"),)),),
        body=(
            Assign(name="v", op=ElementwiseOp("neg"), args=("$0",)),
            Write(output=0, index=(BinOp("*", Var("a"), Literal(2, "int")),), value="v"),
        ),
    )
    consumer = _simple_elementwise("exp", axis_name="c0", extent=8)
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is None


# ---------------------------------------------------------------------------
# Numeric correctness via the numpy loop backend
# ---------------------------------------------------------------------------


def test_merge_numerically_equivalent_elementwise():
    """Run the merged kernel through the numpy loop backend; compare to the composed reference."""
    import numpy as np

    from deplodock.compiler.backend.loop.backend import _exec_launch
    from deplodock.compiler.program.loop import LoopBuffer, LoopLaunch, LoopProgram

    rng = np.random.default_rng(0)
    x = rng.standard_normal(8).astype(np.float32)

    producer = _simple_elementwise("neg")
    consumer = _simple_elementwise("exp")
    merged = merge_loop_ops(producer, producer_output=0, consumer=consumer, consumer_port=0)
    assert merged is not None

    program = LoopProgram(
        name="merge_test",
        buffers=[
            LoopBuffer(name="x", shape=(8,), role="input"),
            LoopBuffer(name="y", shape=(8,), role="output"),
        ],
        launches=[LoopLaunch(loop=merged, input_names=["x"], output_name="y")],
        graph_outputs=["y"],
    )
    merged_out = _exec_launch(program.launches[0], program, {"x": x})

    expected = np.exp(-x)
    np.testing.assert_allclose(merged_out, expected, rtol=1e-5, atol=1e-5)
