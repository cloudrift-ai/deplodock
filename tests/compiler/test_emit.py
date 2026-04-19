"""Tests for the structural CUDA emitter with the grammar-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs. CUDA-specific
by design (source-level assertions on emitted C code); not parameterized
over backends.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor_ir import ElementwiseOp, ReduceOp  # noqa: F401
from deplodock.compiler.program.loop import LoopBuffer, LoopLaunch, LoopProgram

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _pointwise_add_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("y", (4,)), node_id="y")
    g.add_node(op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]
    return g


def _reduce_sum_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("y", (4, 1)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _matmul_graph() -> Graph:
    from deplodock.compiler.ir.frontend_ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _softmax_launch() -> LoopLaunch:
    """Build a LoopLaunch with the softmax SSA pattern directly."""
    from deplodock.compiler.ir.expr import Literal, Var
    from deplodock.compiler.ir.loop_ir import AccumDecl, Assign, Axis, Loop, LoopOp, Port, Update, Write

    a0 = Axis("a0", 4)
    a1 = Axis("a1", 8)
    p = Port(index=(Var("a0"), Var("a1")))
    loop = LoopOp(
        inputs=(p, p),
        body=(
            Loop(
                axis=a0,
                body=(
                    AccumDecl(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
                    AccumDecl(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
                    # Max sweep (K-loop 1).
                    Loop(axis=a1, body=(Update(target="mx", value="$0"),)),
                    # Sum sweep (K-loop 2) — rematerializes exp inside.
                    Loop(
                        axis=a1,
                        body=(
                            Assign(name="sub_s", op=ElementwiseOp("sub"), args=("$0", "mx")),
                            Assign(name="ex_s", op=ElementwiseOp("exp"), args=("sub_s",)),
                            Update(target="sm", value="ex_s"),
                        ),
                    ),
                    # Write sweep (K-loop 3): per-element div.
                    Loop(
                        axis=a1,
                        body=(
                            Assign(name="sub_w", op=ElementwiseOp("sub"), args=("$1", "mx")),
                            Assign(name="ex_w", op=ElementwiseOp("exp"), args=("sub_w",)),
                            Assign(name="out", op=ElementwiseOp("div"), args=("ex_w", "sm")),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="out"),
                        ),
                    ),
                ),
            ),
        ),
    )
    return LoopLaunch(loop=loop, input_names=["x", "x"], output_name="y")


def _softmax_program() -> LoopProgram:
    """Wrap the softmax launch in a standalone LoopProgram for codegen tests."""
    launch = _softmax_launch()
    return LoopProgram(
        name="softmax",
        buffers=[
            LoopBuffer(name="x", shape=(4, 8), role="input"),
            LoopBuffer(name="y", shape=(4, 8), role="output"),
        ],
        launches=[launch],
        graph_inputs=["x"],
        graph_outputs=["y"],
    )


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    assert len(compiled.launches) == 1
    source = compiled.launches[0].kernel_source
    assert "blockIdx.x" in source
    assert "x[" in source and "y[" in source


def test_reduce_emits_k_loop():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    source = compiled.launches[0].kernel_source
    assert "for (int" in source
    assert "+=" in source


def test_contraction_emits_matmul():
    compiled = CudaBackend().compile(_matmul_graph())
    source = compiled.launches[-1].kernel_source
    assert "for (int k" in source
    assert "acc0 +=" in source


def test_buffer_roles():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    roles = {b.name: b.role for b in compiled.buffers}
    assert roles.get("x") == "input"
    assert roles.get("y") == "input"


def test_softmax_emits_multiple_k_loops():
    """Softmax pattern emits separate K-loops for max, sub+exp+sum, div."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel

    loop_program = _softmax_program()
    launch = loop_program.launches[0]
    kdef, arg_order = emit_kernel(launch, "k0_softmax", loop_program)
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as emit_src

    source = emit_src(kdef)
    loop_count = source.count("for (int")
    assert loop_count >= 3, f"expected >= 3 K-loops, got {loop_count}\n{source}"
    assert "fmaxf" in source
    assert "+=" in source


def test_softmax_emits_per_element_store():
    """Softmax output is per-element: the final div stores inside a K-loop."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as emit_src

    loop_program = _softmax_program()
    launch = loop_program.launches[0]
    kdef, _ = emit_kernel(launch, "k0_softmax", loop_program)
    source = emit_src(kdef)
    assert "y[" in source, f"expected output store y[...]\n{source}"


def test_chained_pointwise_single_kernel():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    compiled = CudaBackend().compile(g)
    assert len(compiled.launches) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    result = CudaBackend().run(compiled, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert list(result.outputs.values())[0] == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert list(result.outputs.values())[0] == pytest.approx(expected)


@requires_cuda
def test_softmax_runs_on_gpu():
    """Softmax from a hand-built LoopProgram (no Graph). Uses compile_kernels
    directly — the resulting GpuProgram is what CudaBackend.run expects."""
    import math

    from deplodock.compiler.backend.cuda.emit import compile_kernels

    loop_program = _softmax_program()
    compiled = compile_kernels(loop_program)
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = []
    for row in range(4):
        row_vals = x_data[row * 8 : (row + 1) * 8]
        mx = max(row_vals)
        exps = [math.exp(v - mx) for v in row_vals]
        s = sum(exps)
        expected.extend(e / s for e in exps)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected, rel=1e-3)


@requires_cuda
def test_matmul_runs_on_gpu():
    compiled = CudaBackend().compile(_matmul_graph())
    a_data = [float(i) for i in range(32)]
    b_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"a": a_data, "b": b_data})
    expected = []
    for mi in range(4):
        for ni in range(4):
            s = sum(a_data[mi * 8 + k] * b_data[k * 4 + ni] for k in range(8))
            expected.append(s)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected)


# ---- check_port_bounds: rotate_half / transitive Select gating ----


def _rotate_half_launch() -> LoopLaunch:
    """Build a LoopLaunch mimicking rotary's rotate_half pattern:

      for a0 in 0..64:
          v0 = neg($1)             # $1 at [a0 + 32] — OOB when a0 >= 32 without the Select
          v1 = Select(
                branch(v0,  a0 < 32),   # use negated lower-upper shift when a0 < 32
                branch($2,  Literal(1)),  # else use $2 (shifted down)
              )
          out[a0] = v1

    The port $1 reads position ``a0 + 32`` which is in-bounds ONLY when a0 < 32
    (then positions 32..63). Without the Select's predicate the static checker
    sees max_index=95, out-of-bounds on a 64-element buffer. With the transitive
    tracing (neg → Select branch gated by ``a0 < 32``), the checker sees the
    constraint and the warning is silenced.
    """
    from deplodock.compiler.ir.expr import BinOp, Literal, Var
    from deplodock.compiler.ir.loop_ir import Assign, Axis, Loop, LoopOp, Port, Select, SelectBranch, Write
    from deplodock.compiler.ir.tensor_ir import ElementwiseOp

    axis_a0 = Axis(name="a0", extent=64)
    body = (
        Assign(name="v0", op=ElementwiseOp("neg"), args=("$1",)),
        Select(
            name="v1",
            branches=(
                SelectBranch(value="v0", select=BinOp("<", Var("a0"), Literal(32, "int"))),
                SelectBranch(value="$2", select=Literal(1, "int")),
            ),
        ),
        Write(output=0, index=(Var("a0"),), value="v1"),
    )
    loop = LoopOp(
        inputs=(
            Port(index=(Var("a0"),)),  # $0 — dummy
            Port(index=(BinOp("+", Var("a0"), Literal(32, "int")),)),  # $1 — the OOB-looking one
            Port(index=(BinOp("-", Var("a0"), Literal(32, "int")),)),  # $2 — the other half
        ),
        body=(Loop(axis=axis_a0, body=body),),
    )
    return LoopLaunch(loop=loop, input_names=["x", "x", "x"], output_name="out")


def test_check_port_bounds_recognizes_transitive_select_gating():
    """The rotate_half pattern wraps ``$N`` in an Assign before the Select.
    The static OOB checker must trace through the SSA chain to see that the
    port is effectively gated by the Select's predicate.
    """
    from deplodock.compiler.backend.cuda.emit import check_port_bounds

    launch = _rotate_half_launch()
    program = LoopProgram(
        name="prog",
        buffers=[LoopBuffer(name="x", shape=(64,), role="input"), LoopBuffer(name="out", shape=(64,), role="output")],
        launches=[launch],
    )
    warnings = check_port_bounds(launch, program, launch_idx=0)
    assert warnings == [], f"expected no OOB warnings, got: {warnings}"


def test_check_port_bounds_still_warns_without_gating_select():
    """Sanity: remove the Select and the same OOB read should warn.
    Proves the silence in the previous test comes from the transitive Select
    analysis, not from loose bounds letting everything through.
    """
    from deplodock.compiler.backend.cuda.emit import check_port_bounds
    from deplodock.compiler.ir.expr import BinOp, Literal, Var
    from deplodock.compiler.ir.loop_ir import Axis, Loop, LoopOp, Port, Write

    axis_a0 = Axis(name="a0", extent=64)
    body = (Write(output=0, index=(Var("a0"),), value="$0"),)
    loop = LoopOp(
        inputs=(Port(index=(BinOp("+", Var("a0"), Literal(32, "int")),)),),
        body=(Loop(axis=axis_a0, body=body),),
    )
    launch = LoopLaunch(loop=loop, input_names=["x"], output_name="out")
    program = LoopProgram(
        name="prog",
        buffers=[LoopBuffer(name="x", shape=(64,), role="input"), LoopBuffer(name="out", shape=(64,), role="output")],
        launches=[launch],
    )
    warnings = check_port_bounds(launch, program, launch_idx=0)
    assert len(warnings) == 1 and "out of bounds" in warnings[0]
