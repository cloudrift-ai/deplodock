"""Tests for ``Graph.structural_key()`` — the Merkle-style structural
digest used for candidate dedup in autotuning loops.

The digest is invariant under: graph-internal node-id renames, Tensor
name renames, hint changes, and (transitively, via
``Body.structural_key()``) SSA / axis / commutative-arg / external-buffer
name changes inside body-bearing ops. It is *sensitive* to: op kind,
op body / attrs, output shape / dtype, input wiring (recursive), and
the graph's input / output sequences.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Assign, Load, Loop, Write
from deplodock.compiler.ir.stmt.body import Body

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _input(g: Graph, nid: str, shape: tuple, dtype: str = "f32") -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(nid, shape, dtype), node_id=nid)


def _add_loopop(g: Graph, nid: str, in_x: str, in_y: str, shape: tuple) -> str:
    """LoopOp that does ``out[i] = in_x[i] + in_y[i]`` with ``Load.input``
    referencing the producer node ids ``in_x`` / ``in_y`` (matches Graph
    wiring conventions: buf names == producer node ids)."""
    a = Axis("a", shape[0])
    body = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="lx", input=in_x, index=(Var("a"),)),
                    Load(name="ly", input=in_y, index=(Var("a"),)),
                    Assign(name="z", op="add", args=("lx", "ly")),
                    Write(output=nid, index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    return g.add_node(op=LoopOp(body=body), inputs=[in_x, in_y], output=Tensor(nid, shape), node_id=nid)


def _add_graph(input_names: tuple[str, str], output_name: str, shape: tuple = (4,)) -> Graph:
    g = Graph()
    _input(g, input_names[0], shape)
    _input(g, input_names[1], shape)
    _add_loopop(g, output_name, input_names[0], input_names[1], shape)
    g.inputs = list(input_names)
    g.outputs = [output_name]
    return g


# ---------------------------------------------------------------------------
# Equality cases
# ---------------------------------------------------------------------------


def test_structural_key_equal_for_isomorphic_graphs() -> None:
    a = _add_graph(("x", "y"), "out")
    b = _add_graph(("x", "y"), "out")
    assert a.structural_key() == b.structural_key()


def test_structural_key_invariant_to_node_id_renames() -> None:
    """Same dataflow, different graph-internal node ids → same digest.
    The digest must not depend on user-chosen ids."""
    a = _add_graph(("x", "y"), "out")
    b = _add_graph(("alpha", "beta"), "result")
    assert a.structural_key() == b.structural_key()


def test_structural_key_invariant_to_tensor_name() -> None:
    """``Tensor.name`` is graph-internal label — ignored. Build two
    graphs whose Tensors carry different ``name`` fields but identical
    shape/dtype + identical wiring."""
    a = _add_graph(("x", "y"), "out")
    b = Graph()
    # Use node_id != Tensor.name to decouple the two.
    b.add_node(op=InputOp(), inputs=[], output=Tensor("aaa", (4,)), node_id="x")
    b.add_node(op=InputOp(), inputs=[], output=Tensor("bbb", (4,)), node_id="y")
    a_loop = a.nodes["out"].op  # reuse the same LoopOp body
    b.add_node(op=a_loop, inputs=["x", "y"], output=Tensor("ccc", (4,)), node_id="out")
    b.inputs = ["x", "y"]
    b.outputs = ["out"]
    assert a.structural_key() == b.structural_key()


def test_structural_key_invariant_to_hints() -> None:
    a = _add_graph(("x", "y"), "out")
    b = _add_graph(("x", "y"), "out")
    b.hints.set("cuda.matmul.strategy", "naive")
    b.nodes["out"].hints.set("foo", "bar")
    assert a.structural_key() == b.structural_key()


def test_structural_key_equal_for_swapped_commutative_body_order() -> None:
    """Two LoopOp bodies that read inputs in different orders but
    compute the same commutative add — body normalization erases the
    distinction."""
    g_xy = _add_graph(("x", "y"), "out")
    # Build a graph where the body Loads y first, then x.
    g_yx = Graph()
    _input(g_yx, "x", (4,))
    _input(g_yx, "y", (4,))
    a = Axis("a", 4)
    body = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="ly", input="y", index=(Var("a"),)),
                    Load(name="lx", input="x", index=(Var("a"),)),
                    Assign(name="z", op="add", args=("ly", "lx")),
                    Write(output="out", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    g_yx.add_node(op=LoopOp(body=body), inputs=["x", "y"], output=Tensor("out", (4,)), node_id="out")
    g_yx.inputs = ["x", "y"]
    g_yx.outputs = ["out"]
    assert g_xy.structural_key() == g_yx.structural_key()


# ---------------------------------------------------------------------------
# Inequality cases
# ---------------------------------------------------------------------------


def test_structural_key_distinguishes_op_kind() -> None:
    """Same wiring, same shapes, different op kind on a leaf — digest
    differs because ``InputOp`` vs ``ConstantOp`` is a structural change."""
    a = Graph()
    _input(a, "x", (4,))
    a.inputs = ["x"]
    a.outputs = ["x"]

    b = Graph()
    b.add_node(op=ConstantOp(name="x"), inputs=[], output=Tensor("x", (4,)), node_id="x")
    b.inputs = ["x"]
    b.outputs = ["x"]

    assert a.structural_key() != b.structural_key()


def test_structural_key_distinguishes_shape() -> None:
    a = _add_graph(("x", "y"), "out", shape=(4,))
    b = _add_graph(("x", "y"), "out", shape=(8,))
    assert a.structural_key() != b.structural_key()


def test_structural_key_distinguishes_dtype() -> None:
    a = Graph()
    _input(a, "x", (4,), dtype="f32")
    a.inputs = ["x"]
    a.outputs = ["x"]

    b = Graph()
    _input(b, "x", (4,), dtype="f16")
    b.inputs = ["x"]
    b.outputs = ["x"]

    assert a.structural_key() != b.structural_key()


def test_structural_key_distinguishes_input_order() -> None:
    """Swapping the graph's ``inputs`` sequence — for inputs that are
    structurally distinguishable (different shapes here) the swap is
    observable in the digest. Two structurally-identical InputOps are
    interchangeable by design (graph-isomorphic) — that case is *not*
    distinguished, and shouldn't be: candidates of one source graph
    always share the input list."""
    a = Graph()
    _input(a, "x", (4,))
    _input(a, "y", (8,))
    a.inputs = ["x", "y"]
    a.outputs = ["x"]

    b = Graph()
    _input(b, "x", (4,))
    _input(b, "y", (8,))
    b.inputs = ["y", "x"]
    b.outputs = ["x"]

    assert a.structural_key() != b.structural_key()


def test_structural_key_distinguishes_cross_cluster_body_op() -> None:
    """Same wiring + shapes; ``add`` and ``divide`` live in different
    compute-unit clusters (FMA vs SFU-div) so the bodies hash distinct.
    Within-cluster ops (e.g. add vs multiply) collapse — see
    ``tests/compiler/ir/stmt/test_structural_key.py``."""
    a = _add_graph(("x", "y"), "out")
    b = Graph()
    _input(b, "x", (4,))
    _input(b, "y", (4,))
    ax = Axis("a", 4)
    body = Body(
        (
            Loop(
                axis=ax,
                body=(
                    Load(name="lx", input="x", index=(Var("a"),)),
                    Load(name="ly", input="y", index=(Var("a"),)),
                    Assign(name="z", op="divide", args=("lx", "ly")),
                    Write(output="out", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    b.add_node(op=LoopOp(body=body), inputs=["x", "y"], output=Tensor("out", (4,)), node_id="out")
    b.inputs = ["x", "y"]
    b.outputs = ["out"]
    assert a.structural_key() != b.structural_key()


def test_structural_key_distinguishes_constantop_value() -> None:
    """``ConstantOp.value`` is a non-body attr — must propagate into the
    digest via the dataclass-fields path."""
    a = Graph()
    a.add_node(op=ConstantOp(name="c", value=1.0), inputs=[], output=Tensor("c", (1,)), node_id="c")
    a.inputs = []
    a.outputs = ["c"]

    b = Graph()
    b.add_node(op=ConstantOp(name="c", value=2.0), inputs=[], output=Tensor("c", (1,)), node_id="c")
    b.inputs = []
    b.outputs = ["c"]

    assert a.structural_key() != b.structural_key()


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_structural_key_returns_hex_digest() -> None:
    g = _add_graph(("x", "y"), "out")
    key = g.structural_key()
    assert isinstance(key, str) and len(key) == 64
    int(key, 16)  # valid hex


def test_structural_key_deterministic_across_calls() -> None:
    g = _add_graph(("x", "y"), "out")
    assert g.structural_key() == g.structural_key()


def test_structural_key_changes_after_mutation() -> None:
    """Sanity: mutation changes the digest. Confirms the method is
    actually reading current state, not a stale snapshot."""
    g = _add_graph(("x", "y"), "out")
    k1 = g.structural_key()
    _input(g, "z", (4,))
    g.inputs = ["x", "y", "z"]
    k2 = g.structural_key()
    assert k1 != k2
