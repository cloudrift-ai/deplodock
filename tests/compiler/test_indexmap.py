"""Unit tests for IndexMapOp + coord_expr helpers."""

import pytest
import torch

from deplodock.compiler.backend.ir.expr import BinOp, Literal, Ternary, Var
from deplodock.compiler.coord_expr import (
    PLACEHOLDER_PREFIX,
    compose_index_maps,
    is_placeholder,
    placeholder,
    substitute,
)
from deplodock.compiler.ops import IndexMapOp, IndexSource

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------- placeholder helpers ----------


def test_placeholder_uses_prefix():
    assert placeholder(0).name == f"{PLACEHOLDER_PREFIX}0"
    assert placeholder(3).name == f"{PLACEHOLDER_PREFIX}3"


def test_is_placeholder_unbound():
    assert is_placeholder(placeholder(0))
    assert is_placeholder(placeholder(7))
    assert not is_placeholder(Var("row"))


def test_is_placeholder_specific_axis():
    assert is_placeholder(placeholder(2), d=2)
    assert not is_placeholder(placeholder(2), d=3)


# ---------- substitute ----------


def test_substitute_replaces_var():
    expr = placeholder(0)
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == "row"


def test_substitute_passes_unmapped_vars():
    expr = placeholder(1)
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == placeholder(1).name


def test_substitute_leaves_literal_alone():
    expr = Literal(5, "int")
    assert substitute(expr, {}) is expr


def test_substitute_rewrites_binop():
    expr = placeholder(0) + Literal(7, "int")
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, BinOp) and result.op == "+"
    assert isinstance(result.left, Var) and result.left.name == "row"
    assert isinstance(result.right, Literal) and result.right.value == 7


def test_substitute_rewrites_ternary():
    cond = placeholder(0).lt(Literal(64, "int"))
    expr = Ternary(cond, placeholder(0), placeholder(0) - Literal(64, "int"))
    result = substitute(expr, {placeholder(0).name: Var("col")})
    assert isinstance(result, Ternary)
    assert isinstance(result.if_true, Var) and result.if_true.name == "col"
    assert isinstance(result.if_false, BinOp) and result.if_false.op == "-"


# ---------- compose_index_maps ----------


def test_compose_identity_with_offset():
    # outer: identity (out_coord_0)
    # inner: x[out_coord_0 + 5]  (slice with start=5)
    outer = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0),)),),
    )
    inner = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(5, "int"),)),),
    )
    merged = compose_index_maps(outer, inner)
    assert merged.out_shape == (10,)
    assert len(merged.sources) == 1
    cm = merged.sources[0].coord_map
    assert len(cm) == 1
    # Composed coord should be (out_coord_0 + 5)
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].left, Var) and cm[0].left.name == placeholder(0).name
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 5


def test_compose_two_offsets():
    # outer: x[out_coord_0 + 3]
    # inner: x[out_coord_0 + 5]
    # merged: x[(out_coord_0 + 3) + 5]
    outer = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(3, "int"),)),),
    )
    inner = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(5, "int"),)),),
    )
    merged = compose_index_maps(outer, inner)
    cm = merged.sources[0].coord_map
    # Should be (out_coord_0 + 3) + 5 — outer's expr substituted into inner's placeholder
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 5
    inner_add = cm[0].left
    assert isinstance(inner_add, BinOp) and inner_add.op == "+"
    assert isinstance(inner_add.right, Literal) and inner_add.right.value == 3


def test_compose_transpose_with_slice():
    # outer (slice, dim=1, start=64): x[out_coord_0, out_coord_1 + 64]
    # inner (transpose, swap 0/1): x[out_coord_1, out_coord_0]
    # merged: composing reads inner's coord_map under outer's placeholder mapping
    outer = IndexMapOp(
        out_shape=(8, 64),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1) + Literal(64, "int"))),),
    )
    inner = IndexMapOp(
        out_shape=(8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),),
    )
    merged = compose_index_maps(outer, inner)
    cm = merged.sources[0].coord_map
    # inner read coord_map is (out_coord_1, out_coord_0); we substitute outer's
    # placeholder(d) := outer.coord_map[d]:
    #   placeholder(0) → outer.coord_map[0] = out_coord_0
    #   placeholder(1) → outer.coord_map[1] = out_coord_1 + 64
    # So merged coord_map = (out_coord_1 + 64, out_coord_0)
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].left, Var) and cm[0].left.name == placeholder(1).name
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 64
    assert isinstance(cm[1], Var) and cm[1].name == placeholder(0).name


# ---------- IndexMapOp.is_identity ----------


def test_identity_detects_pure_passthrough():
    op = IndexMapOp(
        out_shape=(8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1))),),
    )
    assert op.is_identity((8, 128))


def test_identity_rejects_shape_change():
    op = IndexMapOp(
        out_shape=(1, 8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1), placeholder(2))),),
    )
    assert not op.is_identity((8, 128))


def test_identity_rejects_offset():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(1, "int"),)),),
    )
    assert not op.is_identity((8,))


def test_identity_rejects_select():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0),), select=placeholder(0).lt(Literal(4, "int"))),),
    )
    assert not op.is_identity((8,))


def test_identity_rejects_multisource():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(
            IndexSource(input_idx=0, coord_map=(placeholder(0),)),
            IndexSource(input_idx=1, coord_map=(placeholder(0),)),
        ),
    )
    assert not op.is_identity((8,))


def test_indexmap_infer_output_shape_returns_out_shape():
    op = IndexMapOp(
        out_shape=(4, 5, 6),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1), placeholder(2))),),
    )
    assert op.infer_output_shape([(99, 99, 99)]) == (4, 5, 6)


# ---------- end-to-end: standalone IndexMap kernel ----------


def _compile_and_run_with_data(graph, input_data):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.plan import plan_graph
    from tests.compiler._fusion_helper import auto_fuse

    fused = auto_fuse(graph)
    plan = plan_graph(fused)
    backend = CudaBackend()
    program = backend.compile(plan)
    return backend.run(program, input_data=input_data).outputs


@requires_cuda
def test_indexmap_transpose_matches_torch():
    """Standalone IndexMap implementing transpose(1, 2) on (1, 8, 28, 64) → (1, 28, 8, 64)."""
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import InputOp

    torch.manual_seed(0)
    x = torch.randn(1, 8, 28, 64).cuda()
    expected = x.transpose(1, 2).contiguous().cpu().flatten().tolist()

    g = Graph()
    x_id = g.add_node(InputOp(), [], Tensor("X", (1, 8, 28, 64)), node_id="X")
    g.inputs = [x_id]
    # Transpose (1, 2): coord_map[i] reads input at axis σ⁻¹(i).
    # σ = (0, 2, 1, 3) (output axis 1 reads input axis 2; output axis 2 reads input axis 1).
    out_id = g.add_node(
        IndexMapOp(
            out_shape=(1, 28, 8, 64),
            sources=(
                IndexSource(
                    input_idx=0,
                    coord_map=(placeholder(0), placeholder(2), placeholder(1), placeholder(3)),
                ),
            ),
        ),
        [x_id],
        Tensor("out", (1, 28, 8, 64)),
        node_id="out",
    )
    g.outputs = [out_id]

    outputs = _compile_and_run_with_data(g, {"X": x.cpu().flatten().tolist()})
    actual = list(outputs.values())[0]
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-5, f"transpose IndexMap mismatch: max_diff={max_diff}"


@requires_cuda
def test_indexmap_slice_matches_torch():
    """Standalone IndexMap implementing slice(dim=-1, start=64, end=128) on (1, 28, 8, 128) → (1, 28, 8, 64)."""
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import InputOp

    torch.manual_seed(0)
    x = torch.randn(1, 28, 8, 128).cuda()
    expected = x[..., 64:128].contiguous().cpu().flatten().tolist()

    g = Graph()
    x_id = g.add_node(InputOp(), [], Tensor("X", (1, 28, 8, 128)), node_id="X")
    g.inputs = [x_id]
    # coord_map: identity for first 3 axes, last axis = out_coord_3 + 64.
    out_id = g.add_node(
        IndexMapOp(
            out_shape=(1, 28, 8, 64),
            sources=(
                IndexSource(
                    input_idx=0,
                    coord_map=(
                        placeholder(0),
                        placeholder(1),
                        placeholder(2),
                        placeholder(3) + Literal(64, "int"),
                    ),
                ),
            ),
        ),
        [x_id],
        Tensor("out", (1, 28, 8, 64)),
        node_id="out",
    )
    g.outputs = [out_id]

    outputs = _compile_and_run_with_data(g, {"X": x.cpu().flatten().tolist()})
    actual = list(outputs.values())[0]
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-5, f"slice IndexMap mismatch: max_diff={max_diff}"


@requires_cuda
def test_indexmap_cat_matches_torch():
    """Standalone IndexMap implementing cat([a, b], dim=-1) on (1, 28, 8, 64) + (1, 28, 8, 64) → (1, 28, 8, 128)."""
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import InputOp

    torch.manual_seed(0)
    a = torch.randn(1, 28, 8, 64).cuda()
    b = torch.randn(1, 28, 8, 64).cuda()
    expected = torch.cat([a, b], dim=-1).contiguous().cpu().flatten().tolist()

    g = Graph()
    a_id = g.add_node(InputOp(), [], Tensor("A", (1, 28, 8, 64)), node_id="A")
    b_id = g.add_node(InputOp(), [], Tensor("B", (1, 28, 8, 64)), node_id="B")
    g.inputs = [a_id, b_id]
    # Two sources: source 0 (A) when out_coord_3 < 64, source 1 (B) at out_coord_3 - 64 otherwise.
    src_a = IndexSource(
        input_idx=0,
        coord_map=(placeholder(0), placeholder(1), placeholder(2), placeholder(3)),
        select=placeholder(3).lt(Literal(64, "int")),
    )
    src_b = IndexSource(
        input_idx=1,
        coord_map=(
            placeholder(0),
            placeholder(1),
            placeholder(2),
            placeholder(3) - Literal(64, "int"),
        ),
    )
    out_id = g.add_node(
        IndexMapOp(out_shape=(1, 28, 8, 128), sources=(src_a, src_b)),
        [a_id, b_id],
        Tensor("out", (1, 28, 8, 128)),
        node_id="out",
    )
    g.outputs = [out_id]

    outputs = _compile_and_run_with_data(
        g,
        {"A": a.cpu().flatten().tolist(), "B": b.cpu().flatten().tolist()},
    )
    actual = list(outputs.values())[0]
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-5, f"cat IndexMap mismatch: max_diff={max_diff}"


@requires_cuda
def test_indexmap_identity_alias_no_kernel():
    """An identity IndexMap should produce a buffer alias (zero kernels)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import InputOp
    from deplodock.compiler.plan import plan_graph
    from tests.compiler._fusion_helper import auto_fuse

    g = Graph()
    x_id = g.add_node(InputOp(), [], Tensor("X", (4, 8)), node_id="X")
    g.inputs = [x_id]
    out_id = g.add_node(
        IndexMapOp(
            out_shape=(4, 8),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1))),),
        ),
        [x_id],
        Tensor("out", (4, 8)),
        node_id="out",
    )
    g.outputs = [out_id]

    fused = auto_fuse(g)
    plan = plan_graph(fused)
    program = CudaBackend().compile(plan)
    assert program.aliases.get("out") == "X"
    assert all(launch.kernel_name != "indexmap" and "indexmap" not in launch.kernel_name for launch in program.launches)


# ---------- in-region fusion ----------


@requires_cuda
def test_transpose_indexmap_fuses_with_elementwise():
    """IndexMap (transpose) inside a fused region: result matches eager."""
    import torch

    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, InputOp, KernelOp
    from tests.compiler._fusion_helper import auto_fuse

    torch.manual_seed(0)
    x = torch.randn(1, 28, 8, 64).cuda()
    y = torch.randn(1, 8, 28, 64).cuda()
    expected = (x.transpose(1, 2) * y).cpu().flatten().tolist()

    g = Graph()
    x_id = g.add_node(InputOp(), [], Tensor("X", (1, 28, 8, 64)), node_id="X")
    y_id = g.add_node(InputOp(), [], Tensor("Y", (1, 8, 28, 64)), node_id="Y")
    g.inputs = [x_id, y_id]
    # Transpose X (axes 1,2) → IndexMap with coord swap
    t_id = g.add_node(
        IndexMapOp(
            out_shape=(1, 8, 28, 64),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(2), placeholder(1), placeholder(3))),),
        ),
        [x_id],
        Tensor("T", (1, 8, 28, 64)),
        node_id="T",
    )
    out_id = g.add_node(ElementwiseOp("mul"), [t_id, y_id], Tensor("out", (1, 8, 28, 64)), node_id="out")
    g.outputs = [out_id]

    fused = auto_fuse(g)
    # The IndexMap should be inside a fused region (not standalone).
    standalone_ims = [n for n in fused.nodes.values() if isinstance(n.op, IndexMapOp)]
    fused_with_im = [
        n for n in fused.nodes.values() if isinstance(n.op, KernelOp) and any(isinstance(o, IndexMapOp) for _, o, _ in n.op.body_ops())
    ]
    assert not standalone_ims, f"IndexMap should be absorbed into fused region; standalone: {[n.id for n in standalone_ims]}"
    assert fused_with_im, "Expected at least one fused region containing the IndexMap"

    outputs = _compile_and_run_with_data(g, {"X": x.cpu().flatten().tolist(), "Y": y.cpu().flatten().tolist()})
    actual = list(outputs.values())[0]
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-4, f"transpose+mul fused max_diff={max_diff}"


@requires_cuda
def test_cat_indexmap_in_fused_region_emits_ternary():
    """Multi-source IndexMap (cat) inside a region: kernel uses Ternary load."""
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, InputOp
    from deplodock.compiler.plan import plan_graph
    from tests.compiler._fusion_helper import auto_fuse

    torch.manual_seed(0)
    a = torch.randn(1, 28, 8, 64).cuda()
    b = torch.randn(1, 28, 8, 64).cuda()
    y = torch.randn(1, 28, 8, 128).cuda()
    expected = (torch.cat([a, b], dim=-1) * y).cpu().flatten().tolist()

    g = Graph()
    a_id = g.add_node(InputOp(), [], Tensor("A", (1, 28, 8, 64)), node_id="A")
    b_id = g.add_node(InputOp(), [], Tensor("B", (1, 28, 8, 64)), node_id="B")
    y_id = g.add_node(InputOp(), [], Tensor("Y", (1, 28, 8, 128)), node_id="Y")
    g.inputs = [a_id, b_id, y_id]
    src_a = IndexSource(
        input_idx=0,
        coord_map=(placeholder(0), placeholder(1), placeholder(2), placeholder(3)),
        select=placeholder(3).lt(Literal(64, "int")),
    )
    src_b = IndexSource(
        input_idx=1,
        coord_map=(placeholder(0), placeholder(1), placeholder(2), placeholder(3) - Literal(64, "int")),
    )
    cat_id = g.add_node(
        IndexMapOp(out_shape=(1, 28, 8, 128), sources=(src_a, src_b)),
        [a_id, b_id],
        Tensor("cat", (1, 28, 8, 128)),
        node_id="cat",
    )
    out_id = g.add_node(ElementwiseOp("mul"), [cat_id, y_id], Tensor("out", (1, 28, 8, 128)), node_id="out")
    g.outputs = [out_id]

    fused = auto_fuse(g)
    plan = plan_graph(fused)
    program = CudaBackend().compile(plan)
    # The cat IndexMap should be inside a fused region; its kernel source should
    # contain a ternary (?:) for the source selection.
    ternary_kernels = [launch for launch in program.launches if "?" in launch.kernel_source and ":" in launch.kernel_source]
    assert ternary_kernels, "Expected a fused-region kernel with a ?: ternary for the cat select"

    result = CudaBackend().run(
        program,
        input_data={
            "A": a.cpu().flatten().tolist(),
            "B": b.cpu().flatten().tolist(),
            "Y": y.cpu().flatten().tolist(),
        },
    )
    actual = list(result.outputs.values())[0]
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-4, f"cat+mul fused max_diff={max_diff}"


def test_qwen_rotary_chain_fuses_into_region():
    """After rewrite + auto_fuse, Qwen's rotary chain absorbs IndexMaps + neg into one region per rotary side."""
    import json
    from pathlib import Path

    from deplodock.compiler.ir import Graph
    from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, KernelOp
    from deplodock.compiler.rewriter import Rewriter
    from tests.compiler._fusion_helper import auto_fuse

    fixture = Path(__file__).parent / "fixtures" / "qwen25_7b_layer0.json"
    with open(fixture) as f:
        g = Graph.from_dict(json.load(f))
    g = Rewriter.from_directory(Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules").apply(g)
    fused = auto_fuse(g)

    # Count regions that absorbed an IndexMap.
    indexmap_regions = [
        n for n in fused.nodes.values() if isinstance(n.op, KernelOp) and any(isinstance(o, IndexMapOp) for _, o, _ in n.op.body_ops())
    ]
    assert len(indexmap_regions) >= 2, f"Expected ≥2 fused regions absorbing IndexMaps (Q + K rotary chains); got {len(indexmap_regions)}"

    # Standalone elementwise count should drop to ~1 (only add_5 residual).
    standalone_em = [nid for nid, n in fused.nodes.items() if isinstance(n.op, ElementwiseOp)]
    assert len(standalone_em) <= 2, f"Too many standalone elementwise ops post-fusion: {standalone_em}"
