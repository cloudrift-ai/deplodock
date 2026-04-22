"""Trace a PyTorch module and convert to Torch IR (faithful capture).

The tracer creates one graph node per FX op, using PyTorch's exact shapes.
No decomposition, no skipping, no shape overrides.  Decomposition into
primitive Deplodock IR ops happens in separate rewriter passes.

Requires PyTorch (optional dependency). All torch imports are guarded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend_ir import (
    CatOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    SdpaOp,
    SliceOp,
    TransposeOp,
    UnsqueezeOp,
)
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor_ir import ElementwiseOp, GatherOp, IndexMapOp, IndexSource, ReduceOp

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


def has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def trace_module(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
) -> Graph:
    """Trace a PyTorch module and convert the FX graph to our IR."""
    graph, _ = trace_module_with_constants(module, example_inputs, kwargs=kwargs)
    return graph


def trace_module_with_constants(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
) -> tuple[Graph, dict[str, str]]:
    """Trace a module and return the IR graph plus a placeholder→attribute map.

    The second return value maps each graph-level constant name (``p_*`` /
    ``b_*``) to the dotted attribute path on ``module`` where the tensor
    lives (e.g. ``self_attn.q_proj.weight``). ``torch.export`` sometimes
    strips prefixes like ``self_`` from the placeholder name, so this map
    is needed to feed constants at runtime.
    """
    import torch

    exported = torch.export.export(module, example_inputs, kwargs=kwargs or {})
    gm = exported.graph_module

    g = Graph()
    node_map: dict[str, str] = {}

    for fx_node in gm.graph.nodes:
        if fx_node.op == "placeholder":
            _handle_placeholder(g, fx_node, node_map, module)
        elif fx_node.op == "call_function":
            _handle_call_function(g, fx_node, node_map)
        elif fx_node.op == "output":
            _handle_output(g, fx_node, node_map)
        else:
            logger.debug("Skipping FX node: %s (op=%s)", fx_node.name, fx_node.op)

    const_targets: dict[str, str] = {}
    sig = exported.graph_signature
    const_targets.update(sig.inputs_to_parameters)
    const_targets.update(sig.inputs_to_buffers)
    return g, const_targets


def _get_shape(fx_node: Any) -> tuple:
    meta = fx_node.meta.get("val", None)
    if meta is not None and hasattr(meta, "shape"):
        return tuple(meta.shape)
    return ()


def _get_dtype(fx_node: Any) -> str:
    meta = fx_node.meta.get("val", None)
    if meta is not None and hasattr(meta, "dtype"):
        return str(meta.dtype).replace("torch.", "")
    return "f32"


def _resolve_inputs(fx_node: Any, node_map: dict[str, str], g: Graph | None = None) -> list[str]:
    """Resolve FX node args to our node IDs. Scalars become ConstantOp nodes."""
    result = []
    for a in fx_node.args:
        if hasattr(a, "name") and a.name in node_map:
            result.append(node_map[a.name])
        elif isinstance(a, (list, tuple)):
            for item in a:
                if hasattr(item, "name") and item.name in node_map:
                    result.append(node_map[item.name])
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and g is not None:
            const_name = f"{fx_node.name}_c{len(result)}"
            const_id = g.add_node(
                op=ConstantOp(name=const_name, value=float(a)),
                inputs=[],
                output=Tensor(const_name, (1,), "f32"),
                node_id=const_name,
            )
            node_map[const_name] = const_id
            result.append(const_id)
    return result


def _handle_placeholder(g: Graph, fx_node: Any, node_map: dict[str, str], module: nn.Module) -> None:
    """Handle placeholder nodes (inputs, parameters, and buffers).

    ``torch.export`` prefixes parameter placeholders with ``p_`` and buffer
    placeholders with ``b_``. Both are bake-in constants from the compiler's
    perspective — only actual user-supplied activations (no prefix) are
    graph inputs.
    """
    name = fx_node.name
    shape = _get_shape(fx_node)
    dtype = _get_dtype(fx_node)
    is_const = name.startswith(("p_", "b_"))

    if is_const:
        nid = g.add_node(op=ConstantOp(name=name), inputs=[], output=Tensor(name, shape, dtype), node_id=name)
    else:
        nid = g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape, dtype), node_id=name)
        g.inputs.append(nid)
    node_map[name] = nid


def _handle_output(g: Graph, fx_node: Any, node_map: dict[str, str]) -> None:
    """Handle output nodes."""
    for arg in fx_node.args[0] if isinstance(fx_node.args[0], (tuple, list)) else [fx_node.args[0]]:
        if hasattr(arg, "name") and arg.name in node_map:
            g.outputs.append(node_map[arg.name])


def _op_name(target: Any) -> str | None:
    """Extract a short op name from an ATen target."""
    s = str(target)
    if "aten." in s:
        parts = s.split(".")
        for i, p in enumerate(parts):
            if p == "aten" and i + 1 < len(parts):
                return parts[i + 1]
    return None


def _get_reduce_axis(fx_node: Any) -> int | str:
    """Extract the reduction axis from an FX node."""
    if len(fx_node.args) > 1:
        axis = fx_node.args[1]
        if isinstance(axis, (list, tuple)):
            return axis[0] if len(axis) == 1 else axis[0]
        if isinstance(axis, int):
            return axis
    return -1


def _keepdim_shape(input_shape: tuple, axis: int | str) -> tuple:
    """Return the keepdim output shape for a reduction over ``axis``."""
    if not isinstance(axis, int) or not input_shape:
        return tuple(input_shape)
    a = axis if axis >= 0 else len(input_shape) + axis
    if a < 0 or a >= len(input_shape):
        return tuple(input_shape)
    return tuple(input_shape[:a]) + (1,) + tuple(input_shape[a + 1 :])


def _squeeze_indexmap(in_shape: tuple, out_shape: tuple, axis: int | str) -> IndexMapOp:
    """Build an IndexMapOp that drops a single size-1 axis.

    ``in_shape`` is the keepdim-reduce output (rank N, dim ``axis`` = 1).
    ``out_shape`` is the non-keepdim shape (rank N-1, ``axis`` removed).
    """
    if not isinstance(axis, int):
        # Symbolic axis fallback — identity map (no safety net).
        coord_map = tuple(placeholder(d) for d in range(len(out_shape)))
        return IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=coord_map),))
    a = axis if axis >= 0 else len(in_shape) + axis
    coord_map = []
    out_d = 0
    for in_d in range(len(in_shape)):
        if in_d == a:
            coord_map.append(Literal(0, "int"))
        else:
            coord_map.append(placeholder(out_d))
            out_d += 1
    return IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),))


def _handle_call_function(g: Graph, fx_node: Any, node_map: dict[str, str]) -> None:
    """Handle call_function nodes — faithful 1:1 capture of FX ops."""
    name = fx_node.name
    shape = _get_shape(fx_node)
    dtype = _get_dtype(fx_node)
    op_name = _op_name(fx_node.target)
    input_ids = _resolve_inputs(fx_node, node_map, g)

    if op_name is None:
        if input_ids:
            node_map[name] = input_ids[0]
        return

    # --- Elementwise ops ---
    _EW_MAP = {
        "add": "add",
        "mul": "mul",
        "sub": "sub",
        "div": "div",
        "neg": "neg",
        "exp": "exp",
        "rsqrt": "rsqrt",
        "reciprocal": "recip",
        "silu": "silu",
        "relu": "relu",
        "tanh": "tanh",
        "abs": "abs",
        "sigmoid": "sigmoid",
        "pow": "pow",
    }
    if op_name in _EW_MAP:
        from deplodock.compiler.ir.broadcast import broadcast_to

        bc_ids = [broadcast_to(g, inp, shape) for inp in input_ids[:2]]
        nid = g.add_node(
            op=ElementwiseOp(fn=_EW_MAP[op_name]),
            inputs=bc_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Linear ---
    if op_name == "linear":
        has_bias = len(input_ids) > 2 and input_ids[2] in g.nodes
        nid = g.add_node(
            op=LinearOp(has_bias=has_bias),
            inputs=input_ids[:3] if has_bias else input_ids[:2],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Matmul ---
    if op_name in ("mm", "matmul"):
        nid = g.add_node(
            op=MatmulOp(),
            inputs=input_ids[:2],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name == "addmm":
        nid = g.add_node(
            op=MatmulOp(has_bias=True),
            inputs=[input_ids[1], input_ids[2], input_ids[0]] if len(input_ids) >= 3 else input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- SDPA ---
    if op_name == "scaled_dot_product_attention":
        # args: (Q, K, V, attn_mask, dropout_p, is_causal)
        is_causal = False
        for a in fx_node.args[3:]:
            if isinstance(a, bool):
                is_causal = a
                break
        nid = g.add_node(
            op=SdpaOp(is_causal=is_causal),
            inputs=input_ids[:3],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Reductions ---
    # Reductions are always emitted as keepdim (rank-preserving). If the
    # traced op was non-keepdim, a squeeze IndexMapOp is inserted afterwards
    # so the downstream graph sees the correct (dropped-axis) shape while the
    # ReduceOp itself stays rank-preserving.
    _RED_OP_MAP = {"sum": "sum", "mean": "mean", "amax": "max", "max": "max"}
    if op_name in _RED_OP_MAP:
        axis = _get_reduce_axis(fx_node)
        x_shape = tuple(g.nodes[input_ids[0]].output.shape) if input_ids else ()
        keepdim_shape = _keepdim_shape(x_shape, axis)
        fn_name = _RED_OP_MAP[op_name]
        if op_name == "mean":
            red_node_op = MeanOp(axis=axis)
        else:
            red_node_op = ReduceOp(fn=fn_name, axis=axis)

        if tuple(shape) == keepdim_shape:
            nid = g.add_node(op=red_node_op, inputs=input_ids[:1], output=Tensor(name, shape, dtype), node_id=name)
        else:
            # Emit keepdim reduce + squeeze IndexMapOp.
            red_id = g.add_node(op=red_node_op, inputs=input_ids[:1], output=Tensor(f"{name}_keepdim", keepdim_shape, dtype))
            nid = g.add_node(
                op=_squeeze_indexmap(keepdim_shape, shape, axis),
                inputs=[red_id],
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
        node_map[name] = nid
        return

    # --- Transpose ---
    if op_name == "transpose":
        dim0 = fx_node.args[1] if len(fx_node.args) > 1 else 0
        dim1 = fx_node.args[2] if len(fx_node.args) > 2 else 1
        nid = g.add_node(
            op=TransposeOp(axes=(dim0, dim1)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name == "t":
        nid = g.add_node(
            op=TransposeOp(axes=(1, 0)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Reshape / view ---
    if op_name in ("view", "reshape", "_unsafe_view"):
        new_shape = fx_node.args[1] if len(fx_node.args) > 1 else shape
        if isinstance(new_shape, (list, tuple)):
            new_shape = tuple(new_shape)
        nid = g.add_node(
            op=ReshapeOp(shape=new_shape),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Unsqueeze ---
    if op_name == "unsqueeze":
        dim = fx_node.args[1] if len(fx_node.args) > 1 and isinstance(fx_node.args[1], int) else 0
        nid = g.add_node(
            op=UnsqueezeOp(dim=dim),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Squeeze / permute / expand ---
    if op_name in ("squeeze", "expand", "permute"):
        nid = g.add_node(
            op=ReshapeOp(shape=shape),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Pass-through ---
    if op_name in ("to", "contiguous", "_assert_tensor_metadata", "clone", "detach", "alias"):
        if input_ids:
            node_map[name] = input_ids[0]
        return

    # --- Slice ---
    if op_name == "slice":
        if input_ids:
            nid = g.add_node(
                op=SliceOp(shape=shape),
                inputs=input_ids,
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
            node_map[name] = nid
        return

    # --- Cat ---
    if op_name == "cat":
        nid = g.add_node(
            op=CatOp(),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Gather ---
    if op_name in ("index_select", "gather", "embedding"):
        axis = fx_node.args[1] if len(fx_node.args) > 1 and isinstance(fx_node.args[1], int) else 0
        nid = g.add_node(
            op=GatherOp(axis=axis),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Fallback ---
    logger.debug("Fallback elementwise for %s (%s)", op_name, fx_node.target)
    if input_ids:
        from deplodock.compiler.ir.broadcast import broadcast_to

        # Fused-op fallbacks (rms_norm, softmax, …) intentionally keep their
        # smaller inputs unbroadcast — their decomposition rule owns the
        # broadcast insertion for the primitives they emit. For any other
        # op (real elementwise), wrap mismatched inputs so ElementwiseOp's
        # matching-shape invariant holds.
        if op_name not in ("rms_norm", "softmax"):
            input_ids = [broadcast_to(g, inp, shape) for inp in input_ids]
        nid = g.add_node(
            op=ElementwiseOp(fn=op_name),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
