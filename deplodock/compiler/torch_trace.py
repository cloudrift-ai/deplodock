"""Trace a PyTorch module and convert to compiler Graph IR.

Requires PyTorch (optional dependency). All torch imports are guarded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    GatherOp,
    InputOp,
    ReduceOp,
    ReshapeOp,
    TransposeOp,
)

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
    """Trace a PyTorch module and convert the FX graph to our IR.

    Args:
        module: PyTorch module to trace (e.g., a single transformer layer).
        example_inputs: Tuple of example input tensors for tracing.
        kwargs: Optional keyword arguments for the module forward.

    Returns:
        Graph in our IR with InputOp, ConstantOp, ElementwiseOp, etc.
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

    return g


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


def _resolve_input(fx_node: Any, node_map: dict[str, str]) -> str | None:
    """Resolve an FX node argument to our node ID."""
    if hasattr(fx_node, "name") and fx_node.name in node_map:
        return node_map[fx_node.name]
    return None


def _resolve_inputs(fx_node: Any, node_map: dict[str, str], g: Graph | None = None) -> list[str]:
    """Resolve FX node args to our node IDs. Scalars become ConstantOp nodes."""
    result = []
    for a in fx_node.args:
        if hasattr(a, "name") and a.name in node_map:
            result.append(node_map[a.name])
        elif isinstance(a, (int, float)) and g is not None:
            # Create a constant node for scalar args (e.g., eps=1e-5).
            const_name = f"{fx_node.name}_c{len(result)}"
            const_id = g.add_node(
                op=ConstantOp(name=const_name),
                inputs=[],
                output=Tensor(const_name, (1,), "f32"),
                node_id=const_name,
            )
            node_map[const_name] = const_id
            result.append(const_id)
    return result


def _handle_placeholder(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, str],
    module: nn.Module,
) -> None:
    """Handle placeholder nodes (inputs and parameters)."""
    name = fx_node.name
    shape = _get_shape(fx_node)
    dtype = _get_dtype(fx_node)

    # Parameters start with "p_" in torch.export convention.
    is_param = name.startswith("p_")

    if is_param:
        nid = g.add_node(
            op=ConstantOp(name=name),
            inputs=[],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
    else:
        nid = g.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
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
    # e.g. "aten.mul.Tensor" → "mul", "aten.linear.default" → "linear"
    if "aten." in s:
        parts = s.split(".")
        for i, p in enumerate(parts):
            if p == "aten" and i + 1 < len(parts):
                return parts[i + 1]
    return None


def _handle_call_function(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, str],
) -> None:
    """Handle call_function nodes (ATen ops)."""
    name = fx_node.name
    shape = _get_shape(fx_node)
    dtype = _get_dtype(fx_node)
    op_name = _op_name(fx_node.target)
    input_ids = _resolve_inputs(fx_node, node_map, g)

    if op_name is None:
        # Skip non-ATen ops (assertions, metadata, etc.)
        if input_ids:
            # Pass through the first input.
            node_map[name] = input_ids[0]
        return

    # --- Elementwise ops ---
    ew_map = {
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
    }
    if op_name in ew_map:
        nid = g.add_node(
            op=ElementwiseOp(fn=ew_map[op_name]),
            inputs=input_ids[:2],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # pow(x, 2) is elementwise square (used in RMSNorm).
    if op_name == "pow":
        nid = g.add_node(
            op=ElementwiseOp(fn="pow"),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Linear (high-level matmul, not decomposed by torch.export) ---
    if op_name == "linear":
        # aten.linear(input, weight, bias=None)
        # Decompose into Elementwise{mul} + Reduce{sum}.
        inp_id = input_ids[0] if len(input_ids) > 0 else None
        w_id = input_ids[1] if len(input_ids) > 1 else None
        if inp_id is None or w_id is None:
            logger.warning("Could not resolve linear inputs for %s", name)
            return

        ew_name = f"{name}_ew"
        ew_id = g.add_node(
            op=ElementwiseOp(fn="mul"),
            inputs=[inp_id, w_id],
            output=Tensor(ew_name, shape + ("K",), dtype),
            node_id=ew_name,
        )
        red_id = g.add_node(
            op=ReduceOp(fn="sum", axis="K"),
            inputs=[ew_id],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = red_id

        # Handle bias if present.
        bias_id = input_ids[2] if len(input_ids) > 2 else None
        if bias_id:
            add_name = f"{name}_bias"
            add_id = g.add_node(
                op=ElementwiseOp(fn="add"),
                inputs=[red_id, bias_id],
                output=Tensor(add_name, shape, dtype),
                node_id=add_name,
            )
            node_map[name] = add_id
        return

    # --- Matmul / mm ---
    if op_name in ("mm", "matmul", "addmm"):
        if op_name == "addmm":
            inp_id = input_ids[1] if len(input_ids) > 1 else None
            w_id = input_ids[2] if len(input_ids) > 2 else None
            bias_id = input_ids[0] if len(input_ids) > 0 else None
        else:
            inp_id = input_ids[0] if len(input_ids) > 0 else None
            w_id = input_ids[1] if len(input_ids) > 1 else None
            bias_id = None

        if inp_id and w_id:
            ew_name = f"{name}_ew"
            ew_id = g.add_node(
                op=ElementwiseOp(fn="mul"),
                inputs=[inp_id, w_id],
                output=Tensor(ew_name, shape + ("K",), dtype),
                node_id=ew_name,
            )
            red_id = g.add_node(
                op=ReduceOp(fn="sum", axis="K"),
                inputs=[ew_id],
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
            node_map[name] = red_id
            if bias_id:
                add_name = f"{name}_bias"
                add_id = g.add_node(
                    op=ElementwiseOp(fn="add"),
                    inputs=[red_id, bias_id],
                    output=Tensor(add_name, shape, dtype),
                    node_id=add_name,
                )
                node_map[name] = add_id
        return

    # --- Reductions ---
    if op_name in ("sum", "mean"):
        axis = _get_reduce_axis(fx_node)
        nid = g.add_node(
            op=ReduceOp(fn="sum", axis=axis),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name in ("amax", "max"):
        axis = _get_reduce_axis(fx_node)
        nid = g.add_node(
            op=ReduceOp(fn="max", axis=axis),
            inputs=input_ids[:1],
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

    # --- Ops that pass through (no-ops for our IR) ---
    if op_name in ("to", "contiguous", "_assert_tensor_metadata", "clone", "detach"):
        if input_ids:
            node_map[name] = input_ids[0]
        return

    # --- Slice / unsqueeze / squeeze / cat: structural ops ---
    if op_name in ("slice", "unsqueeze", "squeeze", "expand", "permute"):
        if input_ids:
            nid = g.add_node(
                op=ReshapeOp(shape=shape),
                inputs=input_ids[:1],
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
            node_map[name] = nid
        return

    if op_name == "cat":
        # Cat with multiple inputs — represent as elementwise for now.
        nid = g.add_node(
            op=ElementwiseOp(fn="cat"),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Scaled dot product attention (high-level, not decomposed) ---
    if op_name == "scaled_dot_product_attention":
        # Only take Q, K, V (first 3 inputs). Extra args (dropout_p, is_causal) are dropped.
        nid = g.add_node(
            op=ElementwiseOp(fn="sdpa"),
            inputs=input_ids[:3],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Gather / index_select ---
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

    # --- Fallback: treat as elementwise ---
    logger.debug("Fallback elementwise for %s (%s)", op_name, fx_node.target)
    if input_ids:
        nid = g.add_node(
            op=ElementwiseOp(fn=op_name),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid


def _get_reduce_axis(fx_node: Any) -> int | str:
    """Extract the reduction axis from an FX node."""
    if len(fx_node.args) > 1:
        axis = fx_node.args[1]
        if isinstance(axis, (list, tuple)):
            return axis[0] if len(axis) == 1 else axis[0]
        if isinstance(axis, int):
            return axis
    return -1
