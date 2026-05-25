"""Trace a PyTorch module and convert to Torch IR (faithful capture).

The tracer creates one graph node per FX op, using PyTorch's exact shapes.
No decomposition, no skipping, no shape overrides.  Decomposition into
primitive Deplodock IR ops happens in separate rewriter passes.

Requires PyTorch (optional dependency). All torch imports are guarded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import (
    CatOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    RmsNormOp,
    SdpaOp,
    SliceOp,
    SoftmaxOp,
    TransposeOp,
    UnsqueezeOp,
)
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, IndexMapOp, IndexSource, ReduceOp

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
    *,
    dynamic_shapes: dict | None = None,
) -> Graph:
    """Trace a PyTorch module and convert the FX graph to our IR.

    ``dynamic_shapes`` flows straight through to ``torch.export.export``.
    Pass a nested dict like ``{"input": {1: torch.export.Dim("seq_len")}}``
    to mark axis 1 of the ``input`` argument as dynamic — the resulting
    graph carries ``Dim('seq_len')`` at every position where torch's
    SymInt propagated, no post-trace shape rewrite needed.
    """
    graph, _ = trace_module_with_constants(module, example_inputs, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    return graph


def trace_module_with_constants(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict | None = None,
) -> tuple[Graph, dict[str, str]]:
    """Trace a module and return the IR graph plus a placeholder→attribute map.

    The second return value maps each graph-level constant name (``p_*`` /
    ``b_*``) to the dotted attribute path on ``module`` where the tensor
    lives (e.g. ``self_attn.q_proj.weight``). ``torch.export`` sometimes
    strips prefixes like ``self_`` from the placeholder name, so this map
    is needed to feed constants at runtime.

    ``dynamic_shapes`` is forwarded to ``torch.export.export``. When set,
    the FX nodes' ``meta["val"]`` shapes contain ``SymInt`` entries for
    every dynamic axis; the FX→IR walker converts those to symbolic
    ``Dim`` instances. Internal names torch assigns (``s0``, ``s1``,
    …) are renamed back to the user-supplied ``Dim`` names from
    ``dynamic_shapes`` so the resulting IR reads as ``Dim('seq_len')``.
    """
    import time

    import torch

    t0 = time.monotonic()
    logger.info("torch.export.export() starting...")
    expanded_dynamic = _expand_dynamic_shapes(module, example_inputs, kwargs or {}, dynamic_shapes) if dynamic_shapes else None
    exported = torch.export.export(module, example_inputs, kwargs=kwargs or {}, dynamic_shapes=expanded_dynamic)
    gm = exported.graph_module
    t1 = time.monotonic()
    n_fx_nodes = sum(1 for _ in gm.graph.nodes)
    logger.info("torch.export.export() done in %.1fs (%d FX nodes)", t1 - t0, n_fx_nodes)

    sym_rename = _sym_rename_map(exported, dynamic_shapes) if dynamic_shapes else {}

    g = Graph()
    node_map: dict[str, str] = {}

    sig = exported.graph_signature
    const_targets: dict[str, str] = {}
    const_targets.update(sig.inputs_to_parameters)
    const_targets.update(sig.inputs_to_buffers)

    for fx_node in gm.graph.nodes:
        if fx_node.op == "placeholder":
            _handle_placeholder(g, fx_node, node_map, module, const_targets, sym_rename=sym_rename)
        elif fx_node.op == "call_function":
            _handle_call_function(g, fx_node, node_map, sym_rename=sym_rename)
        elif fx_node.op == "output":
            _handle_output(g, fx_node, node_map)
        else:
            logger.debug("Skipping FX node: %s (op=%s)", fx_node.name, fx_node.op)
    logger.info("FX→Graph IR walk done in %.1fs (%d IR nodes)", time.monotonic() - t1, len(g.nodes))

    return g, const_targets


def _expand_dynamic_shapes(module, example_inputs: tuple, kwargs: dict, user_dynamic: dict) -> dict:
    """Auto-fill ``None`` for any forward-arg the user didn't mark dynamic.

    ``torch.export.export(dynamic_shapes=...)`` requires the top-level
    dict to match EXACTLY the args / kwargs actually passed (not the
    forward signature). Container-typed args (tuple of tensors for HF's
    ``position_embeddings`` etc.) need a structurally-matching spec,
    not a bare ``None``, or torch raises on the type mismatch."""
    import inspect

    sig = inspect.signature(module.forward)
    sig_params = [n for n in sig.parameters if n != "self"]
    positional_names = sig_params[: len(example_inputs)]
    expected = list(zip(positional_names, example_inputs, strict=True)) + list(kwargs.items())
    out: dict[str, object] = {}
    for name, value in expected:
        out[name] = user_dynamic.get(name) if name in user_dynamic else _static_spec_for(value)
    # Pass through any user keys that weren't expected so torch's error
    # message stays informative (e.g. typo'd input names).
    for name, spec in user_dynamic.items():
        out.setdefault(name, spec)
    return out


def _static_spec_for(value) -> object:
    """Build a structurally-matching ``None`` spec for an input value.

    Bare tensor → ``None``. Tuple / list of tensors → matching tuple /
    list of ``None``. Anything else → ``None`` (torch accepts that for
    non-tensor inputs)."""
    if isinstance(value, tuple):
        return tuple(_static_spec_for(v) for v in value)
    if isinstance(value, list):
        return [_static_spec_for(v) for v in value]
    return None


def _sym_rename_map(exported, dynamic_shapes: dict) -> dict[str, str]:
    """Build a ``{torch-internal-symbol-name: user-Dim-name}`` mapping.

    ``torch.export`` invents internal names (``s0``, ``s1``, …) for each
    dynamic dim, but the user passed in ``torch.export.Dim('seq_len')``
    and expects ``Dim('seq_len')`` to show up in the resulting IR.
    Match by walking ``exported.graph_signature.input_specs`` against the
    ``dynamic_shapes`` keys, then inspecting the placeholder FX node's
    ``meta["val"].shape[axis]`` to find the SymInt torch assigned at
    that position.
    """
    rename: dict[str, str] = {}
    gm = exported.graph_module
    # Map arg-name → placeholder FX node. ``torch.export`` may rename
    # placeholders (strip ``self_`` etc.), but the signature carries
    # the original user-arg-name → placeholder-name mapping.
    sig = exported.graph_signature
    user_to_placeholder: dict[str, str] = {}
    for spec in sig.input_specs:
        # ``user_inputs`` specs have ``arg.name == placeholder.name``;
        # parameter / buffer specs aren't user-facing inputs.
        if getattr(spec, "kind", None) is not None and spec.kind.name == "USER_INPUT":
            user_to_placeholder[spec.arg.name] = spec.arg.name
    placeholder_by_name: dict[str, Any] = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    for arg_name, axis_map in dynamic_shapes.items():
        ph_name = user_to_placeholder.get(arg_name, arg_name)
        ph = placeholder_by_name.get(ph_name)
        if ph is None:
            continue
        val = ph.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            continue
        for axis, user_dim in axis_map.items():
            if axis >= len(val.shape):
                continue
            sym_int = val.shape[axis]
            # ``SymInt`` carries the symbol via ``.node.expr`` (sympy
            # ``Symbol``); ``str(symbol)`` is the internal name like ``s0``.
            sym_name = _symint_name(sym_int)
            if sym_name is None:
                continue
            rename[sym_name] = user_dim.__name__ if hasattr(user_dim, "__name__") else str(user_dim)
    return rename


def _symint_name(value) -> str | None:
    """Return the underlying sympy symbol name for a ``SymInt`` placeholder,
    or ``None`` if ``value`` is a plain int (the static-axis case)."""
    if isinstance(value, int):
        return None
    node = getattr(value, "node", None)
    if node is None:
        return None
    expr = getattr(node, "expr", None) or node
    # Sympy Symbols stringify to their name (``s0``); compound expressions
    # would stringify to e.g. ``2*s0`` — we only support plain symbols today.
    return str(expr)


def _wrap_shape(raw_shape, sym_rename: dict[str, str] | None = None):
    """Convert a torch ``Size`` (possibly containing ``SymInt``) to the
    tuple form our IR expects.

    Plain ints pass through. ``SymInt`` placeholders become ``Dim(name)``
    with ``name`` resolved through ``sym_rename`` (so torch's ``s0``
    becomes the user's ``seq_len``); unrenamed symbols keep their
    torch-internal name. The static-shape case (no SymInt anywhere)
    returns a ``tuple[int, ...]`` — the existing IR construction path
    coerces those to ``Dim(int)`` via ``Tensor.__post_init__``.
    """
    from deplodock.compiler.dim import Dim

    if sym_rename is None:
        sym_rename = {}
    out = []
    for d in raw_shape:
        if isinstance(d, int):
            out.append(d)
        else:
            sym_name = _symint_name(d)
            if sym_name is None:
                # Compound expression we can't represent — fall back to a stringified placeholder.
                out.append(Dim(str(d)))
            else:
                out.append(Dim(sym_rename.get(sym_name, sym_name)))
    return tuple(out)


def _get_shape(fx_node: Any, sym_rename: dict[str, str] | None = None) -> tuple:
    meta = fx_node.meta.get("val", None)
    if meta is not None and hasattr(meta, "shape"):
        return _wrap_shape(meta.shape, sym_rename)
    return ()


def _op_shape(raw_shape, sym_rename: dict[str, str] | None = None):
    """Convert an FX-arg shape (``view``/``reshape``'s second arg, etc.) to
    the ``tuple[int | str, ...]`` form ``ReshapeOp.shape`` /
    ``SliceOp.shape`` expect.

    Ints pass through. ``-1`` (numpy/torch infer-this-dim sentinel) passes
    through. FX node references to ``aten.sym_size.int`` outputs (which
    appear in reshape arg lists as ``[1, sym_size_int_1, 4, -1]``)
    resolve through ``meta["val"]`` to the underlying ``SymInt``.
    ``SymInt`` becomes the renamed symbolic name (``s0`` → ``seq_len``).
    Unknown values get stringified as a fallback so we don't silently
    lose information.
    """
    sym_rename = sym_rename or {}
    out = []
    for d in raw_shape:
        if isinstance(d, int):
            out.append(d)
            continue
        # FX node reference: resolve to ``meta["val"]`` (a SymInt scalar).
        meta = getattr(d, "meta", None)
        if isinstance(meta, dict):
            val = meta.get("val")
            if val is not None:
                d = val
        sym = _symint_name(d)
        if sym is not None:
            out.append(sym_rename.get(sym, sym))
        else:
            out.append(str(d))
    return tuple(out)


def _dim_tuple_to_op_shape(shape):
    """Convert a ``tuple[Dim | int, ...]`` (from ``_wrap_shape`` output) back
    to the ``tuple[int | str, ...]`` form ``ReshapeOp.shape`` /
    ``SliceOp.shape`` carry. ``Dim`` wrappers unwrap to ``.value``."""
    from deplodock.compiler.dim import Dim

    return tuple(d.value if isinstance(d, Dim) else d for d in shape)


def _get_dtype(fx_node: Any) -> str:
    meta = getattr(fx_node, "meta", None)
    val = meta.get("val", None) if isinstance(meta, dict) else None
    if val is not None and hasattr(val, "dtype"):
        return str(val.dtype).replace("torch.", "")
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
            # Inherit dtype from the consuming op's output — scalar literals
            # in mixed-dtype graphs (fp16 * 0.5, etc.) must stay in the
            # consumer's dtype to avoid widening every elementwise step.
            const_id = g.add_node(
                op=ConstantOp(name=const_name, value=float(a)),
                inputs=[],
                output=Tensor(const_name, (1,), _get_dtype(fx_node)),
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
    const_targets: dict[str, str],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Handle placeholder nodes (inputs, parameters, and buffers).

    ``torch.export`` prefixes parameter placeholders with ``p_`` and buffer
    placeholders with ``b_``. Both are bake-in constants from the compiler's
    perspective — only actual user-supplied activations (no prefix) are
    graph inputs.

    Parameter / buffer ``ConstantOp``s carry the source attribute path
    (``source_path``) and pre-chain layout (``source_shape`` / ``source_dtype``)
    so the loader can read them directly from safetensors without a
    side-channel ``const_targets`` dict.
    """
    name = fx_node.name
    shape = _get_shape(fx_node, sym_rename)
    dtype = _get_dtype(fx_node)
    is_const = name.startswith(("p_", "b_"))

    if is_const:
        op = ConstantOp(
            name=name,
            source_path=const_targets.get(name),
            source_shape=tuple(shape) if shape else None,
            source_dtype=dtype,
        )
        nid = g.add_node(op=op, inputs=[], output=Tensor(name, shape, dtype), node_id=name)
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


def _handle_call_function(g: Graph, fx_node: Any, node_map: dict[str, str], *, sym_rename: dict[str, str] | None = None) -> None:
    """Handle call_function nodes — faithful 1:1 capture of FX ops."""
    # ``aten.sym_size.int`` and similar shape-metadata ops return a
    # scalar ``SymInt`` (no tensor shape). They're consumed inline by
    # ``_op_shape`` when a downstream reshape / view references them via
    # ``args``; making them graph nodes would only confuse the matcher.
    val = fx_node.meta.get("val")
    if val is not None and not hasattr(val, "shape"):
        return
    name = fx_node.name
    shape = _get_shape(fx_node, sym_rename)
    dtype = _get_dtype(fx_node)
    op_name = _op_name(fx_node.target)
    input_ids = _resolve_inputs(fx_node, node_map, g)

    if op_name is None:
        if input_ids:
            node_map[name] = input_ids[0]
        return

    # --- Elementwise ops ---
    # Torch's aten-level short names (``sub`` / ``mul`` / ``div`` / ``neg``)
    # get translated to numpy-style long names here so the rest of the
    # pipeline can read our ``ElementwiseOp.op.name`` as a numpy attribute
    # (``np.subtract`` / ``np.multiply`` / …) without further aliasing.
    # Names that already match numpy (``add`` / ``mod`` / ``pow`` / ``exp`` /
    # ``tanh`` / ``abs`` / ``sqrt`` / ``reciprocal`` / …) pass through.
    _ATEN_TO_NUMPY = {
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "neg": "negative",
    }
    _ELEMENTWISE_SOURCES = frozenset(_ATEN_TO_NUMPY) | {
        "add",
        "exp",
        "rsqrt",
        "reciprocal",
        "pow",
        "silu",
        "relu",
        "tanh",
        "abs",
        "sigmoid",
        "gelu",
        "erf",
    }
    if op_name in _ELEMENTWISE_SOURCES:
        from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        canonical = _ATEN_TO_NUMPY.get(op_name, op_name)
        # Disambiguate gelu's tanh approximation from the default erf form
        # — the FX node carries ``kwargs={'approximate': 'tanh'}`` only in
        # that case, and the decomposition rule keys on op name.
        if canonical == "gelu" and (fx_node.kwargs or {}).get("approximate") == "tanh":
            canonical = "gelu_tanh"
        bc_ids = [broadcast_to(g, inp, shape) for inp in input_ids[:2]]
        nid = g.add_node(
            op=ElementwiseOp(op=canonical),
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
    # Torch reduction names pass through to ``ReduceOp`` as-is; ``mean`` is
    # the exception that lands as ``MeanOp`` (decomposition splits it into
    # sum + div). Keeping torch's spelling — ``amax`` stays ``amax``, not
    # mapped to ``max`` — avoids a needless name table.
    if op_name in ("sum", "maximum", "amax", "mean"):
        axis = _get_reduce_axis(fx_node)
        x_shape = tuple(g.nodes[input_ids[0]].output.shape) if input_ids else ()
        keepdim_shape = _keepdim_shape(x_shape, axis)
        if op_name == "mean":
            red_node_op = MeanOp(axis=axis)
        else:
            red_node_op = ReduceOp(op=ElementwiseImpl(op_name), axis=axis)

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
            new_shape = _op_shape(new_shape, sym_rename)
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
        # ``ReshapeOp.shape`` is ``tuple[int | str, ...]`` — unwrap Dim wrappers
        # so the op-level field carries the raw int / symbolic-name form.
        op_shape = _dim_tuple_to_op_shape(shape)
        nid = g.add_node(
            op=ReshapeOp(shape=op_shape),
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
                op=SliceOp(shape=_dim_tuple_to_op_shape(shape)),
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

    # --- Fused frontend ops (decomposed in later passes) ---
    if op_name == "rms_norm":
        # aten.rms_norm: (x, normalized_shape, weight [, eps]). The tracer
        # drops normalized_shape (a list literal), leaving (x, weight) plus
        # an optional eps ConstantOp. eps is the op's own field, not a graph
        # input, so we peel it off here.
        eps_value = 1e-6
        if len(input_ids) >= 3:
            eps_node = g.nodes.get(input_ids[2])
            if eps_node and isinstance(eps_node.op, ConstantOp) and isinstance(eps_node.op.value, (int, float)):
                eps_value = float(eps_node.op.value)
                input_ids = input_ids[:2]
        nid = g.add_node(op=RmsNormOp(eps=eps_value), inputs=input_ids, output=Tensor(name, shape, dtype), node_id=name)
        node_map[name] = nid
        return

    if op_name == "softmax":
        # aten.softmax.int: (x, dim_const). dim becomes the op's field.
        axis: int = -1
        if len(input_ids) >= 2:
            dim_node = g.nodes.get(input_ids[1])
            if dim_node and isinstance(dim_node.op, ConstantOp) and isinstance(dim_node.op.value, (int, float)):
                axis = int(dim_node.op.value)
                input_ids = input_ids[:1]
        nid = g.add_node(op=SoftmaxOp(axis=axis), inputs=input_ids, output=Tensor(name, shape, dtype), node_id=name)
        node_map[name] = nid
        return

    # --- Fallback: unknown op becomes ElementwiseOp by torch-aten name ---
    logger.debug("Fallback elementwise for %s (%s)", op_name, fx_node.target)
    if input_ids:
        from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        input_ids = [broadcast_to(g, inp, shape) for inp in input_ids]
        nid = g.add_node(
            op=ElementwiseOp(op=op_name),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
