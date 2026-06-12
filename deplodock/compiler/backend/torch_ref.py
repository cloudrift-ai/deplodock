"""Execute a frontend Graph with real PyTorch — the eager / torch.compile
reference for ``deplodock run --ir``.

Each frontend / tensor-IR op is mapped to the equivalent torch op (the numpy
``forward()`` has a torch twin), so a dumped ``.torch.json`` reproducer can be
accuracy-checked and timed against torch — including torch.compile fusion — not
just numpy. ``IndexMapOp`` (the post-decomposition layout primitive — broadcast,
transpose/reshape/slice/cat, RoPE rotate) is supported as a vectorized
gather over coordinate grids, so HF norms/RoPE (which trace to primitive
sequences, not fused aten ops) are torch-comparable. Only the data-dependent
``GatherOp`` / ``ScatterOp`` (indices come from a runtime tensor, not from the
output coordinates) stay unsupported: :func:`is_runnable` returns ``False`` and
the caller falls back to deplodock-only benchmarking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from deplodock.compiler.graph import Graph

# Frontend / tensor ops with a torch twin. Everything else (IndexMapOp,
# GatherOp, ScatterOp, ScanOp, …) makes a graph non-runnable as a torch ref.
SUPPORTED = frozenset(
    {
        "TransposeOp",
        "ReshapeOp",
        "SliceOp",
        "CatOp",
        "UnsqueezeOp",
        "LinearOp",
        "MatmulOp",
        "SdpaOp",
        "MeanOp",
        "RmsNormOp",
        "LayerNormOp",
        "SoftmaxOp",
        "ElementwiseOp",
        "ReduceOp",
        "IndexMapOp",
    }
)


def torch_dtype(dtype) -> torch.dtype | None:
    """Graph ``DataType`` token → torch dtype, ``None`` for unknown tokens.

    The graph tokens (``f16`` / ``bf16`` / …) are not torch attribute names,
    and ``DataType.np`` can't be used either (bf16's numpy carrier is uint16) —
    map explicitly."""
    import torch  # noqa: PLC0415

    return {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i32": torch.int32,
        "i64": torch.int64,
    }.get(str(dtype))


def is_runnable(graph: Graph) -> bool:
    """True if every compute op in ``graph`` has a torch mapping."""
    from deplodock.compiler.provenance import is_boundary  # noqa: PLC0415

    return all(is_boundary(n.op) or type(n.op).__name__ in SUPPORTED for n in graph.nodes.values())


def _build_elementwise_table() -> dict[str, Callable]:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    return {
        "add": lambda a: a[0] + a[1],
        "sum": lambda a: a[0] + a[1],
        "subtract": lambda a: a[0] - a[1],
        "sub": lambda a: a[0] - a[1],
        "multiply": lambda a: a[0] * a[1],
        "prod": lambda a: a[0] * a[1],
        "divide": lambda a: a[0] / a[1],
        "true_divide": lambda a: a[0] / a[1],
        "pow": lambda a: a[0] ** a[1],
        "maximum": lambda a: torch.maximum(a[0], a[1]),
        "minimum": lambda a: torch.minimum(a[0], a[1]),
        "negative": lambda a: -a[0],
        "abs": lambda a: torch.abs(a[0]),
        "reciprocal": lambda a: torch.reciprocal(a[0]),
        "sqrt": lambda a: torch.sqrt(a[0]),
        "rsqrt": lambda a: torch.rsqrt(a[0]),
        "exp": lambda a: torch.exp(a[0]),
        "log": lambda a: torch.log(a[0]),
        "sin": lambda a: torch.sin(a[0]),
        "cos": lambda a: torch.cos(a[0]),
        "tanh": lambda a: torch.tanh(a[0]),
        "sigmoid": lambda a: torch.sigmoid(a[0]),
        "silu": lambda a: F.silu(a[0]),
        "relu": lambda a: F.relu(a[0]),
        "erf": lambda a: torch.erf(a[0]),
        "gelu": lambda a: F.gelu(a[0]),
        "gelu_tanh": lambda a: F.gelu(a[0], approximate="tanh"),
        "copy": lambda a: a[0],
    }


# Module-level table — built once on first import (when torch is loaded). The
# per-name lookup happens at ``build_callable`` time, not inside the traced ``fn``,
# so ``torch.compile`` doesn't re-trace per distinct elementwise op (the previous
# in-trace ``if fn not in table`` was the dominant recompile trigger).
_ELEMENTWISE_TABLE: dict[str, Callable] | None = None


def _elementwise_callable(fn: str) -> Callable:
    global _ELEMENTWISE_TABLE
    if _ELEMENTWISE_TABLE is None:
        _ELEMENTWISE_TABLE = _build_elementwise_table()
    op = _ELEMENTWISE_TABLE.get(fn)
    if op is None:
        raise NotImplementedError(f"torch_ref: elementwise {fn!r} unmapped")
    return op


def _elementwise(fn: str, ins: list):
    """Back-compat shim — used by ``_eval`` for one-shot evaluation paths that
    don't go through ``build_callable``'s pre-resolution."""
    return _elementwise_callable(fn)(ins)


def _shape_ints(shape, sym_env: dict[str, int]) -> tuple[int, ...]:
    """Resolve a (possibly symbolic) ``Dim`` shape to concrete ints. Symbolic
    dims eval over ``sym_env`` — the name → runtime-extent map read off the
    supplied input tensors in :func:`build_callable`."""
    return tuple(d.as_static() if d.is_static else int(d.expr.eval(sym_env)) for d in shape)


def _eval(node, ins: list, sym_env: dict[str, int] | None = None):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    sym_env = sym_env or {}
    op = node.op
    name = type(op).__name__
    if name == "ElementwiseOp":
        return _elementwise(op.op.name, ins)
    if name == "ReduceOp":
        x, ax, fn = ins[0], op.axis, op.op.name
        if fn in ("sum", "add"):
            return x.sum(dim=ax, keepdim=True)
        if fn in ("prod", "multiply"):
            return x.prod(dim=ax, keepdim=True)
        if fn in ("maximum", "amax", "max"):
            return x.amax(dim=ax, keepdim=True)
        if fn in ("minimum", "amin", "min"):
            return x.amin(dim=ax, keepdim=True)
        raise NotImplementedError(f"torch_ref: reduce {fn!r} unmapped")
    if name == "LinearOp":
        return F.linear(ins[0], ins[1], ins[2] if op.has_bias else None)
    if name == "MatmulOp":
        out = ins[0] @ ins[1]
        return out + ins[2] if op.has_bias else out
    if name == "SdpaOp":
        q, k, v = ins[0], ins[1], ins[2]
        gqa = q.dim() >= 3 and q.shape[-3] != k.shape[-3]
        try:
            return F.scaled_dot_product_attention(q, k, v, is_causal=op.is_causal, enable_gqa=gqa)
        except TypeError:  # older torch without enable_gqa
            if gqa:
                rep = q.shape[-3] // k.shape[-3]
                k, v = k.repeat_interleave(rep, dim=-3), v.repeat_interleave(rep, dim=-3)
            return F.scaled_dot_product_attention(q, k, v, is_causal=op.is_causal)
    if name == "MeanOp":
        return ins[0].mean(dim=op.axis, keepdim=True)
    if name == "RmsNormOp":
        x, w = ins[0], ins[1]
        try:
            return F.rms_norm(x, (x.shape[-1],), w, op.eps)
        except (AttributeError, RuntimeError):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + op.eps) * w
    if name == "LayerNormOp":
        x = ins[0]
        w = ins[1] if len(ins) > 1 else None
        b = ins[2] if len(ins) > 2 else None
        return F.layer_norm(x, (x.shape[-1],), w, b, op.eps)
    if name == "SoftmaxOp":
        return torch.softmax(ins[0], dim=op.axis)
    if name == "TransposeOp":
        ndim = ins[0].dim()
        if len(op.axes) == ndim:
            return ins[0].permute(*op.axes)
        return ins[0].transpose(op.axes[0], op.axes[1])
    if name == "ReshapeOp":
        return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
    if name == "UnsqueezeOp":
        return ins[0].unsqueeze(op.dim)
    if name == "SliceOp":
        t = ins[0]
        if op.dim is not None:
            dim, start = op.dim, op.start or 0
            extent = op.shape[dim]
            end = start + int(extent) if isinstance(extent, int) else t.shape[dim]
        else:  # legacy constant-input convention (pre-field IR dumps)
            dim = int(ins[1]) if len(ins) > 1 else 0
            start = int(ins[2]) if len(ins) > 2 else 0
            end = int(ins[3]) if len(ins) > 3 else t.shape[dim]
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(start, end)
        return t[tuple(idx)]
    if name == "CatOp":
        tensors = [i for i in ins if isinstance(i, torch.Tensor)]
        dim = next((int(i) for i in ins if not isinstance(i, torch.Tensor)), -1)
        return torch.cat(tensors, dim=dim)
    if name == "IndexMapOp":
        return _index_map(op, ins, sym_env)
    raise NotImplementedError(f"torch_ref: op {name!r} unmapped")


def _idx_expr(e, env: dict):
    """Evaluate a coord/select ``Expr`` over the output-coordinate grids.

    Mirrors ``Expr.eval`` but with torch-/CUDA-safe boolean ops (``Expr.eval``
    routes ``&&``/``||`` through ``np.logical_and``, which can't take a CUDA
    tensor). Falls back to ``Expr.eval`` for leaf node types that don't appear
    in coord maps."""
    cls = type(e).__name__
    if cls == "Var":
        return env[e.name]
    if cls == "Literal":
        return e.value
    if cls == "BinaryExpr":
        lo, ro = _idx_expr(e.left, env), _idx_expr(e.right, env)
        op = e.op
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a // b,
            "//": lambda a, b: a // b,
            "%": lambda a, b: a % b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "&&": lambda a, b: a & b,
            "&": lambda a, b: a & b,
            "||": lambda a, b: a | b,
            "|": lambda a, b: a | b,
            "^": lambda a, b: a ^ b,
        }
        if op in ops:
            return ops[op](lo, ro)
    return e.eval(env)


def _index_map(op, ins: list, sym_env: dict[str, int] | None = None):
    """Run an ``IndexMapOp`` as a vectorized gather over output coordinates.

    For every output cell, the first source whose ``select`` predicate holds
    supplies the value, read from that source at ``coord_map`` (each entry an
    affine ``Expr`` over the output coords). Single-source / ``select=None`` is
    the common case (broadcast, transpose, reshape, slice)."""
    import torch  # noqa: PLC0415

    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX  # noqa: PLC0415

    out_shape = _shape_ints(op.out_shape, sym_env or {})
    # device / dtype from the first tensor-valued source (a source can be a
    # scalar constant — stored as a python float).
    base = next((ins[s.input_idx] for s in op.sources if torch.is_tensor(ins[s.input_idx])), None)
    dev = base.device if base is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype if base is not None else torch.float32
    # Coord / select exprs may reference symbolic extents (e.g. ``coord < seq_len``)
    # besides the output-coordinate placeholders — resolve both from one env.
    env: dict = dict(sym_env or {})
    for i, d in enumerate(out_shape):
        shp = [1] * len(out_shape)
        shp[i] = d
        env[f"{PLACEHOLDER_PREFIX}{i}"] = torch.arange(d, device=dev, dtype=torch.long).reshape(shp).expand(out_shape)

    result = torch.zeros(out_shape, dtype=dtype, device=dev)
    filled = torch.zeros(out_shape, dtype=torch.bool, device=dev)
    for src in op.sources:
        s = ins[src.input_idx]
        if not torch.is_tensor(s):  # scalar constant source → broadcast it
            gathered = torch.full(out_shape, float(s), dtype=dtype, device=dev)
        else:
            idx = []
            for i, c in enumerate(src.coord_map):
                if i < s.dim() and s.shape[i] == 1:  # size-1 source dim → index 0 (mirror lift)
                    idx.append(torch.zeros(out_shape, dtype=torch.long, device=dev))
                else:
                    v = _idx_expr(c, env)
                    v = v if torch.is_tensor(v) else torch.full(out_shape, int(v), dtype=torch.long, device=dev)
                    # Clamp: the gather runs on every cell, but a source's coord_map
                    # may go out of range where its select is false (those values are
                    # discarded by ``take`` below). Clamp so the gather never OOBs.
                    idx.append(v.expand(out_shape).long().clamp(0, s.shape[i] - 1))
            gathered = s[tuple(idx)]
        if src.select is None:
            sel = torch.ones(out_shape, dtype=torch.bool, device=dev)
        else:
            sv = _idx_expr(src.select, env)
            sel = (sv if torch.is_tensor(sv) else torch.full(out_shape, bool(sv), device=dev)).expand(out_shape).bool()
        take = sel & ~filled
        result = torch.where(take, gathered, result)
        filled = filled | take
    return result


def build_callable(graph: Graph, input_tensors: dict[str, torch.Tensor]) -> tuple[Callable, list]:
    """Build a torch callable for ``graph`` plus its positional input list.

    The returned ``fn(*tensors)`` runs the frontend ops in topological order and
    returns the graph's first output; the input list is the ``tensors`` to call
    it with (drawn from ``input_tensors`` in boundary topo order). Scalar
    constants are read inline from the graph (so ``fn`` is a pure function of
    its tensor inputs and ``torch.compile`` can trace it).

    Per-node op callables are resolved **at build time** (not inside ``fn``) so
    ``torch.compile`` sees a stable call sequence — dispatching on the op kind
    inside the traced function used to trigger a dynamo recompile per distinct
    op (``add`` vs ``multiply`` vs …), hitting the recompile limit on big
    graphs."""
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415
    from deplodock.compiler.provenance import is_boundary  # noqa: PLC0415

    order = graph.topological_order()
    tensor_ids = [nid for nid in order if is_boundary(graph.nodes[nid].op) and getattr(graph.nodes[nid].op, "value", None) is None]
    # Bind every symbolic axis name to its concrete extent, read off the
    # supplied tensors (same convention as the CUDA launch). Baked into the
    # per-node callables — ``fn`` stays shape-specialised to these tensors,
    # which is what the bench wants (one concrete size per timed call).
    sym_env: dict[str, int] = {}
    for nid in tensor_ids:
        t = input_tensors[nid]
        for i, d in enumerate(graph.nodes[nid].output.shape):
            if isinstance(getattr(d, "expr", None), Var):
                sym_env.setdefault(d.expr.name, int(t.shape[i]))
    scalars = {
        nid: float(graph.nodes[nid].op.value)
        for nid in order
        if is_boundary(graph.nodes[nid].op) and getattr(graph.nodes[nid].op, "value", None) is not None
    }
    # Pre-resolve a flat list of (nid, op_callable, input_ids, out_dtype).
    # ElementwiseOps get the table lookup once here (the main recompile source);
    # other op kinds wrap ``_eval`` with the node bound so the traced ``fn``
    # invokes a uniform ``op(ins) -> tensor`` callable per step without per-call
    # string dispatch. ``out_dtype`` is the node's declared output dtype: the
    # trace folds HF's explicit casts (e.g. the fp32 RMSNorm body casting back
    # to fp16) into the declared dtype, and the CUDA backend honors it via typed
    # buffers — the torch ref must cast too, or torch's promotion silently runs
    # everything downstream of a mixed-dtype op at fp32.
    compute_steps: list[tuple[str, Callable, list[str], torch.dtype | None]] = []
    for nid in order:
        node = graph.nodes[nid]
        if is_boundary(node.op):
            continue
        if type(node.op).__name__ == "ElementwiseOp":
            op_callable = _elementwise_callable(node.op.op.name)
        else:
            op_callable = (lambda n: lambda ins: _eval(n, ins, sym_env))(node)
        compute_steps.append((nid, op_callable, list(node.inputs), torch_dtype(node.output.dtype)))
    out_id = graph.outputs[0]

    def fn(*tensors):
        env = dict(scalars)
        env.update(zip(tensor_ids, tensors, strict=True))
        for nid, op_callable, in_ids, out_dtype in compute_steps:
            v = op_callable([env[i] for i in in_ids])
            env[nid] = v if out_dtype is None else v.to(out_dtype)
        return env[out_id]

    return fn, [input_tensors[i] for i in tensor_ids]
