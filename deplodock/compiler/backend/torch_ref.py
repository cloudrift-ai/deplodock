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
        "SoftmaxOp",
        "ElementwiseOp",
        "ReduceOp",
        "IndexMapOp",
    }
)

_TORCH_DTYPE = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16", "float64": "float64"}


def is_runnable(graph: Graph) -> bool:
    """True if every compute op in ``graph`` has a torch mapping."""
    from deplodock.compiler.provenance import is_boundary  # noqa: PLC0415

    return all(is_boundary(n.op) or type(n.op).__name__ in SUPPORTED for n in graph.nodes.values())


def _elementwise(fn: str, ins: list):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    table: dict[str, Callable] = {
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
    if fn not in table:
        raise NotImplementedError(f"torch_ref: elementwise {fn!r} unmapped")
    return table[fn](ins)


def _eval(node, ins: list):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

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
    if name == "SoftmaxOp":
        return torch.softmax(ins[0], dim=op.axis)
    if name == "TransposeOp":
        ndim = ins[0].dim()
        if len(op.axes) == ndim:
            return ins[0].permute(*op.axes)
        return ins[0].transpose(op.axes[0], op.axes[1])
    if name == "ReshapeOp":
        return ins[0].reshape(tuple(d.as_static() for d in node.output.shape))
    if name == "UnsqueezeOp":
        return ins[0].unsqueeze(op.dim)
    if name == "SliceOp":
        t = ins[0]
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
        return _index_map(op, ins)
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


def _index_map(op, ins: list):
    """Run an ``IndexMapOp`` as a vectorized gather over output coordinates.

    For every output cell, the first source whose ``select`` predicate holds
    supplies the value, read from that source at ``coord_map`` (each entry an
    affine ``Expr`` over the output coords). Single-source / ``select=None`` is
    the common case (broadcast, transpose, reshape, slice)."""
    import torch  # noqa: PLC0415

    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX  # noqa: PLC0415

    out_shape = tuple(d.as_static() for d in op.out_shape)
    # device / dtype from the first tensor-valued source (a source can be a
    # scalar constant — stored as a python float).
    base = next((ins[s.input_idx] for s in op.sources if torch.is_tensor(ins[s.input_idx])), None)
    dev = base.device if base is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype if base is not None else torch.float32
    env = {}
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
    its tensor inputs and ``torch.compile`` can trace it)."""
    from deplodock.compiler.provenance import is_boundary  # noqa: PLC0415

    order = graph.topological_order()
    tensor_ids = [nid for nid in order if is_boundary(graph.nodes[nid].op) and getattr(graph.nodes[nid].op, "value", None) is None]
    scalars = {
        nid: float(graph.nodes[nid].op.value)
        for nid in order
        if is_boundary(graph.nodes[nid].op) and getattr(graph.nodes[nid].op, "value", None) is not None
    }
    compute = [nid for nid in order if not is_boundary(graph.nodes[nid].op)]
    out_id = graph.outputs[0]

    def fn(*tensors):
        env = dict(scalars)
        env.update(zip(tensor_ids, tensors, strict=True))
        for nid in compute:
            node = graph.nodes[nid]
            env[nid] = _eval(node, [env[i] for i in node.inputs])
        return env[out_id]

    return fn, [input_tensors[i] for i in tensor_ids]
