"""Execute a frontend Graph with real PyTorch — the eager / torch.compile
reference for ``deplodock run --ir``.

Each frontend / tensor-IR op is mapped to the equivalent torch op (the numpy
``forward()`` has a torch twin), so a dumped ``.torch.json`` reproducer can be
accuracy-checked and timed against torch — including torch.compile fusion — not
just numpy. Layout/data-dependent ops that only appear post-decomposition
(``IndexMapOp`` / ``GatherOp`` / ``ScatterOp``) are deliberately unsupported:
:func:`is_runnable` returns ``False`` for such graphs and the caller falls back
to deplodock-only benchmarking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

    from deplodock.compiler.graph import Graph

# Frontend / tensor ops with a torch twin. Everything else (IndexMapOp,
# GatherOp, ScatterOp, ScanOp, …) makes a graph non-runnable as a torch ref.
SUPPORTED = frozenset(
    {
        "TransposeOp", "ReshapeOp", "SliceOp", "CatOp", "UnsqueezeOp",
        "LinearOp", "MatmulOp", "SdpaOp", "MeanOp", "RmsNormOp", "SoftmaxOp",
        "ElementwiseOp", "ReduceOp",
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
    raise NotImplementedError(f"torch_ref: op {name!r} unmapped")


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
    scalars = {nid: float(graph.nodes[nid].op.value) for nid in order
               if is_boundary(graph.nodes[nid].op) and getattr(graph.nodes[nid].op, "value", None) is not None}
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
