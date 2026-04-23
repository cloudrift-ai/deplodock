"""Trace a transformer layer (or an inline torch module) to our Graph IR."""

import ast
import json
import logging
import sys

logger = logging.getLogger(__name__)


def register_trace_command(subparsers):
    parser = subparsers.add_parser(
        "trace",
        help="Trace a transformer layer or inline torch module to Graph IR",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B). Mutually exclusive with --code.",
    )
    parser.add_argument(
        "--code",
        help=(
            "Inline Python expression whose last statement is a call. "
            "The callable may be an nn.Module (e.g. 'nn.RMSNorm(2048)(torch.randn(1,32,2048))') "
            "or a torch function (e.g. 'F.silu(torch.randn(1,32,2048))', "
            "'torch.matmul(torch.randn(4,3), torch.randn(3,2))'). "
            "Call args are used as example inputs. Mutually exclusive with the positional model ID."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to trace. Omit to trace the whole model (input_ids -> logits).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for the example input (default: 32).",
    )
    parser.add_argument("--output", "-o", help="Output JSON path (default: auto-generated)")
    parser.set_defaults(func=handle_trace)


def handle_trace(args):
    if args.code and args.model:
        logger.error("--code and positional model are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.model:
        logger.error("either a positional model ID or --code is required")
        sys.exit(2)

    if args.code:
        _handle_trace_code(args)
    else:
        _handle_trace_model(args)


def _handle_trace_model(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    from deplodock.compiler.trace.torch import trace_module

    logger.info("Loading %s...", args.model)
    dtype = torch.float32 if args.layer is None else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    seq_len = args.seq_len

    if args.layer is None:
        from deplodock.compiler.trace.huggingface import build_full_model_wrapper

        logger.info("Tracing full model (seq_len=%d)...", seq_len)
        wrapper = build_full_model_wrapper(model, seq_len, dtype)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        graph = trace_module(wrapper, (input_ids,))
        basename = f"{args.model.replace('/', '-').lower()}-full-s{seq_len}"
    else:
        layers = model.model.layers
        if args.layer >= len(layers):
            logger.error("Layer %d not found (model has %d layers)", args.layer, len(layers))
            sys.exit(1)

        block = layers[args.layer]
        logger.info("Tracing layer %d...", args.layer)

        hidden_size = model.config.hidden_size
        x = torch.randn(1, seq_len, hidden_size, dtype=dtype)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.model.rotary_emb(x, position_ids)

        graph = trace_module(
            block,
            (x,),
            kwargs={"position_embeddings": (cos, sin)},
        )
        basename = f"{args.model.replace('/', '-').lower()}-layer{args.layer}"

    _save(graph, args, default_basename=basename)


def _handle_trace_code(args):
    graph, slug = graph_from_code(args.code)
    _save(graph, args, default_basename=slug)


def graph_from_code(code: str):
    """Trace an inline torch expression and return ``(graph, filename_slug)``.

    Shared by ``deplodock trace --code`` and ``deplodock compile --code``.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from deplodock.compiler.trace.torch import trace_module

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        logger.error("--code failed to parse: %s", e)
        sys.exit(2)

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        logger.error('--code must end with an expression (e.g. "torch.exp(torch.neg(x))")')
        sys.exit(2)

    final_expr = tree.body[-1].value
    scope = {"torch": torch, "nn": torch.nn, "F": F}
    preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
    exec(compile(preamble, "<--code>", "exec"), scope)  # noqa: S102 — local CLI

    # Fast path: direct call to an nn.Module — trace it straight, preserving
    # the module's parameter capture (weights land as Graph constants).
    if isinstance(final_expr, ast.Call):
        try:
            maybe_mod = eval(compile(ast.Expression(final_expr.func), "<--code>", "eval"), scope)  # noqa: S307
        except Exception:  # noqa: BLE001
            maybe_mod = None
        if isinstance(maybe_mod, torch.nn.Module):
            args = tuple(eval(compile(ast.Expression(a), "<--code>", "eval"), scope) for a in final_expr.args)  # noqa: S307
            kws = {
                kw.arg: eval(compile(ast.Expression(kw.value), "<--code>", "eval"), scope)  # noqa: S307
                for kw in final_expr.keywords
                if kw.arg
            }
            logger.info("Tracing inline module: %s", ast.unparse(final_expr.func))
            graph = trace_module(maybe_mod, args, kwargs=kws or None)
            return graph, _slugify(ast.unparse(final_expr.func))

    # General path: treat the final expression as a function body. Inputs
    # come from two sources: (1) bare Name references to tensors in scope
    # (set up via a preamble like ``x = torch.randn(8)``), and (2) inline
    # tensor-constructor calls (``torch.randn(...)``, etc.) which get
    # eagerly evaluated and bound to fresh placeholder names. Everything
    # else (torch, F, nn, helper modules, scalars) stays in the closure.
    rewritten, tensor_params = _lift_tensor_inputs(final_expr, scope)
    if not tensor_params:
        logger.error("--code expression has no tensor inputs to trace")
        sys.exit(2)

    # Polish synthesized ``_x<N>`` placeholder names: use ``x`` when there's
    # exactly one synthesized input, or ``x0``/``x1``/... for multiple.
    # Names brought in from the preamble (``x = torch.randn(8); ...``) are
    # left alone.
    synth = [k for k in tensor_params if k.startswith("_x")]
    if synth:
        rename = {synth[0]: "x"} if len(synth) == 1 else {old: f"x{i}" for i, old in enumerate(synth)}
        tensor_params = {rename.get(k, k): v for k, v in tensor_params.items()}
        for node in ast.walk(rewritten):
            if isinstance(node, ast.Name) and node.id in rename:
                node.id = rename[node.id]

    expr_src = ast.unparse(rewritten)
    forward_sig = ", ".join(["self", *tensor_params.keys()])
    wrapper_src = f"class _Wrapper(torch.nn.Module):\n    def forward({forward_sig}):\n        return {expr_src}\n"
    exec(wrapper_src, scope)  # noqa: S102 — local CLI
    module = scope["_Wrapper"]()
    example_inputs = tuple(tensor_params.values())
    logger.info("Tracing inline expression: %s", ast.unparse(final_expr))
    graph = trace_module(module, example_inputs)
    return graph, _slugify(ast.unparse(final_expr))


_TENSOR_CTOR_NAMES = frozenset({"randn", "rand", "zeros", "ones", "empty", "full", "arange", "linspace", "tensor", "randint", "eye"})


def _lift_tensor_inputs(expr: "ast.expr", scope: dict) -> tuple["ast.expr", dict]:
    """Rewrite ``expr`` so every tensor input becomes a named placeholder.

    Two kinds of input subtrees are lifted to the returned ``tensor_params``
    dict (preserving order for function-parameter generation):

    * Bare ``Name`` references that resolve to a tensor in ``scope`` — the
      original name is preserved as a parameter.
    * ``Call`` nodes to known tensor constructors (``torch.randn``, etc.)
      with no free tensor refs below them — eagerly evaluated and replaced
      with a fresh ``_x<N>`` placeholder.

    Everything else (non-constructor calls, attribute chains, operators) is
    left intact so torch.export still traces it.
    """
    import copy

    import torch

    tensor_params: dict[str, torch.Tensor] = {}

    def is_constructor_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Accept torch.<ctor> and nn.<ctor>-shaped attribute chains.
        while isinstance(func, ast.Attribute):
            if func.attr in _TENSOR_CTOR_NAMES:
                return True
            func = func.value
        return False

    def fresh_placeholder() -> str:
        i = 0
        while True:
            name = f"_x{i}"
            if name not in tensor_params and name not in scope:
                return name
            i += 1

    def visit(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            val = scope.get(node.id)
            if isinstance(val, torch.Tensor):
                tensor_params.setdefault(node.id, val)
            return node
        if is_constructor_call(node):
            try:
                val = eval(compile(ast.Expression(node), "<--code>", "eval"), scope)  # noqa: S307
            except Exception:  # noqa: BLE001
                val = None
            if isinstance(val, torch.Tensor):
                name = fresh_placeholder()
                tensor_params[name] = val
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        # Otherwise recurse into every child field.
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, visit(value))
        return node

    rewritten = visit(copy.deepcopy(expr))
    ast.fix_missing_locations(rewritten)
    return rewritten, tensor_params


def _slugify(src: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in src).strip("_").lower() or "inline"


def _save(graph, args, default_basename: str) -> None:
    ops_count: dict[str, int] = {}
    for n in graph.nodes.values():
        name = type(n.op).__name__
        ops_count[name] = ops_count.get(name, 0) + 1

    logger.info(
        "Traced: %d nodes (%s)",
        len(graph.nodes),
        ", ".join(f"{v} {k}" for k, v in sorted(ops_count.items())),
    )

    output_path = args.output or f"{default_basename}.json"
    with open(output_path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger.info("Saved: %s", output_path)
