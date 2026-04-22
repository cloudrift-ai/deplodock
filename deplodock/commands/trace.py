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
            "Inline Python expression whose last statement is a call, "
            'e.g. --code "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))". '
            "The callable is traced with the given args as example inputs. "
            "Mutually exclusive with the positional model ID."
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

    from deplodock.compiler.torch_trace import trace_module

    logger.info("Loading %s...", args.model)
    dtype = torch.float32 if args.layer is None else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    seq_len = args.seq_len

    if args.layer is None:
        from deplodock.compiler.model_wrapper import build_full_model_wrapper

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

    from deplodock.compiler.torch_trace import trace_module

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        logger.error("--code failed to parse: %s", e)
        sys.exit(2)

    if not tree.body or not isinstance(tree.body[-1], ast.Expr) or not isinstance(tree.body[-1].value, ast.Call):
        logger.error('--code must end with a call expression, e.g. "m(x)" or "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))"')
        sys.exit(2)

    call = tree.body[-1].value
    scope = {"torch": torch, "nn": torch.nn, "F": F}
    preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
    exec(compile(preamble, "<--code>", "exec"), scope)  # noqa: S102 — local CLI, equivalent to python -c

    module = eval(compile(ast.Expression(call.func), "<--code callable>", "eval"), scope)  # noqa: S307
    example_inputs = tuple(eval(compile(ast.Expression(a), "<--code arg>", "eval"), scope) for a in call.args)  # noqa: S307
    kwargs = {kw.arg: eval(compile(ast.Expression(kw.value), "<--code kwarg>", "eval"), scope) for kw in call.keywords if kw.arg}  # noqa: S307

    logger.info("Tracing inline module: %s", ast.unparse(call.func))
    graph = trace_module(module, example_inputs, kwargs=kwargs or None)

    src = ast.unparse(call.func)
    slug = "".join(c if c.isalnum() else "_" for c in src).strip("_").lower() or "inline"
    return graph, slug


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
