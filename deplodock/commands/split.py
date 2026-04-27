"""Split a fused multi-kernel Graph IR into one self-contained Graph
per kernel. Each split kernel includes its ``LoopOp`` plus standalone
``InputOp`` / ``ConstantOp`` ancestors — any cross-kernel dependency
(consumed output of another LoopOp) is replaced with a fresh
``InputOp`` carrying the producer's tensor metadata. The result is a
JSON suitable for ``deplodock compile <kernel.json> --ir tile/kernel/cuda``
so each hot kernel can be lowered and inspected in isolation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum

logger = logging.getLogger(__name__)


def register_split_command(subparsers):
    parser = subparsers.add_parser(
        "split",
        help="Split a multi-kernel Graph IR into one JSON per LoopOp kernel.",
    )
    parser.add_argument("input", help="Path to a fused Graph JSON (e.g. dump/04_loop_fusion.json).")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: alongside the input).")
    parser.add_argument(
        "--name",
        default=None,
        help="Only emit kernels whose op.name matches this exact string (default: all).",
    )
    parser.set_defaults(func=handle_split)


def handle_split(args):
    src = Path(args.input)
    if not src.exists():
        logger.error("not found: %s", src)
        sys.exit(2)
    out_dir = Path(args.output_dir) if args.output_dir else src.parent / f"{src.stem}_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src) as f:
        data = json.load(f)
    g = Graph.from_dict(data)

    kernel_ids = [nid for nid, n in g.nodes.items() if isinstance(n.op, LoopOp)]
    if not kernel_ids:
        logger.error("no LoopOp nodes in %s — split only works on post-fusion IR", src)
        sys.exit(2)

    written: list[Path] = []
    for nid in kernel_ids:
        kernel_name = _bench_kernel_name(g.nodes[nid])
        if args.name and kernel_name != args.name:
            continue
        sub = _extract_single_kernel(g, nid)
        target = out_dir / f"{kernel_name}.json"
        with open(target, "w") as f:
            json.dump(sub.to_dict(), f, indent=2, default=str)
        written.append(target)
        logger.info("wrote %s", target)

    if not written:
        logger.error("no kernels matched --name=%r (available: %s)", args.name, ", ".join(_kernel_names(g)))
        sys.exit(1)
    print(f"Wrote {len(written)} kernel(s) to {out_dir}")


def _kernel_names(g: Graph) -> list[str]:
    return [_bench_kernel_name(n) for nid, n in g.nodes.items() if isinstance(n.op, LoopOp)]


def _bench_kernel_name(node: Node) -> str:
    """Match the lowering pass's naming: ``k_{dedup(base)}_{reduce|pointwise}``,
    where base is the LoopOp's output tensor name (which the lowering pass
    renames the node to before kernel naming). See
    ``passes/lowering/tile/001_lower_loopop.py``."""
    base = node.output.name or node.id
    suffix = "reduce" if any(isinstance(s, Accum) for s in node.op) else "pointwise"
    return f"k_{_dedup_tokens(base)}_{suffix}"


def _dedup_tokens(name: str) -> str:
    out: list[str] = []
    for tok in name.split("_"):
        if not tok or (out and out[-1] == tok):
            continue
        out.append(tok)
    return "_".join(out) if out else name


def _extract_single_kernel(g: Graph, kernel_id: str) -> Graph:
    """Build a new Graph containing ``kernel_id`` (a LoopOp), with each
    of its inputs preserved as-is when it's an InputOp / ConstantOp,
    and replaced with a fresh InputOp otherwise. The new graph's output
    is the kernel."""
    src_node = g.nodes[kernel_id]
    sub = Graph()
    new_input_ids: list[str] = []

    for inp_id in src_node.inputs:
        src_inp = g.nodes[inp_id]
        if isinstance(src_inp.op, (InputOp, ConstantOp)):
            new_id = sub.add_node(_clone_op(src_inp.op), inputs=[], output=_clone_tensor(src_inp.output), node_id=inp_id)
        else:
            # Cross-kernel edge: stub it as a fresh InputOp.
            new_id = sub.add_node(InputOp(), inputs=[], output=_clone_tensor(src_inp.output), node_id=inp_id)
        new_input_ids.append(new_id)
        if not isinstance(src_inp.op, ConstantOp):
            sub.inputs.append(new_id)

    sub.add_node(_clone_op(src_node.op), inputs=new_input_ids, output=_clone_tensor(src_node.output), node_id=kernel_id)
    sub.outputs = [kernel_id]
    return sub


def _clone_tensor(t: Tensor) -> Tensor:
    return Tensor(name=t.name, shape=t.shape, dtype=t.dtype)


def _clone_op(op):
    """Deep-copy an Op via its dict round-trip so split kernels don't
    share mutable state with the source graph."""
    from copy import deepcopy

    return deepcopy(op)


_ = Node  # silence ruff
