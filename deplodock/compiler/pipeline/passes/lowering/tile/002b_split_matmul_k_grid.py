"""Cross-CTA split-K — split the K-chunk loop into a grid dimension so
multiple CTAs cooperate on each ``(M, N)`` output tile and atomic-add
their partial sums.

Runs after ``002_split_matmul_k`` (which produced the
``Loop(K_o, body=(Loop(K_i, reduce, ...),))`` chunk structure) and
before ``005_blockify_launch`` (so the new ``K_split`` axis enters
``Tile.axes`` and is partitioned to ``BIND_BLOCK`` like any other free
output axis).

Activation:

- ``DEPLODOCK_SPLITK`` env var > 1 — explicit user override (PR-2 scope).
- Auto-heuristic based on grid utilization is intentionally deferred.

Refused when:

- The Tile body has anything between the ``Loop(K_o)`` and the trailing
  ``Write`` (an epilogue stmt would be added ``splitK`` times under
  atomic-add semantics — not safe without the ``K_split == 0`` predicate
  pattern, which PR 3 will introduce).
- ``K_o.extent % splitK != 0`` (would require boundary handling).

Rewrite::

    Loop(K_o:K/BK, body=(Loop(K_i:BK, reduce, B[K → K_o*BK + K_i]),))
    Write(out, idx, acc)
    →
    # K_split lifted into Tile.axes as outermost free axis (→ BLOCK)
    Loop(K_o_new:K/(splitK*BK), body=(Loop(K_i:BK, reduce, B[K_o →
            K_split*K_o_per_split + K_o_new]),))
    Write(out, idx, acc, reduce_op=add)

Output buffer must be zero-initialized (deplodock CUDA backend already
zeros allocations — see ``program.py``).
"""

from __future__ import annotations

import os
from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Loop, Tile, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]


def _splitk_env() -> int:
    raw = os.environ.get("DEPLODOCK_SPLITK")
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def rewrite(graph: Graph, root: Node) -> Graph | None:
    splitk = _splitk_env()
    if splitk <= 1:
        raise RuleSkipped("DEPLODOCK_SPLITK unset or <= 1")
    new_body = _maybe_rewrite(root.op.body, splitk)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body, splitk: int):
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already partitioned — must run before 005")

    new_tile = _split_tile(tile, splitk)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _split_tile(tile: Tile, splitk: int) -> Tile | None:
    """Locate the K_o + Write pair, refuse if epilogue is non-trivial,
    rewrite K_o into K_split outer + K_o_inner inner, lift K_split into
    tile.axes."""
    stmts = list(tile.body)
    k_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Loop) and _is_chunked_matmul(s)), None)
    if k_idx is None:
        raise RuleSkipped("no chunked matmul Loop in tile body")

    write_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Write)), None)
    if write_idx is None:
        raise RuleSkipped("no Write in tile body")
    if write_idx != k_idx + 1:
        raise RuleSkipped(f"epilogue stmts between K_o and Write (write_idx={write_idx}, k_idx={k_idx}) — PR 3 territory")

    k_outer = stmts[k_idx]
    K_o_extent = int(k_outer.axis.extent)
    if K_o_extent % splitk != 0:
        raise RuleSkipped(f"K_o.extent={K_o_extent} not divisible by splitK={splitk}")
    K_o_per_split = K_o_extent // splitk
    if K_o_per_split == 0:
        raise RuleSkipped(f"splitK={splitk} >= K_o.extent={K_o_extent}")

    K_split = Axis(f"{k_outer.axis.name}_split", splitk)
    K_o_new = Axis(k_outer.axis.name, K_o_per_split)
    sigma = Sigma({k_outer.axis.name: Var(K_split.name) * Literal(K_o_per_split, "int") + Var(K_o_new.name)})
    new_inner = tuple(s.rewrite(_id, sigma) for s in k_outer.body)
    new_k_loop = Loop(axis=K_o_new, body=new_inner)

    write = stmts[write_idx]
    new_write = replace(write, reduce_op=ElementwiseImpl("add"))

    new_stmts = stmts[:k_idx] + [new_k_loop, new_write] + stmts[write_idx + 1 :]
    # Lift K_split as outermost axis with the same bind as its siblings
    # (tileify defaults them all to BIND_THREAD). 005_blockify_launch
    # binds the outermost free axes to BLOCK, putting K_split into the
    # grid where each CTA handles one K-chunk.
    bind = tile.axes[0].bind
    new_axes = (BoundAxis(axis=K_split, bind=bind),) + tile.axes
    return Tile(axes=new_axes, body=tuple(new_stmts))


def _is_chunked_matmul(loop: Loop) -> bool:
    """002's output: outer Loop whose body is exactly one inner reduce-Loop."""
    if loop.is_reduce:
        return False
    inner = [s for s in loop.body if isinstance(s, Loop)]
    if len(inner) != 1 or len(loop.body) != 1:
        return False
    return inner[0].is_reduce


def _id(name: str) -> str:
    return name
