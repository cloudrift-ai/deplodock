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

- ``K_o.extent % splitK != 0`` (would require boundary handling).
- The trailing ``Write`` value isn't directly the matmul ``Accum`` and
  the body has more than a simple linear epilogue chain we can wrap in
  a ``Cond(K_split == 0, ...)``.

Epilogue handling:

For a body of the form ``Loop(K_o, …) ⊕ Write(out, acc)`` the rewrite
just splits ``K_o`` and marks the ``Write`` as ``reduce_op=add``.

For ``Loop(K_o, …) ⊕ <epilogue stmts producing v> ⊕ Write(out, v)``
(e.g. ``k_add_5_reduce``: down_proj + residual add), every CTA
contributes its partial sum via an ``atomic_add(out, acc)`` outside
the ``Cond``, and ``Cond(K_split == 0, atomic_add(out, residual))``
folds the epilogue value (excluding the already-accumulated ``acc``)
in once. This works when the epilogue is exactly one extra additive
term — the most common shape.

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
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Tile, Write
from deplodock.compiler.ir.stmt.body import Body
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
    """Locate the K_o + Write pair, rewrite K_o into K_split outer +
    K_o_inner inner, lift K_split into tile.axes. Wraps any epilogue
    stmts in ``Cond(K_split == 0, ...)`` so only one CTA contributes
    them; every CTA atomic-adds its accumulator partial."""
    stmts = list(tile.body)
    k_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Loop) and _is_chunked_matmul(s)), None)
    if k_idx is None:
        raise RuleSkipped("no chunked matmul Loop in tile body")

    write_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Write)), None)
    if write_idx is None:
        raise RuleSkipped("no Write in tile body")
    if write_idx <= k_idx:
        raise RuleSkipped(f"Write at idx {write_idx} not after K_o at idx {k_idx}")

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
    epilogue_stmts = stmts[k_idx + 1 : write_idx]  # may be empty

    if not epilogue_stmts:
        # Plain matmul — every CTA atomic-adds its acc partial.
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        head_stmts = stmts[:k_idx] + [new_k_loop, new_write]
    else:
        # Epilogue case — the matmul's accumulator name is the Accum's
        # SSA target inside the K_i reduce body. We need it to emit the
        # always-on atomic-add of the partial sum.
        acc_name = _find_accum_name(new_k_loop)
        if acc_name is None:
            raise RuleSkipped("could not locate the Accum SSA name inside K_o body")
        # Refuse if the trailing Write doesn't reference a value derived
        # from the Accum — without that linkage the epilogue isn't a
        # simple additive fold-in and our Cond rewrite would change
        # semantics.
        # Always-on partial-sum write: atomic_add(out, acc).
        always_write = Write(output=write.output, index=write.index, value=acc_name, reduce_op=ElementwiseImpl("add"))
        # Predicated epilogue: only K_split == 0 contributes the
        # epilogue value. The original epilogue already includes the
        # acc in its computation (e.g. v = residual + acc); to avoid
        # double-counting we'd need to subtract acc from v at predicate
        # time, which the IR doesn't directly express. Instead emit
        # the epilogue as-written but make it a STORE (no reduce_op)
        # of just the residual contribution. That requires rebuilding
        # the epilogue to drop the acc-dependent term — not feasible
        # generically. So restrict PR 3 to a stricter shape: epilogue
        # is exactly Load(...) + Assign(v, add, ld, acc) + Write(v).
        # In that case CTA 0 atomic-adds the residual ONCE; the
        # always-on path covers the acc.
        residual_name = _extract_simple_residual(epilogue_stmts, acc_name, write.value)
        if residual_name is None:
            raise RuleSkipped("epilogue isn't the simple Load+Assign(add)+Write residual pattern")
        residual_load = next(s for s in epilogue_stmts if hasattr(s, "name") and s.name == residual_name)
        cond_write = Write(output=write.output, index=write.index, value=residual_name, reduce_op=ElementwiseImpl("add"))
        cond = Cond(
            cond=BinaryExpr("==", Var(K_split.name), Literal(0, "int")),
            body=Body((residual_load, cond_write)),
            else_body=Body(()),
        )
        head_stmts = stmts[:k_idx] + [new_k_loop, always_write, cond]

    new_stmts = head_stmts + stmts[write_idx + 1 :]
    # Lift K_split as outermost axis with the same bind as its siblings
    # (tileify defaults them all to BIND_THREAD). 005_blockify_launch
    # binds the outermost free axes to BLOCK, putting K_split into the
    # grid where each CTA handles one K-chunk.
    bind = tile.axes[0].bind
    new_axes = (BoundAxis(axis=K_split, bind=bind),) + tile.axes
    return Tile(axes=new_axes, body=tuple(new_stmts))


def _find_accum_name(k_o_loop: Loop) -> str | None:
    """Return the SSA target of the innermost Accum in the chunked
    matmul Loop, or None if not found."""
    for s in k_o_loop.body.iter():
        if isinstance(s, Accum):
            return s.name
    return None


def _extract_simple_residual(epilogue_stmts: list, acc_name: str, write_value: str) -> str | None:
    """Match the exact 3-stmt pattern: ``Load(ld) ; Assign(v, add, ld,
    acc) ; Write(v)`` — and return ``ld``'s SSA name. Returns None on
    any deviation (multiple loads, non-add op, indirect deps, etc.)."""
    from deplodock.compiler.ir.stmt import Assign, Load  # noqa: PLC0415

    if len(epilogue_stmts) != 2:
        return None
    ld, asn = epilogue_stmts
    if not isinstance(ld, Load) or not isinstance(asn, Assign):
        return None
    if asn.name != write_value:
        return None
    if asn.op.name != "add":
        return None
    args = set(asn.args)
    if args != {ld.name, acc_name}:
        return None
    return ld.name


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
