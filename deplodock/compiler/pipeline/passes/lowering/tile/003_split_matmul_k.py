"""Cross-CTA split-K — split the K-chunk loop into a grid dimension so
multiple CTAs cooperate on each ``(M, N)`` output tile and atomic-add
their partial sums.

Runs after ``002_tile_matmul_k`` (which produced the
``Loop(K_o, body=(Loop(K_i, reduce, ...),))`` chunk structure) and
before ``005_blockify_launch`` (so the new ``K_split`` axis enters
``Tile.axes`` and is partitioned to ``BIND_BLOCK`` like any other free
output axis).

Activation:

- ``DEPLODOCK_SPLITK`` env var > 1 — explicit user override.
- Otherwise, ``tuning.auto_splitk`` picks a value targeting ~3 waves
  per SM. Returns 1 (and the pass skips) when the M-N grid already
  fills the GPU.

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
from deplodock.compiler.tuning import auto_splitk

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already partitioned — must run before 005")

    # Find the chunked matmul Loop to learn K_o.extent for the picker.
    k_outer = next((s for s in tile.body if isinstance(s, Loop) and _is_chunked_matmul(s)), None)
    if k_outer is None:
        raise RuleSkipped("no chunked matmul Loop in tile body")
    splitk = auto_splitk(tile, int(k_outer.axis.extent))
    if splitk <= 1:
        raise RuleSkipped(f"auto-picked splitK={splitk} (grid already fills the GPU or no useful split)")

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

    if not epilogue_stmts and write.value == _find_accum_name(new_k_loop):
        # Plain matmul — every CTA atomic-adds its acc partial.
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        head_stmts = stmts[:k_idx] + [new_k_loop, new_write]
    else:
        acc_name = _find_accum_name(new_k_loop)
        if acc_name is None:
            raise RuleSkipped("could not locate the Accum SSA name inside K_o body")
        head_stmts = _rewrite_epilogue(stmts, k_idx, write_idx, write, epilogue_stmts, new_k_loop, acc_name, K_split)
        if head_stmts is None:
            raise RuleSkipped("epilogue isn't a split-K-safe shape (linear-multiplicative or acc + independent-load)")

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


def _rewrite_epilogue(stmts, k_idx, write_idx, write, epilogue_stmts, new_k_loop, acc_name, K_split):
    """Two split-K-safe epilogue shapes are recognized:

    1. **Linear-multiplicative**: ``Write.value`` is reachable from
       ``acc`` only through ``multiply`` ops whose other operand is
       acc-independent (e.g. ``acc * silu(gate)``). Then
       ``sum_i (c * a_i) = c * sum_i a_i`` distributes — every CTA
       computes its own ``v = c * partial_acc`` and ``atomic_add``s.
       Emit: keep epilogue as-is, mark Write as ``reduce_op=add``.

    2. **Linear-additive (residual)**: epilogue is exactly
       ``Load(r); Assign(v, add, r, acc); Write(v)``. Then
       ``sum_i a_i + r`` doesn't distribute — only one CTA contributes
       ``r``. Emit: always-on ``atomic_add(out, acc)`` plus
       ``Cond(K_split == 0, atomic_add(out, r))``.

    Returns the list of head stmts (everything up to and including the
    rewritten Write/Cond), or ``None`` if neither pattern matches."""
    # Try multiplicative-only chain first (cheap, common — k_mul_8_reduce).
    if _is_linear_multiplicative_chain(epilogue_stmts, acc_name, write.value):
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        return stmts[:k_idx] + [new_k_loop] + list(epilogue_stmts) + [new_write]

    # Fall back to the additive-residual pattern (k_add_5_reduce).
    residual_name = _extract_simple_residual(epilogue_stmts, acc_name, write.value)
    if residual_name is None:
        return None
    residual_load = next(s for s in epilogue_stmts if hasattr(s, "name") and s.name == residual_name)
    always_write = Write(output=write.output, index=write.index, value=acc_name, reduce_op=ElementwiseImpl("add"))
    cond_write = Write(output=write.output, index=write.index, value=residual_name, reduce_op=ElementwiseImpl("add"))
    cond = Cond(
        cond=BinaryExpr("==", Var(K_split.name), Literal(0, "int")),
        body=Body((residual_load, cond_write)),
        else_body=Body(()),
    )
    return stmts[:k_idx] + [new_k_loop, always_write, cond]


def _is_linear_multiplicative_chain(epilogue_stmts: list, acc_name: str, write_value: str) -> bool:
    """Walk the epilogue forward, propagating an ``acc_dep`` SSA-name
    set. A stmt is split-K-safe iff it's:

    - acc-independent (no deps in ``acc_dep``), OR
    - an ``Assign(v, multiply, a, b)`` where exactly one of ``a, b``
      is in ``acc_dep`` (the other is acc-independent).

    Returns True iff every epilogue stmt is safe and ``write_value`` ∈
    ``acc_dep`` (i.e. the Write's value chain back to ``acc``)."""
    from deplodock.compiler.ir.stmt import Assign  # noqa: PLC0415

    acc_dep = {acc_name}
    for s in epilogue_stmts:
        deps = set(s.deps())
        touches_acc = bool(deps & acc_dep)
        if not touches_acc:
            continue
        if not isinstance(s, Assign):
            return False
        if s.op.name != "multiply":
            return False
        if len(s.args) != 2:
            return False
        a, b = s.args
        a_dep = a in acc_dep
        b_dep = b in acc_dep
        if a_dep == b_dep:  # both deps or both indep — both indep can't happen since touches_acc
            return False
        acc_dep.add(s.name)
    return write_value in acc_dep


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
