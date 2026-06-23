"""Realize a complete move choice into the ``TileOp`` tower.

Reproduces the legacy planner's σ-split + tower so the output is accuracy-
equivalent. Each free axis ``A`` splits as ``A → A_b·(T·R) + A_t·R + A_r``
(interleaved ``A_b·(T·R) + A_r·T + A_t`` on a masked *inner* axis), the body is
σ-rewritten, a reduce axis is re-bracketed into a ``K_o`` (serial-outer) /
``K_i`` (stage-inner) tower with an optional ``K_f`` strip-mine, masked axes get
a ``Cond`` store guard, and the layers wrap innermost-first via the shared
:func:`_wrap_tower`.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Init, Loop, Monoid, Select, SelectBranch, Stmt
from deplodock.compiler.ir.tile.ir import RegisterTile
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import IterDag

# One free axis to split: (axis, thread factor, register factor, interleave-when-masked).
_FreeSpec = tuple[Axis, int, int, bool]


def _free_axes(dag: IterDag) -> tuple[Axis, Axis | None, tuple[Loop, ...]]:
    """The free (PARALLEL) axes the builders tile, read off the DAG's parallel
    chain: ``(inner_n axis, outer_m axis | None, extra-outer loops)``. The
    skeleton's ``inner_n`` / ``outer_m`` / ``extra_outer`` are projections of
    exactly these nodes, so reading the DAG is byte-identical."""
    parallel = dag.parallel
    inner_n = parallel[-1].loop.axis
    outer_m = parallel[-2].loop.axis if len(parallel) >= 2 else None
    extra_outer = tuple(n.loop for n in parallel[:-2])
    return inner_n, outer_m, extra_outer


def _split_free_axis(axis: Axis, thread: int, reg: int, *, interleave_when_masked: bool):
    """Build the block/thread/register axes + σ entry + optional store-guard
    bound for one free axis."""
    tr = thread * reg
    masked = (not axis.extent.is_static) or (axis.extent.as_static() % tr != 0)
    src = axis.source_axis or axis
    a_b = Axis(
        f"{axis.name}_b",
        axis.extent.ceil_div(tr) if masked else axis.extent // tr,
        source_axis=src,
        real_extent=axis.extent.as_static() if masked and axis.extent.is_static else None,
    )
    a_t = Axis(f"{axis.name}_t", thread, source_axis=src)
    a_r = Axis(f"{axis.name}_r", reg, source_axis=src)
    block = Var(a_b.name) * Literal(tr, "int")
    if masked and interleave_when_masked:
        expr = block + Var(a_r.name) * Literal(thread, "int") + Var(a_t.name)
    else:
        expr = block + Var(a_t.name) * Literal(reg, "int") + Var(a_r.name)
    bound = axis.extent.expr if masked else None
    return a_b, a_t, a_r, expr, bound


def _replace_k_scalar(
    stmts: tuple[Stmt, ...],
    target_names: frozenset[str],
    k_extent: int,
    bk: int,
    fk: int,
    splitk: int,
    k_s: Axis | None,
    k_bounds: dict | None = None,
) -> tuple[Stmt, ...]:
    """Replace every contraction loop named in ``target_names`` with a ``K_o``
    (serial-outer) / ``K_i`` (stage-inner) tower, σ-mapping
    ``K → K_s·(K_o_ext·bk·fk) + K_o·(bk·fk) + K_f·bk + K_i`` (``K_f`` only when
    ``fk > 1``, ``K_s`` only when ``splitk > 1``). For a plain matmul this is the
    one K loop; a multi-accumulator matmul (gated MLP) has several same-K loops,
    each its own ``K_o``/``K_i`` tower (sharing the ``K_s`` grid axis). The
    ``K_s`` grid axis is added to the outer BLOCK layers by the caller.

    ``k_bounds`` (symbolic flash) maps a streaming-reduce axis name to its runtime
    boundary ``Expr``: that axis' ``K_o`` ceil-divides and each step is wrapped in
    ``Cond(decoded_k < bound)`` — the TWISTED_MONOID identity "skip the fold"
    leaves the online-softmax state (m/l/O) unchanged for an out-of-range key."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            bound = (k_bounds or {}).get(kn)
            if bound is not None:
                # Symbolic streaming axis: ceil-div over the runtime extent.
                k_o_ext = s.axis.extent.ceil_div(splitk * bk * fk)
            else:
                # Each target loop tiles by its OWN extent (a flash nest's KV and
                # nested QK reduces differ); a plain matmul's K matches k_extent.
                this_extent = s.axis.extent.as_static() if (s.axis is not None and s.axis.extent.is_static) else k_extent
                k_o_ext = this_extent // (splitk * bk * fk)
            k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{kn}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * fk, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{kn}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(bk, "int")
            expr = expr + Var(k_i.name)
            if k_s is not None:
                expr = Var(k_s.name) * Literal(k_o_ext * bk * fk, "int") + expr
            sigma_k = Sigma({kn: expr})
            # Recurse first so a nested target loop (flash's QK inside KV) is also
            # split, then σ-rewrite this loop's body.
            inner = _replace_k_scalar(tuple(s.body), target_names, k_extent, bk, fk, splitk, k_s, k_bounds)
            if bound is not None:
                # Symbolic streaming step. A `Cond` around the carrier would
                # block-scope the loop-carried online-softmax state (m/l/O), so
                # instead CLAMP the streaming index in every load (safe read of a
                # duplicate key) and force the masked key's SCORE to the carrier's
                # identity (`-inf`) — `max(m, -inf) = m`, `exp(-inf) = 0`, so the
                # Monoid folds nothing for an out-of-range key (TWISTED_MONOID
                # identity) while the in-place state update stays unconditional.
                decoded_k = sigma_k.apply(Var(kn))
                pred = BinaryExpr("<", decoded_k, bound)
                clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", bound, Literal(1, "int")))
                body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in inner)
                new_body = _mask_flash_monoid(body_c, pred)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
            if k_f is not None:
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def _mask_flash_monoid(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask the streaming flash carrier's score to ``-inf`` past a symbolic-K
    boundary. Before each ``Monoid`` with partial ``(score, value)`` insert
    ``Init(score_kid, op=maximum)`` (seeds ``-inf``, the score component's
    identity) and ``Select(score_km, score if pred else score_kid)``, then fold
    ``score_km``. The masked key contributes nothing (``max(m, -inf) = m``,
    ``exp(-inf) = 0``); the in-place state update stays unconditional."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Monoid):
            score = c.partial[0]
            ident = f"{score}_kid"
            masked = f"{score}_km"
            out.append(Init(name=ident, op=ElementwiseImpl("maximum"), dtype=F32))
            out.append(
                Select(
                    name=masked,
                    branches=(SelectBranch(value=score, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
                )
            )
            out.append(replace(c, partial=(masked, *c.partial[1:])))
        else:
            out.append(c)
    return tuple(out)


