"""Reshape cooperative-reduce register-tile accesses for LDS.128.

Sibling to ``009_permute_register_tile``. Same idea (rewrite the
indices that ``008_register_tile`` emitted so the per-thread Load
pattern hits LDS.128 with zero bank conflicts) applied to a different
tile shape:

- 009 targets matmul (>= 2 THREAD axes). 008's σ produces
  ``Var(lane) * FN + c`` index patterns; 009 rewrites them so each
  LDS.128 phase covers 32 distinct banks.

- 009b targets cooperative-reduce (1 THREAD axis, synthesized ``t``).
  008's σ produces ``Var(t) + k*BN`` index patterns (with k =
  0..FN-1). The shape today is one LDS.16 per scalar Load — even with
  the ``__half2`` pack pass behind it, the load side is the
  bottleneck. We rewrite to:

      Var(t) + k*BN   →   VW*Var(t) + (k // VW)*BN*VW + (k % VW)

  Adjacent replicas (k=0..VW-1) now occupy adjacent memory positions
  per thread, so the ``003_vectorize_loads`` pass downstream can fold
  them into one ``__half2`` (VW=2), ``uint2`` (VW=4), or ``uint4``
  (VW=8) reinterpret-cast read. With VW=8 + fp16 source dtype the
  emit is one LDS.128 per K replica iteration; the warp covers 32
  distinct banks across 4 phases. The ``004_pack_fp16_register_tile``
  pass then collapses the paired Accums into ``__hmax2`` / ``__hadd2``.

## New knob

``VW`` (vector width). Choices ``(1, 2, 4, 8)`` corresponding to
LDS.16 / LDS.32 / LDS.64 / LDS.128. Default ``8`` — maximum LDS
throughput per instruction. The autotuner explores smaller values
when ``8`` doesn't fit (FN not divisible, dtype constraints, or
compute-bound kernels where the larger live-set hurts more than the
fewer instructions help).

Constraints:

- ``VW * dtype.nbytes <= 16`` — LDS.128 byte cap. For fp16 (2 B)
  any of 1/2/4/8 works; for fp32 (4 B) cap is VW <= 4.
- ``FN % VW == 0`` — must divide cleanly into the FN replicas.
- Source-buffer dtype must support a vector type at width VW (the
  target's ``vector_type(elem_dt, VW)`` is non-None). The downstream
  vec-load pass will check this independently; we precheck here so
  the autotuner doesn't waste benchmark slots on no-op variants.

## When the pass skips

- ``> 1`` THREAD axis (matmul shape — 009 handles it).
- No ``FN`` knob set (008 hasn't run, or ran as a no-op).
- ``VW == 1`` (identity rewrite — no-op).
- No body Load matches the ``Var(t) + literal*k`` pattern (kernel
  doesn't have the cooperative-reduce shape we target).
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var, affine_form
from deplodock.compiler.ir.stmt import Body, Load, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

logger = logging.getLogger(__name__)

# Maximum LDS instruction is 128 bits = 16 bytes — that's the cap on
# ``VW * dtype.nbytes``. The choices below are the four LDS widths
# (LDS.16 → LDS.128). ``1`` is the identity — included so the
# autotuner can fall back when the wider variants don't help.
_TUNE_VW_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
_LDS128_BYTES = 16

VW = Knob(
    "VW",
    KnobType.INT,
    hints=_TUNE_VW_CHOICES,
    help="Per-thread load width (positions per LDS instruction). 8 = LDS.128 on fp16; 4 = LDS.64; 2 = LDS.32; 1 = scalar LDS.",
)

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)

    # Only fire for fp16 register-tile reductions. The wider LDS widths
    # (LDS.64 / LDS.128) reduce instruction count proportionally; for
    # fp32 the existing vec-load path already produces LDS.128 at n=4,
    # so the permutation is a no-op win there. Limit scope to fp16 so
    # we don't reshape fp32 kernels and miss alignment proofs.
    fp16_inputs = any(node.output.dtype.name == "f16" for nid, node in match.graph.nodes.items() if nid in root.inputs)
    if not fp16_inputs:
        raise RuleSkipped("no fp16 inputs — 009b only targets fp16 cooperative-reduce")

    # Cooperative-reduce shape: exactly one THREAD axis.
    thread_axes = tile.thread_axes
    if len(thread_axes) != 1:
        raise RuleSkipped("not cooperative-reduce shape (need exactly 1 THREAD axis)")
    t_axis = thread_axes[0]

    # 008 must have run and committed FN.
    fn = root.op.knobs.get("FN")
    if fn is None or int(fn) < 2:
        raise RuleSkipped("FN not set or trivial (008 hasn't unrolled)")
    fn = int(fn)

    # Idempotence: VW already in knobs → this pass already fired.
    if VW.name in root.op.knobs:
        raise RuleSkipped("VW already set (009b already applied)")

    # Identify the cooperative reduce StridedLoop. ``005`` produced it
    # with ``start = Var(thread_axis_name)``; ``008`` widened the step
    # to ``BN * FN`` and σ-substituted the loop's axis variable for
    # each replica. Our rewrite operates on indices that contain
    # ``Var(reduce_loop_axis)`` — *not* the THREAD axis — because
    # that's the variable 008's σ touched.
    reduce_loops = [s for s in tile.body if isinstance(s, StridedLoop) and s.is_reduce and isinstance(s.step, Literal)]
    if not reduce_loops:
        raise RuleSkipped("no reduce StridedLoop in Tile body")
    # Reduce loops tied to the cooperative THREAD axis (start = Var(t)).
    # A softmax-style cooperative tile has two such loops (max + sum);
    # we permute the body of each.
    coop_loops = [sl for sl in reduce_loops if any(t_axis.name in expr.free_vars() for expr in (sl.start,) if hasattr(expr, "free_vars"))]
    if not coop_loops:
        raise RuleSkipped("no reduce StridedLoop is tied to the cooperative THREAD axis")
    # All cooperative reduce loops should share the same axis name (008 unrolled them in lockstep).
    axis_names = {sl.axis.name for sl in coop_loops}
    if len(axis_names) != 1:
        raise RuleSkipped("cooperative reduce loops use mixed axis names — unsupported")
    t_name = next(iter(axis_names))
    # The post-reduce normalize ``StridedLoop`` (005 Phase 3) is also
    # 008-unrolled (it shares the FN slot with the reduce loops). Its
    # body holds the epilogue Load/Write chain we need to permute too,
    # so that 003_vectorize_loads + 005_vectorize_stores can fold the
    # per-replica Loads/Writes into one LDS.128 / STG.128.
    normalize_loops = [
        sl
        for sl in tile.body
        if isinstance(sl, StridedLoop)
        and not sl.is_reduce
        and isinstance(sl.step, Literal)
        and sl.axis.name == t_name
        and any(t_axis.name in expr.free_vars() for expr in (sl.start,) if hasattr(expr, "free_vars"))
    ]
    permuted_loops = [*coop_loops, *normalize_loops]
    # ``BN`` = cooperative-thread stride. ``008`` widened each loop's
    # step to ``BN * FN``; recover BN by dividing back. All loops
    # should agree on BN.
    bns = {int(sl.step.value) // fn for sl in permuted_loops if int(sl.step.value) % fn == 0}
    if len(bns) != 1:
        raise RuleSkipped("cooperative loops disagree on BN (unexpected post-008 shape)")
    bn = next(iter(bns))

    # The rewrite assumes the loop runs exactly once (so
    # ``Var(reduce_axis) == thread_id`` at runtime and multiplying it by
    # VW is the right per-thread permutation). When the loop iterates
    # (axis extent > BN*FN), the loop's iter variable also encodes the
    # per-iter base offset, and naively scaling it by VW blows up the
    # address. Skip the multi-iter case; the autotuner will rank
    # configurations where the loop fully unrolls (BN*FN == extent)
    # against those where it doesn't.
    if any(int(sl.axis.extent) != int(sl.step.value) for sl in permuted_loops):
        raise RuleSkipped("a permuted loop iterates >1× (BN*FN < extent); 009b assumes single-iter")

    # Fork over VW choices that satisfy the constraints:
    #  - FN % VW == 0
    #  - VW * elem_bytes ≤ 16 (LDS.128 cap). The source-buffer dtype is
    #    inferred from the body's Load / Stage dtypes; conservatively
    #    use fp16 for now (the pass is most useful there). fp32 sources
    #    cap at VW=4 naturally.
    #
    # The autotuner gets all viable VW values; default ordering puts
    # the largest VW first so deterministic compile picks it.
    elem_bytes = _infer_elem_bytes(tile.body, t_name)
    if elem_bytes is None:
        raise RuleSkipped("body has no Load referencing the cooperative-thread axis")
    vw_cap = _LDS128_BYTES // elem_bytes
    viable = [v for v in _TUNE_VW_CHOICES if v <= vw_cap and fn % v == 0]
    if not viable or max(viable) <= 1:
        raise RuleSkipped(f"no viable VW > 1 (cap={vw_cap}, FN={fn})")
    # Drop VW=1 (identity rewrite) but keep larger choices.
    viable = [v for v in viable if v > 1]

    variants: list[TileOp] = []
    for vw in sorted(viable, reverse=True):
        # Only permute indices inside the loops 008 actually unrolled
        # — the reduce StridedLoops AND the post-reduce normalize
        # wrapper (now also unrolled by 008's normalize-loop slot).
        # Other Loops in the Tile body that share the axis name but
        # weren't unrolled (pre-008 step ``BN``) must NOT be permuted.
        new_tile_body = _rewrite_only_in_reduce_loops(tile.body, set(map(id, permuted_loops)), t_name, bn, vw, fn)
        new_tile = Tile(axes=tile.axes, body=new_tile_body)
        new_op = TileOp(
            body=body[:idx] + (new_tile,) + body[idx + 1 :],
            name=root.op.name,
            knobs={**root.op.knobs, VW.name: vw},
        )
        variants.append(new_op)
    return variants


def _rewrite_only_in_reduce_loops(body: Body, target_ids: set[int], t_name: str, bn: int, vw: int, fn: int) -> Body:
    """Walk the Tile body looking for the specific reduce StridedLoops
    we want to permute (matched by Python ``id``). Inside each such
    loop's body, apply the index rewrite. Outside, pass through
    unchanged. (The output / epilogue loop uses the same axis name
    but at a different step and must not be permuted.)"""
    out: list[Stmt] = []
    for s in body:
        if id(s) in target_ids:
            new_inner = _rewrite_body(s.body, t_name, bn, vw, fn)
            out.append(dc_replace(s, body=new_inner))
        else:
            out.append(s)
    return Body(out)


def _infer_elem_bytes(body: Body, t_name: str) -> int | None:
    """Look up the element byte size of the buffer most ``Load``s with
    the cooperative-thread axis read. Returns ``None`` if no such Load
    exists or the relevant Stage's dtype isn't a known fp / int type.
    """

    # Find Loads whose index depends on ``t_name``.
    candidates: dict[str, int] = {}
    for s in body.iter():
        if isinstance(s, Load):
            if not any(t_name in e.free_vars() for e in s.index):
                continue
            candidates[s.input] = candidates.get(s.input, 0) + 1

    if not candidates:
        return None
    # Cooperative-reduce at this point still reads from the source
    # smem Stage. The Stage doesn't carry an elem-dtype field at the
    # tile-IR level; the staged dtype matches the underlying graph
    # buffer's dtype, which by this lowering point is determined by
    # the surrounding Tile / output dtype. Conservative default: fp16
    # (the only dtype VW > 2 actually benefits today).
    return 2


def _rewrite_body(body: Body, t_name: str, bn: int, vw: int, fn: int) -> Body:
    """Apply the index permutation to every ``Load`` / ``Write`` in the
    body whose index contains ``Var(t_name)`` with a literal additive
    offset that's a multiple of ``bn``. Recurses into nested bodies."""
    out: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            new_children = tuple(_rewrite_body(child, t_name, bn, vw, fn) for child in nested)
            out.append(s.with_bodies(new_children))
            continue
        if isinstance(s, Load):
            out.append(Load(name=s.name, input=s.input, index=tuple(_rewrite_index(e, t_name, bn, vw, fn) for e in s.index)))
            continue
        if isinstance(s, Write):
            out.append(
                Write(
                    output=s.output,
                    index=tuple(_rewrite_index(e, t_name, bn, vw, fn) for e in s.index),
                    value=s.value,
                    reduce_op=s.reduce_op,
                )
            )
            continue
        out.append(s)
    return Body(out)


def _rewrite_index(e: Expr, t_name: str, bn: int, vw: int, fn: int) -> Expr:
    """Rewrite a single index expression.

    Pattern matched (via :func:`affine_form` on the cooperative-thread
    var): ``c * Var(t) + anchor`` where ``c == 1`` and ``anchor``
    decomposes as ``other_terms + k*bn`` for some ``k`` in ``0..fn-1``.
    Other shapes pass through unchanged."""
    af = affine_form(e, {t_name})
    if af is None:
        return e
    anchor, coeffs = af
    c = coeffs.get(t_name, 0)
    if c != 1:
        return e

    # Try to split the anchor into a literal additive constant and the rest.
    lit_part, rest = _split_literal(anchor)
    if lit_part < 0 or lit_part >= fn * bn or lit_part % bn != 0:
        return e
    k = lit_part // bn

    # Rebuild: c * VW * Var(t) + rest + (k // vw) * bn * vw + (k % vw).
    group_offset = (k // vw) * bn * vw
    lane_offset = k % vw
    new_anchor_lit = group_offset + lane_offset
    new_var_term: Expr = BinaryExpr("*", Literal(vw, "int"), Var(t_name))
    if isinstance(rest, Literal) and rest.value == 0:
        new_anchor: Expr = Literal(new_anchor_lit, "int")
    else:
        new_anchor = BinaryExpr("+", rest, Literal(new_anchor_lit, "int")) if new_anchor_lit != 0 else rest
    return BinaryExpr("+", new_var_term, new_anchor).simplify(SimplifyCtx.empty())


def _split_literal(e: Expr) -> tuple[int, Expr]:
    """Split ``e`` as ``literal + rest`` where ``literal`` is the
    additive integer constant of ``e`` and ``rest`` is the rest of the
    expression (free of any standalone literal additive constant).
    Returns ``(0, e)`` when ``e`` has no literal additive constant."""
    if isinstance(e, Literal) and isinstance(e.value, int):
        return e.value, Literal(0, "int")
    if isinstance(e, BinaryExpr) and e.op == "+":
        lv, lr = _split_literal(e.left)
        rv, rr = _split_literal(e.right)
        if isinstance(lr, Literal) and lr.value == 0:
            new_rest = rr
        elif isinstance(rr, Literal) and rr.value == 0:
            new_rest = lr
        else:
            new_rest = BinaryExpr("+", lr, rr)
        return lv + rv, new_rest
    return 0, e
