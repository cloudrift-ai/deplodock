"""Ping-pong double-buffering for K-outer staged matmul kernels.

For a Tile body shaped like ``Loop(K_outer, body=[Stage*, reduce])``,
this pass replaces each ``Stage`` with a ``BufferedStage`` carrying
``buffer_count = 2`` and a phase expression ``Var(K_outer_axis) % 2``.
The materializer then doubles the smem allocation and prepends the
phase to both the cooperative-load write and every body Load that
reads from the staged buffer.

The win comes from removing the leading ``__syncthreads`` between
prev-compute and next-load: with ping-pong, consecutive iterations
write to *different* physical smem regions, so the next iteration's
load can issue before the previous iteration's compute fully drains.
The trailing Sync (after the cooperative load, before compute reads)
is preserved.

Trigger:

- Tile body has exactly one ``Tile`` (the standard shape).
- The Tile body has a free Loop ``K_outer`` (extent ≥ 2) whose body
  contains ≥ 1 ``Stage`` and exactly one reduce ``Loop``.
- The reduce body has only ``Load`` / ``Assign`` / ``Accum`` stmts,
  defines ≥ 1 ``Accum``, and no body stmt reads any local Accum
  target's running value (rejects in-loop online-softmax-style merges
  where an Assign reads ``acc`` mid-iteration). ``register_tile`` no
  longer enforces this gate; ``014`` keeps it because its rewrite
  reorders smem visibility relative to the FMA chain and that
  reordering compounds fp32 drift on online-softmax-style bodies.
- Smem budget: ``2 × sum(slab_bytes) ≤ ctx.max_dynamic_smem`` — the
  per-block dynamic-smem opt-in cap derived from compute capability
  (99 KB on sm_86/89/120, 163 KB on sm_80, 227 KB on sm_90+).

The ``BUFFER_COMPUTE`` knob forks when a ``ComputeStage`` is present
in the K_o body (i.e. ``007b_hoist`` ran with ``FUSED_PIPELINE=True``):
``False`` (default) leaves the compute stage single-buffered; ``True``
ring-buffers it alongside the transports. The fork only fires when a
``ComputeStage`` is actually present — single-buffer-eligible bodies
emit one variant with ``BUFFER_COMPUTE=False`` stamped for idempotence.

Idempotence: a Stage that's already a ``BufferedStage`` is left alone.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BufferedStage, ComputeStage, Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import collect_invariant_names, single_tile

PATTERN = [Pattern("root", TileOp)]

BUFFER_COMPUTE = Knob(
    "BUFFER_COMPUTE",
    KnobType.BOOL,
    hints=(True, False),
    help=(
        "Promote a ComputeStage (produced by 007b_hoist) to a ring-buffered "
        "Stage so its output is double-buffered alongside the per-source transport stages. "
        "Experimental — lets a downstream pass try to overlap compute and reduce across "
        "K_outer iterations. Default off."
    ),
)

_BUFFER_COUNT = 2

_VARIANT_ORDER: tuple[bool, ...] = (False, True)
"""Emission order for the BUFFER_COMPUTE fork: leave the compute stage
single-buffered first (greedy default — smaller smem), ring-buffer it
second. Only emitted when a ``ComputeStage`` sits inside an eligible
K_o; otherwise the rule emits a single variant with
``BUFFER_COMPUTE=False`` stamped for idempotence."""


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    """Emit BUFFER_COMPUTE polarities for any K-outer Loop carrying a
    ``ComputeStage``; single variant otherwise."""
    if BUFFER_COMPUTE.name in root.op.knobs:
        raise RuleSkipped("double-buffer already applied (idempotence via knob)")

    polarities: tuple[bool, ...] = _VARIANT_ORDER if _has_compute_stage_in_kouter(root.op.body) else (False,)
    variants: list[TileOp] = []
    for buffer_compute in polarities:
        new_body = _maybe_rewrite(root.op.body, ctx.max_dynamic_smem, buffer_compute=buffer_compute)
        if new_body is None:
            continue
        knobs = {**root.op.knobs, BUFFER_COMPUTE.name: buffer_compute}
        variants.append(TileOp(body=new_body, name=root.op.name, knobs=knobs))
    if not variants:
        raise RuleSkipped("no K-outer matmul Loop eligible for double-buffering")
    return variants


def _has_compute_stage_in_kouter(body: Body) -> bool:
    """True if any K-outer Loop in the Tile body contains a ``ComputeStage``
    — the only shape where ``BUFFER_COMPUTE`` polarity affects output."""
    try:
        _, tile = single_tile(body)
    except Exception:
        return False
    for s in tile.body:
        if isinstance(s, Loop) and not s.is_reduce and s.role is Role.SERIAL_OUTER:
            if any(isinstance(c, ComputeStage) for c in s.body):
                return True
    return False


def _maybe_rewrite(body: Body, smem_budget: int, *, buffer_compute: bool) -> Body | None:
    idx, tile = single_tile(body)

    new_tile_body = _process_scope(tile.body, smem_budget, buffer_compute=buffer_compute)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(body: Body, smem_budget: int, *, buffer_compute: bool) -> Body:
    """Walk the body looking for K-outer Loops eligible for double-buffering.

    Tracks ``invariant_names`` — the set of SSA names defined by
    sibling stmts that precede each K-outer Loop. Cross-loop SSA
    reads that resolve into this set are loop-invariant w.r.t. the
    K-outer being considered (a sibling reduce above produced the
    name; the value is finalized before this K-outer starts), so the
    pass can safely double-buffer the loop. Reads that don't resolve
    locally *and* aren't in the invariant set are still rejected —
    those are the in-loop-online-softmax-style merges the original
    gate guarded against.
    """
    new_body: list[Stmt] = list(body)
    changed = False
    invariant_names: set[str] = set()
    for i, s in enumerate(body):
        if isinstance(s, Loop) and not s.is_reduce and _is_kouter_matmul(s, invariant_names):
            if not any(isinstance(st, BufferedStage) for st in s.body):
                stages = [st for st in s.body if isinstance(st, Stage)]
                if _BUFFER_COUNT * sum(st.smem_bytes for st in stages) <= smem_budget:
                    updated = _double_buffer(s, buffer_compute=buffer_compute)
                    if updated is not None:
                        new_body[i] = updated
                        changed = True
        invariant_names.update(collect_invariant_names(s))
    return tuple(new_body) if changed else body


def _is_kouter_matmul(loop: Loop, invariant_names: set[str]) -> bool:
    """K-outer-matmul predicate keyed off planner Roles.

    The planner stamps ``Role.SERIAL_OUTER`` on the K_o loop and
    ``Role.STAGE_INNER`` on the K_i reduce inside it (see
    ``000_partition_planner._build_split_body``). Matching on those
    tags replaces the structural ``is_matmul_k_outer`` walk — only
    planner-recognized matmul K-loops are eligible.

    Layered gates that aren't expressible as roles:

    - ``extent ≥ 2`` (need room for ping-pong).
    - ``≥ 1 Stage`` in the K_o body (after 007_stage_inputs picked
      one — otherwise there's nothing to double-buffer).
    - No in-loop online-softmax-style merge in K_i (non-Accum stmt
      reading a local Accum's running value would compound fp32 drift
      under double-buffering).
    - Cross-loop SSA reads only when the referenced name is in
      ``invariant_names`` — names defined by sibling stmts above this
      K-outer at the enclosing scope. Those values are finalized
      before K-outer starts, so the rewrite can read them from
      registers without phase rotation. (015 carries the same gate.)
    """
    if loop.role is not Role.SERIAL_OUTER:
        return False
    if int(loop.axis.extent) < 2:
        return False
    if not any(isinstance(s, Stage) for s in loop.body):
        return False
    reduces = [c for c in loop.body if isinstance(c, Loop) and c.is_reduce]
    if len(reduces) != 1:
        return False
    k_inner = reduces[0]
    if k_inner.role is not Role.STAGE_INNER:
        return False
    for c in k_inner.body:
        if isinstance(c, Accum):
            continue
        if any(isinstance(s, Accum) for s in k_inner.body.deps_of(c)):
            return False
    for c in k_inner.body:
        for d, defstmt in zip(c.deps(), k_inner.body.deps_of(c), strict=False):
            if defstmt is None and d not in invariant_names:
                return False
    return True


def _double_buffer(loop: Loop, *, buffer_compute: bool) -> Loop | None:
    """Promote each ``Stage`` in the loop body to ``BufferedStage`` with
    ``buffer_count=2`` and a shared phase expression, and rewrite each
    body Load that reads from a staged buffer to prepend the phase
    index. ``buffer_compute`` decides whether a ``ComputeStage``'s own
    output is ring-buffered alongside the transports."""
    phase = Var(loop.axis.name) % Literal(_BUFFER_COUNT, "int")
    staged_names: set[str] = set()

    new_body: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, ComputeStage):
            # ComputeStage: body Loads read sibling Stage smem. Prepend
            # phase so the compute consumes the slot the producer just
            # wrote. When BUFFER_COMPUTE is set, ring-buffer the compute
            # stage's own output too — experimental, lets a downstream
            # pass try to overlap compute and reduce across K_outer
            # iterations.
            new_inner_body = s.body.map(_make_load_rewriter(staged_names, phase))
            if buffer_compute:
                staged_names.add(s.name)
                new_body.append(dc_replace(s, body=new_inner_body, buffer_count=_BUFFER_COUNT, phase=phase))
            else:
                new_body.append(dc_replace(s, body=new_inner_body))
        elif isinstance(s, Stage) and len(s.source_loads) == 1:
            # Single-source transport — standard double-buffer promotion.
            staged_names.add(s.name)
            new_body.append(
                BufferedStage(
                    name=s.name,
                    axes=s.axes,
                    body=s.body,
                    pad=s.pad,
                    buffer_count=_BUFFER_COUNT,
                    phase=phase,
                )
            )
        elif isinstance(s, Loop) and s.is_reduce:
            new_body.append(dc_replace(s, body=s.body.map(_make_load_rewriter(staged_names, phase))))
        else:
            # Multi-source inline-fuse Stage and any other stmt: pass
            # through. The inline-fuse Stage bypasses 010/011/013 by
            # design (gmem Loads inside the body can't be ring-buffered
            # without rewriting the cooperative-load shape).
            new_body.append(s)
    return dc_replace(loop, body=new_body)


def _make_load_rewriter(staged_names: set[str], phase):
    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.input in staged_names:
            return Load(name=s.name, input=s.input, index=(phase, *s.index))
        return s

    return fn
