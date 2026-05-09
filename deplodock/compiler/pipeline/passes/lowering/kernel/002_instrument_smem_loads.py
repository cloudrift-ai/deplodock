"""Instrument smem-Load addresses to ``_debug_buf`` for the bank-conflict
GPU trace.

Off by default. Gated by ``DEPLODOCK_BANK_TRACE=1`` (matches the
``DEPLODOCK_DISABLE_*`` / ``DEPLODOCK_TMA`` convention used by sibling
rules). :func:`deplodock.compiler.diagnostics.bank_conflicts.simulate_graph`
sets the env var around its ``backend.compile()`` call.

When enabled, walks each ``KernelOp`` body, finds every body ``Load``
whose ``input`` matches a sibling ``Smem`` decl, and inserts an
``_AddrTrace`` Stmt right after it. The Stmt subclasses ``Write`` so
``KernelOp.outputs`` auto-picks up ``_debug_buf`` and the cuda-lowering
pass adds it to the kernel arg list.

Slot scheme is fully deterministic — no atomics. For each smem Load:

    slot = load_offset
         + ((iter_flat * num_warps) + warp_id) * 32
         + lane

where ``load_offset`` is the cumulative size of all earlier instrumented
Loads in the same kernel, ``iter_flat`` is the row-major flatten of all
enclosing ``Loop`` / ``StridedLoop`` axis vars over their literal
extents (Horner accumulation), ``num_warps = ceil(blockDim.x / 32)``.
``warp_id = threadIdx.x >> 5``, ``lane = threadIdx.x & 31``. The
recorded value is ``smem_addr_in_elements + 1`` so an unwritten slot
(``0``) is distinguishable from "lane read element 0".

Per-load decoding metadata is stashed on the graph as
``graph._bank_trace_sidecar`` (keyed by ``KernelOp.name``). The driver
reads it after compile to rebuild ``BankConflictResult`` provenance —
KernelOp has been replaced by CudaOp at that point, but the graph
object is the same instance the rule mutated.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import prod

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.stmt import Body, Cond, Load, Loop, RenderCtx, StridedLoop, Tile, Write, _pad, render_index
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

WARP_SIZE = 32

DEBUG_BUF_NAME = "_debug_buf"
SIDECAR_ATTR = "_bank_trace_sidecar"

# Hard cap on the debug buffer size in fp32 elements. Sized once on first
# fire and shared across every instrumented KernelOp in the graph — every
# Load gets a disjoint slot region (offsets accumulate in the sidecar),
# so one shared buffer of this size is enough as long as the sum of
# region sizes stays below the cap. 1M fp32 = 4 MiB.
_MAX_DEBUG_SLOTS = 1 << 20


def rewrite(graph: Graph, root: Node) -> Graph | None:
    if os.environ.get("DEPLODOCK_BANK_TRACE") != "1":
        raise RuleSkipped("DEPLODOCK_BANK_TRACE not set")
    kernel_op = root.op
    if not isinstance(kernel_op, KernelOp):
        return None

    smem_names = kernel_op.smem_names
    if not smem_names:
        return None
    num_warps = _num_warps(kernel_op)
    if num_warps is None:
        return None

    sidecar = _get_sidecar(graph)
    if kernel_op.name in sidecar:
        # Already instrumented (re-fire after some other rule rewrote a
        # sibling node and left this one untouched).
        return None

    state = _State(smem_names=smem_names, num_warps=num_warps, debug_buf=DEBUG_BUF_NAME)
    new_body = _transform_body(kernel_op.body, (), state)
    if not state.records:
        return None

    kernel_op.body = Body(tuple(new_body))
    sidecar[kernel_op.name] = {
        "loads": state.records,
        "buf_size": state.next_offset,
        "num_warps": num_warps,
    }
    _ensure_debug_buf(graph, DEBUG_BUF_NAME)
    return None


def _get_sidecar(graph: Graph) -> dict:
    sidecar = getattr(graph, SIDECAR_ATTR, None)
    if sidecar is None:
        sidecar = {}
        setattr(graph, SIDECAR_ATTR, sidecar)
    return sidecar


def _ensure_debug_buf(graph: Graph, name: str) -> None:
    """Add ``_debug_buf`` as a graph output once per pipeline run.
    Idempotent — multiple instrumented KernelOps share the same buffer."""
    if name not in graph.nodes:
        graph.add_node(
            InputOp(),
            inputs=[],
            output=Tensor(name=name, shape=(_MAX_DEBUG_SLOTS,), dtype="f32"),
            node_id=name,
        )
    if name not in graph.outputs:
        graph.outputs.append(name)


# ---------------------------------------------------------------------------
# Sidecar metadata + walker state
# ---------------------------------------------------------------------------


@dataclass
class LoadRecord:
    """Per-smem-Load metadata stashed on ``OPTIONS.sidecar`` for the
    GPU-trace driver to decode the trace buffer."""

    load_id: int
    smem_name: str
    ssa_name: str
    index: tuple
    enclosing_axes: tuple[Axis, ...]
    iter_extents: tuple[int, ...]
    iter_total: int
    offset: int


@dataclass
class _State:
    smem_names: frozenset
    num_warps: int
    debug_buf: str
    records: dict[int, LoadRecord] = field(default_factory=dict)
    next_offset: int = 0
    next_id: int = 0


# ---------------------------------------------------------------------------
# Body walk
# ---------------------------------------------------------------------------


def _transform_body(body: Iterable, enclosing: tuple[Axis, ...], state: _State) -> list:
    out: list = []
    for s in body:
        if isinstance(s, Load) and s.input in state.smem_names:
            out.append(s)
            out.append(_make_record(s, enclosing, state))
        elif isinstance(s, Tile):
            inner = _transform_body(s.body, enclosing, state)
            out.append(Tile(axes=s.axes, body=Body(tuple(inner))))
        elif isinstance(s, Loop):
            inner = _transform_body(s.body, (*enclosing, s.axis), state)
            out.append(Loop(axis=s.axis, body=Body(tuple(inner)), unroll=s.unroll))
        elif isinstance(s, StridedLoop):
            inner = _transform_body(s.body, (*enclosing, s.axis), state)
            out.append(StridedLoop(axis=s.axis, start=s.start, step=s.step, body=Body(tuple(inner)), unroll=s.unroll))
        elif isinstance(s, Cond):
            inner = _transform_body(s.body, enclosing, state)
            else_inner = _transform_body(s.else_body, enclosing, state) if s.else_body else []
            out.append(Cond(cond=s.cond, body=Body(tuple(inner)), else_body=Body(tuple(else_inner))))
        else:
            out.append(s)
    return out


def _make_record(load: Load, enclosing: tuple[Axis, ...], state: _State) -> _AddrTrace:
    extents = tuple(int(ax.extent) for ax in enclosing)
    iter_total = prod(extents) if extents else 1
    region_size = iter_total * state.num_warps * WARP_SIZE
    rec = LoadRecord(
        load_id=state.next_id,
        smem_name=load.input,
        ssa_name=load.name,
        index=tuple(load.index),
        enclosing_axes=enclosing,
        iter_extents=extents,
        iter_total=iter_total,
        offset=state.next_offset,
    )
    state.records[state.next_id] = rec
    state.next_id += 1
    state.next_offset += region_size
    return _AddrTrace(record=rec, num_warps=state.num_warps, debug_buf=state.debug_buf)


def _num_warps(kernel_op: KernelOp) -> int | None:
    for s in kernel_op.body:
        if isinstance(s, Tile):
            n_threads = prod(int(ax.extent) for ax in s.thread_axes) if s.thread_axes else 1
            return max((n_threads + WARP_SIZE - 1) // WARP_SIZE, 1)
    return None


# ---------------------------------------------------------------------------
# Pass-private Stmt — emits a deterministic-slot Write to ``_debug_buf``.
# Subclasses ``Write`` so ``KernelOp.outputs`` picks up the buffer.
# ---------------------------------------------------------------------------


@dataclass(init=False, frozen=True, eq=False)
class _AddrTrace(Write):
    """Records ``smem_addr_in_elements + 1`` of the immediately-preceding
    smem ``Load`` into ``_debug_buf`` at a deterministic per-(load, iter,
    warp, lane) slot. Subclasses ``Write`` so ``KernelOp.outputs`` picks
    up the buffer; renders directly bypassing ``Write.value``."""

    record: LoadRecord
    num_warps: int
    debug_buf: str

    def __init__(self, record: LoadRecord, num_warps: int, debug_buf: str):
        # Frozen dataclass — must use ``object.__setattr__``. Bypass
        # ``Write.__post_init__`` (we never set ``reduce_op``).
        object.__setattr__(self, "output", debug_buf)
        object.__setattr__(self, "index", ())
        object.__setattr__(self, "value", "")
        object.__setattr__(self, "reduce_op", None)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "num_warps", num_warps)
        object.__setattr__(self, "debug_buf", debug_buf)

    def deps(self) -> tuple[str, ...]:
        return ()

    def has_side_effects(self) -> bool:
        return True

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}_AddrTrace[{self.record.load_id}] {self.debug_buf}<{self.record.ssa_name}@{self.record.smem_name}>"]

    def render(self, ctx: RenderCtx) -> list[str]:
        rec = self.record
        flat_addr = render_index(rec.smem_name, rec.index, ctx)

        if rec.enclosing_axes:
            parts: list[str] = []
            for i, ax in enumerate(rec.enclosing_axes):
                if i == 0:
                    parts.append(ax.name)
                else:
                    parts.append(f"({parts[-1]} * {rec.iter_extents[i]} + {ax.name})")
            iter_flat = parts[-1]
        else:
            iter_flat = "0"

        slot = f"({rec.offset} + ((({iter_flat}) * {self.num_warps} + (threadIdx.x >> 5)) * {WARP_SIZE}) + (threadIdx.x & 31))"
        pad = _pad(ctx.indent)
        return [f"{pad}{self.debug_buf}[{slot}] = (float)(({flat_addr}) + 1);"]
