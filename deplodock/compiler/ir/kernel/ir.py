"""Kernel IR — the fully-scheduled kernel form, just above CUDA source.

Kernel IR sits between Tile IR (schedule decisions as structural Stmts)
and CUDA source (text). Its body contains the explicit hardware
machinery: ``Tile`` (thread/block coord bindings), ``Smem``
(``__shared__`` arrays), ``Sync`` (``__syncthreads`` barriers),
``TreeHalve`` (cross-thread reduction over smem), ``StridedLoop``
(strided per-thread loop).

Pipeline shape::

    Tile IR ──materialize_tile──▶ Kernel IR
                    ──render_kernelop──▶ CUDA source

**Leaf compute reuses Loop IR directly**. ``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond`` / ``Loop`` come straight
from ``ir.loop`` — buf names are strings so they're directly renderable.

Kernel IR deliberately contains no scheduling decisions — those live in
Tile IR and are materialized away before reaching this layer. A
``KernelOp`` is what the CUDA backend turns into a ``RawKernel`` launch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Load,
    Loop,
    RenderCtx,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Tile,
    Write,
    _pad,
    pretty_body,
)
from deplodock.compiler.tensor import Tensor

# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------


@dataclass
class Smem(Stmt):
    """Declare a per-block ``__shared__`` array.

    Renders to ``__shared__ [__align__(N)] <dtype> <name>[<prod(extents)>];``.
    ``extents`` is the multi-dim shape used to flatten ``Load`` / ``Write``
    indices against this buffer. smem_bytes for ``CudaOp`` is computed by
    walking the KernelOp body and summing ``prod(extents) * sizeof(dtype)``
    across distinct ``Smem`` declarations.

    ``align`` is an optional explicit byte alignment. TMA destinations
    (``cp.async.bulk.tensor``) require 16-byte aligned smem; the
    materializer sets ``align=16`` on TMA-target slabs. Default 0 means
    "no explicit alignment" — falls back to the dtype's natural alignment.
    """

    name: str
    extents: tuple[int, ...]
    dtype: str = "float"
    align: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        ext = ", ".join(str(e) for e in self.extents) or "-"
        ali = f" align={self.align}" if self.align else ""
        return [f"{indent}Smem {self.name}[{ext}] ({self.dtype}){ali}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``__shared__ <dtype> <name>[<prod(extents)>];`` and register the
        buffer's shape so subsequent ``Load``/``Write`` flatten correctly.

        When ``ctx.smem_dynamic_offsets`` carries this buffer's offset,
        emit a pointer alias into the shared ``_smem_pool`` (extern
        dynamic smem) instead of a static ``__shared__`` array — the
        renderer falls back to the dynamic path when total per-CTA smem
        would exceed the 48 KB static cap.
        """
        from deplodock.compiler.backend.cuda.dtype import canonical_from_cuda_name  # noqa: PLC0415

        total = 1
        for e in self.extents:
            total *= int(e)
        ctx.shapes[self.name] = tuple(int(e) for e in self.extents)
        # Register the smem buffer's canonical dtype so Load/Write
        # against this name pick the right local C type. Mbarrier slabs
        # (``"unsigned long long"``) map to None and stay out of the
        # dtype-aware path.
        smem_canonical = canonical_from_cuda_name(self.dtype)
        if smem_canonical is not None:
            ctx.buffer_dtypes[self.name] = smem_canonical
        if self.name in ctx.smem_dynamic_offsets:
            offset = ctx.smem_dynamic_offsets[self.name]
            return [f"{_pad(ctx.indent)}{self.dtype}* {self.name} = reinterpret_cast<{self.dtype}*>(_smem_pool + {offset});"]
        ali = f"__align__({self.align}) " if self.align else ""
        return [f"{_pad(ctx.indent)}__shared__ {ali}{self.dtype} {self.name}[{total}];"]


@dataclass
class Sync(Stmt):
    """``__syncthreads();`` — block-wide barrier."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Sync"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}__syncthreads();"]


@dataclass
class CpAsyncCopy(Stmt):
    """Issue one ``cp.async.cg.shared.global`` instruction.

    Replaces the per-thread ``Load(reg) + Write(smem)`` pair in cooperative
    loads on sm_80+. The hardware copies 4 bytes (one fp32) directly from
    global to shared without a register staging slot, freeing one thread
    register and removing the LDG → STS dependency.

    Renders to inline PTX. The asm reads the smem address via
    ``cvta.to.shared.u32`` and the global pointer as a 64-bit value;
    indices flatten via ``render_index`` against the buffer's declared
    shape (same as ``Load`` / ``Write``)."""

    smem: str  # destination smem buffer name
    smem_index: tuple
    src: str  # source global buffer name
    src_index: tuple

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        src_idx = ", ".join(e.pretty() for e in self.src_index)
        return [f"{indent}cp.async {self.smem}[{smem_idx}] <- {self.src}[{src_idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        src_flat = render_index(self.src, self.src_index, ctx)
        pad = _pad(ctx.indent)
        asm = f'asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n" :: "r"(_smem_addr), "l"(&{self.src}[{src_flat}]) : "memory");'
        return [
            f"{pad}{{",
            f"{pad}    unsigned int _smem_addr = __cvta_generic_to_shared(&{self.smem}[{smem_flat}]);",
            f"{pad}    {asm}",
            f"{pad}}}",
        ]


@dataclass
class CpAsyncCommit(Stmt):
    """``cp.async.commit_group;`` — finalize the preceding cp.async copies
    issued by this thread into a commit group. Pairs with
    ``CpAsyncWait`` to wait for that group to drain."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.commit_group"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.commit_group;\\n" ::: "memory");']


@dataclass
class CpAsyncWait(Stmt):
    """``cp.async.wait_group N;`` — block this thread until ≤ N cp.async
    groups remain in flight. ``group=0`` waits for everything (synchronous
    style); larger values stagger waits for software pipelining."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.wait_group({self.group})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.wait_group {self.group};\\n" ::: "memory");']


@dataclass
class TmaDescriptor(Stmt):
    """Host-built ``CUtensorMap`` descriptor for a TMA box copy.

    Declarative — renders nothing inside the kernel. The CUDA backend
    walks the ``KernelOp`` body, calls ``cuTensorMapEncodeTiled`` once
    per distinct ``TmaDescriptor`` at compile time, and binds the
    resulting 128-byte descriptor as a ``__grid_constant__`` kernel
    arg. ``name`` matches the parameter name referenced by sibling
    ``TmaLoad`` stmts. ``src_buf`` names the global buffer the descriptor
    addresses (used by the kernel-arg pipeline to add ``src_buf`` to the
    parameter list).
    """

    name: str
    src_buf: str
    src_shape: tuple[int, ...]
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"
    dtype: str = "float"

    def pretty(self, indent: str = "") -> list[str]:
        shape = ", ".join(str(e) for e in self.src_shape)
        box = ", ".join(str(e) for e in self.box_extents)
        return [f"{indent}TmaDescriptor {self.name} <- {self.src_buf}[{shape}] box=[{box}] swizzle={self.swizzle}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        # Descriptor is a kernel parameter built host-side; no in-kernel render.
        return []


@dataclass
class TmaLoad(Stmt):
    """``cp.async.bulk.tensor.<rank>d`` — single-thread box copy.

    Issued by exactly one thread of the CTA (the materializer wraps this
    in ``Cond(tid==0, [...])``). Source coords come from the staging
    Stmt's ``origin`` (CTA-uniform); destination is the dynamic-phase
    smem slab. Pairs with ``MbarrierArriveExpectTx`` (issued immediately
    before) and ``MbarrierWait`` (issued by every consumer thread)."""

    smem: str
    smem_index: tuple
    desc: str
    coords: tuple
    mbar: str
    mbar_slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        coords = ", ".join(e.pretty() for e in self.coords)
        s = "" if self.mbar_slot is None else f"[{self.mbar_slot.pretty()}]"
        return [f"{indent}TmaLoad {self.smem}[{smem_idx}] <- {self.desc}({coords}) mbar={self.mbar}{s}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        rank = len(self.coords)
        # PTX expects coords in driver order (innermost-first), which is
        # the reverse of our C-order ``coords`` (== ``stage.origin``).
        coord_args = ", ".join(c.render(ctx) for c in reversed(self.coords))
        mbar_addr = "&" + self.mbar if self.mbar_slot is None else f"&{self.mbar}[{self.mbar_slot.render(ctx)}]"
        pad = _pad(ctx.indent)
        # The ``cp_async_bulk_tensor_<rank>d`` helpers are defined in the
        # kernel prelude — single-CTA ``.shared::cta`` qualifier; cluster
        # launches would need a separate helper variant.
        return [
            f"{pad}cp_async_bulk_tensor_{rank}d(&{self.smem}[{smem_flat}], {self.desc}, {coord_args}, {mbar_addr});",
        ]


def _mbar_addr_expr(mbar: str, slot: Expr | None) -> str:
    """``__cvta_generic_to_shared(&<mbar>[<slot>])`` — one mbarrier per slot
    of a ring buffer is the only correct pattern for pipelined TMA, since
    multiple ``arrive.expect_tx`` calls on the *same* mbarrier within one
    phase accumulate into one phase rather than queueing per-slot phases.
    ``slot=None`` means scalar mbar (single-slot, no ring buffering)."""
    if slot is None:
        return f"__cvta_generic_to_shared(&{mbar})"
    return f"__cvta_generic_to_shared(&{mbar}[{{slot_expr}}])".replace("{slot_expr}", "%(slot)s")


@dataclass
class MbarrierInit(Stmt):
    """``mbarrier.init.shared.b64 [&mbar[slot]], count;`` — one-shot init.

    With ``slot=None``: initializes a scalar mbarrier (single-slot path).
    With ``slot=Expr``: initializes one element of a per-slot mbarrier
    array. The materializer emits one ``MbarrierInit`` per literal slot
    index in the buffer count, hoisted to the kernel prologue inside
    ``Cond(threadIdx.x == 0, [...]) + Sync()``."""

    mbar: str
    count: int = 1
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierInit({self.mbar}{s}, count={self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_init({addr}, {self.count});"]


@dataclass
class MbarrierArriveExpectTx(Stmt):
    """``mbarrier.arrive.expect_tx.shared.b64`` — declare the expected
    transaction byte count for the in-flight TMA copy on this barrier.
    With per-slot mbarrier arrays, ``slot`` selects the ring buffer slot."""

    mbar: str
    bytes_: int
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierArriveExpectTx({self.mbar}{s}, bytes={self.bytes_})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_arrive_expect_tx({addr}, {self.bytes_});"]


@dataclass
class MbarrierWait(Stmt):
    """``mbarrier.try_wait.parity.shared.b64`` — block until the barrier's
    parity flips for ``phase``. Replaces ``CpAsyncWait + Sync`` for TMA
    transports — mbarrier arrival already provides CTA-wide visibility.
    ``slot`` selects the ring buffer slot when the mbarrier is an array."""

    mbar: str
    phase: Expr
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierWait({self.mbar}{s}, phase={self.phase.pretty()})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        phase_expr = self.phase.render(ctx)
        return [f"{pad}mbarrier_wait_parity({addr}, {phase_expr});"]


@dataclass
class TreeHalve(Stmt):
    """Cooperative power-of-two tree reduction over a 1D smem buffer.

    Reduces ``buf[0..length)`` into ``buf[0]`` using ``op`` as the combine.
    ``tid_var`` names the cooperative thread axis. ``length`` must be a
    power of two and ``≤ blockDim.x``. ``dtype`` is the element type of
    the smem buffer and the dtype the combine operates in — the renderer
    picks the right intrinsic spelling (``fmaxf`` vs ``__hmax``, etc.)
    via the target.
    """

    buf: str
    op: ElementwiseImpl
    length: int
    tid_var: str
    dtype: DataType = F32

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}TreeHalve({self.buf} :{self.dtype.name}, op={self.op.name}, length={self.length}, tid={self.tid_var})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Power-of-two tree reduction over ``buf[0..length)`` into ``buf[0]``."""
        pad = _pad(ctx.indent)
        inner_pad = _pad(ctx.indent + 1)
        halve_pad = _pad(ctx.indent + 2)
        op_expr = _binary_combine_expr(self.op, f"{self.buf}[{self.tid_var}]", f"{self.buf}[{self.tid_var} + s]", ctx.target, self.dtype.name)
        half = int(self.length) // 2
        return [
            f"{pad}for (int s = {half}; s > 0; s >>= 1) {{",
            f"{inner_pad}if ({self.tid_var} < s) {{",
            f"{halve_pad}{self.buf}[{self.tid_var}] = {op_expr};",
            f"{inner_pad}}}",
            f"{inner_pad}__syncthreads();",
            f"{pad}}}",
        ]


@dataclass
class WarpShuffle(Stmt):
    """Warp-shuffle reduction: combine ``value`` across ``length`` lanes
    via ``__shfl_xor_sync`` and bind the broadcast result to ``name``.

    Renders as a register-only butterfly (no smem, no syncthreads):

        <type> <name> = <value>;
        <name> = <op>(<name>, __shfl_xor_sync(0xffffffff, <name>, length/2));
        <name> = <op>(<name>, __shfl_xor_sync(0xffffffff, <name>, length/4));
        ...
        <name> = <op>(<name>, __shfl_xor_sync(0xffffffff, <name>, 1));

    Used by ``materialize_tile._emit_combine`` when the cooperative
    thread count fits in a single warp (``length ≤ 32``, power of two).
    Replaces the ``Smem`` + ``Sync`` + ``TreeHalve`` + ``Sync`` + ``Load``
    sequence — kills the smem alloc, the two block barriers, and the
    smem-staged broadcast load. ``dtype`` is the parent accumulator's
    dtype; the renderer declares the local + picks the combine intrinsic
    at that dtype via the target.
    """

    name: str
    value: str
    op: ElementwiseImpl
    length: int
    dtype: DataType = F32

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WarpShuffle({self.name} :{self.dtype.name} <- {self.value}, op={self.op.name}, length={self.length})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        acc_dt = self.dtype.name
        value_dt = ctx.ssa_dtypes.get(self.value, acc_dt)
        src_expr = ctx.target.convert(self.value, value_dt, acc_dt)
        ctx.ssa_dtypes[self.name] = acc_dt
        out = [f"{pad}{ctx.type_name(acc_dt)} {self.name} = {src_expr};"]
        s = int(self.length) // 2
        while s > 0:
            shfl = f"__shfl_xor_sync(0xffffffff, {self.name}, {s})"
            combined = _binary_combine_expr(self.op, self.name, shfl, ctx.target, acc_dt)
            out.append(f"{pad}{self.name} = {combined};")
            s >>= 1
        return out


def _binary_combine_expr(op: ElementwiseImpl, a: str, b: str, target=None, dt: str = "f32") -> str:
    """Render a 2-arg combine for ``ElementwiseImpl`` reduce ops at ``dt``.

    ``target`` (optional) provides the dtype-specific intrinsic spelling.
    ``None`` keeps the legacy f32 spellings for callers that haven't
    plumbed a target through yet.
    """
    name = op.name
    if name in ("add", "sum"):
        return f"{a} + {b}"
    if name in ("multiply", "prod"):
        return f"{a} * {b}"
    if name in ("maximum", "amax"):
        spelling = target.intrinsic("fmax", dt) if target is not None else "fmaxf"
        return f"{spelling}({a}, {b})"
    if name == "minimum":
        spelling = target.intrinsic("fmin", dt) if target is not None else "fminf"
        return f"{spelling}({a}, {b})"
    raise ValueError(f"TreeHalve: unsupported op {name!r}")


# ``StridedLoop`` is shared infrastructure — defined in ``ir/stmt.py``
# and re-exported here. Used at Tile IR for cooperative iteration and
# at Kernel IR for cooperative smem loads.


# ---------------------------------------------------------------------------
# Top-level: KernelOp
# ---------------------------------------------------------------------------


# Hard ceiling on the launch grid for ``KernelOp.validate``. The CUDA driver
# allows ~2^31 CTAs per dim; the cap here is sized to allow any realistic
# matmul launch (including K-split fan-in, which multiplies CTA count by
# the split factor — e.g. (M=32, K=18944, N=3584) with BM=BN=16 and
# auto-splitK=37 produces 2 × 224 × 37 ≈ 17 k CTAs of heavy per-CTA work),
# while still rejecting truly degenerate launches that would saturate
# the GPU command processor with light per-CTA work. The autotune-side
# guard against pathological tiny-CTA × huge-grid variants is the
# graduated penalty in ``TileOp.score``, not this cap.
_MAX_CTAS = 65536


@dataclass
class KernelOp(Op):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    Op subclass parallel to ``TileOp`` / ``LoopOp``: lives as a graph
    node, carries a body of Kernel IR stmts plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them at render time, same as ``TileOp``. Kernel signature is derived
    from the body: distinct ``Load.input`` names become kernel input
    params, distinct ``Write.output`` names become writeable output
    params, ordered by first appearance. ``Smem`` buffers are excluded.

    ``input_tensors`` / ``output_tensors`` map a global-buffer name to a
    :class:`Tensor` describing that buffer (its shape + dtype). Populated
    by the CUDA-lowering pass from the surrounding graph; missing entries
    fall back to a unit-shape :data:`F32` tensor so legacy tests that
    build KernelOps directly keep working.
    """

    body: Body = field(default_factory=Body)
    name: str = ""
    input_tensors: dict[str, Tensor] = field(default_factory=dict)
    output_tensors: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body.coerce(self.body)

    def input_tensor(self, name: str) -> Tensor:
        return self.input_tensors.get(name) or Tensor(name, (), F32)

    def output_tensor(self, name: str) -> Tensor:
        return self.output_tensors.get(name) or Tensor(name, (), F32)

    def input_dtype(self, name: str) -> DataType:
        return self.input_tensor(name).dtype

    def output_dtype(self, name: str) -> DataType:
        return self.output_tensor(name).dtype

    def __iter__(self) -> Iterator[Stmt]:
        return self.body.iter()

    def pretty_body(self) -> str:
        """Render as an indented structural listing via per-stmt ``pretty``."""
        sig_in = ", ".join(self.inputs) or "-"
        sig_out = ", ".join(self.outputs) or "-"
        head = f"kernel {self.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}"
        return "\n".join([head, *pretty_body(self.body, "    ")])

    def smem_bytes(self) -> int:
        """Total static + dynamic ``__shared__`` bytes declared in the
        body — sum of ``prod(Smem.extents) * sizeof(Smem.dtype)`` over
        every ``Smem`` decl, **at any nesting depth**. ``Smem`` stmts
        sit inside ``Tile`` / ``Loop`` bodies post-lowering, so the
        top-level-only iterator the engine used to use under-counted by
        the entire footprint and lets oversize kernels slip past
        :meth:`validate`. Use the deep iterator (``for s in self``) to
        match :meth:`smem_names` and the renderer's
        ``_compute_dynamic_smem_offsets``."""
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.backend.cuda.dtype import nbytes_of  # noqa: PLC0415

        total = 0
        for s in self:
            if isinstance(s, Smem):
                elements = prod(int(e) for e in s.extents) if s.extents else 1
                total += elements * nbytes_of(s.dtype)
        return total

    def validate(self, ctx) -> bool:
        """Drop kernels whose launch wouldn't fit the hardware. Runs
        after every tile-level rewrite has settled (the engine calls
        ``validate`` on each op rebind, but only at the ``KernelOp``
        stage are the THREAD axes guaranteed to reflect the final
        per-CTA launch geometry — earlier ``TileOp`` rewrites may still
        split or coalesce them). Three checks:

        - threads ≤ ``ctx.max_threads_per_cta`` (driver-side launch cap),
        - smem ≤ ``ctx.max_dynamic_smem`` (per-block ``cudaFuncSetAttribute``
          opt-in cap; smaller for older / consumer cards),
        - CTAs ≤ ``_MAX_CTAS`` (a hard cap on the launch grid — empirically
          variants with tens of thousands of light CTAs slot into the
          GPU command processor so slowly that the per-launch
          ``_KERNEL_TIMEOUT_MS`` watchdog stops being a useful escape
          hatch; better to drop them at the rule level before benching)."""
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.ir.stmt import Tile  # noqa: PLC0415

        for s in self.body:
            if isinstance(s, Tile):
                threads = prod(int(ba.axis.extent) for ba in s.axes if ba.bind == BIND_THREAD)
                if threads > ctx.max_threads_per_cta:
                    return False
                ctas = prod(int(ba.axis.extent) for ba in s.axes if ba.bind == BIND_BLOCK)
                if ctas > _MAX_CTAS:
                    return False
        if self.smem_bytes() > ctx.max_dynamic_smem:
            return False
        return True

    @property
    def loads(self) -> tuple[Load, ...]:
        return self.body.iter_of_type(Load)

    @property
    def smem_names(self) -> frozenset[str]:
        """Names of all ``__shared__`` buffers declared in the body — these
        are render-internal and are excluded from kernel-parameter inference."""
        return frozenset(s.name for s in self if isinstance(s, Smem))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order — the
        kernel's input parameters. Smem buffers are excluded.

        ``CpAsyncCopy`` stmts also count as global-buffer reads — their
        ``src`` field names a kernel parameter same as ``Load.input`` does
        for the synchronous path. ``TmaDescriptor`` reports its
        ``src_buf`` (the global buffer the descriptor addresses); the
        descriptor parameter itself is appended by the CUDA backend's
        argument pipeline since it's host-built, not a graph buffer."""
        smem = self.smem_names
        names: dict[str, None] = {}
        for s in self:
            if isinstance(s, Load) and s.input not in smem:
                names.setdefault(s.input, None)
            elif isinstance(s, CpAsyncCopy) and s.src not in smem:
                names.setdefault(s.src, None)
            elif isinstance(s, TmaDescriptor) and s.src_buf not in smem:
                names.setdefault(s.src_buf, None)
        return tuple(names)

    @property
    def writes(self) -> tuple[Write, ...]:
        return self.body.iter_of_type(Write)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order —
        the kernel's writeable output parameters. Smem buffers are excluded."""
        smem = self.smem_names
        return tuple(dict.fromkeys(s.output for s in self.writes if s.output not in smem))


# ---------------------------------------------------------------------------
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------


__all__ = [
    # Shared expressions (re-exported)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (reused)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Kernel-IR statements
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "WarpShuffle",
    "CpAsyncCopy",
    "CpAsyncCommit",
    "CpAsyncWait",
    "TmaDescriptor",
    "TmaLoad",
    "MbarrierInit",
    "MbarrierArriveExpectTx",
    "MbarrierWait",
    "StridedLoop",
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "Stmt",
    # Top-level
    "KernelOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff


# ---------------------------------------------------------------------------
# rewrite-dispatch handlers for Kernel-IR stmts
# ---------------------------------------------------------------------------
#
# ``Body.structural_key()`` runs ``normalize_body``, which dispatches
# :func:`deplodock.compiler.ir.stmt.passes.rewrite` over every stmt. The
# default dispatch raises ``NotImplementedError``, so we register a handler
# per Kernel-IR stmt here. Buffer names (``Smem.name``, mbarriers, TMA
# descriptors, etc.) are *not* SSA — the ``rename`` callback only canon-
# icalizes SSA tokens — so they pass through unchanged. ``Expr`` fields go
# through ``sigma.apply``; the leaf-only stmts (``Sync`` / ``CpAsyncCommit``
# / ``CpAsyncWait``) are stateless and return themselves.


from deplodock.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Smem, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: Sync, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncCommit, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncWait, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncCopy, rename, sigma, axis_fn):
    return CpAsyncCopy(
        smem=s.smem,
        smem_index=tuple(sigma.apply(e) for e in s.smem_index),
        src=s.src,
        src_index=tuple(sigma.apply(e) for e in s.src_index),
    )


@_rewrite.register
def _(s: TmaDescriptor, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: TmaLoad, rename, sigma, axis_fn):
    return TmaLoad(
        smem=s.smem,
        smem_index=tuple(sigma.apply(e) for e in s.smem_index),
        desc=s.desc,
        coords=tuple(sigma.apply(e) for e in s.coords),
        mbar=s.mbar,
        mbar_slot=sigma.apply(s.mbar_slot) if s.mbar_slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierInit, rename, sigma, axis_fn):
    return MbarrierInit(
        mbar=s.mbar,
        count=s.count,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierArriveExpectTx, rename, sigma, axis_fn):
    return MbarrierArriveExpectTx(
        mbar=s.mbar,
        bytes_=s.bytes_,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierWait, rename, sigma, axis_fn):
    return MbarrierWait(
        mbar=s.mbar,
        phase=sigma.apply(s.phase),
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: TreeHalve, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: WarpShuffle, rename, sigma, axis_fn):
    # ``name`` is the SSA output; ``value`` is the SSA input — both pass
    # through ``rename`` so the SSA canonicalizer can renumber them.
    return WarpShuffle(name=rename(s.name), value=rename(s.value), op=s.op, length=s.length)
