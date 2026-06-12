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

from dataclasses import dataclass, field
from functools import cached_property

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import Axis
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
    Cond,
    Load,
    Loop,
    Pack,
    RenderCtx,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Unpack,
    Write,
    _pad,
)
from deplodock.compiler.ir.stmt.ir import BodyOp
from deplodock.compiler.ir.tile.ir import GridTile, RegisterTile, SerialTile, StridedTile, ThreadTile

# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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

    def local_decls(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        ext = ", ".join(str(e) for e in self.extents) or "-"
        ali = f" align={self.align}" if self.align else ""
        return [f"{indent}Smem {self.dtype} {self.name}[{ext}]{ali}"]

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


@dataclass(frozen=True)
class Sync(Stmt):
    """Thread-group barrier.

    ``barrier_id == 0`` (default): ``__syncthreads();`` — CTA-wide.

    ``barrier_id > 0`` and ``count`` set: ``bar.sync <id>, <count>;`` —
    named-barrier synchronizing exactly ``count`` threads on barrier id
    ``<id>`` (one of 1..15). Used inside warp-specialized branches where
    only a subset of CTA threads execute the sync — `__syncthreads()`
    would be CUDA UB because the enclosing ``if (warp < P)`` condition
    diverges across warps. The count must equal the number of threads
    that actually arrive at this barrier in flight."""

    barrier_id: int = 0
    count: int | None = None

    def pretty(self, indent: str = "") -> list[str]:
        if self.barrier_id == 0:
            return [f"{indent}Sync"]
        return [f"{indent}Sync(bar={self.barrier_id}, count={self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        if self.barrier_id == 0:
            return [f"{pad}__syncthreads();"]
        if self.count is None:
            raise ValueError(f"Sync(barrier_id={self.barrier_id}) requires count")
        return [f'{pad}asm volatile("bar.sync {self.barrier_id}, {self.count};\\n" ::: "memory");']


@dataclass(frozen=True)
class CpAsyncCopy(Stmt):
    """Issue one ``cp.async.{ca,cg}.shared.global`` instruction.

    Replaces the per-thread ``Load(reg) + Write(smem)`` pair in cooperative
    loads on sm_80+. The hardware copies ``nbytes`` (one contiguous vector of
    ``nbytes / sizeof(dtype)`` elements) directly from global to shared without
    a register staging slot, freeing the thread register and removing the
    LDG → STS dependency.

    ``nbytes`` ∈ {4, 8, 16} (the cp.async copy sizes). 16-byte copies use the
    ``.cg`` (cache-global, bypass-L1) qualifier — the streaming form CUTLASS /
    cuBLAS use, requiring 16-byte-aligned smem + global addresses — and 4/8-byte
    copies use ``.ca``. ``smem_index`` / ``src_index`` address the *start* of
    the contiguous chunk; ``_stage_expand`` picks the widest legal ``nbytes`` so
    e.g. an fp16 tile stages 8 halves per ``cp.async``.

    Renders to inline PTX. The asm reads the smem address via
    ``cvta.to.shared.u32`` and the global pointer as a 64-bit value;
    indices flatten via ``render_index`` against the buffer's declared
    shape (same as ``Load`` / ``Write``)."""

    smem: str  # destination smem buffer name
    smem_index: tuple
    src: str  # source global buffer name
    src_index: tuple
    nbytes: int = 4  # bytes per cp.async (4/8/16); 16 ⇒ .cg, else .ca

    def external_reads(self) -> tuple[str, ...]:
        return (self.src,)

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        src_idx = ", ".join(e.pretty() for e in self.src_index)
        return [f"{indent}cp.async[{self.nbytes}B] {self.smem}[{smem_idx}] <- {self.src}[{src_idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        src_flat = render_index(self.src, self.src_index, ctx)
        pad = _pad(ctx.indent)
        qual = "cg" if self.nbytes == 16 else "ca"  # .cg is 16-byte-only (bypass L1)
        asm = (
            f'asm volatile("cp.async.{qual}.shared.global [%0], [%1], {self.nbytes};\\n" '
            f':: "r"(_smem_addr), "l"(&{self.src}[{src_flat}]) : "memory");'
        )
        return [
            f"{pad}{{",
            f"{pad}    unsigned int _smem_addr = __cvta_generic_to_shared(&{self.smem}[{smem_flat}]);",
            f"{pad}    {asm}",
            f"{pad}}}",
        ]


@dataclass(frozen=True)
class CpAsyncCommit(Stmt):
    """``cp.async.commit_group;`` — finalize the preceding cp.async copies
    issued by this thread into a commit group. Pairs with
    ``CpAsyncWait`` to wait for that group to drain."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.commit_group"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.commit_group;\\n" ::: "memory");']


@dataclass(frozen=True)
class CpAsyncWait(Stmt):
    """``cp.async.wait_group N;`` — block this thread until ≤ N cp.async
    groups remain in flight. ``group=0`` waits for everything (synchronous
    style); larger values stagger waits for software pipelining."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.wait_group({self.group})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.wait_group {self.group};\\n" ::: "memory");']


@dataclass(frozen=True)
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

    def external_reads(self) -> tuple[str, ...]:
        return (self.src_buf,)

    def pretty(self, indent: str = "") -> list[str]:
        shape = ", ".join(str(e) for e in self.src_shape)
        box = ", ".join(str(e) for e in self.box_extents)
        return [f"{indent}TmaDescriptor {self.name} <- {self.src_buf}[{shape}] box=[{box}] swizzle={self.swizzle}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        # Descriptor is a kernel parameter built host-side; no in-kernel render.
        return []


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class MbarrierArrive(Stmt):
    """``mbarrier.arrive.shared.b64`` — simple arrive (no transaction-byte
    count). Used by warp-specialized consumer warps to signal "slot empty"
    so the producer can refill it on the next ring iteration. The producer
    side uses ``MbarrierArriveExpectTx`` instead (which declares the
    incoming TMA byte count)."""

    mbar: str
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierArrive({self.mbar}{s})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_arrive({addr});"]


@dataclass(frozen=True)
class SetMaxNReg(Stmt):
    """``setmaxnreg.{inc,dec}.sync.aligned.u32 N;`` — Hopper+ register-budget
    redistribution. ``direction="dec"`` shrinks the calling warp's max
    register count to ``count`` (returning registers to the pool);
    ``direction="inc"`` claims up to ``count`` registers. Used by warp-
    specialized kernels so producer warps drop registers and consumer
    warps claim them, decoupling occupancy from the consumer's pressure.

    Requires sm_90+. On older targets NVCC rejects the instruction at
    compile time — the materializer only emits this Stmt on the WS=1 path
    where the kernel is already TMA-gated to sm_90+ anyway."""

    count: int
    direction: str  # "inc" or "dec"

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}SetMaxNReg.{self.direction}({self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        return [f'{pad}asm volatile("setmaxnreg.{self.direction}.sync.aligned.u32 {self.count};\\n");']


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    # Named-barrier support: when ``barrier_id > 0`` the per-iter sync
    # renders as ``bar.sync <id>, <count>;`` instead of ``__syncthreads()``.
    # Required when this TreeHalve sits inside a warp-specialized consumer
    # branch — __syncthreads on a warp-divergent condition is UB.
    barrier_id: int = 0
    barrier_count: int | None = None

    def pretty(self, indent: str = "") -> list[str]:
        bar = "" if self.barrier_id == 0 else f", bar={self.barrier_id}/{self.barrier_count}"
        return [f"{indent}TreeHalve({self.dtype.name} {self.buf}, op={self.op.name}, length={self.length}, tid={self.tid_var}{bar})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Power-of-two tree reduction over ``buf[0..length)`` into ``buf[0]``."""
        pad = _pad(ctx.indent)
        inner_pad = _pad(ctx.indent + 1)
        halve_pad = _pad(ctx.indent + 2)
        op_expr = _binary_combine_expr(
            self.op, f"{self.buf}[{self.tid_var}]", f"{self.buf}[{self.tid_var} + s]", ctx.target, self.dtype.name
        )
        half = int(self.length) // 2
        if self.barrier_id == 0:
            sync_line = f"{inner_pad}__syncthreads();"
        else:
            if self.barrier_count is None:
                raise ValueError(f"TreeHalve(barrier_id={self.barrier_id}) requires barrier_count")
            sync_line = f'{inner_pad}asm volatile("bar.sync {self.barrier_id}, {self.barrier_count};\\n" ::: "memory");'
        return [
            f"{pad}for (int s = {half}; s > 0; s >>= 1) {{",
            f"{inner_pad}if ({self.tid_var} < s) {{",
            f"{halve_pad}{self.buf}[{self.tid_var}] = {op_expr};",
            f"{inner_pad}}}",
            sync_line,
            f"{pad}}}",
        ]


@dataclass(frozen=True)
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
        return [f"{indent}WarpShuffle({self.dtype.name} {self.name} <- {self.value}, op={self.op.name}, length={self.length})"]

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


# ---------------------------------------------------------------------------
# Warp-level MMA: ``mma.sync.aligned`` + ``ldmatrix`` (the ``s16816`` path
# cuBLAS / CUTLASS use) — the sole tensor-core Stmt family. Operands are
# *explicit per-thread register arrays* with a PTX-fixed lane→element layout,
# referenced positionally inside inline PTX (``RegFragment`` / ``LdmatrixLoad``
# / ``MmaSyncPtx`` / ``RegStore``), rendered via the ``_MMA_SYNC_PRELUDE``
# helper wrappers (pure PTX — NVRTC-clean, no ``<mma.h>``). Emitted by
# ``kernel/005_lower_atom_tile`` from the ``Mma`` op's ``Atom`` spec.
# (The opaque ``nvcuda::wmma`` node family was removed — the swizzled mma.sync
# slab beat it.)
# ---------------------------------------------------------------------------


def _mma_sync_nregs(role: str, shape: tuple[int, int, int]) -> int:
    """Per-lane register count for an mma.sync operand of cell ``shape``.

    ``a`` (M×K f16) and ``b`` (K×N f16) pack two halfs per 32-bit register;
    ``c`` (M×N f32) holds one float per register. For ``m16n8k16``:
    ``a[4]`` (256 halfs / 32 lanes / 2), ``b[2]`` (128 / 32 / 2), ``c[4]``
    (128 f32 / 32)."""
    m, n, k = shape
    if role == "a":
        return (m * k) // 64
    if role == "b":
        return (k * n) // 64
    if role == "c":
        return (m * n) // 32
    raise ValueError(f"mma.sync: unsupported role {role!r}; expected 'a', 'b', or 'c'")


@dataclass(frozen=True)
class RegFragment(Stmt):
    """Declare an mma.sync per-thread register array.

    mma.sync multiplicands are explicit per-lane register arrays —
    ``unsigned a[4]`` / ``unsigned b[2]`` (f16, two halfs per 32-bit
    reg) — and the accumulator is ``float c[4]`` (f32). ``shape`` is the
    cell ``(M, N, K)``; the count
    derives from ``shape`` + ``role`` via :func:`_mma_sync_nregs`. The
    ``c`` array is zero-initialised at declaration, so the mma.sync path
    needs no separate ``MmaFill``."""

    name: str
    role: str  # "a" / "b" / "c"
    shape: tuple[int, int, int]
    dtype: DataType

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def local_decls(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        return [f"{indent}RegFragment {self.role}:{self.dtype.name} {self.name}[{_mma_sync_nregs(self.role, self.shape)}] ({m}x{n}x{k})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        n_regs = _mma_sync_nregs(self.role, self.shape)
        ctx.ssa_dtypes[self.name] = self.dtype.name
        if self.role == "c":
            zeros = ", ".join(["0.0f"] * n_regs)
            return [f"{_pad(ctx.indent)}float {self.name}[{n_regs}] = {{{zeros}}};"]
        return [f"{_pad(ctx.indent)}unsigned {self.name}[{n_regs}];"]


# Per-lane fp16 element XOR matching the TMA hardware smem swizzle (the
# ldmatrix consumer side). The producer (cuTensorMapEncodeTiled with
# SWIZZLE_{128,64,32}B) permutes 16-byte (8-fp16) chunks within each swizzle
# row by the row-index-mod-2^B; the ldmatrix consumer must apply the same
# permutation to its per-lane element offset. All three modes are CUTLASS
# ``Swizzle<B,4,3>`` (M=4 → 16-byte base, S=3 → 8-row tile), differing ONLY in
# B (3/2/1 XORed bits). In bytes the swizzle is
# ``off ^= ((off >> (M+S)) & (2^B-1)) << M`` = ``((off >> 7) & mask) << 4``;
# halving to fp16 elements turns ``>> 7`` into ``>> 6`` and ``<< 4`` into
# ``<< 3`` — so the element shift is 6 for EVERY mode (a per-mode shift of
# 6/5/4 silently corrupts B64/B32 while leaving B128 correct, which is easy to
# miss since B128 dominates). Only the row mask changes: 0x7 / 0x3 / 0x1. The
# 8-row × atom period divides the slab + slot strides, so the XOR is correct
# measured from the buffer base. Maps mode → (element shift, row mask); the
# chunk delta is always ``<< 3``.
_LDMATRIX_SWIZZLE_XOR: dict[str, tuple[int, int]] = {
    "B128": (6, 0x7),
    "B64": (6, 0x3),
    "B32": (6, 0x1),
}


@dataclass(frozen=True)
class LdmatrixLoad(Stmt):
    """``ldmatrix.sync.aligned.m8n8.x{2,4}[.trans].b16`` — load one operand
    fragment from smem into a per-thread register array.

    ``frag`` is the destination :class:`RegFragment`; ``src_buffer`` /
    ``src_index`` are the cell's smem tile base (the same ``(buffer,
    base-offset)`` ``_mma_src_index`` computes); ``ldm``
    is the smem row stride in elements. Each lane derives its own row
    address from its warp lane id (``threadIdx.x & 31``): the 16×K ``a``
    tile uses ``x4`` (``row = lane%16``, K-col block ``(lane/16)*8``); the
    K×8 ``b`` tile uses ``x2.trans`` (``row = lane%16``) so a row-major
    smem slab feeds the mma's col-major B operand.

    ``staged`` (default ``True``) selects the transport: ``ldmatrix`` is **smem
    only**, so when the operand was NOT staged into shared memory
    (``staged=False``, ``src_buffer`` is the gmem operand) the render emits the
    ``dpl_mma_load_{a,b}_gmem`` helper instead — a gmem-direct fragment load that
    replicates the same lane→element map without ldmatrix. Slower (no smem reuse)
    but correct; ``005_lower_atom_tile`` picks per operand based on whether an
    enclosing ``StageBundle`` staged it.

    ``gmem_guard`` (gmem-direct only) carries a masked-tile boundary as
    ``(base Expr, bound Expr)`` on the operand's lane-varying output axis
    (A's M rows, B's N cols): the render switches to the ``_mclamp`` /
    ``_nclamp`` helper, which clamps the lane coordinate to the ``bound -
    base`` in-range elements — the gmem-direct analogue of the staged
    slab-fill clamp, so an unstaged masked cell still lowers instead of
    reading past the runtime-sized buffer. Clamped lanes read duplicates;
    their stores are masked by the RegStore guards."""

    frag: str
    src_buffer: str
    src_index: tuple
    role: str  # "a" → x4; "b" → x2.trans
    ldm: int = 0
    swizzle: str = "NONE"  # TMA smem swizzle mode the slab was written with
    staged: bool = True  # False → gmem-direct fragment load (operand not in smem)
    gmem_guard: tuple[Expr, Expr] | None = None  # masked-axis (base, bound); gmem-direct only

    def deps(self) -> tuple[str, ...]:
        return (self.frag,)

    def defines(self) -> tuple[str, ...]:
        return (self.frag,)

    def external_reads(self) -> tuple[str, ...]:
        return (self.src_buffer,)

    def exprs(self) -> tuple[Expr, ...]:
        guard = () if self.gmem_guard is None else self.gmem_guard
        return (*self.src_index, *guard)

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.src_index)
        variant = ("x4" if self.role == "a" else "x2.trans") if self.staged else "gmem-direct"
        guard = "" if self.gmem_guard is None else f" guard<{self.gmem_guard[1].pretty()}"
        return [f"{indent}LdmatrixLoad {self.frag} <- {self.src_buffer}[{idx}] ({variant}{guard}, ldm={self.ldm or 'auto'})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index  # noqa: PLC0415

        flat = render_index(self.src_buffer, self.src_index, ctx)
        ldm = self.ldm if self.ldm else _resolve_ldm(self.src_buffer, ctx)
        if not self.staged:
            # Operand not staged in smem — ldmatrix can't reach gmem, so read the
            # fragment straight from gmem (each lane adds its own (row,col) inside
            # the helper). No swizzle: that's a TMA-smem-layout concern only.
            if self.gmem_guard is not None:
                # Masked axis: clamp the lane coordinate to the in-range
                # elements left from the tile base (>= 1 — the boundary Cond
                # admitted the tile).
                base, bound = self.gmem_guard[0].render(ctx), self.gmem_guard[1].render(ctx)
                helper = "dpl_mma_load_a_gmem_mclamp" if self.role == "a" else "dpl_mma_load_b_gmem_nclamp"
                return [f"{_pad(ctx.indent)}{helper}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, ({bound}) - ({base}));"]
            helper = "dpl_mma_load_a_gmem" if self.role == "a" else "dpl_mma_load_b_gmem"
            return [f"{_pad(ctx.indent)}{helper}({self.frag}, &{self.src_buffer}[{flat}], {ldm});"]
        lane = "(threadIdx.x & 31)"
        if self.role == "a":
            # 16×16 A: x4 — lane addresses M-row (lane%16), K-col block (lane/16)*8.
            elem = f"{flat} + ({lane} % 16) * {ldm} + ({lane} / 16) * 8"
            addr = self._swizzled_addr(elem)
            return [f"{_pad(ctx.indent)}dpl_ldmatrix_x4({self.frag}, {addr});"]
        # 16×8 B: x2.trans — lane addresses K-row (lane%16); .trans yields col-major.
        elem = f"{flat} + ({lane} % 16) * {ldm}"
        addr = self._swizzled_addr(elem)
        return [f"{_pad(ctx.indent)}dpl_ldmatrix_x2_trans({self.frag}, {addr});"]

    def _swizzled_addr(self, elem: str) -> str:
        params = _LDMATRIX_SWIZZLE_XOR.get(self.swizzle)
        if params is None:
            return f"&{self.src_buffer}[{elem}]"
        shift, mask = params
        # ``e ^ (((e >> shift) & mask) << 3)`` reproduces the TMA chunk swizzle.
        return f"&{self.src_buffer}[({elem}) ^ (((({elem}) >> {shift}) & {mask}) << 3)]"


@dataclass(frozen=True)
class MmaSyncPtx(Stmt):
    """``mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{ab}.{ab}.f32`` — one
    tensor-core MMA via inline PTX (the ``s16816`` instruction).

    ``c_frag`` is the f32 accumulator array (both input and output —
    ``d = a·b + c``); ``a_frag`` / ``b_frag`` are the 16-bit multiplicand
    arrays filled by :class:`LdmatrixLoad`. ``shape`` spells the M/N/K.
    ``ab_dtype`` (``"f16"`` / ``"bf16"``) tags the operand element type —
    f16 and bf16 share the same fragment layout + ldmatrix path and differ
    only in the PTX instruction's dtype field, so the render just picks the
    matching ``dpl_mma_…`` wrapper. The accumulate is always f32."""

    c_frag: str
    a_frag: str
    b_frag: str
    shape: tuple[int, int, int]
    ab_dtype: str = "f16"

    def deps(self) -> tuple[str, ...]:
        return (self.c_frag, self.a_frag, self.b_frag)

    def defines(self) -> tuple[str, ...]:
        # Accumulates into c_frag in place — a definition, like MmaSync.
        return (self.c_frag,)

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        return [f"{indent}MmaSyncPtx {self.c_frag} += {self.a_frag} @ {self.b_frag} (m{m}n{n}k{k} {self.ab_dtype})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        m, n, k = self.shape
        # ``c`` is passed for both the ``d`` (out) and ``c`` (in) operands.
        return [f"{_pad(ctx.indent)}dpl_mma_m{m}n{n}k{k}_{self.ab_dtype}({self.c_frag}, {self.a_frag}, {self.b_frag}, {self.c_frag});"]


@dataclass(frozen=True)
class EpilogueLoad:
    """One leaf operand of a fused pointwise epilogue (see :class:`RegEpilogue`).

    Not a body :class:`Stmt` — a payload riding the :class:`RegStore`.
    ``index`` is the cell-base coordinate tuple (struct-shared with the
    epilogue's Write); per fragment element the render adds the element's own
    row / col motion on every dim whose ``role`` is ``"m"`` / ``"n"`` (at
    *this* buffer's dim stride — so a transposed or broadcast operand reads
    correctly), and nothing on ``"fixed"`` dims (literals, batch / grid vars —
    uniform across the cell)."""

    name: str
    buffer: str
    index: tuple
    roles: tuple[str, ...]


@dataclass(frozen=True)
class RegEpilogue:
    """A pure pointwise SSA chain folded into the fragment store.

    Captured by ``kernel/005_lower_atom_tile`` from the backward slice between
    the accumulator and the Write (the scalar Load / Assign stmts are stripped
    — the accumulator has no scalar SSA name on the fragment path). The render
    evaluates the chain per fragment element in f32 with ``acc`` substituted
    by the element and each leaf loaded at the element's own (row, col); the
    chain ops reuse the scalar renderer's ``op_to_expr`` translation, so any
    elementwise op with a CUDA spelling works. ``ops`` are ``(name, op_name,
    args)`` in topological (body) order; ``result`` is the SSA name the Write
    stored."""

    acc: str
    loads: tuple[EpilogueLoad, ...]
    ops: tuple[tuple[str, str, tuple[str, ...]], ...]
    result: str


@dataclass(frozen=True)
class RegStore(Stmt):
    """Store an mma.sync f32 accumulator array to the output buffer with a
    per-lane epilogue downconvert.

    mma.sync has no ``store_matrix_sync`` — each lane owns 4 elements of
    the M×N=16×8 C tile (rows ``g`` / ``g+8`` with ``g = lane/4``, cols
    ``(lane%4)*2 + {0,1}``) and writes them directly. ``frag`` is the f32
    ``float c[4]``; when the destination buffer is narrower (``__half*``)
    each value is converted via ``__float2half`` (mirrors
    :class:`MmaStore`'s downconvert). ``ldm`` is the output row stride
    (N) — ``0`` auto-resolves from the buffer's inner extent.

    ``epilogue`` optionally carries a fused pointwise chain
    (:class:`RegEpilogue` — residual adds, bias / scale broadcasts,
    activations): evaluated per fragment element in f32 right before the
    downconvert (the CUTLASS epilogue-visitor pattern).

    ``m_guard`` / ``n_guard`` carry a masked-tile boundary as ``(base Expr,
    bound Expr)`` — the tile's row / col coordinate of fragment element (0,0)
    and the axis's real extent (a ``Literal`` for a static overhang axis, the
    symbolic ``Var`` for a runtime-sized one). The enclosing boundary ``Cond``
    only gates on the tile base (the atom-lane offsets ``_g`` / ``_t`` are
    render-local, invisible to σ), so an atom tile *straddling* the bound
    passes the Cond while its trailing rows / cols are out of range. A guard
    predicates each fragment element's store (and its epilogue gmem reads) at
    its own coordinate: ``base + _g (+8) < bound`` per row, ``base + _t*2
    (+1) < bound`` per col. ``None`` (the default) renders the unguarded
    fast path unchanged. ``m_guard`` alone keeps the vectorized row-pair
    stores (a pair shares a row); an ``n_guard`` forces per-element scalar
    stores (the pair straddles the column bound)."""

    dst_buffer: str
    dst_index: tuple
    frag: str
    shape: tuple[int, int, int]
    ldm: int = 0
    epilogue: RegEpilogue | None = None
    m_guard: tuple[Expr, Expr] | None = None
    n_guard: tuple[Expr, Expr] | None = None

    def deps(self) -> tuple[str, ...]:
        return (self.frag,)

    def external_reads(self) -> tuple[str, ...]:
        # The fused epilogue's leaf loads are gmem reads this stmt performs
        # directly (their original Load stmts were stripped by
        # 005_lower_atom_tile), so they must be declared here for the kernel
        # signature / render shapes to include the buffers.
        if self.epilogue is None:
            return ()
        return tuple(ld.buffer for ld in self.epilogue.loads)

    def external_writes(self) -> tuple[str, ...]:
        return (self.dst_buffer,)

    def exprs(self) -> tuple[Expr, ...]:
        epi = () if self.epilogue is None else tuple(e for ld in self.epilogue.loads for e in ld.index)
        guards = tuple(e for g in (self.m_guard, self.n_guard) if g is not None for e in g)
        return (*self.dst_index, *epi, *guards)

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.dst_index)
        epi = ""
        if self.epilogue is not None:
            chain = ", ".join(op for _, op, _ in self.epilogue.ops)
            bufs = ", ".join(ld.buffer for ld in self.epilogue.loads)
            epi = f" epilogue[{chain}]({bufs or 'no loads'})"
        guards = ""
        if self.m_guard is not None:
            guards += f" m<{self.m_guard[1].pretty()}"
        if self.n_guard is not None:
            guards += f" n<{self.n_guard[1].pretty()}"
        return [f"{indent}RegStore {self.dst_buffer}[{idx}] <- {self.frag}{epi}{guards} (ldm={self.ldm or 'auto'})"]

    def _element_values(self, ctx: RenderCtx) -> tuple[list[list[str]], list[str]]:
        """``(per_element_preamble, values)``: the four per-lane store values,
        plus, per element, the leaf-load / chain-op declaration lines that
        element needs (so a guarded render can scope element ``i``'s gmem
        reads inside element ``i``'s boundary check). Without
        an epilogue the values are the bare ``frag[i]`` and the preambles are
        empty. With one, each element ``i`` (row ``_g``/``_g+8``, col
        ``2_t+{0,1}``) declares its leaf loads (converted to f32; offsets per
        the dim roles at each buffer's own stride) and the chain ops (via
        ``op_to_expr`` — the same translation the scalar ``Assign`` render
        uses), all scoped inside the store's ``{ }`` block. Leaf loads are
        scalar; lanes ``_t = 0..3`` cover 8 contiguous columns, so the warp's
        accesses coalesce regardless."""
        if self.epilogue is None:
            return [[], [], [], []], [f"{self.frag}[{i}]" for i in range(4)]
        from deplodock.compiler.ir.expr import Var  # noqa: PLC0415
        from deplodock.compiler.ir.stmt import render_index  # noqa: PLC0415
        from deplodock.compiler.ir.stmt.base import op_to_expr  # noqa: PLC0415

        epi = self.epilogue
        conv = {"f16": "__half2float({})", "bf16": "__bfloat162float({})"}
        per_elem: list[list[str]] = []
        vals: list[str] = []
        for i in range(4):
            row = "_g" if i < 2 else "(_g + 8)"
            col = f"(_t * 2 + {i & 1})"
            lines: list[str] = []
            env = {epi.acc: f"{self.frag}[{i}]"}
            for ld in epi.loads:
                temp = f"{ld.name}_e{i}"
                if ld.buffer in ctx.literal_constants:
                    lines.append(f"const float {temp} = {float(ctx.literal_constants[ld.buffer])!r}f;")
                    env[ld.name] = temp
                    continue
                flat = render_index(ld.buffer, ld.index, ctx)
                parts = [flat]
                for d, role in enumerate(ld.roles):
                    if role == "fixed":
                        continue
                    stride = _dim_stride(ld.buffer, d, ctx)
                    motion = row if role == "m" else col
                    parts.append(motion if stride == 1 else f"{motion} * {stride}")
                addr = " + ".join(parts)
                dt = ctx.buffer_dtypes.get(ld.buffer, "f32")
                lines.append(f"const float {temp} = {conv.get(dt, '{}').format(f'{ld.buffer}[{addr}]')};")
                env[ld.name] = temp
            for name, op_name, args in epi.ops:
                expr = op_to_expr(op_name, [Var(env[a]) for a in args])
                lines.append(f"const float {name}_e{i} = {expr.render(ctx)};")
                env[name] = f"{name}_e{i}"
            per_elem.append(lines)
            vals.append(env[epi.result])
        return per_elem, vals

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index  # noqa: PLC0415

        flat = render_index(self.dst_buffer, self.dst_index, ctx)
        ldm = self.ldm if self.ldm else _resolve_ldm(self.dst_buffer, ctx)
        dst_dt = ctx.buffer_dtypes.get(self.dst_buffer, "f32")
        pad = _pad(ctx.indent)
        lane = "(threadIdx.x & 31)"
        pre, vals = self._element_values(ctx)
        # C is 16×8: lane owns (row g, cols 2t,2t+1) and (row g+8, cols 2t,2t+1)
        # with g = lane/4, t = lane%4. The two cols per row are CONTIGUOUS, so
        # each row's pair is one vectorized 4-byte store (``__half2`` / ``float2``)
        # rather than two scalar stores — halves the epilogue store count and
        # the address arithmetic. ldm strides over N (the output row width). The
        # base ``flat`` is tile-aligned and ``2t`` is even, so the pair is
        # 4-/8-byte aligned. The ``{ }`` block scopes _g/_t (and the per-element
        # epilogue temps) per RegStore.
        if self.m_guard is not None or self.n_guard is not None:
            return self._render_guarded(ctx, flat=flat, ldm=ldm, dst_dt=dst_dt, pre=pre, vals=vals)
        lines = [f"{pad}{{ const int _g = {lane} >> 2; const int _t = {lane} & 3;"]
        lines += [f"{pad}  {ln}" for group in pre for ln in group]
        vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
        if vec2 is not None:
            packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
            lo = f"{flat} + _g * {ldm} + _t * 2"
            hi = f"{flat} + (_g + 8) * {ldm} + _t * 2"
            lines += [
                f"{pad}  *reinterpret_cast<{vec2}*>(&{self.dst_buffer}[{lo}]) = {packer}({vals[0]}, {vals[1]});",
                f"{pad}  *reinterpret_cast<{vec2}*>(&{self.dst_buffer}[{hi}]) = {packer}({vals[2]}, {vals[3]}); }}",
            ]
            return lines
        # Fallback: per-element scalar stores (dtypes without a 2-vector packer).
        lines += [
            f"{pad}  {self.dst_buffer}[{flat} + _g * {ldm} + _t * 2 + 0] = {vals[0]};",
            f"{pad}  {self.dst_buffer}[{flat} + _g * {ldm} + _t * 2 + 1] = {vals[1]};",
            f"{pad}  {self.dst_buffer}[{flat} + (_g + 8) * {ldm} + _t * 2 + 0] = {vals[2]};",
            f"{pad}  {self.dst_buffer}[{flat} + (_g + 8) * {ldm} + _t * 2 + 1] = {vals[3]}; }}",
        ]
        return lines

    def _render_guarded(self, ctx: RenderCtx, *, flat: str, ldm, dst_dt: str, pre: list[list[str]], vals: list[str]) -> list[str]:
        """Masked-tile store: each fragment element's store (and its epilogue
        gmem reads, which index the same possibly-out-of-range coordinates)
        runs under that element's own boundary check. With only an ``m_guard``
        the two columns of a row share the row predicate, so the vectorized
        row-pair store survives; an ``n_guard`` splits the pair (its columns
        straddle the bound) into per-element scalar stores."""
        pad = _pad(ctx.indent)
        lane = "(threadIdx.x & 31)"
        m_pred: list[str | None] = [None] * 4
        n_pred: list[str | None] = [None] * 4
        if self.m_guard is not None:
            base, bound = self.m_guard[0].render(ctx), self.m_guard[1].render(ctx)
            m_pred = [f"({base}) + _g < ({bound})"] * 2 + [f"({base}) + _g + 8 < ({bound})"] * 2
        if self.n_guard is not None:
            base, bound = self.n_guard[0].render(ctx), self.n_guard[1].render(ctx)
            n_pred = [
                f"({base}) + _t * 2 < ({bound})",
                f"({base}) + _t * 2 + 1 < ({bound})",
                f"({base}) + _t * 2 < ({bound})",
                f"({base}) + _t * 2 + 1 < ({bound})",
            ]
        lines = [f"{pad}{{ const int _g = {lane} >> 2; const int _t = {lane} & 3;"]
        vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
        if self.n_guard is None and vec2 is not None:
            # Row-guarded vectorized pairs: elements {0,1} share row _g,
            # {2,3} share row _g+8; each pair's preamble + store sit inside
            # the pair's row check.
            packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
            for pair, addr_row in ((0, "_g"), (2, "(_g + 8)")):
                addr = f"{flat} + {addr_row} * {ldm} + _t * 2"
                lines.append(f"{pad}  if ({m_pred[pair]}) {{")
                lines += [f"{pad}    {ln}" for ln in (*pre[pair], *pre[pair + 1])]
                lines.append(f"{pad}    *reinterpret_cast<{vec2}*>(&{self.dst_buffer}[{addr}]) = {packer}({vals[pair]}, {vals[pair + 1]});")
                lines.append(f"{pad}  }}")
            lines.append(f"{pad}}}")
            return lines
        # Per-element scalar stores under the conjunction of the live guards.
        for i in range(4):
            row = "_g" if i < 2 else "(_g + 8)"
            preds = " && ".join(p for p in (m_pred[i], n_pred[i]) if p is not None)
            addr = f"{flat} + {row} * {ldm} + _t * 2 + {i & 1}"
            lines.append(f"{pad}  if ({preds}) {{")
            lines += [f"{pad}    {ln}" for ln in pre[i]]
            lines.append(f"{pad}    {self.dst_buffer}[{addr}] = {vals[i]};")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines


def _dim_stride(buffer: str, dim: int, ctx: RenderCtx) -> int:
    """Row-major element stride of ``buffer``'s dim ``dim`` — the product of
    the trailing extents. Used by the fragment-epilogue render to apply a
    fragment element's row / col motion on an operand dim at that operand's
    own layout (a transposed residual's "n" dim strides by its row width,
    not by 1)."""
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"RegStore epilogue: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    stride = 1
    for ext in shape[dim + 1 :]:
        stride *= ext.as_static() if hasattr(ext, "as_static") else int(ext)
    return stride


def _resolve_ldm(buffer: str, ctx: RenderCtx) -> int:
    """Look up the row-major leading-dimension stride (= inner extent)
    for ``buffer`` from the kernel render context. Used by
    :class:`MmaLoad` / :class:`MmaStore` when ``ldm == 0`` (auto).
    Accepts both raw int extents and :class:`Dim` extents (Tensor.shape)
    — for symbolic dims, takes ``.as_static()`` (M9 generalizes to
    runtime-resolved ldm via a kernel-arg int)."""
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"MmaLoad/MmaStore: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    last = shape[-1]
    if hasattr(last, "as_static"):
        return last.as_static()
    return int(last)


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
# enumeration gate (``_enumeration._matmul_thread_gate``) + the learned
# prior, not this cap.
_MAX_CTAS = 65536


def pack_smem(smems) -> tuple[dict[str, int], int]:  # noqa: ANN001 — smems: Iterable[Smem]
    """Pack ``Smem`` buffers into one contiguous pool, padding each to its
    alignment. Walks the buffers in order, aligns the running cursor to
    ``max(nbytes_of(dtype), s.align)`` before placing each buffer, and returns
    ``({name: byte_offset}, total_padded_bytes)``.

    Single source of truth for the per-CTA smem footprint: ``KernelOp.smem_bytes``
    and ``render._compute_dynamic_smem_offsets`` both call this so the
    static-vs-dynamic gate and the launch-time pool size agree (an unpadded sum
    would under-report when a buffer's alignment exceeds its natural stride —
    e.g. a 512/1024 B swizzle-atom-aligned operand)."""
    from math import prod  # noqa: PLC0415

    from deplodock.compiler.backend.cuda.dtype import nbytes_of  # noqa: PLC0415

    offsets: dict[str, int] = {}
    cursor = 0
    for s in smems:
        elements = prod(int(e) for e in s.extents) if s.extents else 1
        align = max(nbytes_of(s.dtype), int(s.align) if s.align else 0)
        if align:
            cursor = (cursor + align - 1) // align * align
        offsets[s.name] = cursor
        cursor += elements * nbytes_of(s.dtype)
    return offsets, cursor


@dataclass
class KernelOp(BodyOp):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    :class:`BodyOp` subclass parallel to ``TileOp`` / ``LoopOp``: lives as
    a graph node, carries a body of Kernel IR stmts plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them at render time, same as ``TileOp``. Kernel signature is derived
    from the body: distinct ``Load.input`` names become kernel input
    params, distinct ``Write.output`` names become writeable output
    params, ordered by first appearance. ``Smem`` buffers are excluded.
    ``CpAsyncCopy.src`` and ``TmaDescriptor.src_buf`` also name kernel
    input parameters (the descriptor parameter itself is host-built and
    appended by the CUDA backend's argument pipeline, not a graph
    buffer)."""

    @cached_property
    def smem_buffers(self) -> dict[str, Smem]:
        """Every ``Smem`` decl in the body, keyed by name. Cached — KernelOp
        is treated as immutable post-construction (passes return new
        instances), so the deep walk runs at most once per op. The deep
        walk matters because ``Smem`` stmts sit inside ``Tile`` / ``Loop``
        bodies post-lowering; a top-level-only iterator would miss them
        and let oversize kernels slip past :meth:`validate`."""
        return {s.name: s for s in self if isinstance(s, Smem)}

    def smem_bytes(self) -> int:
        """Total static + dynamic ``__shared__`` bytes declared in the body —
        the padding-aware pool footprint from :func:`pack_smem` (each buffer
        aligned to ``max(sizeof(dtype), Smem.align)``), so this matches the
        renderer's pool packer and the launch-time pool size."""
        return pack_smem(self.smem_buffers.values())[1]

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

        from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile  # noqa: PLC0415

        for s in self.body:
            if isinstance(s, GridTile):
                ctas = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in s.axes)
                if ctas > _MAX_CTAS:
                    return False
                # ThreadTile lives inside the GridTile's body.
                for child in s.body:
                    if isinstance(child, ThreadTile):
                        threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in child.axes)
                        if threads > ctx.max_threads_per_cta:
                            return False
            elif isinstance(s, ThreadTile):
                threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in s.axes)
                if threads > ctx.max_threads_per_cta:
                    return False
        if self.smem_bytes() > ctx.max_dynamic_smem:
            return False
        return True

    @property
    def smem_names(self) -> frozenset[str]:
        """Names of all ``__shared__`` buffers declared in the body — these
        are render-internal and are excluded from kernel-parameter inference."""
        return frozenset(self.smem_buffers)


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
    "Pack",
    "Unpack",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Kernel-IR statements — typed tile flavor hierarchy (kernel-IR
    # materialization preserves the wrappers Tile IR emits)
    "GridTile",
    "ThreadTile",
    "RegisterTile",
    "SerialTile",
    "StridedTile",
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
    "MbarrierArrive",
    "MbarrierWait",
    "SetMaxNReg",
    "StridedLoop",
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
        nbytes=s.nbytes,
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
def _(s: MbarrierArrive, rename, sigma, axis_fn):
    return MbarrierArrive(
        mbar=s.mbar,
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
def _(s: SetMaxNReg, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: TreeHalve, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: WarpShuffle, rename, sigma, axis_fn):
    # ``name`` is the SSA output; ``value`` is the SSA input — both pass
    # through ``rename`` so the SSA canonicalizer can renumber them.
    return WarpShuffle(name=rename(s.name), value=rename(s.value), op=s.op, length=s.length)


# --- MMA fragment rewrites -------------------------------------------------
#
# Per-cell replication in ``010_split_register_axes`` calls ``s.rewrite(...)``
# on every body Stmt; the dedup pass and ``100_materialize_tile`` do too.
# Each Mma* Stmt routes its SSA fragment names through ``rename`` and its
# Expr-typed offset fields through ``sigma.apply``.


# --- mma.sync (s16816) register-array rewrites -----------------------------


@_rewrite.register
def _(s: RegFragment, rename, sigma, axis_fn):
    return RegFragment(name=rename(s.name), role=s.role, shape=s.shape, dtype=s.dtype)


@_rewrite.register
def _(s: LdmatrixLoad, rename, sigma, axis_fn):
    return LdmatrixLoad(
        frag=rename(s.frag),
        src_buffer=s.src_buffer,
        src_index=tuple(sigma.apply(e) for e in s.src_index),
        role=s.role,
        ldm=s.ldm,
        swizzle=s.swizzle,
        staged=s.staged,
        gmem_guard=None if s.gmem_guard is None else (sigma.apply(s.gmem_guard[0]), sigma.apply(s.gmem_guard[1])),
    )


@_rewrite.register
def _(s: MmaSyncPtx, rename, sigma, axis_fn):
    return MmaSyncPtx(c_frag=rename(s.c_frag), a_frag=rename(s.a_frag), b_frag=rename(s.b_frag), shape=s.shape, ab_dtype=s.ab_dtype)


@_rewrite.register
def _(s: RegStore, rename, sigma, axis_fn):
    # The epilogue's chain SSA names are render-local (scoped per element
    # inside the store's block), so only the load index Exprs σ-substitute —
    # that threads the per-cell replication offsets through, exactly like
    # ``dst_index``.
    epilogue = s.epilogue
    if epilogue is not None:
        epilogue = RegEpilogue(
            acc=epilogue.acc,
            loads=tuple(
                EpilogueLoad(name=ld.name, buffer=ld.buffer, index=tuple(sigma.apply(e) for e in ld.index), roles=ld.roles)
                for ld in epilogue.loads
            ),
            ops=epilogue.ops,
            result=epilogue.result,
        )

    # Guard base/bound Exprs σ-substitute like ``dst_index`` (the per-cell
    # replicator's offsets must reach the boundary predicate; the bound is a
    # Literal or free symbolic Var, which σ passes through).
    def _sub_guard(g):  # noqa: ANN001, ANN202 — tuple[Expr, Expr] | None
        return None if g is None else (sigma.apply(g[0]), sigma.apply(g[1]))

    return RegStore(
        dst_buffer=s.dst_buffer,
        dst_index=tuple(sigma.apply(e) for e in s.dst_index),
        frag=rename(s.frag),
        shape=s.shape,
        ldm=s.ldm,
        epilogue=epilogue,
        m_guard=_sub_guard(s.m_guard),
        n_guard=_sub_guard(s.n_guard),
    )
