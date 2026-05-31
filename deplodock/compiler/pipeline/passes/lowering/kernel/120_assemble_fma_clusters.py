"""Assemble flat matmul cells into structured ``FmaCluster`` nodes.

After ``010_split_register_axes`` unrolls the register tile and
``100_materialize_tile`` lowers to per-thread ``KernelOp`` form, the inner
K-reduce loop body is a flat outer-product cell:

    for a_k in 0..BK:
        Load a0 ← A[k, r0]      # FM A-operand loads (one register row each)
        Load a1 ← A[k, r1]
        Load b0 ← B[k, c0]      # FN B-operand loads (one register col each)
        Load b1 ← B[k, c1]
        Assign v0 = multiply(a0, b0)   # FM*FN products, outer-product grid
        Assign v1 = multiply(a0, b1)
        Assign v2 = multiply(a1, b0)
        Assign v3 = multiply(a1, b1)
        Accum acc0 <- v0               # FM*FN accumulates, row-major acc[m*fn+n]
        Accum acc1 <- v1
        Accum acc2 <- v2
        Accum acc3 <- v3

This pass recognizes that cell — a body of *only* ``Load`` / ``Assign(multiply)``
/ ``Accum(add)`` whose products form a full A×B cross product — and wraps it in
a single :class:`~deplodock.compiler.ir.kernel.ir.FmaCluster`. The K-loop stays;
its body becomes one cluster node.

``FmaCluster.render`` emits one inline-PTX ``asm volatile`` block (``fm*fn``
``fma.rn.f32`` in ``B_INNER`` operand order) whose ordering is *meant* to pin
each source value to a fixed PTX port so ptxas's ``.reuse`` peephole fires.

**Off by default (opt in with ``DEPLODOCK_FMA_CLUSTER=1``).** SASS measurement
on sm_120 (RTX 5090, ``plans/inline-fma-cluster.md`` M4) showed ptxas does *not*
honor the operand-port discipline — it reallocates registers and commutes the
FFMA ``a``/``b`` operands, so ``.reuse`` density moved only 0.762 → 0.777 (the
0.96 the 26×4 B_INNER shape predicts never materialized) and the 2048³ wall-clock
was unchanged (279 µs, 0.96× cuBLAS, identical on vs off). The machinery is
correct and accuracy-verified; it is kept behind the knob as an experiment /
readability switch, not a default-on optimization.

The detector is deliberately conservative: any cell that carries a ``Cond``
(masked-tile boundary guard), a nested loop, an ``Init``, a non-f32 operand
buffer, or whose products are not a clean A×B outer product is left untouched.
"""

from __future__ import annotations

from deplodock import config
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp, Smem
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Stmt
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", KernelOp)]

FMA_CLUSTER = Knob(
    "FMA_CLUSTER",
    KnobType.BOOL,
    hints=(False,),  # OFF by default — opt in with DEPLODOCK_FMA_CLUSTER=1. See note below.
    help=(
        "Assemble the matmul outer-product cell into an inline-PTX FmaCluster (one asm volatile "
        "FFMA block, B_INNER operand order). Opt-in (=1). SASS measurement on sm_120 (RTX 5090) "
        "showed ptxas does not honor the operand-port ordering — .reuse density 0.762 -> 0.777 and "
        "2048^3 wall-clock unchanged (279 us, 0.96x cuBLAS) — so it is off by default; kept as an "
        "experiment / readability switch. See plans/inline-fma-cluster.md M4."
    ),
)


def rewrite(root: Node) -> KernelOp | None:
    # Off by default: the inline-PTX cluster showed no measured .reuse / latency
    # gain on sm_120 (M4). ``DEPLODOCK_FMA_CLUSTER=1`` opts in (env pin honored
    # authoritatively even though only ``False`` is enumerated).
    if not FMA_CLUSTER.narrow((False,))[0]:
        raise RuleSkipped("FMA_CLUSTER disabled (default)")
    _ = config.knob_raw(FMA_CLUSTER.name)  # keep the env-read on the standard path

    f32_bufs = _f32_buffers(root.op)
    new_body, changed = _walk(root.op.body, f32_bufs)
    if not changed:
        raise RuleSkipped("no matmul outer-product cell to assemble")
    return KernelOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _f32_buffers(kernel_op: KernelOp) -> frozenset[str]:
    """Names of buffers provably f32: ``__shared__`` decls with C-type ``float``
    plus kernel inputs/outputs whose ``Tensor`` dtype is F32. The inline-PTX
    emitter is f32-only, so the cluster fires only when *every* operand buffer
    is in this set (a half / bf16 operand would be a constraint type-mismatch)."""
    bufs = {s.name for s in kernel_op.body.iter_of_type(Smem) if s.dtype == "float"}
    for name, tensor in {**kernel_op.inputs, **kernel_op.outputs}.items():
        if tensor.dtype == F32:
            bufs.add(name)
    return frozenset(bufs)


def _walk(body: Body, f32_bufs: frozenset[str]) -> tuple[Body, bool]:
    """Recurse into nested bodies; when a body is a pure outer-product cell,
    replace it with a single ``FmaCluster``."""
    new_stmts: list[Stmt] = []
    changed = False
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b, f32_bufs)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        new_stmts.append(s)
    rebuilt = Body(tuple(new_stmts))
    cluster = _match_outer_product(rebuilt, f32_bufs)
    if cluster is not None:
        return Body((cluster,)), True
    return rebuilt, changed


def _match_outer_product(body: Body, f32_bufs: frozenset[str]) -> FmaCluster | None:
    """Return a ``FmaCluster`` if ``body`` is exactly a clean A×B outer-product
    cell (Loads + multiply Assigns + add Accums, products = full cross product)
    over f32 operand buffers, else ``None``. Conservative: any other statement
    kind, or any non-f32 operand buffer, aborts the match."""
    stmts = tuple(body)
    loads: list[Load] = []
    muls: list[Assign] = []
    accums: list[Accum] = []
    for s in stmts:
        if isinstance(s, Load):
            loads.append(s)
        elif isinstance(s, Assign) and s.op.name == "multiply" and len(s.args) == 2:
            muls.append(s)
        elif isinstance(s, Accum) and s.op.name == "add":
            accums.append(s)
        else:
            return None  # Cond / Init / nested loop / non-mul Assign → not a clean cell

    # Need a real cluster (≥ 2 cells) backed by matching products + accumulators.
    if len(muls) < 2 or len(muls) != len(accums) or not loads:
        return None

    # f32-only: the inline-PTX emitter is ``fma.rn.f32`` with ``"+f"`` / ``"f"``
    # (float-register) constraints. Operand C type lives on the *buffer* (loads
    # are dtype-None; the type comes from the Smem decl / input Tensor), so gate
    # on every operand buffer being provably f32. fp16 / bf16 cells (operands
    # declared ``__half`` etc.) would be a constraint type-mismatch — they stay
    # on the plain-C path (and target the WMMA tensor-core lowering anyway).
    if any(ld.input not in f32_bufs for ld in loads):
        return None

    name_buf: dict[str, str] = {n: ld.input for ld in loads for n in ld.names}
    mul_by_name = {m.name: m for m in muls}

    # acc.value must be a product of two load-defined operands.
    cells: list[tuple[str, str, str]] = []  # (acc_name, operand_x, operand_y)
    for acc in accums:
        m = mul_by_name.get(acc.value)
        if m is None:
            return None
        x, y = m.args
        if x not in name_buf or y not in name_buf:
            return None
        cells.append((acc.name, x, y))

    # The two operands of every cell must come from exactly two buffers (A, B).
    bufs = {name_buf[op] for _, x, y in cells for op in (x, y)}
    if len(bufs) != 2:
        return None
    buf_a, buf_b = sorted(bufs)

    a_names: dict[str, None] = {}  # ordered set of A operands (from buf_a)
    b_names: dict[str, None] = {}  # ordered set of B operands (from buf_b)
    grid: dict[tuple[str, str], str] = {}
    for acc_name, x, y in cells:
        if name_buf[x] == buf_a and name_buf[y] == buf_b:
            a_op, b_op = x, y
        elif name_buf[y] == buf_a and name_buf[x] == buf_b:
            a_op, b_op = y, x
        else:
            return None  # both operands share a buffer — not an A×B outer product
        if (a_op, b_op) in grid:
            return None  # duplicate cell
        grid[(a_op, b_op)] = acc_name
        a_names.setdefault(a_op, None)
        b_names.setdefault(b_op, None)

    fm, fn = len(a_names), len(b_names)
    if len(grid) != fm * fn:
        return None  # not a full cross product (some (a,b) cell missing)
    # Row-major acc[m*fn+n] over the A×B grid.
    acc_names = tuple(grid[(a, b)] for a in a_names for b in b_names)
    return FmaCluster(
        body=stmts,
        a_names=tuple(a_names),
        b_names=tuple(b_names),
        acc_names=acc_names,
        fm=fm,
        fn=fn,
    )
