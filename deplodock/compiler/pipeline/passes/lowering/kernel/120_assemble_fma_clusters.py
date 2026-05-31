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

This is the first milestone (M2) of ``plans/inline-fma-cluster.md``. Here the
cluster is a **behavior-neutral round-trip**: ``FmaCluster.render`` re-emits the
carried cell verbatim, so a ``FMA_CLUSTER=1`` kernel is identical to
``FMA_CLUSTER=0`` (which simply skips this pass). M3 switches the render to a
single inline-PTX ``asm volatile`` block whose operand ordering pins each
source value to a fixed PTX port, so ptxas's ``.reuse`` peephole fires — closing
the register-file port-pressure gap to cuBLAS.

The detector is deliberately conservative: any cell that carries a ``Cond``
(masked-tile boundary guard), a nested loop, an ``Init``, or whose products are
not a clean A×B outer product is left untouched — no cluster, no regression. So
the pass only fires on the clean both-operands-tiled cell (the shape the plan
targets) and is a no-op everywhere else.
"""

from __future__ import annotations

from deplodock import config
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Stmt
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", KernelOp)]

FMA_CLUSTER = Knob(
    "FMA_CLUSTER",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension yet (M8). DEPLODOCK_FMA_CLUSTER=0 pins off.
    help=(
        "Assemble the matmul outer-product cell into an inline-PTX FmaCluster. "
        "Off (=0) keeps the plain-C Load+Accum body — a readability switch for inspecting the kernel."
    ),
)


def rewrite(root: Node) -> KernelOp | None:
    if not FMA_CLUSTER.narrow((True,))[0]:
        raise RuleSkipped("FMA_CLUSTER=0 pinned")
    _ = config.knob_raw(FMA_CLUSTER.name)  # keep the env-read on the standard path

    new_body, changed = _walk(root.op.body)
    if not changed:
        raise RuleSkipped("no matmul outer-product cell to assemble")
    return KernelOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
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
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        new_stmts.append(s)
    rebuilt = Body(tuple(new_stmts))
    cluster = _match_outer_product(rebuilt)
    if cluster is not None:
        return Body((cluster,)), True
    return rebuilt, changed


def _match_outer_product(body: Body) -> FmaCluster | None:
    """Return a ``FmaCluster`` if ``body`` is exactly a clean A×B outer-product
    cell (Loads + multiply Assigns + add Accums, products = full cross product),
    else ``None``. Conservative: any other statement kind aborts the match."""
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
