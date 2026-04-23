"""Template emitter for ``cuda.matmul.strategy=tma_matmul`` LoopOps.

Emits an SMEM-tiled double-register SGEMM kernel as a single ``RawCode``
block. Supports three body shapes:

    body                          emitted kernel
    --------------------------    -------------------------------------
    {loads, mul, accum, write}    bare tiled matmul
    {reduce, epi..., write}       tiled matmul + per-element epilogue
    {pro..., reduce, epi, write}  tiled matmul + per-element fused pro + epi

The prologue/epilogue may include free-axis ``Load``s (e.g. the residual
buffer, gate input) and elementwise ``Assign``s over them + the
accumulator. A scalar ``Load`` outside all free loops (e.g. a broadcast
``1.0``) is hoisted to the kernel preamble.

Layout of the emitted kernel (after K-loop):

.. code-block:: cuda

    for (i = 0..TM) {
        r = ty*TM + i;  // the m_axis
        for (j = 0..TN) {
            c = n0 + tx*TN + j;  // the n_axis
            float <acc_name> = acc[i][j];
            <prologue stmts with axis-vars substituted by r,c>
            <epilogue stmts>
            <OUT buffer>[<flat_index(r,c)>] = <write_value>;
        }
    }

Shape + tile + buffer-role come from the ``cuda.matmul.*`` hints set by
``passes/loop/matmul/001_detect_matmul.py``. ``analyze_matmul`` is the
shared producer/consumer oracle — the detector uses it to verify the
body is emittable, the emitter uses it to extract the stmts it renders.

A genuine TMA path (``cp.async.bulk.tensor.2d`` + ``cuTensorMapEncodeTiled``
descriptors) is a follow-up — the name ``tma_matmul`` is kept as the
strategy token so the intended target is visible in the hint bag.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinOp, Expr, FuncCall, Literal, Var, free_vars
from deplodock.compiler.ir.kernel.ir import GpuKernel, GpuKernelParam, RawCode
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Select, Stmt, Write

# Elementwise ops supported in prologue/epilogue — kept in sync with the
# CUDA spellings in ``_render_elementwise``.
_SUPPORTED_EW = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "exp",
        "recip",
        "relu",
        "sigmoid",
        "tanh",
        "sqrt",
        "rsqrt",
        "log",
        "fabs",
        "fmax",
        "fmin",
        "pow",
    }
)


@dataclass(frozen=True)
class MatmulInfo:
    """Everything the emitter needs to render a matmul-annotated LoopOp.

    Shared by the detector pass (which validates) and the emitter (which
    renders). If ``analyze_matmul`` returns ``None`` the body doesn't fit
    the supported skeleton; the detector skips annotating and the scalar
    emitter takes over.

    ``la`` and ``lb`` are the matmul's two ``Load`` stmts; ``la`` is the
    m-partner (its index depends on ``m_axis``) and ``lb`` is the
    n-partner. Their index expressions may be arbitrary affine / divmod
    chains over ``(m, k)`` / ``(n, k)`` — the emitter substitutes the
    tile-local coords into these to handle scrambled layouts (e.g. an
    attention-output reshape fused into o_proj's A-load).
    """

    m_axis: Axis
    n_axis: Axis
    k_axis: Axis
    a_source: int
    b_source: int
    la: Load
    lb: Load
    top_level: tuple[Stmt, ...]
    prologue: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]
    accum_name: str
    write: Write
    write_value: str


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _active_axes(index: tuple[Expr, ...]) -> list[tuple[int, str]]:
    """Return ``(pos, axis_name)`` for each index dim referencing one axis Var."""
    out = []
    for i, e in enumerate(index):
        fv = free_vars(e)
        if len(fv) == 1:
            out.append((i, next(iter(fv))))
    return out


def _stmts_only_over(stmts: tuple[Stmt, ...], allowed_axes: frozenset[str]) -> bool:
    """Every Load index + every Assign op + no Accum/Select/Loop in stmts."""
    for s in stmts:
        if isinstance(s, Load):
            for e in s.index:
                if not free_vars(e).issubset(allowed_axes):
                    return False
        elif isinstance(s, Assign):
            if s.op.fn not in _SUPPORTED_EW:
                return False
        else:
            return False
    return True


def analyze_matmul(loop_op: LoopOp) -> MatmulInfo | None:
    """Return a ``MatmulInfo`` if the body fits the emitter's skeleton, else None.

    Walks the body and splits it into (top_level, prologue, reduce_core,
    epilogue, write). Validates that:
      - exactly one add-Accum, fed by a mul of two Loads from distinct sources
      - the reduce axis is the trailing active dim of each matmul Load
      - the matmul Loads' other active dims are two distinct free axes (m, n)
      - prologue / epilogue / top_level contain only supported elementwise
        stmts over (m, n) — no extra reductions, Selects, or Loops
      - the Write's active index dims are [m, n] and its value is either
        the accumulator name or an SSA name bound in the epilogue
    """
    reduce_names = loop_op.reduce_axis_names
    if len(reduce_names) != 1:
        return None
    k_name = next(iter(reduce_names))

    # Walk body: separate top-level stmts from the single outer Loop that
    # wraps everything free.
    top_level: list[Stmt] = []
    outer_loop: Loop | None = None
    for s in loop_op.body:
        if isinstance(s, Loop):
            if outer_loop is not None:
                return None
            outer_loop = s
        else:
            top_level.append(s)
    if outer_loop is None:
        return None
    # Top level must be scalar Loads / Assigns only — no axis refs.
    if not _stmts_only_over(tuple(top_level), frozenset()):
        return None

    # Descend through free-axis Loops until we hit the scope holding the reduce.
    # Each step expects exactly one Loop as the "main" stmt (plus no siblings),
    # keeping the free nesting canonical. Stop when we find a sibling list that
    # contains the reduce Loop.
    free_axes: list[Axis] = []
    cur = outer_loop
    reduce_scope: tuple[Stmt, ...] | None = None
    reduce_idx = -1
    while True:
        if cur.axis.name in reduce_names:
            return None  # outer Loop shouldn't be the reduce
        free_axes.append(cur.axis)
        # Search cur.body for the reduce Loop.
        found_reduce_here = -1
        next_free: Loop | None = None
        for i, s in enumerate(cur.body):
            if isinstance(s, Loop) and s.axis.name == k_name:
                found_reduce_here = i
                break
            if isinstance(s, Loop) and s.axis.name not in reduce_names:
                if next_free is not None:
                    return None  # more than one nested free Loop — unsupported
                next_free = s
        if found_reduce_here >= 0:
            reduce_scope = cur.body
            reduce_idx = found_reduce_here
            break
        if next_free is None:
            return None
        cur = next_free

    if reduce_scope is None or len(free_axes) != 2:
        return None
    m_axis, n_axis = free_axes
    axis_by_name = {a.name: a for a in loop_op.axes}
    k_axis = axis_by_name.get(k_name)
    if k_axis is None:
        return None

    prologue = tuple(reduce_scope[:reduce_idx])
    reduce_loop = reduce_scope[reduce_idx]
    post_reduce = tuple(reduce_scope[reduce_idx + 1 :])

    if not isinstance(reduce_loop, Loop) or not post_reduce:
        return None

    # Post-reduce must end with exactly one Write; anything before it is the epilogue.
    write = post_reduce[-1]
    if not isinstance(write, Write):
        return None
    epilogue = post_reduce[:-1]

    # Validate the reduce-loop body is the canonical mul+Accum core.
    rb = reduce_loop.body
    loads: list[Load] = [s for s in rb if isinstance(s, Load)]
    assigns: list[Assign] = [s for s in rb if isinstance(s, Assign)]
    accums: list[Accum] = [s for s in rb if isinstance(s, Accum)]
    others = [s for s in rb if not isinstance(s, (Load, Assign, Accum))]
    if others or len(loads) != 2 or len(assigns) != 1 or len(accums) != 1:
        return None
    mul = assigns[0]
    acc = accums[0]
    if acc.op.fn != "add" or mul.op.fn != "mul" or len(mul.args) != 2:
        return None
    if mul.name != acc.value:
        return None
    load_a_cand = next((ld for ld in loads if ld.name == mul.args[0]), None)
    load_b_cand = next((ld for ld in loads if ld.name == mul.args[1]), None)
    if load_a_cand is None or load_b_cand is None or load_a_cand.source == load_b_cand.source:
        return None
    # Each Load's index may be arbitrary (scrambled by reshape fusion) as
    # long as the set of axis Vars it references is exactly {free, k}.
    # That lets us distinguish A (m-partner) from B (n-partner) even when
    # the index is a divmod chain over multiple buffer dims.

    def _load_axes(ld: Load) -> frozenset[str]:
        out: set[str] = set()
        for e in ld.index:
            out |= set(free_vars(e))
        return frozenset(out)

    la_axes = _load_axes(load_a_cand)
    lb_axes = _load_axes(load_b_cand)
    if k_name not in la_axes or k_name not in lb_axes:
        return None
    la_partner = la_axes - {k_name}
    lb_partner = lb_axes - {k_name}
    if len(la_partner) != 1 or len(lb_partner) != 1:
        return None
    if la_partner | lb_partner != {m_axis.name, n_axis.name}:
        return None
    if next(iter(la_partner)) == m_axis.name:
        la, lb = load_a_cand, load_b_cand
    else:
        la, lb = load_b_cand, load_a_cand
    a_source, b_source = la.source, lb.source

    # Validate prologue / epilogue / top_level stmts.
    mn_set = frozenset({m_axis.name, n_axis.name})
    if not _stmts_only_over(prologue, mn_set) or not _stmts_only_over(epilogue, mn_set):
        return None

    # Write: active index dims must be [m, n] (allowing constant-0 dims).
    w_active = [name for _, name in _active_axes(write.index)]
    if w_active != [m_axis.name, n_axis.name]:
        return None
    # Write value must be defined — either the accumulator, an epilogue
    # Assign, or an epilogue Load.
    defined_names = {acc.name}
    for s in prologue:
        if isinstance(s, (Load, Assign)):
            defined_names.add(s.name)
    for s in epilogue:
        if isinstance(s, (Load, Assign)):
            defined_names.add(s.name)
    if write.value not in defined_names:
        return None

    return MatmulInfo(
        m_axis=m_axis,
        n_axis=n_axis,
        k_axis=k_axis,
        a_source=a_source,
        b_source=b_source,
        la=la,
        lb=lb,
        top_level=tuple(top_level),
        prologue=prologue,
        epilogue=epilogue,
        accum_name=acc.name,
        write=write,
        write_value=write.value,
    )


# ---------------------------------------------------------------------------
# Expression rendering for prologue / epilogue
# ---------------------------------------------------------------------------


def _render_expr(expr: Expr, env: dict[str, str]) -> str:
    """Render an ``Expr`` to a CUDA expression string.

    ``env`` maps axis Var names to local C variable names (e.g. the
    m-axis name to ``"r"``). Unknown Var names render as-is, which lets
    the caller keep already-C-like identifiers (``"r"``, ``"c"``) in the
    environment without rewriting the AST.
    """
    if isinstance(expr, Var):
        return env.get(expr.name, expr.name)
    if isinstance(expr, Literal):
        if isinstance(expr.value, int):
            return str(expr.value)
        return f"{float(expr.value):.6g}f"
    if isinstance(expr, BinOp):
        lhs = _render_expr(expr.left, env)
        rhs = _render_expr(expr.right, env)
        return f"({lhs} {expr.op} {rhs})"
    if isinstance(expr, FuncCall):
        args = ", ".join(_render_expr(a, env) for a in expr.args)
        return f"{expr.name}({args})"
    raise ValueError(f"unsupported expr in matmul epilogue: {type(expr).__name__}")


def _render_elementwise(fn: str, args: list[str]) -> str:
    if fn == "add":
        return f"({args[0]} + {args[1]})"
    if fn == "sub":
        return f"({args[0]} - {args[1]})"
    if fn == "mul":
        return f"({args[0]} * {args[1]})"
    if fn == "div":
        return f"({args[0]} / {args[1]})"
    if fn == "neg":
        return f"(-{args[0]})"
    if fn == "exp":
        return f"expf({args[0]})"
    if fn == "recip":
        return f"(1.0f / {args[0]})"
    if fn == "relu":
        return f"fmaxf(0.0f, {args[0]})"
    if fn == "sigmoid":
        return f"(1.0f / (1.0f + expf(-({args[0]}))))"
    if fn == "tanh":
        return f"tanhf({args[0]})"
    if fn == "sqrt":
        return f"sqrtf({args[0]})"
    if fn == "rsqrt":
        return f"rsqrtf({args[0]})"
    if fn == "log":
        return f"logf({args[0]})"
    if fn == "fabs":
        return f"fabsf({args[0]})"
    if fn == "fmax":
        return f"fmaxf({args[0]}, {args[1]})"
    if fn == "fmin":
        return f"fminf({args[0]}, {args[1]})"
    if fn == "pow":
        return f"powf({args[0]}, {args[1]})"
    raise ValueError(f"unsupported elementwise fn in matmul: {fn}")


def _flatten_index(index: tuple[Expr, ...], shape: tuple, env: dict[str, str]) -> str:
    """Row-major flatten: ``sum_i coord_i * prod(shape[i+1:])``.

    Constant-0 coords collapse the corresponding stride term at compile
    time (C compiler will fold, but we emit the simpler form up front).
    Non-integer dims (symbolic) raise; the detector rejects such buffers.
    """
    if not index:
        return "0"
    parts = []
    for i, e in enumerate(index):
        stride = 1
        for d in shape[i + 1 :]:
            if not isinstance(d, int):
                raise ValueError(f"non-integer dim in matmul buffer shape {shape}")
            stride *= d
        c = _render_expr(e, env)
        if stride == 1:
            parts.append(c)
        else:
            parts.append(f"({c}) * {stride}")
    return " + ".join(parts)


def _render_load(s: Load, node: Node, graph: Graph, env: dict[str, str]) -> str:
    buf_name = node.inputs[s.source]
    buf_node = graph.nodes.get(buf_name)
    buf_shape = tuple(buf_node.output.shape) if buf_node is not None else ()
    flat = _flatten_index(s.index, buf_shape, env)
    return f"float {s.name} = {buf_name}[{flat}];"


def _render_assign(s: Assign) -> str:
    args = list(s.args)
    return f"float {s.name} = {_render_elementwise(s.op.fn, args)};"


def _render_stmts(stmts: tuple[Stmt, ...], node: Node, graph: Graph, env: dict[str, str]) -> list[str]:
    out = []
    for s in stmts:
        if isinstance(s, Load):
            out.append(_render_load(s, node, graph, env))
        elif isinstance(s, Assign):
            out.append(_render_assign(s))
        else:  # pragma: no cover — analyzer rejects other stmts
            raise ValueError(f"unexpected stmt in matmul prologue/epilogue: {type(s).__name__}")
    return out


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def _shape_comment(m: int, n: int, k: int, bm: int, bn: int, bk: int, tm: int, tn: int, threads: int) -> str:
    return f"// matmul M={m} N={n} K={k}  tile=({bm},{bn},{bk})  thread_tile=({tm},{tn})  threads={threads}"


def emit_matmul_kernel(
    node: Node, kernel_name: str, graph: Graph
) -> tuple[GpuKernel, list[str], tuple[int, int, int], tuple[int, int, int]]:
    """Emit a tiled SGEMM (+ optional fused prologue/epilogue) for a matmul node.

    Returns ``(kernel, arg_order, grid, block)``. ``arg_order`` matches the
    scalar emitter (deduped ``node.inputs`` in order + output last). Extra
    inputs referenced only by the prologue/epilogue (e.g. a residual
    buffer) appear in the signature via the same dedup walk.
    """
    info = analyze_matmul(node.op)
    if info is None:
        raise ValueError(f"matmul emitter called on node {node.id} without a valid matmul body")

    h = node.hints
    m = int(h.get("cuda.matmul.m"))
    n = int(h.get("cuda.matmul.n"))
    k = int(h.get("cuda.matmul.k"))
    bm = int(h.get("cuda.matmul.tile_m"))
    bn = int(h.get("cuda.matmul.tile_n"))
    bk = int(h.get("cuda.matmul.block_k"))
    tm = int(h.get("cuda.matmul.thread_m"))
    tn = int(h.get("cuda.matmul.thread_n"))
    threads = int(h.get("cuda.matmul.threads"))

    ty_dim = bm // tm
    tx_dim = bn // tn
    assert ty_dim * tx_dim == threads, f"invalid tile config: BM/TM * BN/TN = {ty_dim}*{tx_dim} != threads={threads}"
    # One block processes the full M dimension — BM must equal M. The 2D-grid
    # case (multiple M-tiles) isn't wired up yet; the detector's tile-config
    # check accepts m % bm == 0, but the template below assumes bm == m.
    assert m == bm, f"matmul template requires M == BM; got M={m}, BM={bm}"

    output_name = node.id
    a_buf = node.inputs[info.a_source]
    b_buf = node.inputs[info.b_source]
    seen: list[str] = []
    for buf in node.inputs:
        if buf not in seen and buf != output_name:
            seen.append(buf)
    params = [GpuKernelParam(dtype="const float*", name=b) for b in seen]
    params.append(GpuKernelParam(dtype="float*", name=output_name))
    arg_order = [*seen, output_name]

    # Render prologue / epilogue with the axis → r/c binding.
    env = {info.m_axis.name: "r", info.n_axis.name: "c"}
    top_level_lines = _render_stmts(info.top_level, node, graph, {})
    pro_lines = _render_stmts(info.prologue, node, graph, env)
    epi_lines = _render_stmts(info.epilogue, node, graph, env)
    write_shape = tuple(graph.nodes[output_name].output.shape)
    write_flat = _flatten_index(info.write.index, write_shape, env)

    # Build the flat-offset expressions for the A- and B-tile loads. The
    # tile-local coords come from the tid-flattened ``idx`` of each tile
    # load; we substitute the axis Vars appearing in ``info.la.index`` /
    # ``info.lb.index`` so scrambled layouts (reshape fused into A / B)
    # still emit a valid gather. For the simple contiguous case this
    # collapses to ``A[m*K + k]`` / ``B[n*K + k]`` as before.
    a_shape = tuple(graph.nodes[a_buf].output.shape)
    b_shape = tuple(graph.nodes[b_buf].output.shape)
    a_tile_env = {info.m_axis.name: f"(idx / {bk})", info.k_axis.name: f"(k0 + idx % {bk})"}
    b_tile_env = {info.n_axis.name: f"(n0 + idx % {bn})", info.k_axis.name: f"(k0 + idx / {bn})"}
    a_flat = _flatten_index(info.la.index, a_shape, a_tile_env)
    b_flat = _flatten_index(info.lb.index, b_shape, b_tile_env)

    indent = "        "
    epi_body_lines: list[str] = []
    epi_body_lines.append(f"float {info.accum_name} = acc[i][j];")
    epi_body_lines.extend(pro_lines)
    epi_body_lines.extend(epi_lines)
    epi_body_lines.append(f"{output_name}[{write_flat}] = {info.write_value};")
    epi_body = f"\n{indent}".join(epi_body_lines)

    top_level_block = "\n".join(top_level_lines)
    if top_level_block:
        top_level_block += "\n"

    code = f"""{_shape_comment(m, n, k, bm, bn, bk, tm, tn, threads)}
// A: {a_buf} shape=({m},{k})   B: {b_buf} shape=({n},{k})   OUT: {output_name} shape=({m},{n})
{top_level_block}__shared__ float A_tile[{bm}][{bk}];
__shared__ float B_tile[{bk}][{bn}];
const int n0 = blockIdx.x * {bn};
const int tid = threadIdx.x;
const int ty = tid / {tx_dim};
const int tx = tid % {tx_dim};
float acc[{tm}][{tn}] = {{}};
for (int k0 = 0; k0 < {k}; k0 += {bk}) {{
    #pragma unroll
    for (int i = 0; i < ({bm} * {bk}) / {threads}; ++i) {{
        int idx = tid + i * {threads};
        A_tile[idx / {bk}][idx % {bk}] = {a_buf}[{a_flat}];
    }}
    #pragma unroll
    for (int i = 0; i < ({bk} * {bn}) / {threads}; ++i) {{
        int idx = tid + i * {threads};
        B_tile[idx / {bn}][idx % {bn}] = {b_buf}[{b_flat}];
    }}
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < {bk}; ++kk) {{
        float a_reg[{tm}];
        float b_reg[{tn}];
        #pragma unroll
        for (int i = 0; i < {tm}; ++i) a_reg[i] = A_tile[ty * {tm} + i][kk];
        #pragma unroll
        for (int j = 0; j < {tn}; ++j) b_reg[j] = B_tile[kk][tx * {tn} + j];
        #pragma unroll
        for (int i = 0; i < {tm}; ++i)
            #pragma unroll
            for (int j = 0; j < {tn}; ++j)
                acc[i][j] += a_reg[i] * b_reg[j];
    }}
    __syncthreads();
}}
#pragma unroll
for (int i = 0; i < {tm}; ++i) {{
    int r = ty * {tm} + i;
    #pragma unroll
    for (int j = 0; j < {tn}; ++j) {{
        int c = n0 + tx * {tn} + j;
        {epi_body}
    }}
}}"""

    kernel = GpuKernel(
        name=kernel_name,
        params=params,
        body=[RawCode(code=code)],
        block_size=(threads, 1, 1),
    )
    grid = (n // bn, 1, 1)
    block = (threads, 1, 1)
    return kernel, arg_order, grid, block


def is_matmul_annotated(node: Node) -> bool:
    """Whether a node should take the matmul template path.

    Requires a ``LoopOp`` payload and a ``cuda.matmul.strategy`` hint whose
    value is ``"tma_matmul"``. Any other value (or missing hint) falls back
    to the scalar emitter.
    """
    if not isinstance(node.op, LoopOp):
        return False
    return node.hints.get("cuda.matmul.strategy") == "tma_matmul"


# ``Select`` isn't used here but its absence in the analyzer's supported set
# is a structural choice — imported so the detector's re-export shares the
# same reference if it grows to accept branchy epilogues later.
__all__ = ["MatmulInfo", "analyze_matmul", "emit_matmul_kernel", "is_matmul_annotated"]

# Satisfy the static checker — ``Select`` is referenced in the pro/epi
# rejection via ``_stmts_only_over``'s ``else`` branch implicitly (any
# non-Load/Assign stmt is refused), but we keep it importable for any
# future analyzer that wants to whitelist it.
_ = Select
