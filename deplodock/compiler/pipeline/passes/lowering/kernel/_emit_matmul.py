"""Template emitter for ``cuda.matmul.strategy=tma_matmul`` LoopOps.

Generalized matmul template: the kernel outputs ``C[batch..., m, n] =
reduce_k(A(batch..., m, k) * B(batch..., n, k))`` where:

- ``k`` is the single reduce axis.
- ``m`` and ``n`` are the distinguishing free axes (m appears only in A's
  chain, n only in B's).
- ``batch_axes`` are any extra free axes that appear in *both* A's and B's
  chains (they're the "grouping" dims — e.g. ``head`` for batched SDPA).
- Each of A and B may be an arbitrary chain of ``Load``s + ``Assign``s +
  ``Select``s — not just a single Load. ``analyze_matmul`` partitions the
  reduce-body stmts into A-set / B-set via backward dataflow from the
  final ``mul``'s two arguments.

Body shapes supported:

    body                                   emitted kernel
    -----------------------------------    -------------------------------------
    {loads, mul, accum, write}             bare tiled matmul
    {reduce, epi..., write}                tiled matmul + per-element epilogue
    {pro..., reduce, epi, write}           with fused pro + epi
    reduce body has RoPE-style loads +     A-chain / B-chain rendered inline
    Selects + multiple Assigns             during tile load

Grid: ``(N/BN, M/BM, prod(batch_extents))``. Block: ``(THREADS,)``. Each
block owns one BM×BN output tile for one batch coord; each thread owns a
TM×TN register tile. A-chain renders inline during A-tile load; B-chain
during B-tile load; the inner K-loop stays a clean
``acc[i][j] += A_reg[i] * B_reg[j]``.

Shape + tile config come from the ``cuda.matmul.*`` hints set by
``passes/loop/matmul/001_detect_matmul.py``. ``analyze_matmul`` is the
shared producer/consumer oracle — the detector uses it to verify the
body is emittable, the emitter uses it to extract the chain stmts it
renders.

A genuine TMA path (``cp.async.bulk.tensor.2d`` + ``cuTensorMapEncodeTiled``
descriptors) is a follow-up — the name ``tma_matmul`` is kept as the
strategy token so the intended target is visible in the hint bag.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, FuncCallExpr, Literal, Var, free_vars
from deplodock.compiler.ir.kernel.ir import GpuKernel, GpuKernelParam, RawCode
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Select, Stmt, Write

# Elementwise ops supported in A-chain / B-chain / prologue / epilogue — kept
# in sync with the CUDA spellings in ``_render_elementwise``.
_SUPPORTED_EW = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "negative",
        "exp",
        "reciprocal",
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

    ``a_chain`` / ``b_chain`` are the reduce-body stmts (in original body
    order) whose SSA results feed the final ``mul(a_result, b_result)``
    that the Accum consumes. A stmt may appear in both chains if it's
    reachable from both sides — renderers duplicate it in each tile-load
    context (compiler CSE folds the redundancy).
    """

    m_axis: Axis
    n_axis: Axis
    k_axis: Axis
    batch_axes: tuple[Axis, ...]
    a_chain: tuple[Stmt, ...]
    b_chain: tuple[Stmt, ...]
    a_result: str
    b_result: str
    top_level: tuple[Stmt, ...]
    prologue: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]
    accum_name: str
    write: Write
    write_value: str


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _stmt_ssa_deps(stmt: Stmt) -> tuple[str, ...]:
    """SSA names this stmt reads (not axis vars / buffer sources)."""
    if isinstance(stmt, Assign):
        return stmt.args
    if isinstance(stmt, Select):
        return tuple(b.value for b in stmt.branches)
    return ()  # Load has no SSA deps; Accum is terminal (we don't walk through it)


def _stmt_axis_refs(stmt: Stmt) -> frozenset[str]:
    """Axis vars directly referenced by a stmt's index / predicate expressions.

    Loads contribute their ``index`` dim exprs; Selects contribute their
    branch ``select`` predicates. Assigns are pure SSA ops — axis deps
    come transitively via the SSA names they read.
    """
    out: set[str] = set()
    if isinstance(stmt, Load):
        for e in stmt.index:
            out |= set(free_vars(e))
    elif isinstance(stmt, Select):
        for b in stmt.branches:
            out |= set(free_vars(b.select))
    return frozenset(out)


def _backward_closure(target_ssa: str, def_map: dict[str, Stmt]) -> set[str]:
    """SSA names transitively feeding ``target_ssa`` within ``def_map``.

    Walks SSA deps only — stops at names not in ``def_map`` (those are
    external / bound at outer scope). Returns the set of names including
    ``target_ssa`` itself.
    """
    seen: set[str] = set()
    stack: list[str] = [target_ssa]
    while stack:
        name = stack.pop()
        if name in seen or name not in def_map:
            seen.add(name)
            continue
        seen.add(name)
        for dep in _stmt_ssa_deps(def_map[name]):
            if dep not in seen:
                stack.append(dep)
    return seen


def _stmts_only_over(stmts: tuple[Stmt, ...], allowed_axes: frozenset[str]) -> bool:
    """Validate prologue / epilogue / top-level region: only Load + Assign
    + Select stmts with supported ops, axis refs within ``allowed_axes``.

    Selects are allowed (causal mask in attention's epilogue looks like
    ``v = lhs when (q<=k) / rhs when 1``). Their predicates must use only
    axes in ``allowed_axes``. Reduce-body A/B chains share the same rule.
    """
    for s in stmts:
        if isinstance(s, Load):
            for e in s.index:
                if not free_vars(e).issubset(allowed_axes):
                    return False
        elif isinstance(s, Assign):
            if s.op.name not in _SUPPORTED_EW:
                return False
        elif isinstance(s, Select):
            for b in s.branches:
                if not free_vars(b.select).issubset(allowed_axes):
                    return False
        else:
            return False
    return True


def analyze_matmul(loop_op: LoopOp) -> MatmulInfo | None:
    """Return a ``MatmulInfo`` if the body fits the generalized skeleton.

    Walks the body, splits into (top_level, [free_loops], reduce_loop,
    epilogue, write). Inside the reduce loop, finds the single add-Accum,
    traces back through its feeding ``Assign(mul, a, b)``, and partitions
    reduce-body stmts into A-chain (feeds ``a``) and B-chain (feeds ``b``).
    Axis partitioning: ``M`` = axes referenced only by A-chain (minus K),
    ``N`` = axes only by B-chain, ``batch_axes`` = axes both reference.
    """
    reduce_names = loop_op.reduce_axis_names
    if len(reduce_names) != 1:
        return None
    k_name = next(iter(reduce_names))

    # Body root: separate top-level stmts from the outer-most free Loop.
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
    if not _stmts_only_over(tuple(top_level), frozenset()):
        return None

    # Descend through free Loops. Each level expects exactly one nested free
    # Loop (or the reduce Loop); no sibling free Loops. Collects all free
    # axes encountered on the path.
    free_axes: list[Axis] = []
    cur = outer_loop
    reduce_scope: tuple[Stmt, ...] | None = None
    reduce_idx = -1
    while True:
        if cur.axis.name in reduce_names:
            return None  # reduce shouldn't wrap free loops in this shape
        free_axes.append(cur.axis)
        found_reduce = -1
        next_free: Loop | None = None
        for i, s in enumerate(cur.body):
            if isinstance(s, Loop):
                if s.axis.name == k_name:
                    found_reduce = i
                    break
                if next_free is not None:
                    return None
                next_free = s
        if found_reduce >= 0:
            reduce_scope = cur.body
            reduce_idx = found_reduce
            break
        if next_free is None:
            return None
        cur = next_free

    if reduce_scope is None or len(free_axes) < 2:
        return None
    axis_by_name = {a.name: a for a in loop_op.axes}
    k_axis = axis_by_name.get(k_name)
    if k_axis is None:
        return None

    prologue = tuple(reduce_scope[:reduce_idx])
    reduce_loop = reduce_scope[reduce_idx]
    post_reduce = tuple(reduce_scope[reduce_idx + 1 :])
    if not isinstance(reduce_loop, Loop) or not post_reduce:
        return None

    # Post-reduce must end with exactly one Write; anything before is epilogue.
    write = post_reduce[-1]
    if not isinstance(write, Write):
        return None
    epilogue = post_reduce[:-1]

    # Reduce body: find exactly one add-Accum, fed by an Assign(mul, [a, b]).
    # Everything else must partition into A-chain (feeds a) or B-chain (b).
    rb = reduce_loop.body
    accums: list[Accum] = [s for s in rb if isinstance(s, Accum)]
    if len(accums) != 1 or accums[0].op.name != "add":
        return None
    acc = accums[0]

    # SSA def map for the reduce body (everything nameable).
    def_map: dict[str, Stmt] = {}
    for s in rb:
        if isinstance(s, (Load, Assign, Select)):
            if s.name in def_map:
                return None  # duplicate SSA def
            def_map[s.name] = s

    mul = def_map.get(acc.value)
    if not isinstance(mul, Assign) or mul.op.name != "multiply" or len(mul.args) != 2:
        return None
    a_result, b_result = mul.args

    a_names = _backward_closure(a_result, def_map)
    b_names = _backward_closure(b_result, def_map)

    # Every non-Accum reduce-body stmt must be reachable from a or b (or
    # both — shared loads like cos/sin). Orphans mean the body has dead
    # code or an unsupported shape; reject.
    reachable = a_names | b_names
    for s in rb:
        if isinstance(s, Accum):
            if s is not acc:
                return None
            continue
        if not isinstance(s, (Load, Assign, Select)):
            return None
        if s.op.name not in _SUPPORTED_EW if isinstance(s, Assign) else False:
            return None
        if s.name not in reachable and s.name != mul.name:
            return None

    # Preserve body order in chain tuples (renderer relies on definition-
    # before-use order).
    a_chain = tuple(s for s in rb if isinstance(s, (Load, Assign, Select)) and s.name in a_names and s.name != mul.name)
    b_chain = tuple(s for s in rb if isinstance(s, (Load, Assign, Select)) and s.name in b_names and s.name != mul.name)

    # Collect axis footprints for each chain. Axis refs come from Load
    # indices and Select predicates; Assigns inherit axes via SSA deps,
    # which we bake in by unioning over the full chain.
    a_axes: set[str] = set()
    for s in a_chain:
        a_axes |= _stmt_axis_refs(s)
    b_axes: set[str] = set()
    for s in b_chain:
        b_axes |= _stmt_axis_refs(s)

    # K must appear in both chains (it's the reduce we're summing over).
    if k_name not in a_axes or k_name not in b_axes:
        return None

    batch_name_set = (a_axes & b_axes) - {k_name}
    m_name_set = a_axes - batch_name_set - {k_name}
    n_name_set = b_axes - batch_name_set - {k_name}
    if len(m_name_set) != 1 or len(n_name_set) != 1:
        return None
    m_name = next(iter(m_name_set))
    n_name = next(iter(n_name_set))

    # Every free axis on the loop nest path must classify as M, N, or batch.
    # No free axis should be unaccounted-for (would mean the axis isn't
    # referenced by either chain, so some output position is undefined).
    free_name_set = {a.name for a in free_axes}
    if free_name_set != (batch_name_set | {m_name, n_name}):
        return None
    m_axis = axis_by_name[m_name]
    n_axis = axis_by_name[n_name]
    batch_axes = tuple(axis_by_name[name] for name in sorted(batch_name_set))

    # Validate prologue / epilogue stmts only use {batch ∪ m ∪ n} axes.
    outer_name_set = frozenset(batch_name_set | {m_name, n_name})
    if not _stmts_only_over(prologue, outer_name_set):
        return None
    if not _stmts_only_over(epilogue, outer_name_set):
        return None

    # Write index: active dims (those referencing one axis Var) must cover
    # exactly {batch_axes ∪ m ∪ n}. The emitter substitutes these axis Vars
    # when flattening the write offset.
    w_active_axes: set[str] = set()
    for e in write.index:
        fv = free_vars(e)
        if fv:
            w_active_axes |= fv
    if w_active_axes != outer_name_set:
        return None

    # Write value must be defined in the outer scope (accumulator name, or
    # a prologue / epilogue Assign / Load).
    defined_names = {acc.name}
    for s in (*prologue, *epilogue):
        if isinstance(s, (Load, Assign)):
            defined_names.add(s.name)
    if write.value not in defined_names:
        return None

    return MatmulInfo(
        m_axis=m_axis,
        n_axis=n_axis,
        k_axis=k_axis,
        batch_axes=batch_axes,
        a_chain=a_chain,
        b_chain=b_chain,
        a_result=a_result,
        b_result=b_result,
        top_level=tuple(top_level),
        prologue=prologue,
        epilogue=epilogue,
        accum_name=acc.name,
        write=write,
        write_value=write.value,
    )


# ---------------------------------------------------------------------------
# Expression rendering
# ---------------------------------------------------------------------------


def _render_expr(expr: Expr, env: dict[str, str]) -> str:
    """Render an ``Expr`` to a CUDA expression string. ``env`` maps axis
    Var names to local C variable names."""
    if isinstance(expr, Var):
        return env.get(expr.name, expr.name)
    if isinstance(expr, Literal):
        if isinstance(expr.value, int):
            return str(expr.value)
        return f"{float(expr.value):.6g}f"
    if isinstance(expr, BinaryExpr):
        lhs = _render_expr(expr.left, env)
        rhs = _render_expr(expr.right, env)
        return f"({lhs} {expr.op} {rhs})"
    if isinstance(expr, FuncCallExpr):
        args = ", ".join(_render_expr(a, env) for a in expr.args)
        return f"{expr.name}({args})"
    raise ValueError(f"unsupported expr in matmul emitter: {type(expr).__name__}")


def _render_elementwise(fn: str, args: list[str]) -> str:
    if fn == "add":
        return f"({args[0]} + {args[1]})"
    if fn == "subtract":
        return f"({args[0]} - {args[1]})"
    if fn == "multiply":
        return f"({args[0]} * {args[1]})"
    if fn == "divide":
        return f"({args[0]} / {args[1]})"
    if fn == "negative":
        return f"(-{args[0]})"
    if fn == "exp":
        return f"expf({args[0]})"
    if fn == "reciprocal":
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
    """Row-major flatten with axis Vars substituted via ``env``."""
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
    return f"float {s.name} = {_render_elementwise(s.op.name, list(s.args))};"


def _render_select(s: Select, env: dict[str, str]) -> str:
    """Render a Select as a chain of C ternaries.

    ``Select(name, branches)``: branches evaluated in order; first
    matching predicate wins. Emitted as
    ``(cond0) ? val0 : ((cond1) ? val1 : ... : 0.0f)``. If the final
    branch's predicate is the tautological ``Literal(1)``, the deepest
    else reduces to that branch's value (no ``0.0f`` fallback).
    """
    if not s.branches:
        raise ValueError("empty Select")
    last = s.branches[-1]
    is_tautology = isinstance(last.select, Literal) and bool(last.select.value)
    if is_tautology:
        result = last.value
        iter_branches = s.branches[:-1]
    else:
        result = "0.0f"
        iter_branches = s.branches
    for br in reversed(iter_branches):
        cond = _render_expr(br.select, env)
        result = f"(({cond}) ? {br.value} : {result})"
    return f"float {s.name} = {result};"


def _render_stmts(stmts: tuple[Stmt, ...], node: Node, graph: Graph, env: dict[str, str]) -> list[str]:
    out: list[str] = []
    for s in stmts:
        if isinstance(s, Load):
            out.append(_render_load(s, node, graph, env))
        elif isinstance(s, Assign):
            out.append(_render_assign(s))
        elif isinstance(s, Select):
            out.append(_render_select(s, env))
        else:  # pragma: no cover
            raise ValueError(f"unexpected stmt in matmul chain/epilogue: {type(s).__name__}")
    return out


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def _batch_decode_lines(batch_axes: tuple[Axis, ...]) -> list[str]:
    """Decode ``blockIdx.z`` into one C int per batch axis.

    With a single batch axis: ``int <name> = blockIdx.z;``. With multiple
    (outer-first), ``blockIdx.z`` is a row-major linear index: the
    innermost axis is ``% extent``, moving outward by dividing.
    """
    if not batch_axes:
        return []
    if len(batch_axes) == 1:
        return [f"const int {batch_axes[0].name} = blockIdx.z;"]
    lines: list[str] = ["const int _bz = blockIdx.z;"]
    stride = 1
    # Decode innermost-first so inner axis reads ``_bz % extent``, then we
    # shift outward by dividing through accumulated stride.
    for ax in reversed(batch_axes):
        if stride == 1:
            lines.append(f"const int {ax.name} = _bz % {ax.extent};")
        else:
            lines.append(f"const int {ax.name} = (_bz / {stride}) % {ax.extent};")
        stride *= int(ax.extent)
    return lines


def _shape_comment(m: int, n: int, k: int, bm: int, bn: int, bk: int, tm: int, tn: int, threads: int, batch: int) -> str:
    return f"// matmul batch={batch} M={m} N={n} K={k}  tile=({bm},{bn},{bk})  thread_tile=({tm},{tn})  threads={threads}"


def emit_matmul_kernel(
    node: Node, kernel_name: str, graph: Graph
) -> tuple[GpuKernel, list[str], tuple[int, int, int], tuple[int, int, int]]:
    """Emit a tiled SGEMM for a matmul-annotated node.

    Returns ``(kernel, arg_order, grid, block)``. ``arg_order`` matches
    the scalar emitter (deduped ``node.inputs`` in order + output last).
    Grid is 3D: ``(N/BN, M/BM, prod(batch_extents))``.
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
    assert m % bm == 0 and n % bn == 0 and k % bk == 0, (
        f"matmul template requires M%BM==N%BN==K%BK==0; got M={m},N={n},K={k},BM={bm},BN={bn},BK={bk}"
    )

    batch_extents = [int(a.extent) for a in info.batch_axes]
    batch_size = 1
    for e in batch_extents:
        batch_size *= e

    output_name = node.id
    seen: list[str] = []
    for buf in node.inputs:
        if buf not in seen and buf != output_name:
            seen.append(buf)
    params = [GpuKernelParam(dtype="const float*", name=b) for b in seen]
    params.append(GpuKernelParam(dtype="float*", name=output_name))
    arg_order = [*seen, output_name]

    # --- Axis environments ---
    # The batch axes keep their original names in the emitted code (one
    # ``int <name> = ...;`` per axis, set from blockIdx.z at kernel entry).
    # M/N map to ``r`` / ``c`` in the per-output-element loop, and to
    # ``(m0 + idx/BK)`` / ``(n0 + idx%BN)`` inside the tile loads. K maps to
    # ``(k0 + ...)`` per tile dim.
    batch_env_epi = {a.name: a.name for a in info.batch_axes}
    batch_env_tile = dict(batch_env_epi)
    epi_env = {**batch_env_epi, info.m_axis.name: "r", info.n_axis.name: "c"}
    a_tile_env = {
        **batch_env_tile,
        info.m_axis.name: f"(m0 + idx / {bk})",
        info.k_axis.name: f"(k0 + idx % {bk})",
    }
    b_tile_env = {
        **batch_env_tile,
        info.n_axis.name: f"(n0 + idx % {bn})",
        info.k_axis.name: f"(k0 + idx / {bn})",
    }

    top_level_lines = _render_stmts(info.top_level, node, graph, {})
    pro_lines = _render_stmts(info.prologue, node, graph, epi_env)
    epi_lines = _render_stmts(info.epilogue, node, graph, epi_env)
    write_shape = tuple(graph.nodes[output_name].output.shape)
    write_flat = _flatten_index(info.write.index, write_shape, epi_env)

    # A-tile and B-tile bodies: render the chain stmts inline, then write
    # the chain's final SSA value into the tile slot.
    a_chain_lines = _render_stmts(info.a_chain, node, graph, a_tile_env)
    b_chain_lines = _render_stmts(info.b_chain, node, graph, b_tile_env)

    batch_decode = _batch_decode_lines(info.batch_axes)

    # --- Template assembly ---
    indent8 = "        "
    indent12 = "            "
    a_chain_block = ("\n" + indent12).join([*a_chain_lines, f"A_tile[idx / {bk}][idx % {bk}] = {info.a_result};"])
    b_chain_block = ("\n" + indent12).join([*b_chain_lines, f"B_tile[idx / {bn}][idx % {bn}] = {info.b_result};"])

    epi_body_lines: list[str] = [f"float {info.accum_name} = acc[i][j];"]
    epi_body_lines.extend(pro_lines)
    epi_body_lines.extend(epi_lines)
    epi_body_lines.append(f"{output_name}[{write_flat}] = {info.write_value};")
    epi_body = ("\n" + indent8).join(epi_body_lines)

    top_level_block = ("\n".join(top_level_lines) + "\n") if top_level_lines else ""
    batch_decode_block = ("\n".join(batch_decode) + "\n") if batch_decode else ""

    code = f"""{_shape_comment(m, n, k, bm, bn, bk, tm, tn, threads, batch_size)}
// A-chain axes: {{{info.m_axis.name}, {info.k_axis.name}}} + batch {{ {", ".join(a.name for a in info.batch_axes)} }}
// B-chain axes: {{{info.n_axis.name}, {info.k_axis.name}}} + batch
{top_level_block}{batch_decode_block}__shared__ float A_tile[{bm}][{bk}];
__shared__ float B_tile[{bk}][{bn}];
const int m0 = blockIdx.y * {bm};
const int n0 = blockIdx.x * {bn};
const int tid = threadIdx.x;
const int ty = tid / {tx_dim};
const int tx = tid % {tx_dim};
float acc[{tm}][{tn}] = {{}};
for (int k0 = 0; k0 < {k}; k0 += {bk}) {{
    #pragma unroll
    for (int i = 0; i < ({bm} * {bk}) / {threads}; ++i) {{
        int idx = tid + i * {threads};
        {{
            {a_chain_block}
        }}
    }}
    #pragma unroll
    for (int i = 0; i < ({bk} * {bn}) / {threads}; ++i) {{
        int idx = tid + i * {threads};
        {{
            {b_chain_block}
        }}
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
    int r = m0 + ty * {tm} + i;
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
    grid = (n // bn, m // bm, batch_size)
    block = (threads, 1, 1)
    return kernel, arg_order, grid, block


def is_matmul_annotated(node: Node) -> bool:
    """Whether a node should take the matmul template path."""
    if not isinstance(node.op, LoopOp):
        return False
    return node.hints.get("cuda.matmul.strategy") == "tma_matmul"


__all__ = ["MatmulInfo", "analyze_matmul", "emit_matmul_kernel", "is_matmul_annotated"]
