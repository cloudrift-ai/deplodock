"""Assemble primitive ops into structural ``LoopOp`` nodes.

The grammar:

    reduce*       all reachable ReduceOps (must share axis + pre-reduce shape)
    elementwise*  all reachable ElementwiseOps
    layout*       same-rank IndexMapOps (for connectivity, become Port.index)
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Expr, Literal, Var, substitute
from deplodock.compiler.ir.graph import Graph, Node, Tensor
from deplodock.compiler.ir.loop import Assign, Axis, LocalBuffer, LoopOp, Port, Select, SelectBranch, Stmt, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production

# Identity values for reductions — lifted from the former REDUCE_REGISTRY.
_REDUCE_IDENTITY: dict[str, float] = {"sum": 0.0, "max": -1e30, "prod": 1.0, "min": 1e30}
# Combine op name for each reduction family.
_REDUCE_COMBINE: dict[str, str] = {"sum": "add", "max": "max", "prod": "mul", "min": "min"}


def _same_rank(op, node, graph):
    """IndexMapOp predicate: absorb IndexMapOps that translate to simple
    per-dim index Exprs — permutations, squeezes, unsqueezes, and linear
    (``+``/``-``) offset slices.

    Each ``coord_map`` entry must be one of:
      - ``Literal(0, "int")`` at a size-1 source dim (broadcast read),
      - ``Var("out_coord_k")`` for some output axis ``k`` (any order —
        transposes and leading/trailing/mid unsqueezes qualify),
      - a ``+``/``-`` combination of placeholders and literals (simple
        slice offsets, e.g. ``out_coord_3 + 32`` for rotary
        ``rotate_half``).

    Multiplicative / modular arithmetic (``* // %``) for GQA replication
    or head-split reshapes falls through to ``003_wrap_indexmap`` as a
    standalone copy kernel. Broadcasts into fresh non-unit output dims
    are also rejected here — they expand the kernel's iteration space in
    ways the region matcher can't currently reconcile with downstream
    reductions.
    """
    if len(op.sources) > 1:
        return False
    in_shape = graph.nodes[node.inputs[0]].output.shape
    return _is_identity_alias(op, in_shape)


def _is_linear_placeholder_expr(expr) -> bool:
    """True iff ``expr`` is a linear (``+``/``-``) combination of placeholders and literals."""
    from deplodock.compiler.ir.expr import BinOp, Cast, Literal, Var

    if isinstance(expr, Literal):
        return True
    if isinstance(expr, Var):
        return expr.name.startswith(PLACEHOLDER_PREFIX)
    if isinstance(expr, BinOp) and expr.op in ("+", "-"):
        return _is_linear_placeholder_expr(expr.left) and _is_linear_placeholder_expr(expr.right)
    if isinstance(expr, Cast):
        return _is_linear_placeholder_expr(expr.expr)
    return False


def _collect_placeholders(expr, used: set[int]) -> None:
    """Collect the set of ``out_coord_k`` indices referenced anywhere in ``expr``."""
    from deplodock.compiler.ir.expr import BinOp, Cast, Var

    if isinstance(expr, Var) and expr.name.startswith(PLACEHOLDER_PREFIX):
        try:
            used.add(int(expr.name[len(PLACEHOLDER_PREFIX) :]))
        except ValueError:
            pass
    elif isinstance(expr, BinOp):
        _collect_placeholders(expr.left, used)
        _collect_placeholders(expr.right, used)
    elif isinstance(expr, Cast):
        _collect_placeholders(expr.expr, used)


def _is_identity_alias(op, in_shape) -> bool:
    """True iff every ``coord_map`` entry is either a plain placeholder, a
    ``Literal(0)`` at a size-1 source dim, or a linear (``+``/``-``)
    expression over placeholders and literals, and every non-size-1
    output dim is covered by at least one placeholder. See
    :func:`_same_rank`.
    """
    from deplodock.compiler.ir.expr import Literal, Var

    src = op.sources[0]
    cm = src.coord_map
    if len(cm) != len(in_shape):
        return False

    used: set[int] = set()
    for i, entry in enumerate(cm):
        if isinstance(entry, Literal):
            if entry.value != 0:
                return False
            if i < len(in_shape) and isinstance(in_shape[i], int) and in_shape[i] != 1:
                return False
        elif isinstance(entry, Var) and entry.name.startswith(PLACEHOLDER_PREFIX):
            try:
                used.add(int(entry.name[len(PLACEHOLDER_PREFIX) :]))
            except ValueError:
                return False
        elif _is_linear_placeholder_expr(entry):
            _collect_placeholders(entry, used)
        else:
            return False

    for i, d in enumerate(op.out_shape):
        if i in used:
            continue
        if isinstance(d, int) and d != 1:
            return False
    return True


GRAMMAR = [
    Production("reduce", ReduceOp, "*", bind={"input_shape": "pre_shape", "axis": "reduce_axis"}),
    Production("elementwise", ElementwiseOp, "*"),
    Production("layout", IndexMapOp, "*", where=_same_rank),
]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    compute_ids = match.get("reduce") + match.get("elementwise")
    layout_ids = set(match.get("layout"))

    if not compute_ids:
        return None

    all_consumed = set(compute_ids)
    output_nid = match.output
    assert output_nid is not None

    external_inputs = _collect_external_inputs(graph, all_consumed)

    # Build the LoopOp's iteration axes. The pre-reduce (broadcast)
    # iteration shape is bound by the matcher for reduce matches;
    # otherwise it's the output shape of the kernel.
    pre_shape, reduce_axis = _iter_geometry(graph, match, output_nid)
    axes = _build_axes(pre_shape, reduce_axis)

    ports, remap, input_names, prelude = _build_input_ports(graph, external_inputs, all_consumed, axes)
    all_consumed |= layout_ids

    output_shape = graph.nodes[output_nid].output.shape
    locals_list, body = _build_body_from_region(graph, all_consumed, remap, output_nid, axes, output_shape)

    if not body:
        return None

    # Select statements absorbed from multi-source IndexMapOp inputs run once
    # per output coord before the region's Assigns, so downstream statements
    # can reference the Select's SSA name.
    body = prelude + body

    last_node = graph.nodes[output_nid]

    # Build fragment: InputOps for external buffers, one LoopOp node.
    frag = Graph()
    for name in input_names:
        if name not in frag.nodes:
            ext = graph.nodes.get(name)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)

    kernel = LoopOp(
        axes=axes,
        inputs=tuple(ports),
        locals=tuple(locals_list),
        body=tuple(body),
    )
    out_id = frag.add_node(kernel, input_names, Tensor(f"kernel_{output_nid}", last_node.output.shape, last_node.output.dtype))
    frag.outputs = [out_id]

    match.consumed = all_consumed
    return frag


# ---------------------------------------------------------------------------
# Iteration-space construction
# ---------------------------------------------------------------------------


def _iter_geometry(graph: Graph, match: ChainMatch, output_nid: str) -> tuple[tuple, int | None]:
    """Return (pre_reduce_shape, reduce_axis_position).

    If the match actually contains a ReduceOp, use the matcher's bound
    ``pre_shape``/``reduce_axis`` (which describe the pre-reduce broadcast
    iteration). Otherwise (pointwise-only region), pre_shape = output shape
    and reduce_axis = None.
    """
    reduce_ids = match.get("reduce") if hasattr(match, "get") else []
    if reduce_ids:
        bindings = getattr(match, "bindings", {}) or {}
        pre_shape = bindings.get("pre_shape")
        reduce_axis = bindings.get("reduce_axis")
        if pre_shape is None:
            # Fall back to looking up the reduce's input shape directly.
            rnode = graph.nodes[reduce_ids[0]]
            pre_shape = rnode.output.shape
            # Try to derive axis from the ReduceOp itself.
            reduce_axis = getattr(rnode.op, "axis", -1)
        ra = int(reduce_axis) if reduce_axis is not None else -1
        if ra < 0:
            ra = len(pre_shape) + ra
        return tuple(pre_shape), ra

    out_shape = graph.nodes[output_nid].output.shape
    return tuple(out_shape), None


def _build_axes(pre_shape: tuple, reduce_axis: int | None) -> tuple[Axis, ...]:
    """Build the tuple of Axis objects for a LoopOp.

    Position ``i`` in ``pre_shape`` maps to ``Axis(f"a{i}", ...)``; kind
    is "reduce" iff ``i == reduce_axis``, else "free". The naming matches
    the ``out_coord_i`` placeholder convention so IndexMapOp coord_maps
    can be absorbed by substitution.
    """
    axes: list[Axis] = []
    for i, dim in enumerate(pre_shape):
        kind = "reduce" if reduce_axis is not None and i == reduce_axis else "free"
        axes.append(Axis(name=f"a{i}", extent=int(dim), kind=kind))
    return tuple(axes)


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body_from_region(
    graph: Graph,
    region: set[str],
    remap: dict[str, str],
    output_nid: str,
    axes: tuple[Axis, ...],
    output_shape: tuple,
) -> tuple[list[LocalBuffer], list[Stmt]]:
    """Lower the region's ops into (LocalBuffers, body statements).

    - ``ElementwiseOp`` nodes become ``Assign`` statements.
    - ``ReduceOp`` nodes become a ``LocalBuffer(name=nid, combine=<op>, init=<identity>)``
      plus an ``Update(target=nid, value=remap[input])`` statement. Downstream
      references to ``nid`` thus read the finalized accumulator.
    - After the final op, a ``Write(output=0, index=<free-axis Vars>, value=...)``
      statement stores the kernel's result.
    """

    def _remap_args(node: Node) -> tuple[str, ...]:
        return tuple(remap.get(inp, inp) for inp in node.inputs)

    locals_list: list[LocalBuffer] = []
    body: list[Stmt] = []

    final_ssa: str | None = None
    for nid in graph.topological_order():
        if nid not in region:
            continue
        node = graph.nodes[nid]
        if isinstance(node.op, ElementwiseOp):
            args = _remap_args(node)
            body.append(Assign(name=nid, op=node.op, args=args))
            final_ssa = nid
        elif isinstance(node.op, ReduceOp):
            fn = node.op.fn
            combine_fn = _REDUCE_COMBINE.get(fn, fn)
            identity = _REDUCE_IDENTITY.get(fn, 0.0)
            locals_list.append(
                LocalBuffer(
                    name=nid,
                    combine=ElementwiseOp(combine_fn),
                    init=Literal(identity),
                )
            )
            args = _remap_args(node)
            body.append(Update(target=nid, value=args[0]))
            final_ssa = nid

    if not body:
        return locals_list, body

    if final_ssa is None:
        final_ssa = output_nid

    write_index = _build_write_index(axes, output_shape)
    body.append(Write(output=0, index=write_index, value=final_ssa))

    return locals_list, body


def _build_write_index(axes: tuple[Axis, ...], output_shape: tuple) -> tuple[Expr, ...]:
    """Build the per-output-dim index for the kernel's final Write.

    Maps the kernel's non-size-1 free axes onto the non-size-1 positions of
    ``output_shape`` in declared order; size-1 output dims receive
    ``Literal(0, "int")``. This handles post-reduce writes (reduce axes
    never appear in the index), keep-dim reductions (size-1 dims from
    collapsed reduce axes), and unsqueeze-style absorbed size-1 dims
    uniformly.
    """
    free_axes = tuple(a for a in axes if a.kind == "free")
    non_unit_free = [a for a in free_axes if a.extent != 1]
    non_unit_out_count = sum(1 for d in output_shape if not (isinstance(d, int) and d == 1))

    if len(non_unit_free) == non_unit_out_count:
        out: list[Expr] = []
        nu_idx = 0
        for d in output_shape:
            if isinstance(d, int) and d == 1:
                out.append(Literal(0, "int"))
            else:
                out.append(Var(non_unit_free[nu_idx].name))
                nu_idx += 1
        return tuple(out)

    # Fallbacks for edge cases where the non-unit counts don't line up.
    if len(output_shape) == len(free_axes):
        return tuple(Var(a.name) for a in free_axes)
    if len(output_shape) == len(axes):
        out = []
        for a, dim in zip(axes, output_shape, strict=True):
            if isinstance(dim, int) and dim == 1 and a.extent != 1:
                out.append(Literal(0, "int"))
            else:
                out.append(Var(a.name))
        return tuple(out)
    n_extra = len(output_shape) - len(free_axes)
    out = [Literal(0, "int") for _ in range(max(0, n_extra))]
    for a in free_axes:
        out.append(Var(a.name))
    return tuple(out)


# ---------------------------------------------------------------------------
# Input-port construction
# ---------------------------------------------------------------------------


def _collect_external_inputs(graph: Graph, consumed: set[str]) -> list[str]:
    seen: list[str] = []
    for nid in graph.topological_order():
        if nid not in consumed:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            continue
        for inp in node.inputs:
            if inp not in consumed and inp not in seen:
                seen.append(inp)
    return seen


def _build_input_ports(
    graph: Graph,
    external_inputs: list[str],
    consumed: set[str],
    axes: tuple[Axis, ...],
) -> tuple[list[Port], dict[str, str], list[str], list[Stmt]]:
    """Build Ports with explicit index Exprs, plus any prelude body stmts.

    Returns ``(ports, remap, input_names, prelude)``. ``prelude`` contains
    ``Select`` statements emitted for absorbed multi-source IndexMapOps;
    callers prepend it to the kernel body so downstream Assigns can
    reference the Select's SSA name.
    """
    ports: list[Port] = []
    remap: dict[str, str] = {}
    input_names: list[str] = []
    prelude: list[Stmt] = []
    idx = 0

    for buf_id in external_inputs:
        result = _emit_port_for_external(graph, buf_id, consumed, remap, axes, idx)
        if result is None:
            continue
        port_specs, remap_val, stmts = result
        for port, input_name in port_specs:
            ports.append(port)
            input_names.append(input_name)
        remap[buf_id] = remap_val
        prelude.extend(stmts)
        idx += len(port_specs)

    return ports, remap, input_names, prelude


def _emit_port_for_external(
    graph: Graph,
    buf_id: str,
    consumed: set[str],
    remap: dict[str, str],
    axes: tuple[Axis, ...],
    start_idx: int,
) -> tuple[list[tuple[Port, str]], str, list[Stmt]] | None:
    """Build port(s) + optional prelude statements for one external input.

    Returns ``(port_specs, remap_value, prelude_stmts)`` where:
      - ``port_specs`` is a list of ``(Port, input_name)`` pairs — one entry
        for simple/single-source IndexMapOp inputs, multiple for absorbed
        multi-source IndexMapOps (cat/concat).
      - ``remap_value`` is what ``buf_id`` should map to in the body:
        ``f"${start_idx}"`` for a single port, or a ``Select`` SSA name
        when multiple ports + a ``Select`` prelude are emitted.
      - ``prelude_stmts`` is the list of ``Stmt``s to prepend to the body
        (always empty except for absorbed multi-source IndexMapOps).

    Returns ``None`` when the input has been folded into an existing $N
    reference (IndexMap chain terminating at an already-consumed node).
    """
    node = graph.nodes.get(buf_id)

    if node is not None and isinstance(node.op, IndexMapOp) and len(node.op.sources) == 1:
        # Follow IndexMapOp chain to find ultimate non-IndexMapOp source.
        chain = [buf_id]
        cur_id = node.inputs[node.op.sources[0].input_idx]
        while cur_id in graph.nodes:
            cur_node = graph.nodes[cur_id]
            if isinstance(cur_node.op, IndexMapOp) and len(cur_node.op.sources) == 1:
                chain.append(cur_id)
                cur_id = cur_node.inputs[cur_node.op.sources[0].input_idx]
            else:
                break
        if cur_id in consumed:
            # Internal chain — remap all to the consumed source's $N.
            src_ref = remap.get(cur_id, cur_id)
            for nid in chain:
                consumed.add(nid)
                remap[nid] = src_ref
            return None
        # External source — absorb this IndexMapOp into Port.index. Only mark
        # ``buf_id`` as consumed when we're its sole consumer; otherwise the
        # IndexMapOp stays in the graph for the other consumers (which will
        # get their own standalone copy kernel from ``003_wrap_indexmap``),
        # while this kernel still reads the ultimate source directly.
        if len(graph.consumers(buf_id)) == 1:
            consumed.add(buf_id)
        src_id = node.inputs[node.op.sources[0].input_idx]
        src_node = graph.nodes.get(src_id)
        src_shape = src_node.output.shape if src_node is not None else ()
        index = _indexmap_to_port_index(node.op, axes, src_shape)
        return [(Port(index=index), src_id)], f"${start_idx}", []

    if node is not None and isinstance(node.op, IndexMapOp) and len(node.op.sources) > 1:
        # Multi-source IndexMapOp (cat/concat): absorb as one Port per source
        # plus a ``Select`` prelude statement that picks the right branch.
        mapping = _build_axis_mapping(axes, node.op.out_shape)
        port_specs: list[tuple[Port, str]] = []
        branches: list[SelectBranch] = []
        for i, src in enumerate(node.op.sources):
            src_id = node.inputs[src.input_idx]
            src_node = graph.nodes.get(src_id)
            if src_node is None:
                return None  # source not in graph; bail
            src_shape = src_node.output.shape
            port_index = _source_to_port_index(src, mapping, src_shape)
            port_specs.append((Port(index=port_index), src_id))
            sel_expr: Expr = substitute(src.select, mapping) if src.select is not None else Literal(1, "int")
            branches.append(SelectBranch(value=f"${start_idx + i}", select=sel_expr))
        if len(graph.consumers(buf_id)) == 1:
            consumed.add(buf_id)
        sel_name = f"sel_{buf_id}"
        return port_specs, sel_name, [Select(name=sel_name, branches=tuple(branches))]

    # Plain port (identity load on all axes matching source buffer rank).
    src_shape = node.output.shape if node is not None else ()
    index = _identity_index_for_shape(src_shape, axes)
    return [(Port(index=index), buf_id)], f"${start_idx}", []


def _build_axis_mapping(axes: tuple[Axis, ...], out_shape: tuple) -> dict[str, Expr]:
    """Build a placeholder → axis-Var mapping for an IndexMapOp's ``out_shape``.

    Walks right-to-left, matching each non-size-1 output dim to the next
    axis with the same extent. Size-1 output dims (and unmatched ones)
    map to ``Literal(0)``.
    """
    mapping: dict[str, Expr] = {}
    axis_cursor = len(axes) - 1
    for out_i in range(len(out_shape) - 1, -1, -1):
        out_d = out_shape[out_i]
        if isinstance(out_d, int) and out_d == 1:
            mapping[f"{PLACEHOLDER_PREFIX}{out_i}"] = Literal(0, "int")
            continue
        while axis_cursor >= 0 and int(axes[axis_cursor].extent) != int(out_d):
            axis_cursor -= 1
        if axis_cursor < 0:
            mapping[f"{PLACEHOLDER_PREFIX}{out_i}"] = Literal(0, "int")
            continue
        mapping[f"{PLACEHOLDER_PREFIX}{out_i}"] = Var(axes[axis_cursor].name)
        axis_cursor -= 1
    return mapping


def _source_to_port_index(src, mapping: dict[str, Expr], src_shape: tuple) -> tuple[Expr, ...]:
    """Substitute placeholders in one ``IndexSource.coord_map`` → Port index.

    Source-buffer size-1 dims are forced to ``Literal(0)`` (clip-to-bounds,
    matching ``IndexMapOp.forward``).
    """
    out: list[Expr] = []
    for i, c in enumerate(src.coord_map):
        if i < len(src_shape) and isinstance(src_shape[i], int) and src_shape[i] == 1:
            out.append(Literal(0, "int"))
        else:
            out.append(substitute(c, mapping))
    return tuple(out)


def _indexmap_to_port_index(op: IndexMapOp, axes: tuple[Axis, ...], src_shape: tuple) -> tuple[Expr, ...]:
    """Single-source IndexMapOp → Port.index (delegates to ``_source_to_port_index``)."""
    return _source_to_port_index(op.sources[0], _build_axis_mapping(axes, op.out_shape), src_shape)


def _identity_index_for_shape(src_shape: tuple, axes: tuple[Axis, ...]) -> tuple[Expr, ...]:
    """Build an identity-like index for a buffer of ``src_shape`` under ``axes``.

    Aligns ``src_shape``'s non-size-1 dims to ``axes``'s extents by walking
    right-to-left and, for each src dim, selecting the next axis (also
    right-to-left) with matching extent. Axes whose extent doesn't match
    any remaining src dim are skipped — they may be reduce axes or extra
    iteration dims (absorbed unsqueezes). Size-1 source dims become
    ``Literal(0)`` (broadcast).
    """
    if not src_shape:
        return ()

    result: list[Expr] = [Literal(0, "int")] * len(src_shape)
    axis_cursor = len(axes) - 1  # right-to-left pointer into axes

    for i in range(len(src_shape) - 1, -1, -1):
        dim = src_shape[i]
        if isinstance(dim, int) and dim == 1:
            continue
        # Walk axes right-to-left looking for matching extent.
        while axis_cursor >= 0 and int(axes[axis_cursor].extent) != int(dim):
            axis_cursor -= 1
        if axis_cursor < 0:
            break
        result[i] = Var(axes[axis_cursor].name)
        axis_cursor -= 1
    return tuple(result)
