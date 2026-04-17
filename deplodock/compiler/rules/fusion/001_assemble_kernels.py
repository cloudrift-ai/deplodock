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
from deplodock.compiler.ir.loop import Assign, Axis, LocalBuffer, LoopOp, Port, Stmt, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production

# Identity values for reductions — lifted from the former REDUCE_REGISTRY.
_REDUCE_IDENTITY: dict[str, float] = {"sum": 0.0, "max": -1e30, "prod": 1.0, "min": 1e30}
# Combine op name for each reduction family.
_REDUCE_COMBINE: dict[str, str] = {"sum": "add", "max": "max", "prod": "mul", "min": "min"}


def _same_rank(op, node, graph):
    """IndexMapOp predicate: pass through single-source same-rank IndexMapOps.

    Multi-source IndexMapOps (cat) have Mux-like semantics that can't be
    folded into a simple Port.index — they need their own kernel.
    """
    if len(op.sources) > 1:
        return False
    in_shape = graph.nodes[node.inputs[0]].output.shape
    return len(op.out_shape) <= len(in_shape)


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

    ports, remap, input_names = _build_input_ports(graph, external_inputs, all_consumed, axes)
    all_consumed |= layout_ids

    output_shape = graph.nodes[output_nid].output.shape
    locals_list, body = _build_body_from_region(graph, all_consumed, remap, output_nid, axes, output_shape)

    if not body:
        return None

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

    When the output shape matches the full pre-reduce rank, each dim is:
    - ``Var(axis.name)`` if the output keeps that axis's full extent
      (softmax-like per-element output that varies with the reduce axis), or
    - ``Literal(0, "int")`` if the output has size 1 at that position
      (keepdim-collapsed reduce dim).

    For lower-rank outputs (drop-axis reduces), use only free-axis Vars.
    """
    free_axes = tuple(a for a in axes if a.kind == "free")
    if len(output_shape) == len(free_axes):
        return tuple(Var(a.name) for a in free_axes)
    if len(output_shape) == len(axes):
        out: list[Expr] = []
        for a, dim in zip(axes, output_shape, strict=True):
            if isinstance(dim, int) and dim == 1 and a.extent != 1:
                out.append(Literal(0, "int"))
            else:
                out.append(Var(a.name))
        return tuple(out)
    # Fallback: right-align free axes onto the output dims, pad leading with 0.
    n_extra = len(output_shape) - len(free_axes)
    out = []
    for _ in range(n_extra):
        out.append(Literal(0, "int"))
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
) -> tuple[list[Port], dict[str, str], list[str]]:
    """Build Ports with explicit index Exprs and a remap dict.

    Returns (ports, remap, input_names).
    """
    ports: list[Port] = []
    remap: dict[str, str] = {}
    input_names: list[str] = []
    idx = 0

    for buf_id in external_inputs:
        new_port = _emit_port_for_external(graph, buf_id, consumed, remap, axes)
        if new_port is None:
            continue
        port, input_name = new_port
        ports.append(port)
        input_names.append(input_name)
        remap[buf_id] = f"${idx}"
        idx += 1

    return ports, remap, input_names


def _emit_port_for_external(
    graph: Graph,
    buf_id: str,
    consumed: set[str],
    remap: dict[str, str],
    axes: tuple[Axis, ...],
) -> tuple[Port, str] | None:
    """Build one Port for a single external input.

    Returns ``(port, input_name)`` where ``port.index`` is a tuple of
    Exprs over axis Vars (one Expr per source-buffer dim) and ``input_name``
    is the buffer name the Port binds to at the program level.

    Returns ``None`` when the input has been folded into an internal
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
        # External source — absorb this IndexMapOp into Port.index.
        if len(graph.consumers(buf_id)) == 1:
            consumed.add(buf_id)
            src_id = node.inputs[node.op.sources[0].input_idx]
            src_node = graph.nodes.get(src_id)
            src_shape = src_node.output.shape if src_node is not None else ()
            index = _indexmap_to_port_index(node.op, axes, src_shape)
            return Port(index=index), src_id

    # Plain port (identity load on all axes matching source buffer rank).
    src_shape = node.output.shape if node is not None else ()
    index = _identity_index_for_shape(src_shape, axes)
    return Port(index=index), buf_id


def _indexmap_to_port_index(op: IndexMapOp, axes: tuple[Axis, ...], src_shape: tuple) -> tuple[Expr, ...]:
    """Translate an IndexMapOp's coord_map (single source) into a Port.index.

    The coord_map uses ``Var("out_coord_i")`` placeholders referring to
    position ``i`` in ``op.out_shape``. We map those positions onto the
    kernel's axis space:

    - If ``len(op.out_shape) == len(axes)``: pre-reduce (broadcast) shape
      — placeholder(i) → axes[i].
    - If ``len(op.out_shape) == len(free_axes)``: post-reduce shape —
      placeholder(i) → free_axes[i].
    - Otherwise: right-align onto the full axes (pad with zeros for
      missing leading positions).

    For source-buffer dims of size 1 (broadcast semantics), the index is
    forced to ``Literal(0, "int")`` regardless of what the coord_map
    produces — matching the clip-to-in-bounds behavior of
    ``IndexMapOp.forward``.
    """
    src = op.sources[0]
    free_axes = tuple(a for a in axes if a.kind == "free")
    out_rank = len(op.out_shape)

    mapping: dict[str, Expr] = {}
    if out_rank == len(axes):
        for i, a in enumerate(axes):
            mapping[f"{PLACEHOLDER_PREFIX}{i}"] = Var(a.name)
    elif out_rank == len(free_axes):
        for i, a in enumerate(free_axes):
            mapping[f"{PLACEHOLDER_PREFIX}{i}"] = Var(a.name)
    else:
        # Right-align: the last out_rank axes map onto placeholders 0..out_rank-1.
        tail = axes[-out_rank:] if out_rank > 0 else ()
        for i, a in enumerate(tail):
            mapping[f"{PLACEHOLDER_PREFIX}{i}"] = Var(a.name)

    out: list[Expr] = []
    for i, c in enumerate(src.coord_map):
        if i < len(src_shape) and isinstance(src_shape[i], int) and src_shape[i] == 1:
            out.append(Literal(0, "int"))
        else:
            out.append(substitute(c, mapping))
    return tuple(out)


def _identity_index_for_shape(src_shape: tuple, axes: tuple[Axis, ...]) -> tuple[Expr, ...]:
    """Build an identity-like index for a buffer of ``src_shape`` under ``axes``.

    Two alignment options, picked by rank match:

    - ``len(src_shape) == len(axes)``: the buffer matches the full
      pre-reduce iteration space. Map dim i → axes[i].
    - ``len(src_shape) == len(free_axes)``: the buffer matches the
      post-reduce output shape. Map dim i → free_axes[i] (skips reduce
      axis).
    - Otherwise: right-align onto the full axes tuple (padding at the
      left); size-1 broadcast dims get ``Literal(0, "int")``.
    """
    if not src_shape:
        return ()
    free_axes = tuple(a for a in axes if a.kind == "free")
    rank = len(src_shape)
    n_axes = len(axes)

    def _build_index(target_axes: tuple[Axis, ...]) -> tuple[Expr, ...]:
        out: list[Expr] = []
        for i, dim in enumerate(src_shape):
            axis = target_axes[i]
            if isinstance(dim, int) and dim == 1 and axis.extent != 1:
                out.append(Literal(0, "int"))
            else:
                out.append(Var(axis.name))
        return tuple(out)

    if rank == n_axes:
        return _build_index(axes)
    if rank == len(free_axes):
        return _build_index(free_axes)
    if rank > n_axes:
        index: list[Expr] = []
        for i in range(rank):
            if i < n_axes:
                index.append(Var(axes[i].name))
            else:
                index.append(Literal(0, "int"))
        return tuple(index)
    pad = n_axes - rank
    out: list[Expr] = []
    for i, dim in enumerate(src_shape):
        axis = axes[pad + i]
        if isinstance(dim, int) and dim == 1 and axis.extent != 1:
            out.append(Literal(0, "int"))
        else:
            out.append(Var(axis.name))
    return tuple(out)
