"""Tile analysis: classify a FusedRegionOp's computation pattern.

Walks the ops, identifies reduction axes, op phases, input access patterns,
and classifies the region as one of: pointwise, row_reduce,
reduce_broadcast, or contraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.ops import ContractionCore, KernelOp, ReduceOp, ReduceStage


def _is_broadcast_compatible(small_shape: tuple, large_shape: tuple) -> bool:
    """Check if small broadcasts to large via NumPy-style rules."""
    if len(small_shape) > len(large_shape):
        return False
    offset = len(large_shape) - len(small_shape)
    for i, s in enumerate(small_shape):
        large_dim = large_shape[offset + i]
        if not isinstance(s, int) or not isinstance(large_dim, int):
            continue
        if s != 1 and s != large_dim:
            return False
    return True


# Type alias for region op tuples: (node_id, op, input_ids)
RegionEntry = tuple[str, object, list[str]]


@dataclass
class OpPhases:
    """Ops split into prologue (before first reduce), reduces, epilogue (after last reduce).

    For multi-reduce patterns (e.g. softmax max+sum), inter_reduce[i] holds the
    ops between reduces[i] and reduces[i+1].
    """

    prologue: list[RegionEntry]
    reduces: list[RegionEntry]
    epilogue: list[RegionEntry]
    inter_reduce: list[list[RegionEntry]] = field(default_factory=list)


@dataclass
class AccessPattern:
    """How a single input tensor is accessed within the kernel."""

    shape: tuple[int, ...]
    size: int  # total elements
    is_scalar: bool  # size == 1 (broadcast everywhere)
    is_row_vector: bool  # 1D, indexed by column only
    is_2d: bool  # indexed by both row and column
    is_per_row: bool = False  # last dim == 1, indexed by row only (e.g., (N,1) or (1,28,32,1))
    is_broadcast: bool = False  # smaller input that broadcasts to output (indexed via modulo)


@dataclass
class TileAnalysis:
    """Analysis result for a FusedRegionOp.

    Captures everything needed to choose a tiling strategy and generate
    the kernel, without re-walking the ops.
    """

    pattern: str  # "pointwise" | "row_reduce" | "reduce_broadcast" | "contraction"
    op_phases: OpPhases
    output_shape: tuple[int, ...]
    reduce_fns: list[str]  # ["sum"], ["max", "sum"], etc.
    input_access: dict[str, AccessPattern]  # per external input
    # Dimensions (concrete ints, derived from shapes).
    rows: int  # product of non-reduced dims (M for contraction, rows for reduce)
    cols: int  # last dim of the pre-reduction tensor (N for contraction, cols for reduce)
    k_dim: int  # shared/reduced dimension (K for contraction, same as cols for reduce)
    # For contraction only: names of the two matmul operands.
    contraction_a: str | None = None
    contraction_b: str | None = None
    # Whether the epilogue needs a second per-element pass over inputs.
    epilogue_needs_per_element: bool = False
    # Batch dimensions for batched contractions (e.g. multi-head attention).
    batch_dims: tuple[int, ...] = ()
    batch_size: int = 1
    # GQA / broadcast batch: when one operand has fewer batch elements,
    # its batch index is divided by this factor.  E.g. 28 Q heads / 4 KV heads = 7.
    # "b_batch_group" means B's batch index = batch // b_batch_group.
    # 1 means both operands use the same batch index (no broadcast).
    a_batch_group: int = 1
    b_batch_group: int = 1
    # Per-input indexmaps carried on Port.indexmap. When set, the load path
    # substitutes the placeholder coord_map with the kernel's runtime
    # indices to build the actual input address (transpose-into-matmul).
    port_indexmaps: dict = field(default_factory=dict)


def flat_region_ops(kernel: KernelOp) -> list:
    """Walk kernel body nodes in topo order, return (id, op, inputs) tuples.

    Replaces the old ``KernelOp.region_ops`` compat property: dedups across
    prologue, core (ContractionCore.mul/reduce + post_stages, or
    tuple[ReduceStage] pre_ops/reduce), and epilogue by node id.
    """
    seen: set[str] = set()
    out: list = []

    def emit(node) -> None:
        if node is None or node.id in seen:
            return
        seen.add(node.id)
        out.append((node.id, node.op, list(node.inputs)))

    for n in kernel.prologue:
        emit(n)
    if isinstance(kernel.core, ContractionCore):
        emit(kernel.core.mul)
        emit(kernel.core.reduce)
        for stage in kernel.core.post_stages:
            if not isinstance(stage, ReduceStage):
                continue
            for pre in stage.pre_ops:
                emit(pre)
            emit(stage.reduce)
    elif isinstance(kernel.core, tuple):
        for stage in kernel.core:
            if not isinstance(stage, ReduceStage):
                continue
            for pre in stage.pre_ops:
                emit(pre)
            emit(stage.reduce)
    for n in kernel.epilogue:
        emit(n)
    return out


def analyze(region: KernelOp, shapes: dict[str, tuple]) -> TileAnalysis:
    """Analyze a FusedRegionOp and classify its computation pattern.

    Args:
        region: The fused region containing primitive ops in topo order.
        shapes: Map of node_id/buffer_name -> shape tuple.

    Returns:
        TileAnalysis with pattern classification and metadata.
    """
    # Split ops into phases directly from structured kernel fields.
    phases = _split_phases(region)

    # Determine output shape.
    out_id = [p.buffer_id for p in region.outputs][0]
    out_shape = shapes.get(out_id, (1,))

    # Collect reduce function names.
    reduce_fns = [op.fn for _, op, _ in phases.reduces]

    # Collect per-input indexmaps (set by 070_absorb_indexmap_into_port when
    # a pure-view IndexMap absorbs into a kernel's Port). An empty dict
    # means no absorption happened and loads use the natural indices.
    port_indexmaps: dict = {p.buffer_id: p.indexmap for p in region.inputs if p.indexmap is not None}

    # Build input access patterns.
    input_access = {}
    for inp in [p.buffer_id for p in region.inputs]:
        inp_shape = shapes.get(inp, (1,))
        inp_size = math.prod(d for d in inp_shape if isinstance(d, int))
        has_symbolic = any(isinstance(d, str) for d in inp_shape)
        # Per-row scalar: last dim is 1 with >1 total elements.
        # Covers both (N, 1) and (1, 28, 32, 1) — indexed by row only.
        last_dim = inp_shape[-1] if inp_shape else 1
        last_dim_is_one = isinstance(last_dim, int) and last_dim == 1
        is_per_row = last_dim_is_one and inp_size > 1 and len(inp_shape) >= 2
        out_size = math.prod(d for d in out_shape if isinstance(d, int))
        is_broadcast = (
            inp_size > 1
            and inp_size < out_size
            and not is_per_row
            and len(inp_shape) >= 2
            and _is_broadcast_compatible(inp_shape, out_shape)
        )
        input_access[inp] = AccessPattern(
            shape=inp_shape,
            size=inp_size,
            is_scalar=(inp_size == 1 and not has_symbolic),
            is_row_vector=(len(inp_shape) == 1 and (inp_size > 1 or has_symbolic)),
            is_2d=(len(inp_shape) >= 2 and (inp_size > 1 or has_symbolic) and not is_per_row and not is_broadcast),
            is_per_row=is_per_row,
            is_broadcast=is_broadcast,
        )

    # No reduces → pointwise.
    if not phases.reduces:
        total = math.prod(d for d in out_shape if isinstance(d, int))
        return TileAnalysis(
            port_indexmaps=port_indexmaps,
            pattern="pointwise",
            op_phases=phases,
            output_shape=out_shape,
            reduce_fns=[],
            input_access=input_access,
            rows=1,
            cols=total,
            k_dim=0,
        )

    # Has reduces — determine the pre-reduction tensor shape.
    first_reduce_input = phases.reduces[0][2][0]  # first reduce's first input_id
    pre_shape = shapes.get(first_reduce_input, out_shape)

    # Check for contraction pattern: exactly 2 ops (mul + sum), two 2D inputs
    # sharing a dimension that gets reduced, producing a 2D output.
    is_contraction, a_name, b_name, m, n, k, batch_dims, batch_size, a_bg, b_bg = _detect_contraction(region, phases, shapes, input_access)

    if is_contraction:
        epilogue_per_elem = _epilogue_needs_per_element(region, phases, shapes, input_access)
        return TileAnalysis(
            port_indexmaps=port_indexmaps,
            pattern="contraction",
            op_phases=phases,
            output_shape=out_shape,
            reduce_fns=reduce_fns,
            input_access=input_access,
            rows=m,
            cols=n,
            k_dim=k,
            contraction_a=a_name,
            contraction_b=b_name,
            epilogue_needs_per_element=epilogue_per_elem,
            batch_dims=batch_dims,
            batch_size=batch_size,
            a_batch_group=a_bg,
            b_batch_group=b_bg,
        )

    # Row reduction patterns — extract rows/cols from pre-reduction shape.
    if len(pre_shape) >= 2:
        rows = math.prod(d for d in pre_shape[:-1] if isinstance(d, int))
        cols = pre_shape[-1] if isinstance(pre_shape[-1], int) else 1
    else:
        rows = 1
        cols = math.prod(d for d in pre_shape if isinstance(d, int))

    epilogue_per_elem = _epilogue_needs_per_element(region, phases, shapes, input_access)

    if epilogue_per_elem:
        pattern = "reduce_broadcast"
    else:
        pattern = "row_reduce"

    return TileAnalysis(
        port_indexmaps=port_indexmaps,
        pattern=pattern,
        op_phases=phases,
        output_shape=out_shape,
        reduce_fns=reduce_fns,
        input_access=input_access,
        rows=rows,
        cols=cols,
        k_dim=cols,  # for row reductions, k == cols
        epilogue_needs_per_element=epilogue_per_elem,
    )


def _node_entry(n) -> RegionEntry:
    """Convert a Node to a (id, op, inputs) tuple."""
    return (n.id, n.op, list(n.inputs))


def _split_phases(kernel: KernelOp) -> OpPhases:
    """Derive OpPhases directly from a KernelOp's structured fields.

    No scanning: each ``core`` variant maps deterministically to the four
    phases (prologue, reduces, inter_reduce, epilogue).

    - ``ContractionCore``: prologue + [mul] feed [reduce]; epilogue follows.
    - ``tuple[ReduceStage, ...]`` (ReduceCore): each stage contributes one
      reduce and its pre_ops as inter_reduce[i-1]. Nodes between the last
      stage's reduce and the end of prologue are the row-reduce epilogue.
    - ``None``: pure pointwise — all nodes in prologue.
    """
    if isinstance(kernel.core, ContractionCore):
        # Walk kernel.prologue in order. Pre-reduce ops feed the K-loop
        # (phases.prologue). After the contraction reduce, ops are either
        # inter_reduce pre-chains (between stages in post_stages) or the
        # per-row epilogue. post_stages defines which reduces are part of
        # the downstream chain and their pre_ops.
        mul_id = kernel.core.mul.id if kernel.core.mul is not None else None
        reduce_id = kernel.core.reduce.id if kernel.core.reduce is not None else None
        post_reduce_ids = {s.reduce.id for s in kernel.core.post_stages if isinstance(s, ReduceStage) and s.reduce is not None}

        prologue: list[RegionEntry] = []
        seen_reduce = False
        tail_nodes: list = []  # nodes after contraction reduce (to be split below)
        for n in kernel.prologue:
            if n.id == mul_id or n.id == reduce_id:
                if n.id == reduce_id:
                    seen_reduce = True
                continue
            if seen_reduce:
                tail_nodes.append(n)
            else:
                prologue.append(_node_entry(n))
        if kernel.core.mul is not None:
            prologue.append(_node_entry(kernel.core.mul))

        reduces: list[RegionEntry] = []
        if kernel.core.reduce is not None:
            reduces.append(_node_entry(kernel.core.reduce))

        inter_reduce: list[list[RegionEntry]] = []
        current_inter: list[RegionEntry] = []
        for n in tail_nodes:
            entry = _node_entry(n)
            if n.id in post_reduce_ids:
                inter_reduce.append(current_inter)
                current_inter = []
                reduces.append(entry)
            else:
                current_inter.append(entry)
        # Leftover current_inter = per-row epilogue (nothing consumes it
        # as a reduce input).
        epilogue = current_inter + [_node_entry(n) for n in kernel.epilogue]
        return OpPhases(prologue=prologue, reduces=reduces, epilogue=epilogue, inter_reduce=inter_reduce)

    if isinstance(kernel.core, tuple) and kernel.core:
        # ReduceCore: the reduce rule leaves Nodes in kernel.prologue and
        # annotates boundaries via stages. Find stage reduce positions.
        stages = kernel.core
        reduce_node_ids = {s.reduce.id for s in stages}
        # First stage's reduce marks end-of-prologue.
        first_reduce_id = stages[0].reduce.id
        prologue: list[RegionEntry] = []
        seen_first_reduce = False
        tail: list = []  # nodes in kernel.prologue after last reduce
        last_reduce_id = stages[-1].reduce.id
        seen_last_reduce = False
        for n in kernel.prologue:
            if n.id == first_reduce_id:
                seen_first_reduce = True
                if n.id == last_reduce_id:
                    seen_last_reduce = True
                continue
            if not seen_first_reduce:
                prologue.append(_node_entry(n))
                continue
            if n.id in reduce_node_ids:
                if n.id == last_reduce_id:
                    seen_last_reduce = True
                continue
            if seen_last_reduce:
                tail.append(_node_entry(n))
        # Reduces + inter-reduce chains come straight from the stages.
        reduces_entries = [_node_entry(s.reduce) for s in stages]
        inter_reduce: list[list[RegionEntry]] = [[_node_entry(pn) for pn in s.pre_ops] for s in stages[1:]]
        epilogue = tail + [_node_entry(n) for n in kernel.epilogue]
        return OpPhases(prologue=prologue, reduces=reduces_entries, epilogue=epilogue, inter_reduce=inter_reduce)

    # core is None — unstructured. Scan kernel.prologue + epilogue to find
    # reduce boundaries, matching the legacy behavior for kernels that
    # haven't been classified by the fusion rules (e.g. tests that construct
    # KernelOps by hand).
    prologue_out: list[RegionEntry] = []
    reduces_out: list[RegionEntry] = []
    inter_reduce_out: list[list[RegionEntry]] = []
    epilogue_out: list[RegionEntry] = []
    current_inter: list[RegionEntry] = []
    phase = "prologue"
    flat_nodes = list(kernel.prologue) + list(kernel.epilogue)
    for n in flat_nodes:
        entry = _node_entry(n)
        if isinstance(n.op, ReduceOp):
            if phase == "epilogue":
                inter_reduce_out.append(current_inter)
                current_inter = []
            reduces_out.append(entry)
            phase = "epilogue"
        elif phase == "prologue":
            prologue_out.append(entry)
        else:
            current_inter.append(entry)
    epilogue_out = current_inter
    return OpPhases(prologue=prologue_out, reduces=reduces_out, epilogue=epilogue_out, inter_reduce=inter_reduce_out)


def _needed_by(ops: list) -> set[str]:
    """Return set of node_ids referenced as inputs by the given ops."""
    needed = set()
    for _, _, input_ids in ops:
        needed.update(input_ids)
    return needed


def _detect_contraction(
    region: KernelOp,
    phases: OpPhases,
    shapes: dict[str, tuple],
    input_access: dict[str, AccessPattern],
) -> tuple[bool, str | None, str | None, int, int, int, tuple[int, ...], int]:
    """Determine matmul metadata from a KernelOp's structured ``core``.

    If ``kernel.core`` is a ``ContractionCore``, read a/b buffer IDs from
    its Ports and derive M, N, K, batch_dims, batch_size (plus GQA batch
    groups) from the Port shapes — no scanning. Otherwise not a contraction.

    Returns: (is_contraction, a_name, b_name, M, N, K, batch_dims, batch_size,
              a_batch_group, b_batch_group).
    """
    if not isinstance(region.core, ContractionCore):
        return False, None, None, 0, 0, 0, (), 1, 1, 1
    a_id = region.core.a.buffer_id
    b_id = region.core.b.buffer_id

    if a_id not in input_access or b_id not in input_access:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    a_acc = input_access[a_id]
    b_acc = input_access[b_id]
    if not a_acc.is_2d or not b_acc.is_2d:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    a_shape = a_acc.shape
    b_shape = b_acc.shape
    if len(a_shape) < 2 or len(b_shape) < 2:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    # Detect batch dimensions: leading dims that match or broadcast between A and B.
    batch_dims: tuple[int, ...] = ()
    batch_size = 1
    a_batch_group = 1
    b_batch_group = 1
    if len(a_shape) > 2 and len(b_shape) > 2:
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        if a_batch == b_batch:
            batch_dims = a_batch
        else:
            # Broadcast batch dims (e.g. GQA: 28 Q heads vs 4 KV heads).
            # Pad the shorter batch tuple with leading 1s so both have the same ndim,
            # then check each dim matches or one divides the other.
            max_len = max(len(a_batch), len(b_batch))
            a_padded = (1,) * (max_len - len(a_batch)) + a_batch
            b_padded = (1,) * (max_len - len(b_batch)) + b_batch
            merged_batch: list[int] = []
            for ad, bd in zip(a_padded, b_padded, strict=True):
                if not isinstance(ad, int) or not isinstance(bd, int):
                    return False, None, None, 0, 0, 0, (), 1, 1, 1
                if ad == bd:
                    merged_batch.append(ad)
                elif ad > bd and bd > 0 and ad % bd == 0:
                    merged_batch.append(ad)
                elif bd > ad and ad > 0 and bd % ad == 0:
                    merged_batch.append(bd)
                else:
                    return False, None, None, 0, 0, 0, (), 1, 1, 1
            batch_dims = tuple(merged_batch)
            a_batch_size = math.prod(d for d in a_padded if isinstance(d, int))
            b_batch_size = math.prod(d for d in b_padded if isinstance(d, int))
            if a_batch_size >= b_batch_size and b_batch_size > 0:
                b_batch_group = a_batch_size // b_batch_size
            elif a_batch_size > 0:
                a_batch_group = b_batch_size // a_batch_size
        batch_size = math.prod(d for d in batch_dims if isinstance(d, int))
        # Batched: A(B..., M, K) @ B(B..., K, N)
        a_k = a_shape[-1]
        b_k = b_shape[-2]
    else:
        # 2D: A(M, K) @ B(K, N)
        a_k = a_shape[-1]
        b_k = b_shape[0]

    # K dimension must match (both int or both same symbolic string).
    if a_k != b_k:
        return False, None, None, 0, 0, 0, (), 1, 1, 1

    m = a_shape[-2] if isinstance(a_shape[-2], int) else 1
    if not batch_dims:
        # 2D: M = product of all dims except last
        m = math.prod(d for d in a_shape[:-1] if isinstance(d, int)) if any(isinstance(d, int) for d in a_shape[:-1]) else 1
    k = a_k if isinstance(a_k, int) else 1
    n = b_shape[-1] if isinstance(b_shape[-1], int) else 1

    return True, a_id, b_id, m, n, k, batch_dims, batch_size, a_batch_group, b_batch_group


def _epilogue_needs_per_element(
    region: KernelOp,
    phases: OpPhases,
    shapes: dict[str, tuple],
    input_access: dict[str, AccessPattern],
) -> bool:
    """Check if the epilogue requires a second per-element pass.

    This is true when epilogue ops (or prologue ops they depend on) need
    to read per-element values from 2D inputs — e.g., rmsnorm epilogue
    needs the original x values to multiply by the normalization factor.
    """
    if not phases.epilogue:
        return False

    epilogue_needs = _needed_by(phases.epilogue) | _needed_by(
        phases.prologue if any(node_id in _needed_by(phases.epilogue) for node_id, _, _ in phases.prologue) else []
    )

    # Check if any external input needed by epilogue requires per-element
    # access (is_2d). Per-row and scalar inputs are available during the
    # reduce pass and don't need a second per-element loop.
    for inp in [p.buffer_id for p in region.inputs]:
        if inp in epilogue_needs:
            acc = input_access.get(inp)
            if acc and acc.is_2d:
                return True

    # Check if any prologue op needed by epilogue (transitive 2D dependency).
    for node_id, _, _ in phases.prologue:
        if node_id in _needed_by(phases.epilogue):
            return True

    return False
