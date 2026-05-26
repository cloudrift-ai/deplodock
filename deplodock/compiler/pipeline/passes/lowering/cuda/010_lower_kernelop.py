"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` CUDA source string
and mutates the node's op payload in place. Grid / block geometry is
derived from the ``Tile`` in the body; ``smem_bytes`` is summed
over ``Smem`` decls. ``arg_order`` is ``kernel_op.inputs +
kernel_op.outputs`` keys — matches the kernel signature emitted by
``render_kernelop``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.cuda import CudaOp, TmaDescMeta
from deplodock.compiler.ir.cuda.ir import GridDimSpec
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import TmaDescriptor
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", KernelOp)]

_BLOCK = 256


def rewrite(match: Match, root: Node) -> Graph | None:  # noqa: ARG001 — match required by rule dispatch signature
    # Per-buffer Tensor descriptors (shape + dtype) come straight off
    # ``root.op.inputs`` / ``root.op.outputs`` — the matcher snapped them
    # to the surrounding graph's Tensors via ``populate_io``, including
    # the ``constant`` / ``value`` flags for ConstantOp predecessors.
    tensors: dict[str, Tensor] = {**root.op.inputs, **root.op.outputs}

    # Scalar ConstantOp inputs get embedded as float literals in the kernel
    # body — no kernel parameter, no buffer load.
    literal_constants: dict[str, float] = {n: float(t.value) for n, t in root.op.inputs.items() if t.constant and t.value is not None}

    runtime_inputs = tuple(b for b in root.op.inputs if b not in literal_constants)

    grid, block, runtime_args = _launch_geometry(root.op)
    # TMA descriptors are kernel parameters that come *after* the buffer
    # args (matching the signature emitted by ``render_kernelop``).
    # Collect dedup'd metadata so the backend can encode + bind them.
    desc_stmts = root.op.body.iter_of_type(TmaDescriptor)
    seen: set[str] = set()
    descriptors: list[TmaDescMeta] = []
    desc_names: list[str] = []
    for s in desc_stmts:
        if s.name in seen:
            continue
        seen.add(s.name)
        descriptors.append(
            TmaDescMeta(name=s.name, src_buf=s.src_buf, box_extents=s.box_extents, swizzle=s.swizzle),
        )
        desc_names.append(s.name)
    # Outputs receiving atomic-reduction writes (cross-CTA split-K) must be
    # zero-initialized before each launch so per-CTA partials accumulate
    # cleanly. Anything else can keep its prior contents. Helper-driven:
    # a Write is atomic iff some enclosing block axis is missing from its
    # index (see ``body.coordination``).
    escape = root.op.body.coordination
    atomic_outputs: list[str] = []
    seen_atomic: set[str] = set()
    output_set = set(root.op.outputs)
    for s in escape.writes:
        if escape.atomic_axes(s) and s.output in output_set and s.output not in seen_atomic:
            seen_atomic.add(s.output)
            atomic_outputs.append(s.output)
    return CudaOp(
        kernel_source=render_kernelop(root.op, tensors=tensors, literal_constants=literal_constants, runtime_args=runtime_args),
        kernel_name=root.op.name,
        arg_order=(*runtime_inputs, *root.op.outputs, *desc_names),
        grid=grid,
        block=block,
        smem_bytes=root.op.smem_bytes(),
        tma_descriptors=tuple(descriptors),
        zero_outputs=tuple(atomic_outputs),
        runtime_args=runtime_args,
    )


def _launch_geometry(
    kernel_op: KernelOp,
) -> tuple[tuple[GridDimSpec, GridDimSpec, GridDimSpec], tuple[GridDimSpec, GridDimSpec, GridDimSpec], tuple[str, ...]]:
    """Pick (grid_spec, block_spec, runtime_args) from the outermost tile.

    Symbolic ``Axis.extent`` values flow through as ``str`` entries in the
    grid / block factor tuples; static extents collapse to a single int
    factor (or the empty product ``(1,)``). ``runtime_args`` lists every
    symbolic axis name referenced anywhere in the body (grid / thread
    tiles AND inner serial loops over symbolic reduce axes), in
    first-seen order — these become ``int`` kernel parameters and
    arg-pack entries.

    - ``GridTile`` (cooperative): one CTA per block-axis tuple; per-CTA
      threads = product of inner ``ThreadTile``'s axes.
    - Standalone ``ThreadTile`` (pointwise): flatten all axes into a
      linear ``tid = blockIdx.x * blockDim.x + threadIdx.x`` and launch
      ``ceil(n / _BLOCK)`` CTAs of ``_BLOCK`` threads (only legal when
      that prod is fully static — block count must be known at launch
      time and we don't have a ceil-div spec yet).
    """
    seen = _collect_symbolic_axis_names(kernel_op)

    def _axes_spec(axes) -> GridDimSpec:
        from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

        factors: list = []
        for a in axes:
            if a.extent.is_static:
                v = a.extent.as_static()
                if v != 1:
                    factors.append(v)
            elif isinstance(a.extent.expr, Var):
                name = a.extent.as_atom_name()
                seen.setdefault(name, None)
                factors.append(name)
            else:
                # Composite symbolic extent (ceil-div block axis for a
                # hint-driven masked tile): carry the Expr; the launch resolver
                # evals it. Its free names become ``int`` runtime args.
                expr = a.extent.expr
                for name in expr.free_vars():
                    seen.setdefault(name, None)
                factors.append(expr)
        return tuple(factors) if factors else (1,)

    for s in kernel_op.body:
        if isinstance(s, GridTile):
            grid_spec = _axes_spec(s.axes)
            block_spec: GridDimSpec = (1,)
            for child in s.body:
                if isinstance(child, ThreadTile):
                    block_spec = _axes_spec(child.axes)
                    break
            return (grid_spec, (1,), (1,)), (block_spec, (1,), (1,)), tuple(seen)
        if isinstance(s, ThreadTile):
            from math import prod  # noqa: PLC0415

            if all(a.extent.is_static for a in s.axes):
                n_threads = max(prod(a.extent.as_static() for a in s.axes), 1)
                grid = (((n_threads + _BLOCK - 1) // _BLOCK,), (1,), (1,))
                return grid, ((_BLOCK,), (1,), (1,)), tuple(seen)
            # Symbolic flattened pointwise: grid = ceil_div(prod(extents), _BLOCK).
            # ``Dim`` arithmetic folds the static factors and carries the
            # symbolic ones; the launch resolver evals the Expr per launch.
            from deplodock.compiler.dim import Dim  # noqa: PLC0415

            nthreads = Dim(1)
            for a in s.axes:
                nthreads = nthreads * a.extent
                if not a.extent.is_static:
                    for name in a.extent.expr.free_vars():
                        seen.setdefault(name, None)
            grid_expr = ((nthreads + (_BLOCK - 1)) // _BLOCK).expr
            return ((grid_expr,), (1,), (1,)), ((_BLOCK,), (1,), (1,)), tuple(seen)
    return ((1,), (1,), (1,)), ((1,), (1,), (1,)), tuple(seen)


def _collect_symbolic_axis_names(kernel_op: KernelOp) -> dict[str, None]:
    """Walk the entire body and collect every symbolic axis name (via
    ``Axis.extent.as_atom_name()``) referenced by any tile or loop. Dict
    (not set) so insertion order matches first-seen order — kernel param
    signature is stable across runs of the same kernel."""
    from deplodock.compiler.ir.axis import Axis  # noqa: PLC0415
    from deplodock.compiler.ir.stmt.base import Stmt  # noqa: PLC0415

    seen: dict[str, None] = {}

    def visit(stmt: Stmt) -> None:
        for ax in _stmt_axes(stmt):
            if isinstance(ax, Axis) and not ax.extent.is_static:
                # Composite extents (ceil-div block axes) contribute every free
                # name (``free_vars``), not just an atomic one — each becomes a
                # runtime arg.
                for name in ax.extent.expr.free_vars():
                    seen.setdefault(name, None)
        for body in stmt.nested():
            for child in body:
                visit(child)

    for stmt in kernel_op.body:
        visit(stmt)
    return seen


def _stmt_axes(stmt) -> tuple:
    """Surface every ``Axis`` directly carried by ``stmt`` (``ParallelTile``
    axes tuple or ``Loop`` / ``SerialTile`` / ``StridedTile`` single
    axis). Excludes axes inside nested bodies — those get visited
    recursively by the caller."""
    axes = getattr(stmt, "axes", None)
    if axes is not None:
        return tuple(axes)
    axis = getattr(stmt, "axis", None)
    if axis is not None:
        return (axis,)
    return ()
