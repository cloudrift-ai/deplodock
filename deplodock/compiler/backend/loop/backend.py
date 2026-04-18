"""Loop backend: numpy interpreter for a ``LoopProgram``.

Walks each ``LoopLaunch`` in topological order, evaluating its ``LoopOp``
as whole-tensor numpy operations. Body statements dispatch on type:

- ``Assign``: pure computation via ``ElementwiseOp.forward``.
- ``Update``: fold a value into a ``LocalBuffer`` accumulator; done as a
  numpy reduction (``np.sum``/``np.max``/etc.) over the reduce axis since
  this is whole-tensor evaluation, not coord-by-coord.
- ``Write``: stash an SSA value into the output buffer at the computed
  position.
- ``Select``: coord-predicated merge via ``np.where``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, ProgramResult
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Assign, Axis, LocalBuffer, LoopOp, Port, Select, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.loop import LoopLaunch, LoopProgram

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph


class LoopBackend(Backend):
    """Execute a ``LoopProgram`` via numpy whole-tensor operations."""

    def compile(self, graph: Graph) -> LoopProgram:
        return compile_graph(graph)

    def run(self, compiled: LoopProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        t0 = time.perf_counter()
        arrays = _execute(compiled, input_data or {})
        elapsed = (time.perf_counter() - t0) * 1000
        return ProgramResult(outputs=arrays, time_ms=elapsed)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def _concrete_shape(shape: tuple) -> tuple[int, ...]:
    return tuple(int(d) for d in shape)


def _execute(program: LoopProgram, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Walk the LoopProgram, returning {graph_output_name: ndarray}."""
    buffers: dict[str, np.ndarray] = {}

    for b in program.buffers:
        if b.role == "input":
            if b.name not in input_data:
                raise KeyError(f"Missing input {b.name!r} for LoopProgram")
            buffers[b.name] = np.asarray(input_data[b.name], dtype=np.float32).reshape(_concrete_shape(b.shape))
        elif b.role == "constant":
            if b.name in input_data:
                buffers[b.name] = np.asarray(input_data[b.name], dtype=np.float32).reshape(_concrete_shape(b.shape))
            elif b.name in program.constant_values:
                buffers[b.name] = np.array([program.constant_values[b.name]], dtype=np.float32)

    for launch in program.launches:
        buffers[launch.output_name] = _exec_launch(launch, program, buffers)

    return {name: buffers[name] for name in program.graph_outputs}


def _exec_launch(launch: LoopLaunch, program: LoopProgram, buffers: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate one launch. Returns ndarray with shape == program.output_shape(launch)."""
    out_shape = _concrete_shape(program.output_shape(launch))

    # Non-LoopOp launches (ops fusion didn't wrap) go through Op.forward.
    if not isinstance(launch.loop, LoopOp):
        args = [buffers[n] for n in launch.input_names]
        result = launch.loop.forward(*args)
        return np.asarray(result, dtype=np.float32).reshape(out_shape)

    input_arrays = [buffers[n] for n in launch.input_names]
    return execute_loop_op(launch.loop, input_arrays, out_shape)


def execute_loop_op(
    loop: LoopOp,
    input_arrays: list[np.ndarray],
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """Numpy interpreter for one ``LoopOp``.

    ``input_arrays[i]`` is the external buffer that feeds ``loop.inputs[i]``
    (the ``$i`` Port). ``out_shape`` is the shape of the output buffer the
    kernel writes into. Returns an ndarray of shape ``out_shape``.

    Shared between ``LoopBackend`` (which resolves buffers from a
    ``LoopProgram``) and ``LoopOp.forward`` (which is called directly from
    the graph-level numpy walker).
    """
    dollar = _bind_inputs(loop, input_arrays)

    reduce_axis_positions = tuple(i for i, a in enumerate(loop.axes) if a.kind == "reduce")
    values: dict[str, np.ndarray] = dict(dollar)
    lb_map: dict[str, LocalBuffer] = {lb.name: lb for lb in loop.locals}
    for lb in loop.locals:
        init_val = lb.init.eval({}) if lb.init is not None else 0.0
        values[lb.name] = np.asarray(init_val, dtype=np.float32)

    output_array = np.zeros(out_shape, dtype=np.float32)
    output_written = False

    for stmt in loop.body:
        if isinstance(stmt, Assign):
            args = [values[a] for a in stmt.args]
            assert isinstance(stmt.op, ElementwiseOp)
            values[stmt.name] = stmt.op.forward(*_align_ranks(args))
        elif isinstance(stmt, Update):
            lb = lb_map[stmt.target]
            src = values[stmt.value]
            reduced = _fold_to_accumulator(src, lb, reduce_axis_positions)
            values[stmt.target] = reduced
        elif isinstance(stmt, Select):
            axis_env = _broadcast_axis_env(loop.axes)
            branch_vals = [values[b.value] for b in stmt.branches]
            # Walk branches in reverse: the last branch is the catch-all, earlier
            # branches override where their predicate holds. Let numpy broadcasting
            # compute the natural output shape (only the axes that the predicates
            # and branch values actually depend on) — don't force the full iter
            # shape, so row-space Selects stay row-shaped.
            result: np.ndarray | None = None
            for b, val in zip(reversed(stmt.branches), reversed(branch_vals), strict=True):
                mask = b.select.eval(axis_env) if b.select is not None else True
                val_arr = np.asarray(val, dtype=np.float32)
                if result is None:
                    if isinstance(mask, np.ndarray):
                        result = np.broadcast_to(val_arr, np.broadcast_shapes(val_arr.shape, mask.shape)).astype(np.float32, copy=True)
                    else:
                        result = val_arr.astype(np.float32, copy=True)
                else:
                    result = np.where(mask, val_arr, result).astype(np.float32, copy=False)
            values[stmt.name] = result
        elif isinstance(stmt, Write):
            val = values[stmt.value]
            _write_output(output_array, out_shape, stmt, val, loop.axes)
            output_written = True

    if output_written:
        return output_array.astype(np.float32).reshape(out_shape)

    # Fallback: no explicit Write (shouldn't happen post-commit-3 but kept for safety).
    if loop.body:
        for stmt in reversed(loop.body):
            if isinstance(stmt, Assign):
                arr = np.asarray(values[stmt.name], dtype=np.float32)
                while arr.ndim > len(out_shape):
                    size1 = [i for i in range(arr.ndim) if arr.shape[i] == 1]
                    if not size1:
                        break
                    arr = np.squeeze(arr, axis=size1[-1])
                return arr.reshape(out_shape)
    if dollar:
        arr = np.asarray(next(iter(dollar.values())), dtype=np.float32)
        return arr.reshape(out_shape)
    raise ValueError("LoopOp produced no output")


def _fold_to_accumulator(src: np.ndarray, lb: LocalBuffer, reduce_axis_positions: tuple[int, ...]) -> np.ndarray:
    """Reduce ``src`` along the reduce axes using the LocalBuffer's combine op.

    Returns a ndarray of the reduced shape (keepdims=True preserved for
    downstream broadcast compatibility).
    """
    if not reduce_axis_positions:
        return np.asarray(src, dtype=np.float32)
    combine = lb.combine
    axes = tuple(reduce_axis_positions)
    if combine is None:
        return np.asarray(src, dtype=np.float32)
    fn = combine.fn
    if fn == "add":
        return np.sum(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "max":
        return np.max(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "min":
        return np.min(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "mul":
        return np.prod(src, axis=axes, keepdims=True).astype(np.float32)
    raise NotImplementedError(f"_fold_to_accumulator: unknown combine fn {fn!r}")


def _iter_shape(axes: tuple[Axis, ...]) -> tuple[int, ...]:
    return tuple(int(a.extent) for a in axes)


def _broadcast_axis_env(axes: tuple[Axis, ...]) -> dict[str, object]:
    """Return axis-name → broadcast-shaped arange array."""
    env: dict[str, object] = {}
    for i, a in enumerate(axes):
        shape = [1] * len(axes)
        shape[i] = int(a.extent)
        env[a.name] = np.arange(int(a.extent)).reshape(shape)
    return env


def _write_output(
    output_array: np.ndarray,
    out_shape: tuple[int, ...],
    write: Write,
    value: np.ndarray,
    axes: tuple[Axis, ...],
) -> None:
    """Apply ``write`` to ``output_array``.

    The write's index Exprs may reference any axis in the LoopOp (free or
    reduce); when reduce axes appear, the write iterates over the full
    pre-reduce space (e.g. softmax's per-element final division). The
    effective write shape is the per-axis extent for each referenced axis.
    """
    axis_names = {a.name for a in axes}
    used: set[str] = set()
    for e in write.index:
        _collect_used_axis_names(e, axis_names, used)

    # Per-axis broadcast shape: extent if referenced, else 1.
    wshape = tuple(int(a.extent) if a.name in used else 1 for a in axes)

    # Axis env: referenced axes get arange arrays; others get 0.
    env: dict[str, object] = {}
    for i, a in enumerate(axes):
        if a.name in used:
            shape = [1] * len(axes)
            shape[i] = int(a.extent)
            env[a.name] = np.arange(int(a.extent)).reshape(shape)
        else:
            env[a.name] = 0

    coord_arrs = []
    for e in write.index:
        arr = e.eval(env)
        if isinstance(arr, np.ndarray):
            arr = np.broadcast_to(arr, wshape).astype(np.intp)
        else:
            arr = np.full(wshape, int(arr), dtype=np.intp)
        coord_arrs.append(arr)

    # Reshape value to match wshape: val may have size-1 axes inserted at
    # reduce positions (from keepdim reductions); we just broadcast to wshape.
    val = np.asarray(value, dtype=np.float32)
    # If val has the full iter shape (N-D matching axes), broadcast directly.
    # If val has fewer dims (post-reduce scalar), also broadcasts via numpy.
    val = np.broadcast_to(val, wshape).astype(np.float32)

    if coord_arrs:
        if len(coord_arrs) == output_array.ndim:
            output_array[tuple(coord_arrs)] = val
        else:
            # Write.index has more (or fewer) dims than output_array. Flatten to
            # a row-major offset using output_array's strides, mirroring CUDA's
            # ``_flatten_coords``. The trailing coords contribute with stride=1
            # until we reach output_array.ndim coords, then standard row-major
            # strides apply. This handles reshape-merging IndexMapOps absorbed
            # into the kernel (e.g. (H, D) merged into (H*D) in the output).
            dims = list(output_array.shape)
            flat = np.zeros_like(coord_arrs[0])
            stride = 1
            for d in range(len(coord_arrs) - 1, -1, -1):
                flat = flat + coord_arrs[d] * stride
                if 0 < d < len(dims):
                    stride *= dims[d]
            flat_out = output_array.reshape(-1)
            flat_out[flat] = val
    else:
        output_array[...] = val


def _bind_inputs(loop: LoopOp, input_arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Bind each ``$i`` port to a broadcast-ready array.

    Ports load in declaration order; each loaded value is exposed as
    ``$i`` to later ports so data-dependent access patterns (e.g. gather)
    can reference an earlier port's value in their index Expr.
    """
    dollar: dict[str, np.ndarray] = {}
    for i, port in enumerate(loop.inputs):
        dollar[f"${i}"] = _apply_port_index(port, input_arrays[i], loop.axes, dollar)
    return dollar


def _apply_port_index(port: Port, base: np.ndarray, axes: tuple[Axis, ...], dollar: dict[str, np.ndarray]) -> np.ndarray:
    """Return the array the body sees for this Port (rank == len(axes))."""
    if not port.index:
        return base

    axis_names = {a.name for a in axes}
    used_axes: set[str] = set()
    for e in port.index:
        _collect_used_axis_names(e, axis_names, used_axes)

    # A reference to another port (``$N``) implicitly spans the full iter
    # shape — conservatively extend used_axes to cover all axes in that case.
    if any(_references_dollar(e) for e in port.index):
        used_axes = set(axis_names)

    bshape = tuple(int(a.extent) if a.name in used_axes else 1 for a in axes)

    if _is_identity(port, axes, base.shape):
        return base.reshape(bshape)

    axis_env: dict[str, object] = {}
    for i, a in enumerate(axes):
        if a.name in used_axes:
            shape = [1] * len(axes)
            shape[i] = int(a.extent)
            axis_env[a.name] = np.arange(int(a.extent)).reshape(shape)
        else:
            axis_env[a.name] = 0
    # Expose earlier ports' loaded values so e.g. gather can reference ``$0``.
    axis_env.update(dollar)

    coord_arrs = []
    for dim, e in enumerate(port.index):
        arr = e.eval(axis_env)
        if isinstance(arr, np.ndarray):
            arr = np.broadcast_to(arr, bshape).astype(np.intp)
        else:
            arr = np.full(bshape, int(arr), dtype=np.intp)
        # Clip to valid buffer range: out-of-bounds coords occur when a
        # Select branch's predicate masks those positions but the base
        # load is still materialized. Clipped reads are safe because the
        # Select.where in the caller overwrites them. Matches
        # IndexMapOp.forward's clip-to-bounds convention.
        if dim < base.ndim:
            arr = np.clip(arr, 0, base.shape[dim] - 1)
        coord_arrs.append(arr)

    return base[tuple(coord_arrs)]


def _is_identity(port: Port, axes: tuple[Axis, ...], base_shape: tuple) -> bool:
    axis_names_by_position: dict[str, int] = {a.name: i for i, a in enumerate(axes)}
    if len(port.index) != len(base_shape):
        return False
    prev_pos = -1
    for i, e in enumerate(port.index):
        if not isinstance(e, Var):
            return False
        pos = axis_names_by_position.get(e.name)
        if pos is None or pos <= prev_pos:
            return False
        if int(base_shape[i]) != int(axes[pos].extent):
            return False
        prev_pos = pos
    return True


def _references_dollar(expr) -> bool:
    """True if ``expr`` contains any ``Var($N)`` reference to a prior port."""
    if isinstance(expr, Var):
        return expr.name.startswith("$")
    for attr in ("left", "right", "cond", "if_true", "if_false", "expr"):
        c = getattr(expr, attr, None)
        if c is not None and _references_dollar(c):
            return True
    args = getattr(expr, "args", None)
    if isinstance(args, (list, tuple)):
        return any(_references_dollar(a) for a in args)
    return False


def _collect_used_axis_names(expr, axis_names: set[str], out: set[str]) -> None:
    if isinstance(expr, Var):
        if expr.name in axis_names:
            out.add(expr.name)
        return
    for attr in ("left", "right", "cond", "if_true", "if_false", "expr"):
        c = getattr(expr, attr, None)
        if c is not None:
            _collect_used_axis_names(c, axis_names, out)
    for attr in ("args",):
        c = getattr(expr, attr, None)
        if isinstance(c, (list, tuple)):
            for child in c:
                _collect_used_axis_names(child, axis_names, out)


def _align_ranks(args: list[np.ndarray]) -> list[np.ndarray]:
    """Bring args to a broadcast-compatible rank."""
    if len(args) <= 1:
        return args
    min_rank = min(a.ndim for a in args)
    if all(a.ndim == min_rank for a in args):
        return args
    aligned: list[np.ndarray] = []
    for a in args:
        while a.ndim > min_rank:
            size1 = [i for i in range(a.ndim) if a.shape[i] == 1]
            if not size1:
                break
            a = np.squeeze(a, axis=size1[-1])
        aligned.append(a)
    return aligned


# Keep Assign imported for IDE cross-references; the interpreter dispatches by isinstance above.
_ = Assign

__all__ = ["LoopBackend", "execute_loop_op"]
