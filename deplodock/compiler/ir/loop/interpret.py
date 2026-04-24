"""Numpy interpreter for a single ``LoopOp`` body.

Walks an SSA ``LoopOp`` body (``Load`` / ``Assign`` / ``Accum`` / ``Write``
/ ``Select``) against pre-provided input ndarrays and returns the output
ndarray. Used by ``LoopOp.forward`` (which makes every LoopOp node
executable via the generic ``Op.forward`` dispatch) and therefore by
any backend that interprets graphs through that dispatch (both the
numpy and loop backends).

Implementation uses whole-tensor numpy operations: Loads apply an index
expression against the external buffer, Accums fold along a reduce
axis via ``np.sum`` / ``np.max`` / ..., and Writes scatter into the
output buffer at evaluated coordinates.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop.ir import Accum, Assign, Axis, Load, LoopOp, Select, Write


def execute_loop_op(
    loop: LoopOp,
    input_arrays: dict[str, np.ndarray],
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """Numpy interpreter for one ``LoopOp``.

    ``input_arrays[buf_name]`` is the external buffer keyed by the
    ``Load.source`` strings in the body. ``out_shape`` is the shape of the
    output buffer the kernel writes into. Returns an ndarray of shape
    ``out_shape``.
    """
    axis_position = {a.name: i for i, a in enumerate(loop.axes)}
    meta = loop.analyze()
    accum_axis_position: dict[str, int] = {name: axis_position[axis.name] for name, axis in meta.reduce_axes.items()}
    values: dict[str, np.ndarray] = {}
    acc_map: dict[str, Accum] = {decl.name: decl for decl in loop.accums}
    for decl in loop.accums:
        init_val = decl.init.eval({})
        values[decl.name] = np.asarray(init_val, dtype=np.float32)

    output_array = np.zeros(out_shape, dtype=np.float32)
    output_written = False

    for stmt in loop:
        if isinstance(stmt, Assign):
            args = [values[a] for a in stmt.args]
            assert isinstance(stmt.op, ElementwiseImpl)
            values[stmt.name] = stmt.op(*_align_ranks(args))
        elif isinstance(stmt, Load):
            if stmt.input not in input_arrays:
                raise ValueError(f"Load source {stmt.input!r} not found in input_arrays (have {sorted(input_arrays)})")
            values[stmt.name] = _apply_load_index(stmt.index, input_arrays[stmt.input], loop.axes, values)
        elif isinstance(stmt, Accum):
            acc = acc_map[stmt.name]
            src = values[stmt.value]
            own_pos = accum_axis_position.get(stmt.name)
            positions = (own_pos,) if own_pos is not None else ()
            reduced = _fold_to_accumulator(src, acc, positions)
            values[stmt.name] = reduced
        elif isinstance(stmt, Select):
            axis_env = _broadcast_axis_env(loop.axes)
            branch_vals = [values[b.value] for b in stmt.branches]
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
    raise ValueError("LoopOp produced no output")


def _fold_to_accumulator(src: np.ndarray, acc: Accum, reduce_axis_positions: tuple[int, ...]) -> np.ndarray:
    if not reduce_axis_positions:
        return np.asarray(src, dtype=np.float32)
    axes = tuple(reduce_axis_positions)
    fn = acc.op.name
    if fn == "add":
        return np.sum(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "maximum":
        return np.max(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "minimum":
        return np.min(src, axis=axes, keepdims=True).astype(np.float32)
    if fn == "multiply":
        return np.prod(src, axis=axes, keepdims=True).astype(np.float32)
    raise NotImplementedError(f"_fold_to_accumulator: unknown combine fn {fn!r}")


def _broadcast_axis_env(axes: tuple[Axis, ...]) -> dict[str, object]:
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
    axis_names = {a.name for a in axes}
    used: set[str] = set()
    for e in write.index:
        _collect_used_axis_names(e, axis_names, used)

    wshape = tuple(int(a.extent) if a.name in used else 1 for a in axes)

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

    val = np.asarray(value, dtype=np.float32)
    val = np.broadcast_to(val, wshape).astype(np.float32)

    if coord_arrs:
        if len(coord_arrs) == output_array.ndim:
            output_array[tuple(coord_arrs)] = val
        else:
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


def _apply_load_index(
    index: tuple,
    base: np.ndarray,
    axes: tuple[Axis, ...],
    bindings: dict[str, np.ndarray],
) -> np.ndarray:
    if not index:
        return base

    axis_names = {a.name for a in axes}
    used_axes: set[str] = set()
    for e in index:
        _collect_used_axis_names(e, axis_names, used_axes)

    if any(_references_ssa(e, set(bindings)) for e in index):
        used_axes = set(axis_names)

    bshape = tuple(int(a.extent) if a.name in used_axes else 1 for a in axes)

    if _is_identity_index(index, axes, base.shape):
        return base.reshape(bshape)

    axis_env: dict[str, object] = {}
    for i, a in enumerate(axes):
        if a.name in used_axes:
            shape = [1] * len(axes)
            shape[i] = int(a.extent)
            axis_env[a.name] = np.arange(int(a.extent)).reshape(shape)
        else:
            axis_env[a.name] = 0
    axis_env.update(bindings)

    coord_arrs = []
    for dim, e in enumerate(index):
        arr = e.eval(axis_env)
        if isinstance(arr, np.ndarray):
            arr = np.broadcast_to(arr, bshape).astype(np.intp)
        else:
            arr = np.full(bshape, int(arr), dtype=np.intp)
        if dim < base.ndim:
            arr = np.clip(arr, 0, base.shape[dim] - 1)
        coord_arrs.append(arr)

    return base[tuple(coord_arrs)]


def _is_identity_index(index: tuple, axes: tuple[Axis, ...], base_shape: tuple) -> bool:
    axis_names_by_position: dict[str, int] = {a.name: i for i, a in enumerate(axes)}
    if len(index) != len(base_shape):
        return False
    prev_pos = -1
    for i, e in enumerate(index):
        if not isinstance(e, Var):
            return False
        pos = axis_names_by_position.get(e.name)
        if pos is None or pos <= prev_pos:
            return False
        if int(base_shape[i]) != int(axes[pos].extent):
            return False
        prev_pos = pos
    return True


def _references_ssa(expr, ssa_names: set[str]) -> bool:
    if isinstance(expr, Var):
        return expr.name in ssa_names
    for attr in ("left", "right", "cond", "if_true", "if_false", "expr"):
        c = getattr(expr, attr, None)
        if c is not None and _references_ssa(c, ssa_names):
            return True
    args = getattr(expr, "args", None)
    if isinstance(args, (list, tuple)):
        return any(_references_ssa(a, ssa_names) for a in args)
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
