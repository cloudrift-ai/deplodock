"""Loop backend: numpy interpreter for a ``LoopProgram``.

Walks each ``LoopLaunch`` in topological order, evaluating its ``LoopOp``
as whole-tensor numpy operations — no explicit coord loops. Broadcasting
handles per-element semantics; ``keepdims=True`` reductions keep the
axis-1 shape invariant the SSA body expects.

The interpreter is intentionally a faithful mirror of ``backend/cuda/emit``'s
semantics: same ``$N`` port-indexing convention, same ``Port.index`` axis
substitution. Disagreement between ``LoopBackend`` and ``CudaBackend`` on
the same ``LoopProgram`` implicates codegen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, ProgramResult
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, LoopOp, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.loop import LoopLaunch, LoopProgram

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph


class LoopBackend(Backend):
    """Execute a ``LoopProgram`` via numpy whole-tensor operations.

    The compiled artifact is just the ``LoopProgram`` itself — no wrapping.
    """

    def compile(self, graph: Graph) -> LoopProgram:
        return compile_graph(graph)

    def run(self, compiled: LoopProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        t0 = time.perf_counter()
        arrays = _execute(compiled, input_data or {})
        elapsed = (time.perf_counter() - t0) * 1000
        return ProgramResult(outputs=arrays, time_ms=elapsed)

    # ``benchmark`` inherits the default wall-time loop from ``Backend``.


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def _concrete_shape(shape: tuple) -> tuple[int, ...]:
    """Coerce a shape to concrete ints (required for numpy reshape)."""
    return tuple(int(d) for d in shape)


def _execute(program: LoopProgram, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Walk the LoopProgram, returning {graph_output_name: ndarray}."""
    buffers: dict[str, np.ndarray] = {}

    # Seed inputs.
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

    # Non-LoopOp launches (ops fusion didn't wrap — multi-source IndexMapOp
    # for cat, non-2-axis TransposeOp, GatherOp, etc.) go through Op.forward.
    if not isinstance(launch.loop, LoopOp):
        args = [buffers[n] for n in launch.input_names]
        result = launch.loop.forward(*args)
        return np.asarray(result, dtype=np.float32).reshape(out_shape)

    loop = launch.loop

    # Bind $N → ndarray via Port.index substitution.
    dollar = _bind_inputs(loop, launch, buffers)

    # Evaluate SSA body. ReduceOps use keepdims=True (matches LoopOp.infer_shapes);
    # ElementwiseOp args may need rank-alignment to reconcile keepdim-1 axes
    # that downstream consumers don't carry (see _align_ranks below).
    values: dict[str, np.ndarray] = dict(dollar)
    for assign in loop.body:
        args = [values[a] for a in assign.args]
        if isinstance(assign.op, ReduceOp):
            values[assign.name] = _reduce_keepdims(assign.op, args[0])
        else:
            assert isinstance(assign.op, ElementwiseOp)
            values[assign.name] = assign.op.forward(*_align_ranks(args))

    if loop.body:
        result = values[loop.body[-1].name]
    elif dollar:
        result = next(iter(dollar.values()))
    else:
        raise ValueError("LoopOp has neither inputs nor body")

    arr = np.asarray(result, dtype=np.float32)
    if arr.shape != out_shape:
        while arr.ndim > len(out_shape):
            size1 = [i for i in range(arr.ndim) if arr.shape[i] == 1]
            if not size1:
                break
            arr = np.squeeze(arr, axis=size1[-1])
    return arr.reshape(out_shape)


def _bind_inputs(
    loop: LoopOp,
    launch: LoopLaunch,
    buffers: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bind each ``$i`` port to a numpy view / gather from its external buffer.

    For identity-like Ports (index is ``(Var(a0), Var(a1), ...)`` matching
    the LoopOp's axes), returns the buffer directly. For non-identity
    patterns (transpose, broadcast, slice), materializes via advanced
    indexing: each Expr in ``port.index`` is evaluated under an axis env
    of broadcast-shaped ``arange`` arrays, producing one coord-array per
    dim; the result is indexed from the buffer.
    """
    dollar: dict[str, np.ndarray] = {}
    for i, port in enumerate(loop.inputs):
        key = f"${i}"
        buf_name = launch.input_names[i]
        base = buffers[buf_name]
        dollar[key] = _apply_port_index(port, base, loop.axes)
    return dollar


def _apply_port_index(port: Port, base: np.ndarray, axes: tuple[Axis, ...]) -> np.ndarray:
    """Return the array that the body sees for this Port.

    The returned array has rank ``len(axes)`` with shape ``(extent_i if
    axis i is used by port.index else 1)``, producing a broadcast-ready
    view. Values at iteration coord (c0, ..., cN) equal ``base[e0, ..., em]``
    where each ``ek`` is ``port.index[k]`` evaluated under the current
    axis coords.
    """
    if not port.index:
        return base

    axis_names = {a.name for a in axes}
    used_axes: set[str] = set()
    for e in port.index:
        _collect_used_axis_names(e, axis_names, used_axes)

    # Per-axis broadcast dim size.
    bshape = tuple(int(a.extent) if a.name in used_axes else 1 for a in axes)

    # Identity fast-path: buffer already matches broadcast shape (reshape only).
    if _is_identity(port, axes, base.shape):
        return base.reshape(bshape)

    # Axis env: used axes get arange-broadcast arrays; unused get 0.
    axis_env: dict[str, object] = {}
    for i, a in enumerate(axes):
        if a.name in used_axes:
            shape = [1] * len(axes)
            shape[i] = int(a.extent)
            axis_env[a.name] = np.arange(int(a.extent)).reshape(shape)
        else:
            axis_env[a.name] = 0

    coord_arrs = []
    for e in port.index:
        arr = e.eval(axis_env)
        if isinstance(arr, np.ndarray):
            arr = np.broadcast_to(arr, bshape).astype(np.intp)
        else:
            arr = np.full(bshape, int(arr), dtype=np.intp)
        coord_arrs.append(arr)

    return base[tuple(coord_arrs)]


def _is_identity(port: Port, axes: tuple[Axis, ...], base_shape: tuple) -> bool:
    """True if port.index is identity across all axes used (no transpose/permute)."""
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


def _collect_used_axis_names(expr, axis_names: set[str], out: set[str]) -> None:
    if isinstance(expr, Var):
        if expr.name in axis_names:
            out.add(expr.name)
        return
    for attr in ("left", "right", "cond", "if_true", "if_false"):
        c = getattr(expr, attr, None)
        if c is not None:
            _collect_used_axis_names(c, axis_names, out)
    for attr in ("args",):
        c = getattr(expr, attr, None)
        if isinstance(c, (list, tuple)):
            for child in c:
                _collect_used_axis_names(child, axis_names, out)


def _align_ranks(args: list[np.ndarray]) -> list[np.ndarray]:
    """Bring args to a broadcast-compatible rank.

    When a ReduceOp with keepdim=True produces an extra size-1 axis and the
    downstream consumer has the same rank *minus* that axis (e.g. SDPA's
    ``mul(qk_summed, scale)``), numpy's native broadcast doesn't line up the
    axes the way the compiler intends. We squeeze size-1 axes from
    higher-rank args until all args share the minimum rank.
    """
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


def _reduce_keepdims(op: ReduceOp, x: np.ndarray) -> np.ndarray:
    """Evaluate a ReduceOp with keepdims=True (matches LoopOp.infer_shapes)."""
    fn = op.fn
    axis = op.axis if isinstance(op.axis, int) else None
    if fn == "sum":
        return np.sum(x, axis=axis, keepdims=True)
    if fn == "max":
        return np.max(x, axis=axis, keepdims=True)
    if fn == "prod":
        return np.prod(x, axis=axis, keepdims=True)
    raise NotImplementedError(f"_reduce_keepdims: unknown fn {fn!r}")


__all__ = ["LoopBackend"]
