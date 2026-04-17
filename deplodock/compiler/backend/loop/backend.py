"""Loop backend: numpy interpreter for a ``LoopProgram``.

Walks each ``LoopLaunch`` in topological order, evaluating its ``LoopOp``
as whole-tensor numpy operations — no explicit coord loops. Broadcasting
handles per-element semantics; ``keepdims=True`` reductions keep the
axis-1 shape invariant the SSA body expects.

The interpreter is intentionally a faithful mirror of ``backend/cuda/emit``'s
semantics: same `$N` port-indexing convention, same treatment of
``Port.indexmap`` and ``Mux.select``. Disagreement between ``LoopBackend``
and ``CudaBackend`` on the same ``LoopProgram`` implicates codegen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, ProgramResult
from deplodock.compiler.ir.loop import Combine, LoopInput, LoopOp, Mux, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.loop import LoopLaunch, LoopProgram

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph


@dataclass
class WrappedLoopProgram:
    """Compiled-loop artifact: wraps a post-fusion ``LoopProgram``."""

    program: LoopProgram


class LoopBackend(Backend):
    """Execute a ``LoopProgram`` via numpy whole-tensor operations."""

    def compile(self, graph: Graph) -> WrappedLoopProgram:
        return WrappedLoopProgram(compile_graph(graph))

    def run(self, compiled: WrappedLoopProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        t0 = time.perf_counter()
        arrays = _execute(compiled.program, input_data or {})
        elapsed = (time.perf_counter() - t0) * 1000
        outputs = {n: np.asarray(v, dtype=np.float32).flatten().tolist() for n, v in arrays.items()}
        return ProgramResult(outputs=outputs, time_ms=elapsed)

    def run_arrays(self, compiled: WrappedLoopProgram, *, input_data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        return _execute(compiled.program, input_data or {})

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
            # Other constants (weights supplied by caller via input_data) must have been covered above.

    # Execute each launch in order.
    for launch in program.launches:
        buffers[launch.output_name] = _exec_launch(launch, program, buffers)

    return {name: buffers[name] for name in program.graph_outputs}


def _exec_launch(launch: LoopLaunch, program: LoopProgram, buffers: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate one launch. Returns ndarray with shape == program.output_shape(launch)."""
    out_shape = _concrete_shape(program.output_shape(launch))

    # Non-LoopOp launches (TransposeOp with >2 axes, GatherOp, etc. that fusion
    # didn't wrap) are evaluated via Op.forward directly.
    if not isinstance(launch.loop, LoopOp):
        args = [buffers[n] for n in launch.input_names]
        result = launch.loop.forward(*args)
        return np.asarray(result, dtype=np.float32).reshape(out_shape)

    loop = launch.loop

    # Build $N → ndarray map by walking the LoopOp's input tree.
    dollar: dict[str, np.ndarray] = {}
    port_idx = [0]

    def bind(inp: LoopInput) -> np.ndarray:
        if isinstance(inp, Port):
            key = f"${port_idx[0]}"
            buf_name = launch.input_names[port_idx[0]]
            port_idx[0] += 1
            base = buffers[buf_name]
            arr = inp.indexmap.forward(base) if inp.indexmap is not None else base
            dollar[key] = arr
            return arr
        if isinstance(inp, Combine):
            vals = [bind(s) for s in inp.sources]
            acc = vals[0]
            val_iter = iter(vals[1:])
            for op in inp.ops:
                assert isinstance(op, ElementwiseOp)
                if op.info.arity == 1:
                    acc = op.forward(acc)
                else:
                    acc = op.forward(acc, next(val_iter))
            return acc
        if isinstance(inp, Mux):
            return _eval_mux(inp, out_shape, bind_for_mux_branch=bind)
        raise TypeError(f"Unknown LoopInput variant: {type(inp).__name__}")

    top_vals: list[np.ndarray] = [bind(inp) for inp in loop.inputs]

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
    elif top_vals:
        # Copy kernel — no body, pass through the (single) top-level input.
        result = top_vals[0]
    else:
        raise ValueError("LoopOp has neither inputs nor body")

    # Reshape to declared output shape (reconciles any broadcast/keepdim drift).
    arr = np.asarray(result, dtype=np.float32)
    if arr.shape != out_shape:
        # Squeeze keepdim-1 axes that exceed the declared output rank before reshape.
        while arr.ndim > len(out_shape):
            size1 = [i for i in range(arr.ndim) if arr.shape[i] == 1]
            if not size1:
                break
            arr = np.squeeze(arr, axis=size1[-1])
    return arr.reshape(out_shape)


def _align_ranks(args: list[np.ndarray]) -> list[np.ndarray]:
    """Bring args to a broadcast-compatible rank.

    When a ReduceOp with keepdim=True produces an extra size-1 axis and the
    downstream consumer has the same rank *minus* that axis (e.g. SDPA's
    ``mul(qk_summed, scale)``), numpy's native broadcast doesn't line up the
    axes the way the compiler intends. We squeeze size-1 axes from
    higher-rank args until all args share the minimum rank.

    The RMSNorm case (reduce→keepdim then broadcast against a same-rank
    tensor via IndexMap expansion) is untouched: args already have matching
    rank so no squeezing happens.
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
                break  # no size-1 axis to squeeze; leave as-is (broadcast may still fail later)
            # Squeeze the rightmost size-1 axis — typically the most recent keepdim artifact.
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


def _eval_mux(mux: Mux, out_shape: tuple[int, ...], *, bind_for_mux_branch) -> np.ndarray:
    """Evaluate a Mux at the output coordinate grid.

    Each branch's ``select`` is an ``Expr`` over ``out_coord_N`` placeholders.
    Build a bool mask per branch by substituting coord-grid ndarrays into
    the select, then fold with ``np.where`` (first-match semantics).

    Branches are ``bind``-evaluated in forward order so the enclosing
    port_idx walker reads ``input_names[i]`` for each branch's Port in
    the correct sequence; the ``np.where`` fold then runs in reverse so
    earlier branches override later (catch-all) ones.
    """
    from deplodock.compiler.ir.expr import eval_expr

    # Coord grids: one ndarray per output axis.
    if out_shape:
        grids = np.meshgrid(*[np.arange(d) for d in out_shape], indexing="ij")
    else:
        grids = []
    env = {f"out_coord_{i}": grids[i] for i in range(len(grids))}

    # Forward-order bind to advance the enclosing port_idx in the right sequence.
    branch_vals = [bind_for_mux_branch(branch.input) for branch in mux.branches]

    # Reverse-order mask fold: last branch is the catch-all, earlier override.
    result: np.ndarray | None = None
    for branch, val in zip(reversed(mux.branches), reversed(branch_vals), strict=True):
        mask = eval_expr(branch.select, env) if branch.select is not None else True
        if result is None:
            result = np.broadcast_to(val, out_shape).astype(np.float32, copy=True)
        else:
            result = np.where(mask, np.broadcast_to(val, out_shape), result)
    assert result is not None
    return result


# Re-export IndexMapOp for any consumers that need it.
__all__ = ["LoopBackend", "WrappedLoopProgram", "IndexMapOp"]
