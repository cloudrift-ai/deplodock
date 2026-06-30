"""JIT-compile and execute LoopOp kernels via cppyy / Cling.

A ``LoopOp`` plus the runtime input shapes uniquely determine a C++
source string. We render that source by walking the body via each
``Stmt.render``, hand the result to ``cppyy.cppdef`` (Cling JIT-compiles
+ links it into the running process), then call the resulting function
with raw pointers to numpy arrays.

Module-level state:

- ``_PRELUDE_INSTALLED``: ``cppyy.cppdef(PRELUDE)`` is idempotent in
  effect but emits warnings on redefinition; we install the prelude
  once.
- ``_FN_CACHE``: keyed by the rendered C++ source string (which folds in
  the loop body + input/output shapes) so repeated calls with an identical
  kernel don't re-JIT. NOT keyed on ``id(loop)`` — object addresses are
  recycled by CPython, so a GC'd LoopOp's id can alias a later same-shape
  one and serve the wrong cached kernel.
- ``_FN_COUNTER``: monotonic id used to give every kernel a unique C
  symbol name; cppyy / Cling can't redefine a function once it's been
  compiled in the same process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from emmy.compiler.ir.stmt import RenderCtx, render_body

if TYPE_CHECKING:
    from emmy.compiler.ir.loop.ir import LoopOp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C++ source generation
# ---------------------------------------------------------------------------


_INTRINSICS_CPP: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf_",  # libm has no rsqrtf; PRELUDE provides one
    "tanh": "tanhf",
    "fabs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
}


PRELUDE = """\
#include <cmath>
#include <algorithm>
static inline float rsqrtf_(float x) { return 1.0f / sqrtf(x); }
"""


def render_loopop_cpp(loop: LoopOp, fn_name: str, input_shapes: dict[str, tuple[int, ...]], output_shape: tuple[int, ...]) -> str:
    """Emit a complete ``extern "C" void <fn_name>(...)`` definition.

    Inputs become ``const float*`` params in ``loop.inputs`` order; the
    sole output (first ``loop.outputs`` key) becomes a trailing ``float*`` param.
    """
    output_name = next(iter(loop.outputs))
    shapes: dict[str, tuple[int, ...]] = {**input_shapes, output_name: output_shape}
    ctx = RenderCtx(shapes=shapes, indent=1, intrinsics=_INTRINSICS_CPP)

    sig_parts = [f"const float* {n}" for n in loop.inputs]
    sig_parts.append(f"float* {output_name}")
    params_text = ", ".join(sig_parts)

    body_text = "\n".join(render_body(loop.body, ctx))
    return f'extern "C" void {fn_name}({params_text}) {{\n{body_text}\n}}\n'


# ---------------------------------------------------------------------------
# JIT compile + execute
# ---------------------------------------------------------------------------


_PRELUDE_INSTALLED = False
_FN_CACHE: dict[tuple, object] = {}
_FN_COUNTER = 0


def _ensure_prelude() -> None:
    global _PRELUDE_INSTALLED
    if _PRELUDE_INSTALLED:
        return
    import cppyy

    cppyy.cppdef(PRELUDE)
    _PRELUDE_INSTALLED = True


def execute_loop_op_cpp(
    loop: LoopOp,
    input_arrays: dict[str, np.ndarray],
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """JIT-compile ``loop`` to C++ if needed, then run it on ``input_arrays``.

    Returns a fresh ``np.ndarray`` of shape ``out_shape``. ``input_arrays``
    is keyed by the ``Load.input`` strings; the ordered list comes from
    ``loop.inputs`` (matcher-populated when run from the pipeline;
    body-derived placeholders otherwise).
    """
    import cppyy

    _ensure_prelude()

    bufs = tuple(loop.inputs)
    input_shapes = {name: tuple(int(d) for d in input_arrays[name].shape) for name in bufs}
    # Key on the rendered source, NOT ``id(loop)``: a LoopOp plus its runtime
    # shapes uniquely determine the C++ string (see module docstring), and the
    # object address is unsafe — CPython reuses the id of a GC'd LoopOp, so a
    # later same-shape LoopOp (e.g. tanh after a freed negate) could collide on
    # ``id`` and be served the stale kernel. Render with a fixed placeholder
    # name so the key is stable across calls; the JIT'd symbol gets its unique
    # ``_FN_COUNTER`` name only on a miss.
    cache_key = render_loopop_cpp(loop, "loopop_kern", input_shapes, out_shape)

    fn = _FN_CACHE.get(cache_key)
    if fn is None:
        global _FN_COUNTER
        _FN_COUNTER += 1
        fn_name = f"loopop_kern_{_FN_COUNTER}"
        src = render_loopop_cpp(loop, fn_name, input_shapes, out_shape)
        logger.debug("JIT-compiling LoopOp kernel %s:\n%s", fn_name, src)
        cppyy.cppdef(src)
        fn = getattr(cppyy.gbl, fn_name)
        _FN_CACHE[cache_key] = fn

    output = np.zeros(out_shape, dtype=np.float32)

    # cppyy accepts numpy arrays of matching dtype as ``const float*`` /
    # ``float*`` parameters via the buffer protocol. Ensure C-contiguous
    # float32 so the buffer matches the parameter type.
    coerced = [np.ascontiguousarray(input_arrays[n], dtype=np.float32) for n in bufs]
    fn(*coerced, output)
    return output


__all__ = ["execute_loop_op_cpp", "render_loopop_cpp", "PRELUDE"]
