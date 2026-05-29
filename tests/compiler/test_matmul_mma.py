"""End-to-end MMA matmul correctness — M8 of
``plans/mma-fragment-factorization.md``.

Verifies the WMMA F16 path produces correct output across realistic
shapes. Pins ``DEPLODOCK_MMA=1`` and lets the planner pick the
best-scoring warp-tier variant; compiles via ``nvcc --cubin`` (NVRTC
doesn't ship ``<crt/mma.h>``; see plan's Failure modes) and compares
against the f32 matmul reference within fp16 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_wmma() -> bool:
    if not _has_cuda():
        return False
    import cupy as cp

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 70


def _matmul_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    i = Axis("i", M)
    j = Axis("j", N)
    k = Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _matmul_graph(*, M: int, N: int, K: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=_matmul_loop_op(M=M, N=N, K=K), inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F32), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
def test_mma_matmul_matches_f32_reference(M: int, N: int, K: int, monkeypatch):
    """A square F16 matmul compiled via the MMA path agrees with the f32
    PyTorch reference within fp16 tolerance."""
    import cupy as cp

    from deplodock.compiler.backend.cuda.nvcc import compile_to_cubin
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "1")

    g = _matmul_graph(M=M, N=N, K=K)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16", "expected warp-tier MMA variant"

    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "wmma::mma_sync" in src
    assert "#include <mma.h>" in src

    cap = cp.cuda.Device().compute_capability
    cubin_path = compile_to_cubin(src, kop.name, arch=f"sm_{cap}")
    mod = cp.RawModule(path=str(cubin_path))
    k = mod.get_function(kop.name)

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    a_cp = cp.asarray(a)
    b_cp = cp.asarray(b)
    c_cp = cp.zeros((M, N), dtype=cp.float32)

    # Derive launch geometry from the knobs.
    knobs = kop.knobs
    wm, wn = int(knobs["WM"]), int(knobs["WN"])
    fm, fn = int(knobs["FM"]), int(knobs["FN"])
    splitk = int(knobs.get("SPLITK", 1))
    atom_m = atom_n = 16  # wmma_m16n16k16_f16
    m_b = max(1, M // (wm * fm * atom_m))
    n_b = max(1, N // (wn * fn * atom_n))
    grid_x = m_b * n_b * splitk
    threads_per_cta = wm * wn * 32

    k((grid_x,), (threads_per_cta,), (a_cp, b_cp, c_cp))

    expected = a.astype(np.float32) @ b.astype(np.float32)
    diff = np.abs(c_cp.get() - expected).max()
    # f16 operands × f32 accumulator: tolerance ≈ K * max-product-mag * f16
    # ulp. For our 0.1-scale inputs and K ≤ 128, ~1e-2 is comfortable.
    assert diff < 1e-2, f"M={M} N={N} K={K} max-abs-err {diff}"
