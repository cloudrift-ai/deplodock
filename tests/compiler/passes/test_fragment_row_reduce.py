"""The fragment row-reduction kernel-IR op (``FragmentRowReduce``) — the flash
fragment-softmax primitive (Phase 3 of ``plans/tensor-core-streaming-flash-mma.md``).

The op renders the ``rowmax`` / ``rowsum`` over an ``mma.sync`` ``m16n8`` C-fragment's
N (kv) lanes — the same logic the validated reference kernel
(``tests/compiler/e2e/test_flash_tensorcore_reference.py``) uses. This test renders the
op and runs the emitted CUDA on real hardware: each lane loads its 4 C-fragment elements
(per the documented layout) for a ``BN``-wide score tile, the rendered reduction produces
the two per-row results, and they match numpy's row reduction exactly. Locks the op's
codegen against the PTX C-layout — the warp-chain build emits this op.
"""

from __future__ import annotations

import numpy as np

from ..conftest import requires_cuda


def _rendered(op_name: str, n_tiles: int) -> str:
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.kernel.ir import FragmentRowReduce
    from deplodock.compiler.ir.stmt.base import RenderCtx

    frags = tuple(f"cf{t}" for t in range(n_tiles))
    op = FragmentRowReduce(top="top", bot="bot", frags=frags, op=ElementwiseImpl(op_name), group=4)
    return "\n".join(op.render(RenderCtx(indent=1)))


@requires_cuda
def test_fragment_rowmax_and_rowsum_match_numpy():
    import cupy as cp

    from deplodock.compiler.backend.cuda import nvcc

    BN = 16  # 2 N-atoms of 8
    n_tiles = BN // 8
    # Per lane, load the 4 C-fragment elements of each N-atom: rows g/g+8 (g=lane/4),
    # cols (lane%4)*2+{0,1}, the N-atom offset nt*8 added to the column.
    load = ""
    for nt in range(n_tiles):
        load += (
            f"    float cf{nt}[4];\n"
            f"    cf{nt}[0]=S[g*{BN} + {nt}*8 + c0+0]; cf{nt}[1]=S[g*{BN} + {nt}*8 + c0+1];\n"
            f"    cf{nt}[2]=S[(g+8)*{BN} + {nt}*8 + c0+0]; cf{nt}[3]=S[(g+8)*{BN} + {nt}*8 + c0+1];\n"
        )
    for kind, op_name, ref in (("max", "maximum", "max"), ("sum", "add", "sum")):
        body = _rendered(op_name, n_tiles)
        src = (
            f'extern "C" __global__ void frag_{kind}(const float* S, float* out){{\n'
            f"    int lane=threadIdx.x&31; int g=lane/4; int c0=(lane%4)*2;\n"
            f"{load}"
            f"{body}\n"
            f"    if(lane%4==0){{ out[g]=top; out[g+8]=bot; }}\n"
            f"}}\n"
        )
        fn = nvcc.load_function(src, f"frag_{kind}", "", uses_tma=False)
        np.random.seed(0)
        S = np.random.randn(16, BN).astype(np.float32)
        d_S = cp.asarray(S)
        d_out = cp.zeros(16, cp.float32)
        fn((1,), (32,), (d_S, d_out))
        got = cp.asnumpy(d_out)
        want = getattr(S, ref)(axis=1)
        assert np.allclose(got, want, atol=1e-4), f"{kind}: max|diff|={np.max(np.abs(got - want)):.2e}"


def test_fragment_row_reduce_renders_the_shuffle_butterfly():
    """Structural: the op emits the in-lane combine + the ``__shfl_xor`` butterfly over
    the 4-lane column group (``xor 2``, ``xor 1``) — the validated pattern, no GPU."""
    src = _rendered("maximum", 2)
    assert "__shfl_xor_sync(0xffffffff, top, 2)" in src
    assert "__shfl_xor_sync(0xffffffff, top, 1)" in src
    assert "__shfl_xor_sync(0xffffffff, bot, 2)" in src
    assert "fmaxf" in src
