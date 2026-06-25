"""The warp-chain assembler — emit the fused tensor-core flash kernel
(``plans/tensor-core-streaming-flash-mma.md`` Phase 2.3 + Phase 3).

Generates the FA-2 kernel validated end-to-end in
``tests/compiler/e2e/test_flash_tensorcore_reference.py``, generalized over the flash
shape ``(B, H, S, D)``: one warp per 16 query rows, the QK^T `mma` (Q ``ldmatrix.x4`` A
over ``D/16`` K-tiles, K transposed-B native pack) → the score C-fragment → the
fragment online-softmax (the validated ``FragmentRowReduce`` op for ``rowmax``/``rowsum``
+ the per-row ``m``/``l``/``α`` recurrence in fragment-distributed form) → the C→A handoff
(``P`` C-fragment → smem row-major → ``ldmatrix.x4`` A) → the P@V `mma` over ``D/8``
N-tiles. The two ``mma`` cells are exactly the atom-layer's QK^T (fragment-output) +
P@V (fragment-``A``) — the reuse boundary Phase 2.1/2.2 fixed.

v1 scope: **fp16, non-causal, equal-head (no GQA), `D % 16 == 0`, `S % 16 == 0`**.
:func:`warp_chain_eligible` gates it; an out-of-scope flash falls back to the scalar
chain (``chain_build``). Masking / GQA / the symbolic-`seq_len` stream are follow-ups.
"""

from __future__ import annotations

import math

from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.kernel.ir import FragmentRowReduce
from deplodock.compiler.ir.kernel.render import _MMA_SYNC_PRELUDE
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt.base import RenderCtx

# Reuse the project's SHARED tensor-core codegen (the exact ``dpl_mma_m16n8k16_*`` /
# ``dpl_ldmatrix_x4`` / ``dpl_ldmatrix_x2_trans`` helpers ``render_kernelop`` emits for the
# warp-tier matmul) — the QK^T / P@V mma + the A / V ldmatrix loads fall out of the same
# ops as the matmul. Only the transposed-B (Q@K^T) native smem pack is bespoke: the shared
# lib lowers a transposed-B operand gmem-direct (``ir/kernel`` raises on a staged transposed-B
# ldmatrix), so the smem-staged native pack is the one genuinely-new primitive here.
_PRELUDE = (
    "\n#include <cuda_fp16.h>\n"
    + _MMA_SYNC_PRELUDE
    + r"""
__device__ __forceinline__ void dpl_wc_load_b_native(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; int n=lane/4; int kb=(lane%4)*2;
  __half2 h0=__halves2half2(sm[n*ldm+kb+0], sm[n*ldm+kb+1]);
  __half2 h1=__halves2half2(sm[n*ldm+kb+8+0], sm[n*ldm+kb+8+1]);
  r[0]=*reinterpret_cast<unsigned*>(&h0); r[1]=*reinterpret_cast<unsigned*>(&h1);
}
"""
)


def warp_chain_eligible(*, B: int, H: int, S: int, D: int, group: int, causal: bool, mask: bool, symbolic: bool) -> bool:
    """The v1 fused-TC-flash scope: static fp16, non-causal, equal-head, both extents a
    multiple of 16. Everything else falls back to the scalar chain."""
    return not symbolic and not causal and not mask and group == 1 and D % 16 == 0 and S % 16 == 0 and 16 <= D <= 256 and S >= 16


def _reduce_lines(top: str, bot: str, frags: tuple[str, ...], op: str, ctx: RenderCtx) -> str:
    """Render the validated ``FragmentRowReduce`` op into the generated source — the
    same primitive ``test_fragment_row_reduce.py`` pins."""
    return "\n".join(FragmentRowReduce(top=top, bot=bot, frags=frags, op=ElementwiseImpl(op), group=4).render(ctx))


def warp_chain_kernel_source(kname: str, *, B: int, H: int, S: int, D: int, scale: float) -> str:
    """The fused TC flash ``__global__`` source for ``(B,H,S,D)`` fp16, non-causal."""
    kt = D // 16  # QK^T K-tiles (reduce over D)
    nd = D // 8  # P@V N-tiles (output over D)
    ctx = RenderCtx(indent=2)

    rowmax = _reduce_lines("r0", "r1", ("Sf[0]", "Sf[1]"), "maximum", ctx)
    rowsum = _reduce_lines("s0", "s1", ("Pf[0]", "Pf[1]"), "add", ctx)

    qa_load = "\n".join(f"    dpl_ldmatrix_x4(qa[{t}], &qs[(lane%16)*{D} + (lane/16)*8 + {t * 16}]);" for t in range(kt))
    of_decl = ", ".join(["{0,0,0,0}"] * nd)
    # QK^T: 2 N-atoms (kv 0-7 / 8-15), accumulate over kt K-tiles.
    qk = []
    for nt in range(2):
        qk.append("      { float acc[4]={0,0,0,0};")
        for t in range(kt):
            off = nt * 8 * D + t * 16
            qk.append(f"        {{ unsigned kb[2]; dpl_wc_load_b_native(kb, ks + {off}, {D});")
            qk.append(f"          dpl_mma_m16n8k16_f16(acc, qa[{t}], kb, acc); }}")
        qk.append(f"        for(int e=0;e<4;e++) Sf[{nt}][e]=acc[e]*{scale!r}f; }}")
    qk_body = "\n".join(qk)
    # P (exp(S - m)) per N-atom + rowsum partials.
    p_body = "\n".join(
        f"      Pf[{nt}][0]=__expf(Sf[{nt}][0]-mn0); Pf[{nt}][1]=__expf(Sf[{nt}][1]-mn0);"
        f" Pf[{nt}][2]=__expf(Sf[{nt}][2]-mn1); Pf[{nt}][3]=__expf(Sf[{nt}][3]-mn1);"
        for nt in range(2)
    )
    rescale = "\n".join(f"      Of[{n}][0]*=a0; Of[{n}][1]*=a0; Of[{n}][2]*=a1; Of[{n}][3]*=a1;" for n in range(nd))
    # Write P C-fragment to smem row-major [16][16] (C->A handoff).
    p_store = "\n".join(
        f"      ps[g*16 + {nt}*8 + c0+0]=__float2half(Pf[{nt}][0]); ps[g*16 + {nt}*8 + c0+1]=__float2half(Pf[{nt}][1]);"
        f" ps[(g+8)*16 + {nt}*8 + c0+0]=__float2half(Pf[{nt}][2]); ps[(g+8)*16 + {nt}*8 + c0+1]=__float2half(Pf[{nt}][3]);"
        for nt in range(2)
    )
    pv = "\n".join(
        f"      {{ unsigned vb[2]; dpl_ldmatrix_x2_trans(vb, &vs[(lane%16)*{D} + {n * 8}]);\n"
        f"        dpl_mma_m16n8k16_f16(Of[{n}], pa, vb, Of[{n}]); }}"
        for n in range(nd)
    )
    store = "\n".join(
        f"    O[base + ((qb*16)+g)*{D} + {n}*8 + c0+0]=__float2half(Of[{n}][0]/l0);"
        f" O[base + ((qb*16)+g)*{D} + {n}*8 + c0+1]=__float2half(Of[{n}][1]/l0);"
        f" O[base + ((qb*16)+g+8)*{D} + {n}*8 + c0+0]=__float2half(Of[{n}][2]/l1);"
        f" O[base + ((qb*16)+g+8)*{D} + {n}*8 + c0+1]=__float2half(Of[{n}][3]/l1);"
        for n in range(nd)
    )
    return f"""{_PRELUDE}
extern "C" __global__ void {kname}(const __half* Q,const __half* K,const __half* V,__half* O){{
  int nqb={S // 16}; int blk=blockIdx.x; int qb=blk%nqb; int bh=blk/nqb;
  long base=(long)bh*{S}*{D};
  int lane=threadIdx.x&31; int g=lane/4; int c0=(lane%4)*2;
  __shared__ __half qs[16*{D}], ks[16*{D}], vs[16*{D}], ps[16*16];
  for(int i=lane;i<16*{D};i+=32) qs[i]=Q[base + (qb*16)*{D} + i];
  __syncwarp();
  unsigned qa[{kt}][4];
{qa_load}
  float m0=-1e30f,m1=-1e30f,l0=0,l1=0;
  float Of[{nd}][4]={{{of_decl}}};
  for(int kv0=0; kv0<{S}; kv0+=16){{
    for(int i=lane;i<16*{D};i+=32){{ ks[i]=K[base+kv0*{D}+i]; vs[i]=V[base+kv0*{D}+i]; }}
    __syncwarp();
    float Sf[2][4];
{qk_body}
{rowmax}
    float mn0=fmaxf(m0,r0), mn1=fmaxf(m1,r1);
    float a0=__expf(m0-mn0), a1=__expf(m1-mn1);
    float Pf[2][4];
{p_body}
{rowsum}
    l0=l0*a0+s0; l1=l1*a1+s1;
{rescale}
{p_store}
    __syncwarp();
    unsigned pa[4]; dpl_ldmatrix_x4(pa, &ps[(lane%16)*16 + (lane/16)*8]);
{pv}
    m0=mn0; m1=mn1;
  }}
{store}
}}
"""


def assemble_warp_chain(loop_op: LoopOp, *, B: int, H: int, S: int, D: int) -> Graph:
    """Build a single-``CudaOp`` ``Graph`` fragment for the fused TC flash — the q/k/v
    inputs + one kernel node. Spliced by the engine in place of the streaming LoopOp."""
    # Q/K/V are the rank-4 inputs in declared order (the scale/ninf constants are rank-1).
    rank4 = [n for n, t in loop_op.inputs.items() if len(t.shape) == 4]
    q_id, k_id, v_id = rank4[0], rank4[1], rank4[2]
    out = next(iter(loop_op.outputs.values()))
    scale = 1.0 / math.sqrt(D)
    kname = loop_op.name if loop_op.name.startswith("k_") else f"k_{loop_op.name}"
    src = warp_chain_kernel_source(kname, B=B, H=H, S=S, D=D, scale=scale)

    g = Graph()
    for nid in (q_id, k_id, v_id):
        t = loop_op.inputs[nid]
        g.add_node(op=InputOp(), inputs=[], output=Tensor(nid, tuple(t.shape), F16), node_id=nid)
    g.add_node(
        op=CudaOp(
            kernel_source=src,
            kernel_name=kname,
            arg_order=(q_id, k_id, v_id, out.name),
            grid=((B * H * (S // 16),), (1,), (1,)),
            block=((32,), (1,), (1,)),
        ),
        inputs=[q_id, k_id, v_id],
        output=Tensor(out.name, tuple(out.shape), out.dtype),
        node_id=out.name,
    )
    g.outputs = [out.name]
    return g
