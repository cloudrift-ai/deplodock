"""Validated reference kernel for the **fused tensor-core flash** — the executable
target the warp-chain codegen must generate.

This is NOT the compiler's output: it is a hand-written FA-2 kernel that proves the
whole design works end-to-end on real hardware, and pins the lane-layout contracts the
Phase-3 codegen relies on. Each section maps to a plan phase:

- **QK^T `mma`** (Phase 2.1) — ``Q`` loaded as the ``ldmatrix.x4`` A fragment (m16k16),
  ``K`` as the **transposed-B** native col-major B (manual pack, ``n=lane/4``,
  ``k=(lane%4)*2{+8}``) — the same atom + `b_trans` the `atomize_cell` `out_index` path
  produces. Output: the score ``S[16,16]`` as two ``m16n8`` C-fragments (kv 0-7 / 8-15).
- **fragment online-softmax** (Phase 3) — ``rowmax`` / ``rowsum`` over the C-fragment's
  N (kv) lanes: combine the two N-tiles + the in-lane col pair, then a ``__shfl_xor``
  butterfly over the 4-lane col group (``xor 2``, ``xor 1``). The C-fragment layout is
  rows ``g`` / ``g+8`` (``g=lane/4``), cols ``(lane%4)*2+{0,1}`` — the same layout
  ``ir/kernel`` documents for ``RegStore``. The per-row ``m`` / ``l`` and the ``α``
  rescale are the carrier's ``merge`` / ``combine_states`` in fragment-distributed form.
- **C→A handoff** (Phase 3, v1 SMEM) — the probability ``P`` C-fragment is written
  row-major to smem and ``ldmatrix.x4``-loaded back as the P@V A operand (avoids the v2
  register shuffle).
- **P@V `mma`** (Phase 2.2) — ``A=P`` (from smem), ``B=V`` canonical col-major
  (``ldmatrix.x2.trans``), accumulating ``O[16,D]`` across the KV stream.

If this diverges from torch, the codegen target is wrong; keep it green as the spec.
One warp per 16 query rows, ``D=16``, fp16 in / f32 accumulate, non-causal.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import requires_cuda

_KERNEL = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ void mma_m16n8k16(float* d, const unsigned* a, const unsigned* b, const float* c){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]),
      "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
// A (m16k16): ldmatrix.x4 — row=lane%16, k-block=(lane/16)*8.
__device__ __forceinline__ void ldm_a(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; unsigned addr=__cvta_generic_to_shared(sm + (lane%16)*ldm + (lane/16)*8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(addr));
}
// canonical B[k,n] k-major: ldmatrix.x2.trans -> col-major; row=lane%16.
__device__ __forceinline__ void ldm_b_trans(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; unsigned addr=__cvta_generic_to_shared(sm + (lane%16)*ldm);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
    :"=r"(r[0]),"=r"(r[1]):"r"(addr));
}
// transposed-B (Q@K^T) native col-major: manual pack. n=lane/4, k=(lane%4)*2{+8}.
__device__ __forceinline__ void load_b_native(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; int n=lane/4; int kb=(lane%4)*2;
  __half2 h0=__halves2half2(sm[n*ldm+kb+0], sm[n*ldm+kb+1]);
  __half2 h1=__halves2half2(sm[n*ldm+kb+8+0], sm[n*ldm+kb+8+1]);
  r[0]=*reinterpret_cast<unsigned*>(&h0); r[1]=*reinterpret_cast<unsigned*>(&h1);
}

extern "C" __global__ void fa2(const __half* Q,const __half* K,const __half* V,float* O,int S,float scale){
  int qb = blockIdx.x; int lane=threadIdx.x&31; const int D=16;
  __shared__ __half qs[16*16], ks[16*16], vs[16*16], ps[16*16];
  for(int i=lane;i<16*D;i+=32){ qs[i]=Q[(qb*16)*D + i]; }
  __syncwarp();
  unsigned qa[4]; ldm_a(qa, qs, D);                 // Q -> A fragment, once per query tile
  float m0=-1e30f,m1=-1e30f,l0=0,l1=0;              // online stats, rows g / g+8 per lane
  float Of[2][4]={{0,0,0,0},{0,0,0,0}};             // O[16,D] accumulator (2 N-tiles of d)
  int g = lane/4;
  for(int kv0=0; kv0<S; kv0+=16){                   // KV stream
    for(int i=lane;i<16*D;i+=32){ ks[i]=K[(kv0)*D+i]; vs[i]=V[(kv0)*D+i]; }
    __syncwarp();
    float Sf[2][4];                                 // QK^T mma -> score C-fragments
    for(int nt=0;nt<2;nt++){
      unsigned kb[2]; load_b_native(kb, ks + nt*8*D, D);
      float z[4]={0,0,0,0}; mma_m16n8k16(Sf[nt], qa, kb, z);
      for(int e=0;e<4;e++) Sf[nt][e]*=scale;
    }
    float r0=fmaxf(fmaxf(Sf[0][0],Sf[0][1]),fmaxf(Sf[1][0],Sf[1][1]));   // fragment rowmax
    float r1=fmaxf(fmaxf(Sf[0][2],Sf[0][3]),fmaxf(Sf[1][2],Sf[1][3]));
    r0=fmaxf(r0,__shfl_xor_sync(-1,r0,2)); r0=fmaxf(r0,__shfl_xor_sync(-1,r0,1));
    r1=fmaxf(r1,__shfl_xor_sync(-1,r1,2)); r1=fmaxf(r1,__shfl_xor_sync(-1,r1,1));
    float mn0=fmaxf(m0,r0), mn1=fmaxf(m1,r1);
    float a0=__expf(m0-mn0), a1=__expf(m1-mn1);      // α rescale (combine_states)
    float Pf[2][4]; float s0=0,s1=0;
    for(int nt=0;nt<2;nt++){
      Pf[nt][0]=__expf(Sf[nt][0]-mn0); Pf[nt][1]=__expf(Sf[nt][1]-mn0);
      Pf[nt][2]=__expf(Sf[nt][2]-mn1); Pf[nt][3]=__expf(Sf[nt][3]-mn1);
      s0+=Pf[nt][0]+Pf[nt][1]; s1+=Pf[nt][2]+Pf[nt][3];
    }
    s0+=__shfl_xor_sync(-1,s0,2); s0+=__shfl_xor_sync(-1,s0,1);   // fragment rowsum
    s1+=__shfl_xor_sync(-1,s1,2); s1+=__shfl_xor_sync(-1,s1,1);
    l0=l0*a0+s0; l1=l1*a1+s1;
    for(int nt=0;nt<2;nt++){ Of[nt][0]*=a0;Of[nt][1]*=a0;Of[nt][2]*=a1;Of[nt][3]*=a1; }
    int c0=(lane%4)*2;                               // C->A handoff: P C-frag -> smem row-major
    for(int nt=0;nt<2;nt++){
      ps[g*16 + nt*8 + c0+0]=__float2half(Pf[nt][0]);  ps[g*16 + nt*8 + c0+1]=__float2half(Pf[nt][1]);
      ps[(g+8)*16 + nt*8 + c0+0]=__float2half(Pf[nt][2]); ps[(g+8)*16 + nt*8 + c0+1]=__float2half(Pf[nt][3]);
    }
    __syncwarp();
    unsigned pa[4]; ldm_a(pa, ps, 16);               // P@V mma: A=P (ldmatrix), B=V canonical
    for(int nt=0;nt<2;nt++){
      unsigned vb[2]; ldm_b_trans(vb, vs + nt*8, D);
      mma_m16n8k16(Of[nt], pa, vb, Of[nt]);
    }
    m0=mn0; m1=mn1;
  }
  int c0=(lane%4)*2;                                 // epilogue O/l + store (C-frag layout)
  for(int nt=0;nt<2;nt++){
    O[((qb*16)+g)*D + nt*8 + c0+0]=Of[nt][0]/l0;   O[((qb*16)+g)*D + nt*8 + c0+1]=Of[nt][1]/l0;
    O[((qb*16)+g+8)*D + nt*8 + c0+0]=Of[nt][2]/l1; O[((qb*16)+g+8)*D + nt*8 + c0+1]=Of[nt][3]/l1;
  }
}
"""


@requires_cuda
@pytest.mark.parametrize("S", [16, 32, 64, 128])
def test_fused_tensorcore_flash_reference_matches_torch(S):
    """The hand-written fused tensor-core flash matches torch SDPA across the KV stream
    (1–8 tiles). The validated spec for the warp-chain codegen — every lane layout (the
    A/B fragments, the C-fragment row reduction, the C→A handoff) is exercised here."""
    import cupy as cp
    import torch

    from emmy.compiler.backend.cuda import nvcc

    fn = nvcc.load_function(_KERNEL, "fa2", "", uses_tma=False)
    torch.manual_seed(S)
    D = 16
    q, k, v = (torch.randn(S, D, dtype=torch.float16) for _ in range(3))
    dq, dk, dv = (cp.asarray(t.numpy()) for t in (q, k, v))
    d_out = cp.zeros((S, D), cp.float32)
    fn((S // 16,), (32,), (dq, dk, dv, d_out, np.int32(S), np.float32(1.0 / np.sqrt(D))))
    got = torch.from_numpy(cp.asnumpy(d_out))
    ref = torch.nn.functional.scaled_dot_product_attention(q.cuda().float(), k.cuda().float(), v.cuda().float()).cpu()
    max_diff = float((got - ref).abs().max())
    assert max_diff < 2e-3, f"fused TC flash S={S} max_diff={max_diff:.2e}"
