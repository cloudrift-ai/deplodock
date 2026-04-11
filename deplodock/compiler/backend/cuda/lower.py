"""Lower graph IR to CUDA IR (KernelDef)."""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.backend.cuda.ir import (
    ArrayAccess,
    Assign,
    AugAssign,
    BinOp,
    CudaBuiltin,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    Var,
    VarDecl,
)
from deplodock.compiler.ir import Graph
from deplodock.compiler.ops import FusedReduceElementwiseOp, MatmulOp


@dataclass
class MatmulConfig:
    """Tunable parameters for matmul lowering.

    `threads_y` / `threads_x` describe the **launched CUDA block dimensions**
    (blockDim.y, blockDim.x) — they are NOT the tile shape in the M/N
    directions. For "one thread per output element" strategies (`naive`,
    `smem_tiled`, `register_blocked`) the tile happens to equal the thread
    dim because `thread_m = thread_n = 1`; for tile-coarsened strategies the
    tile is `threads_y * thread_m` × `threads_x * thread_n`.

    The `tma_db` strategy currently hardcodes `(32, 8)` in its kernel template
    and does not read these fields at lowering time — the `_tma()` factory in
    `tuning.py` still records `(32, 8)` here so the JSON trace is honest about
    the launched block.
    """

    strategy: str = "naive"
    threads_y: int = 16
    threads_x: int = 16
    block_k: int = 16
    thread_m: int = 1
    thread_n: int = 1
    vectorize: bool = False
    unroll_k: bool = False
    double_buffer: bool = False
    smem_pad: int = 1
    coarsen_rows: int = 1
    coarsen_cols: int = 1
    assume_aligned: bool = False
    k_splits: int = 1
    batch_count: int = 1


def lower_graph(graph: Graph, config: MatmulConfig | None = None) -> KernelDef:
    """Lower a fused graph to a CUDA kernel definition.

    Currently handles the matmul case: a single FusedReduceElementwiseOp
    output node with reduce_fn=sum, elementwise_fn=mul, and two 2D inputs.
    """
    # Find the output node.
    if len(graph.outputs) != 1:
        raise ValueError(f"Expected exactly 1 output, got {len(graph.outputs)}")
    out_node = graph.nodes[graph.outputs[0]]

    if isinstance(out_node.op, MatmulOp):
        pass  # MatmulOp is always sum/mul — proceed directly
    elif isinstance(out_node.op, FusedReduceElementwiseOp):
        if out_node.op.reduce_fn != "sum" or out_node.op.elementwise_fn != "mul":
            raise ValueError(f"Only sum/mul fusion supported, got {out_node.op.reduce_fn}/{out_node.op.elementwise_fn}")
    else:
        raise ValueError(f"Expected MatmulOp or FusedReduceElementwiseOp output, got {type(out_node.op).__name__}")

    if len(out_node.inputs) != 2:
        raise ValueError(f"Expected 2 inputs, got {len(out_node.inputs)}")

    config = config or MatmulConfig()
    match config.strategy:
        case "naive":
            return _lower_matmul_naive(graph, out_node, config)
        case "tma_db":
            return _lower_matmul_tma_db(graph, out_node, config)
        case "tma_db_tf32":
            return _lower_matmul_tma_db_tf32(graph, out_node, config)
        case "tma_db_fma_tf32":
            return _lower_matmul_tma_db_fma_tf32(graph, out_node, config)
        case _:
            raise ValueError(f"Unknown strategy: {config.strategy}")


def _lower_matmul_naive(graph, out_node, config):
    """Generate a naive matmul kernel: each thread computes one C element."""
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]

    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    # Thread-to-element mapping.
    row_var = VarDecl(
        "int",
        "row",
        BinOp(
            "+",
            BinOp("*", CudaBuiltin("blockIdx.y"), CudaBuiltin("blockDim.y")),
            CudaBuiltin("threadIdx.y"),
        ),
    )
    col_var = VarDecl(
        "int",
        "col",
        BinOp(
            "+",
            BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")),
            CudaBuiltin("threadIdx.x"),
        ),
    )

    # Accumulator.
    acc_decl = VarDecl("float", "acc", Literal(0.0))

    # K-loop body: acc += A[row*K+k] * B[k*N+col]
    a_index = BinOp("+", BinOp("*", Var("row"), Var("K")), Var("k"))
    b_index = BinOp("+", BinOp("*", Var("k"), Var("N")), Var("col"))
    k_loop = ForLoop(
        var="k",
        start=Literal(0, dtype="int"),
        end=Var("K"),
        body=[
            AugAssign(
                "acc",
                "+=",
                BinOp("*", ArrayAccess(a_name, a_index), ArrayAccess(b_name, b_index)),
            )
        ],
    )

    # Write result: C[row*N+col] = acc
    c_index = BinOp("+", BinOp("*", Var("row"), Var("N")), Var("col"))
    write_result = Assign(ArrayAccess(c_name, c_index), Var("acc"))

    # Bounds check.
    bounds = IfStmt(
        cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", Var("col"), Var("N"))),
        body=[acc_decl, k_loop, write_result],
    )

    return KernelDef(
        name="fused_matmul",
        params=[
            KernelParam("float*", a_name),
            KernelParam("float*", b_name),
            KernelParam("float*", c_name),
            KernelParam("int", "M"),
            KernelParam("int", "N"),
            KernelParam("int", "K"),
        ],
        body=[row_var, col_var, bounds],
        block_size=(config.threads_x, config.threads_y, 1),
    )


def _lower_matmul_tma_db(graph, out_node, config):
    """TMA double-buffer FP32 SGEMM with mbarrier pipelining.

    Uses cp.async.bulk.tensor.2d for loading A and B tiles.
    Double-buffer: 2 shared memory stages, TMA loading overlaps with FMA compute.
    Beats cuBLAS at 512 (132%) and 1024 (112%) with exact FP32 accuracy.
    Tile: 64×128, BK=config.block_k (default 32), 256 threads, 8×4 per thread.
    Requires CUtensorMap descriptors passed via __grid_constant__.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.backend.cuda.ir import RawCode

    bk = config.block_k if config.block_k >= 16 else 32
    tm = config.thread_m if config.thread_m > 1 else 8
    tn = 4
    tx, ty = 32, 8
    bm, bn = ty * tm, tx * tn
    a_size = bm * bk
    b_size = bk * bn
    stage = a_size + b_size

    # Generate FMA block: B load first, then interleave A load + FMA per row
    fma_lines = []
    fma_lines.append(f"            float b0=cbs[kk*{bn}+tc],b1=cbs[kk*{bn}+tc+1],b2=cbs[kk*{bn}+tc+2],b3=cbs[kk*{bn}+tc+3];")
    for i in range(tm):
        fma_lines.append(f"            float a{i}=cas[(tr+{i})*{bk}+kk];")
        fma_lines.append(f"            c{i}0+=a{i}*b0;c{i}1+=a{i}*b1;c{i}2+=a{i}*b2;c{i}3+=a{i}*b3;")
    fma_block = "\n".join(fma_lines)

    # Generate write block
    write_lines = []
    for i in range(tm):
        write_lines.append(f"    W({i},c{i}0,c{i}1,c{i}2,c{i}3)")
    write_block = "\n".join(write_lines)

    # Generate accumulator declarations
    acc_decl = "float " + ",".join(f"c{i}{j}=0" for i in range(tm) for j in range(tn)) + ";"

    use_k_splits = config.k_splits > 1
    use_batch = config.batch_count > 1
    if use_k_splits and use_batch:
        # Both modes consume blockIdx.z, and the runner only dispatches one of
        # them on grid.z, so combining them silently corrupts results. Batched
        # GEMM already saturates the grid via the batch dim, so k-splits is
        # redundant — callers must collapse k_splits to 1 when batching.
        raise ValueError(
            f"MatmulConfig: k_splits={config.k_splits} and batch_count={config.batch_count} "
            "cannot both be > 1 (blockIdx.z collision). Set k_splits=1 for batched GEMM."
        )

    # TMA descriptor reference: indexed by batch when batched
    tma_a_ref = f"&{a_name}_tma[batch]" if use_batch else f"&{a_name}_tma"
    tma_b_ref = f"&{b_name}_tma[batch]" if use_batch else f"&{b_name}_tma"
    batch_setup = "int batch=blockIdx.z;\n" if use_batch else ""
    # C pointer: offset by batch stride when batched
    c_ptr = f"({c_name}+batch*M*N)" if use_batch else c_name

    # K-range: compile-time when k_splits==1, runtime when k_splits>1
    if use_k_splits:
        k_range_code = f"int k_per_split=(K/{bk}/k_splits)*{bk};\nint k_start=blockIdx.z*k_per_split;\nint k_end=(blockIdx.z==k_splits-1)?K:k_start+k_per_split;"
        nt_expr = f"(k_end-k_start)/{bk}"
        first_k = "k_start"
        next_k = f"k_start+(t+1)*{bk}"
        write_macro = f"""#if (M % {bm} == 0 && N % {bn} == 0)
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);int gc=bn+tc;float*Cout={c_ptr}; \
atomicAdd(&Cout[gr*N+gc],v0);atomicAdd(&Cout[gr*N+gc+1],v1);atomicAdd(&Cout[gr*N+gc+2],v2);atomicAdd(&Cout[gr*N+gc+3],v3);}}}}
#else
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);if(gr<M){{{{int gc=bn+tc;float*Cout={c_ptr}; \
if(gc<N)atomicAdd(&Cout[gr*N+gc],v0);if(gc+1<N)atomicAdd(&Cout[gr*N+gc+1],v1);if(gc+2<N)atomicAdd(&Cout[gr*N+gc+2],v2);if(gc+3<N)atomicAdd(&Cout[gr*N+gc+3],v3);}}}}}}}}
#endif"""
    else:
        k_range_code = ""
        nt_expr = f"K/{bk}"
        first_k = "0"
        next_k = f"(t+1)*{bk}"
        # Bounds checks eliminated at compile time when M/N are tile-aligned
        write_macro = f"""#if (M % {bm} == 0 && N % {bn} == 0)
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);int gc=bn+tc;float*Cout={c_ptr}; \
Cout[gr*N+gc]=v0;Cout[gr*N+gc+1]=v1;Cout[gr*N+gc+2]=v2;Cout[gr*N+gc+3]=v3;}}}}
#else
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);if(gr<M){{{{int gc=bn+tc;float*Cout={c_ptr}; \
if(gc<N)Cout[gr*N+gc]=v0;if(gc+1<N)Cout[gr*N+gc+1]=v1;if(gc+2<N)Cout[gr*N+gc+2]=v2;if(gc+3<N)Cout[gr*N+gc+3]=v3;}}}}}}}}
#endif"""

    kernel_code = f"""\
extern __shared__ __align__(128) char dsmem[];
float*smem=(float*)dsmem;
uint64_t*mbar=(uint64_t*)(dsmem+2*{stage}*4);
const int as0=(int)__cvta_generic_to_shared(&smem[0]);
const int bs0=(int)__cvta_generic_to_shared(&smem[{a_size}]);
const int as1=(int)__cvta_generic_to_shared(&smem[{stage}]);
const int bs1=(int)__cvta_generic_to_shared(&smem[{stage}+{a_size}]);
const int mb0=(int)__cvta_generic_to_shared(&mbar[0]);
const int mb1=(int)__cvta_generic_to_shared(&mbar[1]);
int tid=threadIdx.y*{tx}+threadIdx.x;
int tr=threadIdx.y*{tm},tc=threadIdx.x*{tn};
{batch_setup}// CTA swizzle: rasterize in groups of SWIZ row-blocks for L2 A-tile reuse
const int SWIZ=8;
int ntx=(N+{bn - 1})/{bn};
int nty=(M+{bm - 1})/{bm};
int pid=blockIdx.x+blockIdx.y*gridDim.x;
int grp=pid/(ntx*SWIZ);
int rem=pid%(ntx*SWIZ);
int by_s=grp*SWIZ+rem%SWIZ;
int bx_s=rem/SWIZ;
if(by_s>=nty||bx_s>=ntx)return;
int bm=by_s*{bm},bn=bx_s*{bn};
{k_range_code}
if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));asm volatile("fence.mbarrier_init.release.cluster;");}}
__syncthreads();
{acc_decl}
const int bytes={stage * 4};
int p0=0,p1=0,nt={nt_expr};
if(nt>0&&tid==0){{asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(as0),"l"({tma_a_ref}),"r"({first_k}),"r"(bm),"r"(mb0):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(bs0),"l"({tma_b_ref}),"r"(bn),"r"({first_k}),"r"(mb0):"memory");}}
for(int t=0;t<nt;t++){{
    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;
    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;
    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));
    if(s==0)p0^=1;else p1^=1;
    if(tid==0&&t+1<nt){{int nk={next_k};asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(nm),"r"(bytes):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(na),"l"({tma_a_ref}),"r"(nk),"r"(bm),"r"(nm):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(nb),"l"({tma_b_ref}),"r"(bn),"r"(nk),"r"(nm):"memory");}}
    float*cas=&smem[s*{stage}];float*cbs=&smem[s*{stage}+{a_size}];
    #pragma unroll
    for(int kk=0;kk<{bk};kk++){{
{fma_block}
    }}
    __syncthreads();
}}
{write_macro}
{write_block}"""

    params = [
        KernelParam("float*", c_name),
    ]
    if use_k_splits:
        params.append(KernelParam("int", "k_splits"))

    return KernelDef(
        name="fused_matmul",
        params=params,
        body=[RawCode(kernel_code)],
        block_size=(tx, ty, 1),
        includes=["cuda.h"],
        tile_m=bm,
        tile_n=bn,
        tma_params=[f"{a_name}_tma", f"{b_name}_tma"],
        batched=use_batch,
    )


def _lower_matmul_tma_db_tf32(graph, out_node, config):
    """Pure TF32 SGEMM with TMA double-buffer loads.

    Same FP32 in/out interface as cublasSgemm. The kernel TMA-loads FP32
    A and B tiles into shared memory, then loads them as TF32 wmma fragments
    (load_matrix_sync with precision::tf32 truncates the FP32 mantissa to 10
    bits at load time — no separate quantization step needed). Each output
    fragment is computed by a single wmma::mma_sync per K-chunk, accumulating
    into a single FP32 fragment. Trivial epilogue: one store_matrix_sync per
    fragment.

    This is the simplest possible tensor-core SGEMM kernel. It serves two
    roles:
        1. Reference baseline for what our codegen can achieve when there's
           no emulation overhead.
        2. The TF32 half of a future FFMA + TF32 hybrid (which gives up some
           precision in part of the matrix in exchange for using both pipes
           concurrently).

    Layout: configurable BM, BN, BK via DEPLODOCK_TF32_BM/BN/BK env vars.
    Defaults: BM=128, BN=128, BK=16, 8 warps in 4 row x 2 col layout (each
    warp owns 2 row frags x 4 col frags = 8 fragments).

    Precision: TF32 has ~3.3 decimal digits of mantissa precision (10 bits),
    less than FP32's ~7 (23 bits). For ML / well-conditioned numerical work
    this is usually fine; for strict bit-equivalence to FP32 it isn't.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    import os as _os

    from deplodock.compiler.backend.cuda.ir import RawCode

    # Defaults found by sweep at 8192: 128x64x8 + min_blocks=2 hits ~51.5 TFLOPS
    # at 96.4% tensor pipe util. Bottleneck is long_scoreboard stalls (62%) =
    # waiting on operand collector during ldmatrix from row-major smem.
    bm = int(_os.environ.get("DEPLODOCK_TF32_BM", "128"))
    bn = int(_os.environ.get("DEPLODOCK_TF32_BN", "64"))
    bk = int(_os.environ.get("DEPLODOCK_TF32_BK", "8"))
    tx, ty = 32, 8
    warp_rows = 4
    warp_cols = 2
    assert warp_rows * warp_cols == 8
    # TF32 wmma is 16x16x8 (k=8), not 16x16x16 like BF16/FP16. Adjust.
    wmma_m, wmma_n, wmma_k = 16, 16, 8
    frag_rows_per_warp = bm // (warp_rows * wmma_m)
    frag_cols_per_warp = bn // (warp_cols * wmma_n)
    assert warp_rows * frag_rows_per_warp * wmma_m == bm, f"BM={bm} not divisible by warp_rows*wmma_m={warp_rows * wmma_m}"
    assert warp_cols * frag_cols_per_warp * wmma_n == bn, f"BN={bn} not divisible by warp_cols*wmma_n={warp_cols * wmma_n}"
    assert bk % wmma_k == 0, f"BK={bk} must be a multiple of wmma_k={wmma_k}"
    n_kchunks = bk // wmma_k

    # One FP32 accumulator fragment per (warp, row_frag, col_frag).
    acc_decl_lines = []
    for ri in range(frag_rows_per_warp):
        for ci in range(frag_cols_per_warp):
            acc_decl_lines.append(f"wmma::fragment<wmma::accumulator,{wmma_m},{wmma_n},{wmma_k},float> hc{ri}_{ci};")
            acc_decl_lines.append(f"wmma::fill_fragment(hc{ri}_{ci},0.0f);")
    acc_decl = "\n".join(acc_decl_lines)

    # Inner loop: per K-tile, for each k_chunk in [0, BK/wmma_k), load A frags
    # and B frags via wmma TF32 (auto-truncates from FP32 in smem), then 1 mma
    # per (row_frag, col_frag). Interleaved load/mma pattern lets each mma
    # start as soon as its operands arrive.
    inner_lines = []
    inner_lines.append(f"        wmma::fragment<wmma::matrix_a,{wmma_m},{wmma_n},{wmma_k},wmma::precision::tf32,wmma::row_major> ha;")
    inner_lines.append(f"        wmma::fragment<wmma::matrix_b,{wmma_m},{wmma_n},{wmma_k},wmma::precision::tf32,wmma::row_major> hb;")
    for kc in range(n_kchunks):
        for ri in range(frag_rows_per_warp):
            a_row_offset = f"(warp_row*{frag_rows_per_warp * wmma_m}+{ri * wmma_m})"
            inner_lines.append(f"        wmma::load_matrix_sync(ha,&A_smem[{a_row_offset}*{bk}+{kc * wmma_k}],{bk});")
            for ci in range(frag_cols_per_warp):
                b_col_offset = f"(warp_col*{frag_cols_per_warp * wmma_n}+{ci * wmma_n})"
                inner_lines.append(f"        wmma::load_matrix_sync(hb,&B_smem[{kc * wmma_k}*{bn}+{b_col_offset}],{bn});")
                inner_lines.append(f"        wmma::mma_sync(hc{ri}_{ci},ha,hb,hc{ri}_{ci});")
    inner_block = "\n".join(inner_lines)

    a_size = bm * bk
    b_size = bk * bn
    stage = a_size + b_size  # FP32 floats
    tma_bytes = stage * 4

    use_batch = config.batch_count > 1
    tma_a_ref = f"&{a_name}_tma[batch]" if use_batch else f"&{a_name}_tma"
    tma_b_ref = f"&{b_name}_tma[batch]" if use_batch else f"&{b_name}_tma"
    batch_setup = "int batch=blockIdx.z;\n" if use_batch else ""
    c_ptr = f"({c_name}+batch*M*N)" if use_batch else c_name

    # Trivial epilogue: one store_matrix_sync per fragment, with edge guards.
    # Uses c_ptr so batched mode offsets correctly.
    epilogue_lines = []
    for ri in range(frag_rows_per_warp):
        for ci in range(frag_cols_per_warp):
            row_expr = f"(bm+warp_row*{frag_rows_per_warp * wmma_m}+{ri * wmma_m})"
            col_expr = f"(bn+warp_col*{frag_cols_per_warp * wmma_n}+{ci * wmma_n})"
            epilogue_lines.append(
                f"    if({row_expr}+{wmma_m}<=M&&{col_expr}+{wmma_n}<=N) wmma::store_matrix_sync(&{c_ptr}[{row_expr}*N+{col_expr}],hc{ri}_{ci},N,wmma::mem_row_major);"
            )
    epilogue_block = "\n".join(epilogue_lines)
    epilogue_block_b = epilogue_block

    kernel_code = f"""\
extern __shared__ __align__(128) char dsmem[];
float*smem=(float*)dsmem;
uint64_t*mbar=(uint64_t*)(dsmem+2*{stage}*4);
const int as0=(int)__cvta_generic_to_shared(&smem[0]);
const int bs0=(int)__cvta_generic_to_shared(&smem[{a_size}]);
const int as1=(int)__cvta_generic_to_shared(&smem[{stage}]);
const int bs1=(int)__cvta_generic_to_shared(&smem[{stage}+{a_size}]);
const int mb0=(int)__cvta_generic_to_shared(&mbar[0]);
const int mb1=(int)__cvta_generic_to_shared(&mbar[1]);
int tid=threadIdx.y*{tx}+threadIdx.x;
int wid=threadIdx.y;
int warp_row=wid/{warp_cols};
int warp_col=wid%{warp_cols};
{batch_setup}const int SWIZ=8;
int ntx=(N+{bn - 1})/{bn};
int nty=(M+{bm - 1})/{bm};
int pid=blockIdx.x+blockIdx.y*gridDim.x;
int grp=pid/(ntx*SWIZ);
int rem=pid%(ntx*SWIZ);
int by_s=grp*SWIZ+rem%SWIZ;
int bx_s=rem/SWIZ;
if(by_s>=nty||bx_s>=ntx)return;
int bm=by_s*{bm},bn=bx_s*{bn};
if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));asm volatile("fence.mbarrier_init.release.cluster;");}}
__syncthreads();
{acc_decl}
const int bytes={tma_bytes};
int p0=0,p1=0,nt=K/{bk};
if(nt>0&&tid==0){{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(as0),"l"({tma_a_ref}),"r"(0),"r"(bm),"r"(mb0):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(bs0),"l"({tma_b_ref}),"r"(bn),"r"(0),"r"(mb0):"memory");
}}
for(int t=0;t<nt;t++){{
    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;
    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;
    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));
    if(s==0)p0^=1;else p1^=1;
    if(tid==0&&t+1<nt){{
        int nk=(t+1)*{bk};
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(nm),"r"(bytes):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(na),"l"({tma_a_ref}),"r"(nk),"r"(bm),"r"(nm):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(nb),"l"({tma_b_ref}),"r"(bn),"r"(nk),"r"(nm):"memory");
    }}
    float*A_smem=&smem[s*{stage}];
    float*B_smem=&smem[s*{stage}+{a_size}];
{inner_block}
    __syncthreads();
}}
{epilogue_block_b}"""

    return KernelDef(
        name="fused_matmul",
        params=[KernelParam("float*", c_name)],
        body=[RawCode(kernel_code)],
        block_size=(tx, ty, 1),
        includes=["cuda.h", "mma.h"],
        tile_m=bm,
        tile_n=bn,
        tma_params=[f"{a_name}_tma", f"{b_name}_tma"],
        batched=use_batch,
        min_blocks_per_sm=int(_os.environ.get("DEPLODOCK_TF32_MIN_BLOCKS", "2")),
    )


def _lower_matmul_tma_db_fma_tf32(graph, out_node, config):
    """FMA + TF32 hybrid SGEMM: native FFMA on top rows, TF32 on bottom rows.

    Splits the 8 warps in a CTA into:
        * FFMA group (warps 0..ffma_warps-1) using the FP32 FMA pipe to
          compute the top FFMA_BM rows of each output tile in bit-grade FP32.
        * TF32 group (warps ffma_warps..7) using the tensor pipe with
          wmma::precision::tf32 to compute the bottom TF32_BM rows.

    The two pipes (FMA and tensor) are physically independent, so the warp
    schedulers can issue both concurrently on the same SM. Goal: stack the
    throughputs and exceed either alone.

    Both groups read the same FP32 A/B tiles from TMA double-buffered smem.
    The FFMA group runs the standard tma_db inner loop. The TF32 group runs
    wmma::load_matrix_sync (auto-truncate FP32 to TF32 at load time) +
    mma_sync. No quantization, no scale management.

    Precision: top FFMA_BM rows are bit-grade FP32 (FFMA), bottom TF32_BM
    rows are TF32-precision (~3.3 sig digits). The kernel is NOT bit-
    equivalent to native FP32 in the bottom rows — same contract as the
    pure tma_db_tf32 strategy for that portion.

    Default layout: BM=128, BN=128, BK=8, 4 FFMA + 4 TF32 warps.
    FFMA group: 4 warps × 32 threads × tm=16 × tn=4 → 64×128 (top half)
    TF32 group: 2x2 warp grid × 2 row frags × 4 col frags × 16 → 64×128 (bottom half)
    All configurable via DEPLODOCK_FMATF32_* env vars.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    import os as _os

    from deplodock.compiler.backend.cuda.ir import RawCode

    # Hybrid layout. Defaults tuned across the (size, batch) sweep matrix:
    # 4 FFMA + 4 TF32 warps, tm=24, bk=32, tf32_frag_rows=1.
    #   ffma_bm = 96, tf32_bm = 32, bm = 128.  75/25 work split.
    # The smaller per-thread tile (vs the previous tm=28 single-8192-tuned
    # default) generalizes better across grid-saturation regimes — wins in
    # 6/9 of the (size, batch) sweep cases.
    ffma_warps = int(_os.environ.get("DEPLODOCK_FMATF32_FFMA_WARPS", "4"))
    tf32_warps = 8 - ffma_warps
    assert tf32_warps >= 1, f"need at least 1 tf32 warp; got ffma_warps={ffma_warps}"
    tx, ty = 32, 8
    tm = int(_os.environ.get("DEPLODOCK_FMATF32_TM", "24"))
    tn = 4
    bk = int(_os.environ.get("DEPLODOCK_FMATF32_BK", "32"))
    # FFMA group thread layout: ffma_warps row warps × tx=32 col threads
    # FFMA tile = ffma_warps * tm rows × tx * tn = 128 cols
    ffma_bm = ffma_warps * tm  # 4*16 = 64
    bn = tx * tn  # 128
    # TF32 group warp layout: 2x2 grid (configurable)
    tf32_warp_rows = int(_os.environ.get("DEPLODOCK_FMATF32_TF32_WARP_ROWS", "2"))
    tf32_warp_cols = tf32_warps // tf32_warp_rows
    assert tf32_warp_rows * tf32_warp_cols == tf32_warps, f"tf32_warps={tf32_warps} not divisible by tf32_warp_rows={tf32_warp_rows}"
    wmma_m, wmma_n, wmma_k = 16, 16, 8
    # TF32 fragment grid that tiles BN exactly
    tf32_frag_cols_per_warp = bn // (tf32_warp_cols * wmma_n)
    assert tf32_warp_cols * tf32_frag_cols_per_warp * wmma_n == bn, (
        f"BN={bn} not divisible by tf32_warp_cols*wmma_n={tf32_warp_cols * wmma_n}"
    )
    # TF32 BM (bottom rows) = configurable via frag_rows_per_warp
    tf32_frag_rows_per_warp = int(_os.environ.get("DEPLODOCK_FMATF32_TF32_FRAG_ROWS", "1"))
    tf32_bm = tf32_warp_rows * tf32_frag_rows_per_warp * wmma_m
    bm = ffma_bm + tf32_bm
    assert bk % wmma_k == 0
    n_kchunks = bk // wmma_k
    a_size = bm * bk
    b_size = bk * bn
    stage = a_size + b_size

    use_batch = config.batch_count > 1
    tma_a_ref = f"&{a_name}_tma[batch]" if use_batch else f"&{a_name}_tma"
    tma_b_ref = f"&{b_name}_tma[batch]" if use_batch else f"&{b_name}_tma"
    batch_setup = "int batch=blockIdx.z;\n" if use_batch else ""
    c_ptr = f"({c_name}+batch*M*N)" if use_batch else c_name

    # FFMA accumulator declarations and inner block.
    ffma_acc_decl = "float " + ",".join(f"c{i}_{j}=0.0f" for i in range(tm) for j in range(tn)) + ";"

    fma_lines = []
    fma_lines.append(f"            float b0=B_smem[kk*{bn}+tc],b1=B_smem[kk*{bn}+tc+1],b2=B_smem[kk*{bn}+tc+2],b3=B_smem[kk*{bn}+tc+3];")
    for i in range(tm):
        fma_lines.append(f"            float a{i}=A_smem[(tr+{i})*{bk}+kk];")
        fma_lines.append(f"            c{i}_0+=a{i}*b0;c{i}_1+=a{i}*b1;c{i}_2+=a{i}*b2;c{i}_3+=a{i}*b3;")
    fma_block = "\n".join(fma_lines)

    # FFMA epilogue: write top FFMA_BM rows
    write_macro = f"""#if (M % {bm} == 0 && N % {bn} == 0)
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);int gc=bn+tc;float*Cout={c_ptr}; \
Cout[gr*N+gc]=v0;Cout[gr*N+gc+1]=v1;Cout[gr*N+gc+2]=v2;Cout[gr*N+gc+3]=v3;}}}}
#else
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);if(gr<M){{{{int gc=bn+tc;float*Cout={c_ptr}; \
if(gc<N)Cout[gr*N+gc]=v0;if(gc+1<N)Cout[gr*N+gc+1]=v1;if(gc+2<N)Cout[gr*N+gc+2]=v2;if(gc+3<N)Cout[gr*N+gc+3]=v3;}}}}}}}}
#endif"""
    ffma_write_lines = [f"        W({i},c{i}_0,c{i}_1,c{i}_2,c{i}_3)" for i in range(tm)]
    ffma_write_block = "\n".join(ffma_write_lines)

    # TF32 accumulator declarations
    tf32_acc_lines = []
    for ri in range(tf32_frag_rows_per_warp):
        for ci in range(tf32_frag_cols_per_warp):
            tf32_acc_lines.append(f"wmma::fragment<wmma::accumulator,{wmma_m},{wmma_n},{wmma_k},float> hc{ri}_{ci};")
            tf32_acc_lines.append(f"wmma::fill_fragment(hc{ri}_{ci},0.0f);")
    tf32_acc_decl = "\n".join(tf32_acc_lines)

    # TF32 inner loop
    tf32_inner_lines = []
    tf32_inner_lines.append(f"        wmma::fragment<wmma::matrix_a,{wmma_m},{wmma_n},{wmma_k},wmma::precision::tf32,wmma::row_major> ha;")
    tf32_inner_lines.append(f"        wmma::fragment<wmma::matrix_b,{wmma_m},{wmma_n},{wmma_k},wmma::precision::tf32,wmma::row_major> hb;")
    for kc in range(n_kchunks):
        for ri in range(tf32_frag_rows_per_warp):
            # TF32 group's row strip starts at row FFMA_BM in the CTA tile
            a_row_offset = f"({ffma_bm}+t_warp_row*{tf32_frag_rows_per_warp * wmma_m}+{ri * wmma_m})"
            tf32_inner_lines.append(f"        wmma::load_matrix_sync(ha,&A_smem[{a_row_offset}*{bk}+{kc * wmma_k}],{bk});")
            for ci in range(tf32_frag_cols_per_warp):
                b_col_offset = f"(t_warp_col*{tf32_frag_cols_per_warp * wmma_n}+{ci * wmma_n})"
                tf32_inner_lines.append(f"        wmma::load_matrix_sync(hb,&B_smem[{kc * wmma_k}*{bn}+{b_col_offset}],{bn});")
                tf32_inner_lines.append(f"        wmma::mma_sync(hc{ri}_{ci},ha,hb,hc{ri}_{ci});")
    tf32_inner_block = "\n".join(tf32_inner_lines)

    # TF32 epilogue: write bottom rows (uses c_ptr for batched mode)
    tf32_write_lines = []
    for ri in range(tf32_frag_rows_per_warp):
        for ci in range(tf32_frag_cols_per_warp):
            row_expr = f"(bm+{ffma_bm}+t_warp_row*{tf32_frag_rows_per_warp * wmma_m}+{ri * wmma_m})"
            col_expr = f"(bn+t_warp_col*{tf32_frag_cols_per_warp * wmma_n}+{ci * wmma_n})"
            tf32_write_lines.append(
                f"        if({row_expr}+{wmma_m}<=M&&{col_expr}+{wmma_n}<=N) wmma::store_matrix_sync(&{c_ptr}[{row_expr}*N+{col_expr}],hc{ri}_{ci},N,wmma::mem_row_major);"
            )
    tf32_write_block = "\n".join(tf32_write_lines)

    tma_bytes = stage * 4
    write_macro_b = write_macro
    tf32_write_block_b = tf32_write_block

    kernel_code = f"""\
extern __shared__ __align__(128) char dsmem[];
float*smem=(float*)dsmem;
uint64_t*mbar=(uint64_t*)(dsmem+2*{stage}*4);
const int as0=(int)__cvta_generic_to_shared(&smem[0]);
const int bs0=(int)__cvta_generic_to_shared(&smem[{a_size}]);
const int as1=(int)__cvta_generic_to_shared(&smem[{stage}]);
const int bs1=(int)__cvta_generic_to_shared(&smem[{stage}+{a_size}]);
const int mb0=(int)__cvta_generic_to_shared(&mbar[0]);
const int mb1=(int)__cvta_generic_to_shared(&mbar[1]);
int tid=threadIdx.y*{tx}+threadIdx.x;
int wid=threadIdx.y;
int lane=threadIdx.x;
bool is_tf32=(wid>={ffma_warps});
// FFMA group thread coords
int tr=wid*{tm};
int tc=lane*{tn};
// TF32 group warp coords (warps {ffma_warps}..7)
int t_wid=wid-{ffma_warps};
int t_warp_row=t_wid/{tf32_warp_cols};
int t_warp_col=t_wid%{tf32_warp_cols};
{batch_setup}const int SWIZ=8;
int ntx=(N+{bn - 1})/{bn};
int nty=(M+{bm - 1})/{bm};
int pid=blockIdx.x+blockIdx.y*gridDim.x;
int grp=pid/(ntx*SWIZ);
int rem=pid%(ntx*SWIZ);
int by_s=grp*SWIZ+rem%SWIZ;
int bx_s=rem/SWIZ;
if(by_s>=nty||bx_s>=ntx)return;
int bm=by_s*{bm},bn=bx_s*{bn};
if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));asm volatile("fence.mbarrier_init.release.cluster;");}}
__syncthreads();
{ffma_acc_decl}
{tf32_acc_decl}
const int bytes={tma_bytes};
int p0=0,p1=0,nt=K/{bk};
if(nt>0&&tid==0){{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(as0),"l"({tma_a_ref}),"r"(0),"r"(bm),"r"(mb0):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(bs0),"l"({tma_b_ref}),"r"(bn),"r"(0),"r"(mb0):"memory");
}}
for(int t=0;t<nt;t++){{
    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;
    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;
    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));
    if(s==0)p0^=1;else p1^=1;
    if(tid==0&&t+1<nt){{
        int nk=(t+1)*{bk};
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(nm),"r"(bytes):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(na),"l"({tma_a_ref}),"r"(nk),"r"(bm),"r"(nm):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(nb),"l"({tma_b_ref}),"r"(bn),"r"(nk),"r"(nm):"memory");
    }}
    float*A_smem=&smem[s*{stage}];
    float*B_smem=&smem[s*{stage}+{a_size}];
    if(!is_tf32){{
        #pragma unroll
        for(int kk=0;kk<{bk};kk++){{
{fma_block}
        }}
    }}else{{
{tf32_inner_block}
    }}
    __syncthreads();
}}
{write_macro_b}
if(!is_tf32){{
{ffma_write_block}
}}else{{
{tf32_write_block_b}
}}"""

    return KernelDef(
        name="fused_matmul",
        params=[KernelParam("float*", c_name)],
        body=[RawCode(kernel_code)],
        block_size=(tx, ty, 1),
        includes=["cuda.h", "mma.h"],
        tile_m=bm,
        tile_n=bn,
        tma_params=[f"{a_name}_tma", f"{b_name}_tma"],
        batched=use_batch,
        min_blocks_per_sm=int(_os.environ.get("DEPLODOCK_FMATF32_MIN_BLOCKS", "0")),
    )
