"""Lower graph IR to CUDA IR (KernelDef)."""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.cuda.ir import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinOp,
    CudaBuiltin,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    PragmaUnroll,
    SyncThreads,
    Var,
    VarDecl,
)
from deplodock.compiler.ir import Graph
from deplodock.compiler.ops import FusedReduceElementwiseOp


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

    if not isinstance(out_node.op, FusedReduceElementwiseOp):
        raise ValueError(f"Expected FusedReduceElementwiseOp output, got {type(out_node.op).__name__}")

    if out_node.op.reduce_fn != "sum" or out_node.op.elementwise_fn != "mul":
        raise ValueError(f"Only sum/mul fusion supported, got {out_node.op.reduce_fn}/{out_node.op.elementwise_fn}")

    if len(out_node.inputs) != 2:
        raise ValueError(f"Expected 2 inputs, got {len(out_node.inputs)}")

    config = config or MatmulConfig()
    match config.strategy:
        case "naive":
            return _lower_matmul_naive(graph, out_node, config)
        case "smem_tiled":
            return _lower_matmul_smem_tiled(graph, out_node, config)
        case "register_blocked":
            return _lower_matmul_register_blocked(graph, out_node, config)
        case "coarsened_f4":
            return _lower_matmul_coarsened_f4(graph, out_node, config)
        case "coarsened_2r4c":
            return _lower_matmul_coarsened_2r4c(graph, out_node, config)
        case "hybrid_smem_f4":
            return _lower_matmul_hybrid_smem_f4(graph, out_node, config)
        case "flat_scalar":
            return _lower_matmul_flat_scalar(graph, out_node, config)
        case "flat_f4":
            return _lower_matmul_flat_f4(graph, out_node, config)
        case "hybrid_1r_f4":
            return _lower_matmul_hybrid_1r_f4(graph, out_node, config)
        case "smem_ab_blocked":
            return _lower_matmul_smem_ab_blocked(graph, out_node, config)
        case "wmma_bf16":
            return _lower_matmul_wmma_bf16(graph, out_node, config)
        case "tma_db":
            return _lower_matmul_tma_db(graph, out_node, config)
        case "tma_db_hybrid":
            return _lower_matmul_tma_db_hybrid(graph, out_node, config)
        case "tma_db_tf32":
            return _lower_matmul_tma_db_tf32(graph, out_node, config)
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


def _lower_matmul_smem_tiled(graph, out_node, config):
    """Shared-memory tiled matmul kernel with cooperative loading.

    Each thread block loads BM×BK of A and BK×BN of B into shared memory
    using all threads cooperatively (strided loop over linear thread ID).
    Iterates K in BK steps. Each thread computes one output element.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    bm = config.threads_y
    bn = config.threads_x
    bk = config.block_k
    pad = config.smem_pad
    num_threads = bm * bn

    from deplodock.compiler.cuda.ir import Ternary

    # Global row/col for this thread's output element.
    row_var = VarDecl(
        "int",
        "row",
        BinOp("+", BinOp("*", CudaBuiltin("blockIdx.y"), Literal(bm, "int")), CudaBuiltin("threadIdx.y")),
    )
    col_var = VarDecl(
        "int",
        "col",
        BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(bn, "int")), CudaBuiltin("threadIdx.x")),
    )

    # Linear thread ID for cooperative loading.
    tid_var = VarDecl("int", "tid", BinOp("+", BinOp("*", CudaBuiltin("threadIdx.y"), Literal(bn, "int")), CudaBuiltin("threadIdx.x")))

    # Block corner in global matrix.
    block_row_var = VarDecl("int", "block_row", BinOp("*", CudaBuiltin("blockIdx.y"), Literal(bm, "int")))
    block_col_var = VarDecl("int", "block_col", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(bn, "int")))

    # Shared memory tiles (1D, padded).
    a_stride = bk + pad
    b_stride = bn + pad
    as_decl = ArrayDecl("__shared__ float", "As", [bm * a_stride])
    bs_decl = ArrayDecl("__shared__ float", "Bs", [bk * b_stride])

    acc_decl = VarDecl("float", "acc", Literal(0.0))

    # --- Tile loop (steps by BK through K) ---
    # Cooperative A-tile load: BM*BK elements, strided by num_threads.
    a_tile_size = bm * bk
    a_load_row = VarDecl("int", "a_r", BinOp("/", Var("i"), Literal(bk, "int")))
    a_load_col = VarDecl("int", "a_c", BinOp("%", Var("i"), Literal(bk, "int")))
    a_g_row = BinOp("+", Var("block_row"), Var("a_r"))
    a_g_col = BinOp("+", Var("tile_k"), Var("a_c"))
    a_global = BinOp("+", BinOp("*", a_g_row, Var("K")), a_g_col)
    a_bounds = BinOp("&&", BinOp("<", a_g_row, Var("M")), BinOp("<", a_g_col, Var("K")))
    a_smem_idx = BinOp("+", BinOp("*", Var("a_r"), Literal(a_stride, "int")), Var("a_c"))
    a_store = Assign(
        ArrayAccess("As", a_smem_idx),
        Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0)),
    )
    a_load_loop = ForLoop(
        "i",
        Var("tid"),
        Literal(a_tile_size, "int"),
        [a_load_row, a_load_col, a_store],
        step=Literal(num_threads, "int"),
    )

    # Cooperative B-tile load: BK*BN elements, strided by num_threads.
    b_tile_size = bk * bn
    b_load_row = VarDecl("int", "b_r", BinOp("/", Var("j"), Literal(bn, "int")))
    b_load_col = VarDecl("int", "b_c", BinOp("%", Var("j"), Literal(bn, "int")))
    b_g_row = BinOp("+", Var("tile_k"), Var("b_r"))
    b_g_col = BinOp("+", Var("block_col"), Var("b_c"))
    b_global = BinOp("+", BinOp("*", b_g_row, Var("N")), b_g_col)
    b_bounds = BinOp("&&", BinOp("<", b_g_row, Var("K")), BinOp("<", b_g_col, Var("N")))
    b_smem_idx = BinOp("+", BinOp("*", Var("b_r"), Literal(b_stride, "int")), Var("b_c"))
    b_store = Assign(
        ArrayAccess("Bs", b_smem_idx),
        Ternary(b_bounds, ArrayAccess(b_name, b_global), Literal(0.0)),
    )
    b_load_loop = ForLoop(
        "j",
        Var("tid"),
        Literal(b_tile_size, "int"),
        [b_load_row, b_load_col, b_store],
        step=Literal(num_threads, "int"),
    )

    sync1 = SyncThreads()

    # Inner k-loop: acc += As[ty][kk] * Bs[kk][tx]
    ty_expr = CudaBuiltin("threadIdx.y")
    tx_expr = CudaBuiltin("threadIdx.x")
    as_read = ArrayAccess("As", BinOp("+", BinOp("*", ty_expr, Literal(a_stride, "int")), Var("kk")))
    bs_read = ArrayAccess("Bs", BinOp("+", BinOp("*", Var("kk"), Literal(b_stride, "int")), tx_expr))
    inner_body = [AugAssign("acc", "+=", BinOp("*", as_read, bs_read))]

    inner_loop_stmts: list = []
    if config.unroll_k:
        inner_loop_stmts.append(PragmaUnroll())
    inner_loop_stmts.append(ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), inner_body))

    sync2 = SyncThreads()

    tile_loop = ForLoop(
        "tile_k",
        Literal(0, "int"),
        Var("K"),
        [a_load_loop, b_load_loop, sync1, *inner_loop_stmts, sync2],
        step=Literal(bk, "int"),
    )

    # Write result with bounds check.
    c_index = BinOp("+", BinOp("*", Var("row"), Var("N")), Var("col"))
    write_result = IfStmt(
        cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", Var("col"), Var("N"))),
        body=[Assign(ArrayAccess(c_name, c_index), Var("acc"))],
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
        body=[row_var, col_var, tid_var, block_row_var, block_col_var, as_decl, bs_decl, acc_decl, tile_loop, write_result],
        block_size=(bn, bm, 1),
    )


def _lower_matmul_register_blocked(graph, out_node, config):
    """Register-blocked matmul: each thread computes TM×TN output elements.

    Block of (BN/TN, BM/TM) threads computes BM×BN output tile.
    Shared memory holds BM×BK of A and BK×BN of B.
    Inner loop loads TM A-values and TN B-values into registers,
    then does TM×TN FMAs (outer product).
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    bm = config.threads_y
    bn = config.threads_x
    bk = config.block_k
    tm = config.thread_m
    tn = config.thread_n
    pad = config.smem_pad

    threads_x = bn // tn  # threads per block in x
    threads_y = bm // tm  # threads per block in y
    num_threads = threads_x * threads_y

    from deplodock.compiler.cuda.ir import Ternary

    body: list = []

    # Thread ID and block offsets.
    body.append(
        VarDecl("int", "tid", BinOp("+", BinOp("*", CudaBuiltin("threadIdx.y"), Literal(threads_x, "int")), CudaBuiltin("threadIdx.x")))
    )
    body.append(VarDecl("int", "block_row", BinOp("*", CudaBuiltin("blockIdx.y"), Literal(bm, "int"))))
    body.append(VarDecl("int", "block_col", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(bn, "int"))))

    # Thread's starting row/col within the block tile.
    body.append(VarDecl("int", "thread_row", BinOp("*", CudaBuiltin("threadIdx.y"), Literal(tm, "int"))))
    body.append(VarDecl("int", "thread_col", BinOp("*", CudaBuiltin("threadIdx.x"), Literal(tn, "int"))))

    # Shared memory.
    a_stride = bk + pad
    b_stride = bn + pad
    body.append(ArrayDecl("__shared__ float", "As", [bm * a_stride]))
    body.append(ArrayDecl("__shared__ float", "Bs", [bk * b_stride]))

    # Register accumulators: acc[TM * TN] = 0.
    body.append(ArrayDecl("float", "acc", [tm * tn]))
    # Zero accumulators.
    body.append(ForLoop("i", Literal(0, "int"), Literal(tm * tn, "int"), [Assign(ArrayAccess("acc", Var("i")), Literal(0.0))]))

    # Register arrays for A and B fragments.
    body.append(ArrayDecl("float", "regA", [tm]))
    body.append(ArrayDecl("float", "regB", [tn]))

    # === Tile loop ===
    tile_body: list = []

    # Cooperative A-tile load: BM * BK elements.
    a_tile_size = bm * bk
    a_load_body = [
        VarDecl("int", "a_r", BinOp("/", Var("li"), Literal(bk, "int"))),
        VarDecl("int", "a_c", BinOp("%", Var("li"), Literal(bk, "int"))),
    ]
    a_g_row = BinOp("+", Var("block_row"), Var("a_r"))
    a_g_col = BinOp("+", Var("tile_k"), Var("a_c"))
    a_bounds = BinOp("&&", BinOp("<", a_g_row, Var("M")), BinOp("<", a_g_col, Var("K")))
    a_global = BinOp("+", BinOp("*", a_g_row, Var("K")), a_g_col)
    a_smem = BinOp("+", BinOp("*", Var("a_r"), Literal(a_stride, "int")), Var("a_c"))
    a_load_body.append(
        Assign(
            ArrayAccess("As", a_smem),
            Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0)),
        )
    )
    tile_body.append(ForLoop("li", Var("tid"), Literal(a_tile_size, "int"), a_load_body, step=Literal(num_threads, "int")))

    # Cooperative B-tile load: BK * BN elements.
    b_tile_size = bk * bn
    b_load_body = [
        VarDecl("int", "b_r", BinOp("/", Var("lj"), Literal(bn, "int"))),
        VarDecl("int", "b_c", BinOp("%", Var("lj"), Literal(bn, "int"))),
    ]
    b_g_row = BinOp("+", Var("tile_k"), Var("b_r"))
    b_g_col = BinOp("+", Var("block_col"), Var("b_c"))
    b_bounds = BinOp("&&", BinOp("<", b_g_row, Var("K")), BinOp("<", b_g_col, Var("N")))
    b_global = BinOp("+", BinOp("*", b_g_row, Var("N")), b_g_col)
    b_smem = BinOp("+", BinOp("*", Var("b_r"), Literal(b_stride, "int")), Var("b_c"))
    b_load_body.append(
        Assign(
            ArrayAccess("Bs", b_smem),
            Ternary(b_bounds, ArrayAccess(b_name, b_global), Literal(0.0)),
        )
    )
    tile_body.append(ForLoop("lj", Var("tid"), Literal(b_tile_size, "int"), b_load_body, step=Literal(num_threads, "int")))

    tile_body.append(SyncThreads())

    # Inner k-loop over BK dimension.
    inner_body: list = []

    # Load TM values from As column kk into regA.
    # regA[m] = As[(thread_row + m) * a_stride + kk]
    reg_a_load_body = [
        Assign(
            ArrayAccess("regA", Var("m")),
            ArrayAccess("As", BinOp("+", BinOp("*", BinOp("+", Var("thread_row"), Var("m")), Literal(a_stride, "int")), Var("kk"))),
        ),
    ]
    # All inner loops get #pragma unroll for maximum performance.
    inner_body.append(PragmaUnroll())
    inner_body.append(ForLoop("m", Literal(0, "int"), Literal(tm, "int"), reg_a_load_body))

    # Load TN values from Bs row kk into regB.
    # regB[n] = Bs[kk * b_stride + thread_col + n]
    reg_b_load_body = [
        Assign(
            ArrayAccess("regB", Var("n")),
            ArrayAccess("Bs", BinOp("+", BinOp("*", Var("kk"), Literal(b_stride, "int")), BinOp("+", Var("thread_col"), Var("n")))),
        ),
    ]
    inner_body.append(PragmaUnroll())
    inner_body.append(ForLoop("n", Literal(0, "int"), Literal(tn, "int"), reg_b_load_body))

    # Outer product: acc[m*TN+n] = acc[m*TN+n] + regA[m]*regB[n]
    acc_idx_expr = BinOp("+", BinOp("*", Var("m"), Literal(tn, "int")), Var("n"))
    fma_body = [
        Assign(
            ArrayAccess("acc", acc_idx_expr),
            BinOp("+", ArrayAccess("acc", acc_idx_expr), BinOp("*", ArrayAccess("regA", Var("m")), ArrayAccess("regB", Var("n")))),
        ),
    ]
    inner_body.append(PragmaUnroll())
    fma_n_loop = ForLoop("n", Literal(0, "int"), Literal(tn, "int"), fma_body)
    inner_body.append(PragmaUnroll())
    fma_m_loop = ForLoop("m", Literal(0, "int"), Literal(tm, "int"), [PragmaUnroll(), fma_n_loop])
    inner_body.append(fma_m_loop)

    # Always unroll the k-loop for register-blocked kernels.
    k_loop_stmts: list = [PragmaUnroll()]
    k_loop_stmts.append(ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), inner_body))

    tile_body.extend(k_loop_stmts)
    tile_body.append(SyncThreads())

    tile_loop = ForLoop("tile_k", Literal(0, "int"), Var("K"), tile_body, step=Literal(bk, "int"))
    body.append(tile_loop)

    # Write results: C[(block_row+thread_row+m)*N + block_col+thread_col+n] = acc[m*TN+n]
    write_inner = []
    g_row = BinOp("+", BinOp("+", Var("block_row"), Var("thread_row")), Var("wm"))
    g_col = BinOp("+", BinOp("+", Var("block_col"), Var("thread_col")), Var("wn"))
    c_idx = BinOp("+", BinOp("*", g_row, Var("N")), g_col)
    acc_rd = BinOp("+", BinOp("*", Var("wm"), Literal(tn, "int")), Var("wn"))
    write_inner.append(
        IfStmt(
            cond=BinOp("&&", BinOp("<", g_row, Var("M")), BinOp("<", g_col, Var("N"))),
            body=[Assign(ArrayAccess(c_name, c_idx), ArrayAccess("acc", acc_rd))],
        )
    )
    write_n_loop = ForLoop("wn", Literal(0, "int"), Literal(tn, "int"), write_inner)
    write_m_loop = ForLoop("wm", Literal(0, "int"), Literal(tm, "int"), [write_n_loop])
    body.append(write_m_loop)

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
        body=body,
        block_size=(threads_x, threads_y, 1),
    )


def _lower_matmul_coarsened_f4(graph, out_node, config):
    """Coarsened matmul with float4 vectorized B loads.

    Each thread computes 4 consecutive output columns using float4 loads
    for B, sharing one A load across all 4 FMAs. Block: (32, 8).
    Achieves ~90% of cuBLAS on RTX 5090.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.cuda.ir import FieldAccess, VectorLoad

    body: list = []

    # Thread mapping: each thread handles 4 consecutive columns.
    body.append(
        VarDecl("int", "row", BinOp("+", BinOp("*", CudaBuiltin("blockIdx.y"), CudaBuiltin("blockDim.y")), CudaBuiltin("threadIdx.y")))
    )
    body.append(
        VarDecl(
            "int",
            "col_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")), CudaBuiltin("threadIdx.x")),
                Literal(4, "int"),
            ),
        )
    )

    # Accumulators for 4 columns.
    body.append(VarDecl("float", "acc0", Literal(0.0)))
    body.append(VarDecl("float", "acc1", Literal(0.0)))
    body.append(VarDecl("float", "acc2", Literal(0.0)))
    body.append(VarDecl("float", "acc3", Literal(0.0)))

    # K-loop body.
    a_load = ArrayAccess(a_name, BinOp("+", BinOp("*", Var("row"), Var("K")), Var("k")))
    k_body: list = [
        VarDecl("float", "a_val", a_load),
        # float4 b4 = *(float4*)(&B[k * N + col_base])
        VarDecl("float4", "b4", VectorLoad(b_name, BinOp("+", BinOp("*", Var("k"), Var("N")), Var("col_base")), 4)),
        # acc0 += a_val * b4.x; etc.
        AugAssign("acc0", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "x"))),
        AugAssign("acc1", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "y"))),
        AugAssign("acc2", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "z"))),
        AugAssign("acc3", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "w"))),
    ]

    # Scalar fallback k-loop for edge columns.
    k_body_scalar: list = [VarDecl("float", "a_val", a_load)]
    b_row_s = BinOp("*", Var("k"), Var("N"))
    for i in range(4):
        col_i = BinOp("+", Var("col_base"), Literal(i, "int"))
        k_body_scalar.append(
            IfStmt(
                cond=BinOp("<", col_i, Var("N")),
                body=[AugAssign(f"acc{i}", "+=", BinOp("*", Var("a_val"), ArrayAccess(b_name, BinOp("+", b_row_s, col_i))))],
            )
        )

    # Bounds-checked k-loop with float4 fast path and scalar fallback.
    row_ok = BinOp("<", Var("row"), Var("M"))
    n_aligned = BinOp("==", BinOp("%", Var("N"), Literal(4, "int")), Literal(0, "int"))
    col_f4_ok = BinOp("&&", BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")), n_aligned)
    col_any_ok = BinOp("<", Var("col_base"), Var("N"))
    body.append(
        IfStmt(
            cond=BinOp("&&", row_ok, col_f4_ok),
            body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body)],
            else_body=[
                IfStmt(
                    cond=BinOp("&&", row_ok, col_any_ok),
                    body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body_scalar)],
                ),
            ],
        )
    )

    # Write results with per-element bounds.
    for i, acc in enumerate(["acc0", "acc1", "acc2", "acc3"]):
        col_i = BinOp("+", Var("col_base"), Literal(i, "int"))
        c_idx = BinOp("+", BinOp("*", Var("row"), Var("N")), col_i)
        body.append(
            IfStmt(
                cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", col_i, Var("N"))),
                body=[Assign(ArrayAccess(c_name, c_idx), Var(acc))],
            )
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
        body=body,
        block_size=(config.threads_x, config.threads_y, 1),
    )


def _lower_matmul_coarsened_2r4c(graph, out_node, config):
    """Each thread computes 2 rows × 4 columns = 8 elements.

    Uses float4 loads for B (4 cols at once) and loads 2 A values per k-step.
    The B float4 is shared across both rows, giving 8 FMAs per 6 loads.
    Beats cuBLAS by ~18% on RTX 5090.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.cuda.ir import FieldAccess, VectorLoad

    body: list = []

    # row_base = (blockIdx.y * blockDim.y + threadIdx.y) * 2
    body.append(
        VarDecl(
            "int",
            "row_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.y"), CudaBuiltin("blockDim.y")), CudaBuiltin("threadIdx.y")),
                Literal(2, "int"),
            ),
        )
    )
    # col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4
    body.append(
        VarDecl(
            "int",
            "col_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")), CudaBuiltin("threadIdx.x")),
                Literal(4, "int"),
            ),
        )
    )

    # 8 accumulators: acc[row][col] for row in 0..1, col in 0..3
    acc_names = [f"acc{r}{c}" for r in range(2) for c in range(4)]
    for name in acc_names:
        body.append(VarDecl("float", name, Literal(0.0)))

    # K-loop body.
    k_body: list = []
    # Load 2 A values.
    k_body.append(VarDecl("float", "a0", ArrayAccess(a_name, BinOp("+", BinOp("*", Var("row_base"), Var("K")), Var("k")))))
    k_body.append(
        VarDecl(
            "float", "a1", ArrayAccess(a_name, BinOp("+", BinOp("*", BinOp("+", Var("row_base"), Literal(1, "int")), Var("K")), Var("k")))
        )
    )
    # Load float4 B.
    k_body.append(VarDecl("float4", "b4", VectorLoad(b_name, BinOp("+", BinOp("*", Var("k"), Var("N")), Var("col_base")), 4)))
    # 8 FMAs: row 0
    for c, field in enumerate(["x", "y", "z", "w"]):
        k_body.append(AugAssign(f"acc0{c}", "+=", BinOp("*", Var("a0"), FieldAccess(Var("b4"), field))))
    # 8 FMAs: row 1
    for c, field in enumerate(["x", "y", "z", "w"]):
        k_body.append(AugAssign(f"acc1{c}", "+=", BinOp("*", Var("a1"), FieldAccess(Var("b4"), field))))

    # Scalar fallback k-loop for edge columns/rows.
    k_body_scalar: list = []
    # Load A values with bounds checks.
    a0_load = ArrayAccess(a_name, BinOp("+", BinOp("*", Var("row_base"), Var("K")), Var("k")))
    a1_load = ArrayAccess(a_name, BinOp("+", BinOp("*", BinOp("+", Var("row_base"), Literal(1, "int")), Var("K")), Var("k")))
    k_body_scalar.append(VarDecl("float", "a0", a0_load))
    k_body_scalar.append(VarDecl("float", "a1", a1_load))
    b_row_s = BinOp("*", Var("k"), Var("N"))
    for c in range(4):
        col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
        b_scalar = ArrayAccess(b_name, BinOp("+", b_row_s, col_c))
        scalar_fma: list = [VarDecl("float", f"bv{c}", b_scalar)]
        for r in range(2):
            row_r = BinOp("+", Var("row_base"), Literal(r, "int"))
            scalar_fma.append(
                IfStmt(
                    cond=BinOp("<", row_r, Var("M")),
                    body=[AugAssign(f"acc{r}{c}", "+=", BinOp("*", Var(f"a{r}"), Var(f"bv{c}")))],
                )
            )
        k_body_scalar.append(IfStmt(cond=BinOp("<", col_c, Var("N")), body=scalar_fma))

    # Bounds-checked k-loop: float4 fast path + scalar fallback.
    f4_cond = BinOp(
        "&&",
        BinOp("<", BinOp("+", Var("row_base"), Literal(1, "int")), Var("M")),
        BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")),
    )
    any_cond = BinOp(
        "&&",
        BinOp("<", Var("row_base"), Var("M")),
        BinOp("<", Var("col_base"), Var("N")),
    )
    body.append(
        IfStmt(
            cond=f4_cond,
            body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body)],
            else_body=[
                IfStmt(cond=any_cond, body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body_scalar)]),
            ],
        )
    )

    # Write results with per-element bounds.
    for r in range(2):
        for c in range(4):
            row_r = BinOp("+", Var("row_base"), Literal(r, "int"))
            col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
            c_idx = BinOp("+", BinOp("*", row_r, Var("N")), col_c)
            body.append(
                IfStmt(
                    cond=BinOp("&&", BinOp("<", row_r, Var("M")), BinOp("<", col_c, Var("N"))),
                    body=[Assign(ArrayAccess(c_name, c_idx), Var(f"acc{r}{c}"))],
                )
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
        body=body,
        block_size=(32, 8, 1),
    )


def _lower_matmul_hybrid_smem_f4(graph, out_node, config):
    """Hybrid: shared memory A + float4 B loads + 2-row coarsening.

    Each thread computes 2 rows x 4 cols. A is loaded into shared memory
    (eliminates redundant DRAM reads across threads in x-dimension).
    B is loaded via float4 directly from global memory.
    Block: (32, 8). BK = config.block_k (default 32).
    Beats cuBLAS by 30-42% on RTX 5090 across all tested sizes.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name
    bk = config.block_k

    from deplodock.compiler.cuda.ir import FieldAccess, Ternary, VectorLoad

    body: list = []
    rows_per_thread = config.thread_m if config.thread_m > 1 else 2
    cols_per_thread = 4
    threads_y = config.threads_y if config.threads_y != 16 else 8
    threads_x = config.threads_x if config.threads_x != 16 else 32
    smem_rows = threads_y * rows_per_thread  # 16
    smem_stride = bk + 1  # padding to avoid bank conflicts

    # Thread mapping.
    body.append(
        VarDecl(
            "int",
            "row_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.y"), CudaBuiltin("blockDim.y")), CudaBuiltin("threadIdx.y")),
                Literal(rows_per_thread, "int"),
            ),
        )
    )
    body.append(
        VarDecl(
            "int",
            "col_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")), CudaBuiltin("threadIdx.x")),
                Literal(cols_per_thread, "int"),
            ),
        )
    )

    # Shared memory for A.
    body.append(ArrayDecl("__shared__ float", "As", [smem_rows * smem_stride]))

    # Shared memory row indices for this thread's output rows.
    body.append(VarDecl("int", "sr0", BinOp("*", CudaBuiltin("threadIdx.y"), Literal(rows_per_thread, "int"))))
    for r in range(1, rows_per_thread):
        body.append(VarDecl("int", f"sr{r}", BinOp("+", Var("sr0"), Literal(r, "int"))))

    # Pre-computed smem base addresses for A reads (avoids multiply in inner loop).
    for r in range(rows_per_thread):
        body.append(VarDecl("int", f"ab{r}", BinOp("*", Var(f"sr{r}"), Literal(smem_stride, "int"))))

    # Accumulators.
    for r in range(rows_per_thread):
        for c in range(cols_per_thread):
            body.append(VarDecl("float", f"acc{r}{c}", Literal(0.0)))

    # === Tile loop body ===
    tile_body: list = []

    # Load A into shared memory.
    # BK <= 32: each thread loads at most 1 element (guard if BK < 32).
    # BK > 32: strided loop so each thread loads BK/32 elements.
    sr_pairs = [(f"sr{r}", r) for r in range(rows_per_thread)]
    aligned = config.assume_aligned
    if bk <= threads_x:
        a_col_expr = BinOp("+", Var("tile_k"), CudaBuiltin("threadIdx.x"))
        a_load_stmts: list = []
        for sr_name, row_off in sr_pairs:
            g_row = BinOp("+", Var("row_base"), Literal(row_off, "int"))
            a_global = BinOp("+", BinOp("*", g_row, Var("K")), a_col_expr)
            smem_idx = BinOp("+", BinOp("*", Var(sr_name), Literal(smem_stride, "int")), CudaBuiltin("threadIdx.x"))
            if aligned:
                a_load_stmts.append(Assign(ArrayAccess("As", smem_idx), ArrayAccess(a_name, a_global)))
            else:
                a_bounds = BinOp("&&", BinOp("<", g_row, Var("M")), BinOp("<", a_col_expr, Var("K")))
                a_load_stmts.append(Assign(ArrayAccess("As", smem_idx), Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0))))
        if bk < threads_x:
            tile_body.append(IfStmt(cond=BinOp("<", CudaBuiltin("threadIdx.x"), Literal(bk, "int")), body=a_load_stmts))
        else:
            tile_body.extend(a_load_stmts)
    else:
        a_col_strided = BinOp("+", Var("tile_k"), Var("a_s"))
        stride_body: list = []
        for sr_name, row_off in sr_pairs:
            g_row = BinOp("+", Var("row_base"), Literal(row_off, "int"))
            a_global = BinOp("+", BinOp("*", g_row, Var("K")), a_col_strided)
            smem_idx = BinOp("+", BinOp("*", Var(sr_name), Literal(smem_stride, "int")), Var("a_s"))
            if aligned:
                stride_body.append(Assign(ArrayAccess("As", smem_idx), ArrayAccess(a_name, a_global)))
            else:
                a_bounds = BinOp("&&", BinOp("<", g_row, Var("M")), BinOp("<", a_col_strided, Var("K")))
                stride_body.append(Assign(ArrayAccess("As", smem_idx), Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0))))
        tile_body.append(ForLoop("a_s", CudaBuiltin("threadIdx.x"), Literal(bk, "int"), stride_body, step=Literal(threads_x, "int")))

    tile_body.append(SyncThreads())

    # Inner k-loop: float4 fast path + scalar fallback for edge columns.
    a_reads = []
    for r in range(rows_per_thread):
        a_reads.append(VarDecl("float", f"a{r}", ArrayAccess("As", BinOp("+", Var(f"ab{r}"), Var("kk")))))

    # Float4 path (col_base + 3 < N and N%4==0).
    k_body_f4: list = list(a_reads)
    k_body_f4.append(
        VarDecl(
            "float4", "b4", VectorLoad(b_name, BinOp("+", BinOp("*", BinOp("+", Var("tile_k"), Var("kk")), Var("N")), Var("col_base")), 4)
        )
    )
    for r in range(rows_per_thread):
        for c, field in enumerate(["x", "y", "z", "w"]):
            k_body_f4.append(AugAssign(f"acc{r}{c}", "+=", BinOp("*", Var(f"a{r}"), FieldAccess(Var("b4"), field))))

    # Scalar path (edge columns where float4 would go out of bounds).
    k_body_scalar: list = list(a_reads)
    b_row_expr = BinOp("*", BinOp("+", Var("tile_k"), Var("kk")), Var("N"))
    for c in range(cols_per_thread):
        col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
        b_scalar = ArrayAccess(b_name, BinOp("+", b_row_expr, col_c))
        scalar_fmas = [VarDecl("float", f"bv{c}", b_scalar)]
        for r in range(rows_per_thread):
            scalar_fmas.append(AugAssign(f"acc{r}{c}", "+=", BinOp("*", Var(f"a{r}"), Var(f"bv{c}"))))
        k_body_scalar.append(IfStmt(cond=BinOp("<", col_c, Var("N")), body=scalar_fmas))

    if aligned:
        # Skip all bounds checks — assume M, N, K are multiples of tile dims.
        tile_body.append(PragmaUnroll())
        tile_body.append(ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), k_body_f4))
    else:
        n_aligned = BinOp("==", BinOp("%", Var("N"), Literal(4, "int")), Literal(0, "int"))
        col_ok = BinOp("&&", BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")), n_aligned)
        col_any = BinOp("<", Var("col_base"), Var("N"))
        tile_body.append(
            IfStmt(
                cond=col_ok,
                body=[PragmaUnroll(), ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), k_body_f4)],
                else_body=[
                    IfStmt(cond=col_any, body=[ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), k_body_scalar)]),
                ],
            )
        )
    tile_body.append(SyncThreads())

    body.append(ForLoop("tile_k", Literal(0, "int"), Var("K"), tile_body, step=Literal(bk, "int")))

    # Write results with per-element bounds checking.
    for r in range(rows_per_thread):
        for c in range(cols_per_thread):
            row_r = BinOp("+", Var("row_base"), Literal(r, "int"))
            col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
            c_idx = BinOp("+", BinOp("*", row_r, Var("N")), col_c)
            if aligned:
                body.append(Assign(ArrayAccess(c_name, c_idx), Var(f"acc{r}{c}")))
            else:
                body.append(
                    IfStmt(
                        cond=BinOp("&&", BinOp("<", row_r, Var("M")), BinOp("<", col_c, Var("N"))),
                        body=[Assign(ArrayAccess(c_name, c_idx), Var(f"acc{r}{c}"))],
                    )
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
        body=body,
        block_size=(threads_x, threads_y, 1),
    )


def _lower_matmul_flat_scalar(graph, out_node, config):
    """Flat 1D scalar kernel for small matrices.

    Maps threads in a 2D grid: row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x.
    No shared memory, no coarsening. Maximizes block count for SM occupancy.
    Block: (block_n, 1, 1) where block_n defaults to 128.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    threads_x = config.threads_x  # 128 by default

    body: list = []

    # Thread mapping: row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x
    body.append(VarDecl("int", "row", CudaBuiltin("blockIdx.y")))
    body.append(
        VarDecl(
            "int",
            "col",
            BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(threads_x, "int")), CudaBuiltin("threadIdx.x")),
        )
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
    body.append(bounds)

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
        body=body,
        block_size=(threads_x, 1, 1),
    )


def _lower_matmul_flat_f4(graph, out_node, config):
    """Flat 1D kernel with float4 B loads, 4 cols per thread.

    Maps row = blockIdx.y, col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4.
    No shared memory. Float4 vectorized B loads. Scalar fallback for edge columns.
    Block: (block_n, 1, 1) where block_n defaults to 32.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.cuda.ir import FieldAccess, VectorLoad

    threads_x = config.threads_x  # 32 by default

    body: list = []

    # Thread mapping: row = blockIdx.y, col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4
    body.append(VarDecl("int", "row", CudaBuiltin("blockIdx.y")))
    body.append(
        VarDecl(
            "int",
            "col_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(threads_x, "int")), CudaBuiltin("threadIdx.x")),
                Literal(4, "int"),
            ),
        )
    )

    # 4 accumulators.
    for i in range(4):
        body.append(VarDecl("float", f"acc{i}", Literal(0.0)))

    # K-loop with float4 B loads.
    a_load = ArrayAccess(a_name, BinOp("+", BinOp("*", Var("row"), Var("K")), Var("k")))
    k_body: list = [
        VarDecl("float", "a_val", a_load),
        VarDecl("float4", "b4", VectorLoad(b_name, BinOp("+", BinOp("*", Var("k"), Var("N")), Var("col_base")), 4)),
        AugAssign("acc0", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "x"))),
        AugAssign("acc1", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "y"))),
        AugAssign("acc2", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "z"))),
        AugAssign("acc3", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), "w"))),
    ]

    # Scalar fallback for edge columns.
    k_body_scalar: list = [VarDecl("float", "a_val", a_load)]
    b_row_s = BinOp("*", Var("k"), Var("N"))
    for i in range(4):
        col_i = BinOp("+", Var("col_base"), Literal(i, "int"))
        k_body_scalar.append(
            IfStmt(
                cond=BinOp("<", col_i, Var("N")),
                body=[AugAssign(f"acc{i}", "+=", BinOp("*", Var("a_val"), ArrayAccess(b_name, BinOp("+", b_row_s, col_i))))],
            )
        )

    row_ok = BinOp("<", Var("row"), Var("M"))
    n_aligned = BinOp("==", BinOp("%", Var("N"), Literal(4, "int")), Literal(0, "int"))
    col_f4_ok = BinOp("&&", BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")), n_aligned)
    col_any_ok = BinOp("<", Var("col_base"), Var("N"))
    body.append(
        IfStmt(
            cond=BinOp("&&", row_ok, col_f4_ok),
            body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body)],
            else_body=[
                IfStmt(
                    cond=BinOp("&&", row_ok, col_any_ok),
                    body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body_scalar)],
                ),
            ],
        )
    )

    # Write results with per-element bounds.
    for i in range(4):
        col_i = BinOp("+", Var("col_base"), Literal(i, "int"))
        c_idx = BinOp("+", BinOp("*", Var("row"), Var("N")), col_i)
        body.append(
            IfStmt(
                cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", col_i, Var("N"))),
                body=[Assign(ArrayAccess(c_name, c_idx), Var(f"acc{i}"))],
            )
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
        body=body,
        block_size=(threads_x, 1, 1),
    )


def _lower_matmul_hybrid_1r_f4(graph, out_node, config):
    """Hybrid: shared memory A + float4 B + 1-row (no row coarsening).

    Like hybrid_smem_f4 but each thread computes 1 row x 4 cols.
    Doubles the grid in y compared to 2-row variant, better for medium sizes.
    Block: (32, threads_y). BK = config.block_k.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name
    bk = config.block_k

    from deplodock.compiler.cuda.ir import FieldAccess, Ternary, VectorLoad

    body: list = []
    cols_per_thread = 4
    threads_y = config.threads_y
    threads_x = config.threads_x
    smem_rows = threads_y
    smem_stride = bk + 1

    body.append(
        VarDecl("int", "row", BinOp("+", BinOp("*", CudaBuiltin("blockIdx.y"), Literal(threads_y, "int")), CudaBuiltin("threadIdx.y")))
    )
    body.append(
        VarDecl(
            "int",
            "col_base",
            BinOp(
                "*",
                BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), Literal(threads_x, "int")), CudaBuiltin("threadIdx.x")),
                Literal(cols_per_thread, "int"),
            ),
        )
    )

    body.append(ArrayDecl("__shared__ float", "As", [smem_rows * smem_stride]))
    body.append(VarDecl("int", "sr", CudaBuiltin("threadIdx.y")))

    for c in range(cols_per_thread):
        body.append(VarDecl("float", f"acc{c}", Literal(0.0)))

    tile_body: list = []

    if bk <= threads_x:
        a_col_expr = BinOp("+", Var("tile_k"), CudaBuiltin("threadIdx.x"))
        a_global = BinOp("+", BinOp("*", Var("row"), Var("K")), a_col_expr)
        a_bounds = BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", a_col_expr, Var("K")))
        smem_idx = BinOp("+", BinOp("*", Var("sr"), Literal(smem_stride, "int")), CudaBuiltin("threadIdx.x"))
        a_store = Assign(ArrayAccess("As", smem_idx), Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0)))
        if bk < threads_x:
            tile_body.append(IfStmt(cond=BinOp("<", CudaBuiltin("threadIdx.x"), Literal(bk, "int")), body=[a_store]))
        else:
            tile_body.append(a_store)
    else:
        a_col_strided = BinOp("+", Var("tile_k"), Var("a_s"))
        a_global = BinOp("+", BinOp("*", Var("row"), Var("K")), a_col_strided)
        a_bounds = BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", a_col_strided, Var("K")))
        smem_idx = BinOp("+", BinOp("*", Var("sr"), Literal(smem_stride, "int")), Var("a_s"))
        stride_body = [Assign(ArrayAccess("As", smem_idx), Ternary(a_bounds, ArrayAccess(a_name, a_global), Literal(0.0)))]
        tile_body.append(ForLoop("a_s", CudaBuiltin("threadIdx.x"), Literal(bk, "int"), stride_body, step=Literal(threads_x, "int")))

    tile_body.append(SyncThreads())

    k_body_f4: list = [
        VarDecl("float", "a_val", ArrayAccess("As", BinOp("+", BinOp("*", Var("sr"), Literal(smem_stride, "int")), Var("kk")))),
        VarDecl(
            "float4",
            "b4",
            VectorLoad(b_name, BinOp("+", BinOp("*", BinOp("+", Var("tile_k"), Var("kk")), Var("N")), Var("col_base")), 4),
        ),
    ]
    for c, field in enumerate(["x", "y", "z", "w"]):
        k_body_f4.append(AugAssign(f"acc{c}", "+=", BinOp("*", Var("a_val"), FieldAccess(Var("b4"), field))))

    k_body_scalar: list = [
        VarDecl("float", "a_val", ArrayAccess("As", BinOp("+", BinOp("*", Var("sr"), Literal(smem_stride, "int")), Var("kk")))),
    ]
    b_row_expr = BinOp("*", BinOp("+", Var("tile_k"), Var("kk")), Var("N"))
    for c in range(cols_per_thread):
        col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
        b_load = ArrayAccess(b_name, BinOp("+", b_row_expr, col_c))
        k_body_scalar.append(
            IfStmt(
                cond=BinOp("<", col_c, Var("N")),
                body=[AugAssign(f"acc{c}", "+=", BinOp("*", Var("a_val"), b_load))],
            )
        )

    n_aligned = BinOp("==", BinOp("%", Var("N"), Literal(4, "int")), Literal(0, "int"))
    col_ok = BinOp("&&", BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")), n_aligned)
    col_any = BinOp("<", Var("col_base"), Var("N"))
    tile_body.append(
        IfStmt(
            cond=col_ok,
            body=[PragmaUnroll(), ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), k_body_f4)],
            else_body=[IfStmt(cond=col_any, body=[ForLoop("kk", Literal(0, "int"), Literal(bk, "int"), k_body_scalar)])],
        )
    )
    tile_body.append(SyncThreads())

    body.append(ForLoop("tile_k", Literal(0, "int"), Var("K"), tile_body, step=Literal(bk, "int")))

    for c in range(cols_per_thread):
        col_c = BinOp("+", Var("col_base"), Literal(c, "int"))
        c_idx = BinOp("+", BinOp("*", Var("row"), Var("N")), col_c)
        body.append(
            IfStmt(
                cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", col_c, Var("N"))),
                body=[Assign(ArrayAccess(c_name, c_idx), Var(f"acc{c}"))],
            )
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
        body=body,
        block_size=(threads_x, threads_y, 1),
    )


def _lower_matmul_smem_ab_blocked(graph, out_node, config):
    """Shared memory A+B with register blocking.

    Classic high-performance GEMM: tile both A and B into shared memory,
    then each thread computes a TM x (TN*4) register-blocked output tile.
    BM = block_m * thread_m, BN = block_n * thread_n * 4.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.cuda.ir import RawCode

    bk = config.block_k
    tm = config.thread_m
    tn = config.thread_n
    threads_x = config.threads_x
    threads_y = config.threads_y
    num_threads = threads_x * threads_y
    bm = threads_y * tm
    bn = threads_x * tn * 4
    pad = 1

    a_stride = bk + pad
    b_stride = bn + pad
    cols_per_t = tn * 4
    a_tile_count = bm * bk
    b_tile_count = bk * bn

    # Generate scalar accumulator declarations and FMA chains.
    acc_decls = "\n".join(f"float acc_{r}_{c} = 0.0f;" for r in range(tm) for c in range(cols_per_t))
    fma_chain = "\n".join(f"        acc_{r}_{c} += regA_{r} * regB_{c};" for r in range(tm) for c in range(cols_per_t))
    reg_a_loads = "\n".join(f"        float regA_{m} = As[(thread_row + {m}) * {a_stride} + kk];" for m in range(tm))
    reg_b_loads = "\n".join(f"        float regB_{n} = Bs[kk * {b_stride} + thread_col + {n}];" for n in range(cols_per_t))
    write_back = "\n".join(
        f"if (block_row + thread_row + {r} < M && block_col + thread_col + {c} < N)\n"
        f"    {c_name}[(block_row + thread_row + {r}) * N + block_col + thread_col + {c}] = acc_{r}_{c};"
        for r in range(tm)
        for c in range(cols_per_t)
    )

    kernel_code = (
        f"int tid = threadIdx.y * {threads_x} + threadIdx.x;\n"
        f"int block_row = blockIdx.y * {bm};\n"
        f"int block_col = blockIdx.x * {bn};\n"
        f"__shared__ float As[{bm * a_stride}];\n"
        f"__shared__ float Bs[{bk * b_stride}];\n"
        f"{acc_decls}\n"
        f"int thread_row = threadIdx.y * {tm};\n"
        f"int thread_col = threadIdx.x * {cols_per_t};\n"
        f"for (int tile_k = 0; tile_k < K; tile_k += {bk}) {{\n"
        f"    for (int i = tid; i < {a_tile_count}; i += {num_threads}) {{\n"
        f"        int r = i / {bk}, c = i % {bk};\n"
        f"        int gr = block_row + r, gc = tile_k + c;\n"
        f"        As[r * {a_stride} + c] = (gr < M && gc < K) ? {a_name}[gr * K + gc] : 0.0f;\n"
        f"    }}\n"
        f"    for (int i = tid; i < {b_tile_count}; i += {num_threads}) {{\n"
        f"        int r = i / {bn}, c = i % {bn};\n"
        f"        int gr = tile_k + r, gc = block_col + c;\n"
        f"        Bs[r * {b_stride} + c] = (gr < K && gc < N) ? {b_name}[gr * N + gc] : 0.0f;\n"
        f"    }}\n"
        f"    __syncthreads();\n"
        f"    #pragma unroll\n"
        f"    for (int kk = 0; kk < {bk}; kk++) {{\n"
        f"{reg_a_loads}\n"
        f"{reg_b_loads}\n"
        f"{fma_chain}\n"
        f"    }}\n"
        f"    __syncthreads();\n"
        f"}}\n"
        f"{write_back}"
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
        body=[RawCode(kernel_code)],
        block_size=(threads_x, threads_y, 1),
    )


def _lower_matmul_wmma_bf16(graph, out_node, config):
    """BF16 WMMA tensor core matmul with shift-based cooperative loading.

    Uses BF16 tensor cores for ~118% of cuBLAS FP32 SGEMM throughput.
    Accuracy: ~0.024% relative error (BF16 truncation, not exact FP32).
    Tile: 64x64, BK=16, 4 warps (128 threads), 2x2 WMMA 16x16x16 per warp.
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    from deplodock.compiler.cuda.ir import RawCode

    bm, bn = 64, 64
    bk_raw = config.block_k if config.block_k >= 16 else 16
    # BK must be power of 2 for shift-based loading
    bk = 1 << (bk_raw.bit_length() - 1) if bk_raw & (bk_raw - 1) else bk_raw
    num_threads = 128
    a_loads = (bm * bk) // num_threads  # 8
    b_loads = (bk * bn) // num_threads  # 8
    a_shift = bk.bit_length() - 1  # log2(BK)
    a_mask = bk - 1
    b_shift = 6  # log2(BN=64)
    b_mask = bn - 1

    kernel_code = (
        f"int warp_id = threadIdx.x / 32;\n"
        f"int warp_m = warp_id / 2, warp_n = warp_id % 2;\n"
        f"int block_m = blockIdx.y * {bm}, block_n = blockIdx.x * {bn};\n"
        f"int tid = threadIdx.x;\n"
        f"__shared__ __nv_bfloat16 As[{bm * bk}];\n"
        f"__shared__ __nv_bfloat16 Bs[{bk * bn}];\n"
        f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> c00, c01, c10, c11;\n"
        f"wmma::fill_fragment(c00, 0.0f);\n"
        f"wmma::fill_fragment(c01, 0.0f);\n"
        f"wmma::fill_fragment(c10, 0.0f);\n"
        f"wmma::fill_fragment(c11, 0.0f);\n"
        f"for (int tile_k = 0; tile_k < K; tile_k += {bk}) {{\n"
    )
    # A loading: shift-based, no div/mod
    kernel_code += "    #pragma unroll\n"
    kernel_code += f"    for (int p = 0; p < {a_loads}; p++) {{\n"
    kernel_code += f"        int idx = tid + p * {num_threads};\n"
    kernel_code += f"        int r = idx >> {a_shift}, c = idx & {a_mask};\n"
    kernel_code += "        int gr = block_m + r, gc = tile_k + c;\n"
    kernel_code += f"        As[r * {bk} + c] = __float2bfloat16((gr < M && gc < K) ? {a_name}[gr * K + gc] : 0.0f);\n"
    kernel_code += "    }\n"
    # B loading
    kernel_code += "    #pragma unroll\n"
    kernel_code += f"    for (int p = 0; p < {b_loads}; p++) {{\n"
    kernel_code += f"        int idx = tid + p * {num_threads};\n"
    kernel_code += f"        int r = idx >> {b_shift}, c = idx & {b_mask};\n"
    kernel_code += "        int gr = tile_k + r, gc = block_n + c;\n"
    kernel_code += f"        Bs[r * {bn} + c] = __float2bfloat16((gr < K && gc < N) ? {b_name}[gr * N + gc] : 0.0f);\n"
    kernel_code += "    }\n"
    kernel_code += "    __syncthreads();\n"
    # WMMA compute: 2x2 tiles per warp, with K-stepping for BK > 16
    wmma_k = 16  # BF16 WMMA K dimension
    k_steps = bk // wmma_k
    kernel_code += "    int wm = warp_m * 32, wn = warp_n * 32;\n"
    kernel_code += (
        "    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;\n"
        "    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;\n"
    )
    if k_steps > 1:
        kernel_code += f"    #pragma unroll\n    for (int wk = 0; wk < {bk}; wk += {wmma_k}) {{\n"
        for wi, wj, cfrag in [(0, 0, "c00"), (0, 1, "c01"), (1, 0, "c10"), (1, 1, "c11")]:
            kernel_code += (
                f"    wmma::load_matrix_sync(af, &As[(wm+{wi * 16})*{bk}+wk], {bk});\n"
                f"    wmma::load_matrix_sync(bf, &Bs[wk*{bn}+wn+{wj * 16}], {bn});\n"
                f"    wmma::mma_sync({cfrag}, af, bf, {cfrag});\n"
            )
        kernel_code += "    }\n"
    else:
        for wi, wj, cfrag in [(0, 0, "c00"), (0, 1, "c01"), (1, 0, "c10"), (1, 1, "c11")]:
            kernel_code += (
                f"    wmma::load_matrix_sync(af, &As[(wm+{wi * 16})*{bk}], {bk});\n"
                f"    wmma::load_matrix_sync(bf, &Bs[wn+{wj * 16}], {bn});\n"
                f"    wmma::mma_sync({cfrag}, af, bf, {cfrag});\n"
            )
    kernel_code += "    __syncthreads();\n"
    kernel_code += "}\n"
    # Store results
    kernel_code += "int wm = warp_m * 32, wn = warp_n * 32;\n"
    for wi, wj, cfrag in [(0, 0, "c00"), (0, 1, "c01"), (1, 0, "c10"), (1, 1, "c11")]:
        kernel_code += (
            f"if (block_m+wm+{wi * 16}+16 <= M && block_n+wn+{wj * 16}+16 <= N)\n"
            f"    wmma::store_matrix_sync(&{c_name}[(block_m+wm+{wi * 16})*N + block_n+wn+{wj * 16}], {cfrag}, N, wmma::mem_row_major);\n"
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
        body=[RawCode(kernel_code)],
        block_size=(num_threads, 1, 1),
        includes=["cuda_bf16.h", "mma.h"],
        tile_m=bm,
        tile_n=bn,
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

    from deplodock.compiler.cuda.ir import RawCode

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


def _lower_matmul_tma_db_hybrid(graph, out_node, config):
    """TMA double-buffer hybrid SGEMM: FFMA warps + HMMA (TF32) warps in same CTA.

    Splits the 8 warps in a CTA into an FFMA group (warps 0..ffma_warps-1) using
    the native FP32 FMA pipe and an HMMA group (remaining warps) using tensor
    cores via wmma in TF32 precision (1xTF32 — single mma per fragment, no
    hi/lo emulation; the HMMA-owned output rows are TF32-precision, not FP32).

    Goal: exercise both FMA and tensor pipes concurrently per SM. The TMA load
    path is identical to tma_db so wall-time differences attribute to compute.

    Hardcoded layout: 6 FFMA + 2 HMMA warps, BM=128 BN=128 BK=32, FFMA tm=16
    covers top 96 rows, HMMA covers bottom 32 rows (2 warps × 16-row fragments).
    """
    input_a = graph.nodes[out_node.inputs[0]]
    input_b = graph.nodes[out_node.inputs[1]]
    a_name = input_a.output.name
    b_name = input_b.output.name
    c_name = out_node.output.name

    import os as _os

    from deplodock.compiler.cuda.ir import RawCode

    # Hybrid layout — read from env vars so we can sweep without recompiling
    ffma_warps = int(_os.environ.get("DEPLODOCK_HYBRID_FFMA_WARPS", "6"))
    hmma_warps = 8 - ffma_warps
    if hmma_warps < 1:
        raise ValueError(f"ffma_warps={ffma_warps} leaves no HMMA warps")
    hmma_frag_rows = int(_os.environ.get("DEPLODOCK_HYBRID_HMMA_FRAG_ROWS", "1"))
    tx, ty = 32, 8  # 8 warps, 256 threads
    bk = int(_os.environ.get("DEPLODOCK_HYBRID_BK", "32"))
    tm = int(_os.environ.get("DEPLODOCK_HYBRID_TM", "16"))
    tn = 4
    bn = tx * tn  # 128
    ffma_bm = ffma_warps * tm
    hmma_bm = hmma_warps * hmma_frag_rows * 16
    bm = ffma_bm + hmma_bm
    a_size = bm * bk
    b_size = bk * bn
    stage = a_size + b_size
    n_frag_cols = bn // 16  # 8
    n_kchunks = bk // 8  # 4

    # NOTE: A 3xTF32 emulation mode used to live here (DEPLODOCK_HYBRID_3XTF32),
    # which split A and B into hi/lo TF32 halves and ran 3 mma_sync calls per
    # fragment to recover FP32 precision. It was correct (max_rel_err 3e-5) but
    # ~2x slower than baseline tma_db: the 3x more HMMA work overwhelmed the
    # 2-warp HMMA latency budget, making HMMA the wall-time bottleneck. Removed
    # in favor of an INT8 emulation path which uses the much-higher-throughput
    # integer tensor pipe (~980 TOPS vs ~498 TFLOPS BF16 measured on sm_120).
    extra_smem = 0

    # FFMA inner-loop block (only ffma warps execute it)
    fma_lines = []
    fma_lines.append(f"            float b0=cbs[kk*{bn}+tc],b1=cbs[kk*{bn}+tc+1],b2=cbs[kk*{bn}+tc+2],b3=cbs[kk*{bn}+tc+3];")
    for i in range(tm):
        fma_lines.append(f"            float a{i}=cas[(tr+{i})*{bk}+kk];")
        fma_lines.append(f"            c{i}0+=a{i}*b0;c{i}1+=a{i}*b1;c{i}2+=a{i}*b2;c{i}3+=a{i}*b3;")
    fma_block = "\n".join(fma_lines)

    # FFMA accumulator + write
    acc_decl = "float " + ",".join(f"c{i}{j}=0" for i in range(tm) for j in range(tn)) + ";"
    write_lines = [f"        W({i},c{i}0,c{i}1,c{i}2,c{i}3)" for i in range(tm)]
    write_block = "\n".join(write_lines)

    # HMMA accumulator fragments: hmma_frag_rows × n_frag_cols per warp
    hmma_acc_lines = []
    for ri in range(hmma_frag_rows):
        for j in range(n_frag_cols):
            hmma_acc_lines.append(f"wmma::fragment<wmma::accumulator,16,16,8,float> hc{ri}_{j};")
            hmma_acc_lines.append(f"wmma::fill_fragment(hc{ri}_{j},0.0f);")
    hmma_acc_decl = "\n".join(hmma_acc_lines)

    # HMMA inner loop: per K-tile, n_kchunks 8-wide chunks. For each chunk,
    # for each fragment row owned by this warp, load A then sweep N fragments.
    # Currently uses 1xTF32 (single mma per fragment, TF32 precision).
    hmma_inner_lines = []
    hmma_inner_lines.append("        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ha;")
    hmma_inner_lines.append("        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> hb;")
    for kc in range(n_kchunks):
        for ri in range(hmma_frag_rows):
            hmma_inner_lines.append(
                f"        wmma::load_matrix_sync(ha,&cas[({ffma_bm}+(h_wid*{hmma_frag_rows}+{ri})*16)*{bk}+{kc * 8}],{bk});"
            )
            for j in range(n_frag_cols):
                hmma_inner_lines.append(f"        wmma::load_matrix_sync(hb,&cbs[{kc * 8}*{bn}+{j * 16}],{bn});")
                hmma_inner_lines.append(f"        wmma::mma_sync(hc{ri}_{j},ha,hb,hc{ri}_{j});")
    hmma_inner = "\n".join(hmma_inner_lines)

    # HMMA epilogue — guarded against M/N tile-overflow at the matrix edge
    hmma_write_lines = []
    for ri in range(hmma_frag_rows):
        for j in range(n_frag_cols):
            row_expr = f"(bm+{ffma_bm}+(h_wid*{hmma_frag_rows}+{ri})*16)"
            col_expr = f"(bn+{j * 16})"
            hmma_write_lines.append(
                f"        if({row_expr}+16<=M&&{col_expr}+16<=N) wmma::store_matrix_sync(&{c_name}[{row_expr}*N+{col_expr}],hc{ri}_{j},N,wmma::mem_row_major);"
            )
    hmma_write = "\n".join(hmma_write_lines)

    # Write macro for FFMA group (same as tma_db, no batch / k_splits support yet)
    write_macro = f"""#if (M % {bm} == 0 && N % {bn} == 0)
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);int gc=bn+tc;float*Cout={c_name}; \
Cout[gr*N+gc]=v0;Cout[gr*N+gc+1]=v1;Cout[gr*N+gc+2]=v2;Cout[gr*N+gc+3]=v3;}}}}
#else
#define W(r,v0,v1,v2,v3) {{{{int gr=bm+tr+(r);if(gr<M){{{{int gc=bn+tc;float*Cout={c_name}; \
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
int wid=threadIdx.y;
bool is_hmma=(wid>={ffma_warps});
int h_wid=wid-{ffma_warps};
int tr=threadIdx.y*{tm},tc=threadIdx.x*{tn};
// CTA swizzle: rasterize in groups of SWIZ row-blocks for L2 A-tile reuse
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
if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));asm volatile("fence.mbarrier_init.release.cluster;");}}
__syncthreads();
{acc_decl}
{hmma_acc_decl}
const int bytes={stage * 4};
int p0=0,p1=0,nt=K/{bk};
if(nt>0&&tid==0){{asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(as0),"l"(&{a_name}_tma),"r"(0),"r"(bm),"r"(mb0):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(bs0),"l"(&{b_name}_tma),"r"(bn),"r"(0),"r"(mb0):"memory");}}
for(int t=0;t<nt;t++){{
    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;
    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;
    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));
    if(s==0)p0^=1;else p1^=1;
    if(tid==0&&t+1<nt){{int nk=(t+1)*{bk};asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(nm),"r"(bytes):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(na),"l"(&{a_name}_tma),"r"(nk),"r"(bm),"r"(nm):"memory");asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(nb),"l"(&{b_name}_tma),"r"(bn),"r"(nk),"r"(nm):"memory");}}
    float*cas=&smem[s*{stage}];float*cbs=&smem[s*{stage}+{a_size}];
    if(!is_hmma){{
        #pragma unroll
        for(int kk=0;kk<{bk};kk++){{
{fma_block}
        }}
    }}else{{
{hmma_inner}
    }}
    __syncthreads();
}}
{write_macro}
if(!is_hmma){{
{write_block}
}}else{{
{hmma_write}
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
        batched=False,
        extra_smem_bytes=extra_smem,
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

    from deplodock.compiler.cuda.ir import RawCode

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

    # Trivial epilogue: one store_matrix_sync per fragment, with edge guards.
    epilogue_lines = []
    for ri in range(frag_rows_per_warp):
        for ci in range(frag_cols_per_warp):
            row_expr = f"(bm+warp_row*{frag_rows_per_warp * wmma_m}+{ri * wmma_m})"
            col_expr = f"(bn+warp_col*{frag_cols_per_warp * wmma_n}+{ci * wmma_n})"
            epilogue_lines.append(
                f"    if({row_expr}+{wmma_m}<=M&&{col_expr}+{wmma_n}<=N) wmma::store_matrix_sync(&{c_name}[{row_expr}*N+{col_expr}],hc{ri}_{ci},N,wmma::mem_row_major);"
            )
    epilogue_block = "\n".join(epilogue_lines)

    a_size = bm * bk
    b_size = bk * bn
    stage = a_size + b_size  # FP32 floats
    tma_bytes = stage * 4

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
if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));asm volatile("fence.mbarrier_init.release.cluster;");}}
__syncthreads();
{acc_decl}
const int bytes={tma_bytes};
int p0=0,p1=0,nt=K/{bk};
if(nt>0&&tid==0){{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(as0),"l"(&{a_name}_tma),"r"(0),"r"(bm),"r"(mb0):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(bs0),"l"(&{b_name}_tma),"r"(bn),"r"(0),"r"(mb0):"memory");
}}
for(int t=0;t<nt;t++){{
    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;
    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;
    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));
    if(s==0)p0^=1;else p1^=1;
    if(tid==0&&t+1<nt){{
        int nk=(t+1)*{bk};
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(nm),"r"(bytes):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(na),"l"(&{a_name}_tma),"r"(nk),"r"(bm),"r"(nm):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"::"r"(nb),"l"(&{b_name}_tma),"r"(bn),"r"(nk),"r"(nm):"memory");
    }}
    float*A_smem=&smem[s*{stage}];
    float*B_smem=&smem[s*{stage}+{a_size}];
{inner_block}
    __syncthreads();
}}
{epilogue_block}"""

    return KernelDef(
        name="fused_matmul",
        params=[KernelParam("float*", c_name)],
        body=[RawCode(kernel_code)],
        block_size=(tx, ty, 1),
        includes=["cuda.h", "mma.h"],
        tile_m=bm,
        tile_n=bn,
        tma_params=[f"{a_name}_tma", f"{b_name}_tma"],
        batched=False,
        min_blocks_per_sm=int(_os.environ.get("DEPLODOCK_TF32_MIN_BLOCKS", "2")),
    )
