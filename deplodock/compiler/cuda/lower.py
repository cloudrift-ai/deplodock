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
    """Tunable parameters for matmul lowering."""

    strategy: str = "naive"
    block_m: int = 16
    block_n: int = 16
    block_k: int = 16
    thread_m: int = 1
    thread_n: int = 1
    vectorize: bool = False
    unroll_k: bool = False
    double_buffer: bool = False
    smem_pad: int = 1


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
        block_size=(config.block_n, config.block_m, 1),
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

    bm = config.block_m
    bn = config.block_n
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

    bm = config.block_m
    bn = config.block_n
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

    # Bounds-checked k-loop.
    k_loop = IfStmt(
        cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N"))),
        body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body)],
    )
    body.append(k_loop)

    # Write results.
    write_body: list = []
    for i, acc in enumerate(["acc0", "acc1", "acc2", "acc3"]):
        c_idx = BinOp("+", BinOp("*", Var("row"), Var("N")), BinOp("+", Var("col_base"), Literal(i, "int")))
        write_body.append(Assign(ArrayAccess(c_name, c_idx), Var(acc)))

    body.append(
        IfStmt(
            cond=BinOp("&&", BinOp("<", Var("row"), Var("M")), BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N"))),
            body=write_body,
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

    # Bounds-checked k-loop.
    bounds_cond = BinOp(
        "&&",
        BinOp("<", BinOp("+", Var("row_base"), Literal(1, "int")), Var("M")),
        BinOp("<", BinOp("+", Var("col_base"), Literal(3, "int")), Var("N")),
    )
    body.append(IfStmt(cond=bounds_cond, body=[ForLoop("k", Literal(0, "int"), Var("K"), k_body)]))

    # Write results.
    write_body: list = []
    for r in range(2):
        for c in range(4):
            c_idx = BinOp(
                "+", BinOp("*", BinOp("+", Var("row_base"), Literal(r, "int")), Var("N")), BinOp("+", Var("col_base"), Literal(c, "int"))
            )
            write_body.append(Assign(ArrayAccess(c_name, c_idx), Var(f"acc{r}{c}")))
    body.append(IfStmt(cond=bounds_cond, body=write_body))

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
