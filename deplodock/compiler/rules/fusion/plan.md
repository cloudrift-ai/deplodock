 Ready to code?                                                                                                                                                                                                  
                                                                                                                                                                                                                 
 Here is Claude's plan:                                                                                                                                                                                          
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: General Fusion + Kernel Generator                         

 Context

 Replace hand-written fusion rules AND kernel templates with:
 1. Auto-fusion: discovers fusion regions from intermediate tensor sizes
 2. Kernel generator: generates CUDA code from a fused region's primitive ops

 Goal: the system automatically discovers flash attention from primitives AND generates the tiled online-softmax kernel. No hand-written .cu templates for fused ops.

 Pipeline

 torch.export → Primitives (decomposition pass)
     ↓
 auto_fuse(graph) → groups ops into fusion regions
     ↓
 For each region: generate_kernel(region) → CUDA source
     ↓
 FusedRegionOp (carries generated kernel source)
     ↓
 plan_graph → CudaBackend → Program → GPU

 Phase 1: auto_fuse — region discovery

 New file: compiler/fusion.py

 def auto_fuse(graph: Graph) -> Graph:
     """Discover fusion regions and replace them with FusedRegionOp nodes."""

 Algorithm

 1. Score edges: for each single-consumer edge, score = product(intermediate_shape). Higher = more bandwidth saved.
 2. Greedy merge: process edges highest-score first. Merge producer's region with consumer's region if:
   - No cycle created (check via topological sort on region DAG)
   - Compatible op types (pointwise+pointwise, pointwise+reduce, reduce+pointwise, matmul+pointwise epilogue)
 3. Structural ops (reshape, transpose) always merge with their neighbors — zero cost.
 4. Output: each region becomes a FusedRegionOp node. Internal nodes deleted, external inputs/outputs preserved.

 Data structures

 class UnionFind:
     """Disjoint set for tracking regions."""
     def find(self, x): ...
     def merge(self, x, y): ...
     def members(self, x) -> set: ...

 @dataclass
 class FusedRegionOp(Op):
     """A fused region of primitive ops."""
     region_ops: list[tuple[str, Op, list[str]]]  # [(node_id, op, input_ids), ...]
     kernel_source: str = ""  # filled by kernel generator

 Tests

 - On decomposed TinyLlama: verify regions match expected groupings
 - Matmul pair (Q @ K^T then scores @ V) groups with softmax between them
 - Pointwise chains (neg → exp → add → recip → mul) merge into one region
 - Single matmul stays as one region

 Phase 2: kernel_gen — CUDA code generation from regions

 New file: compiler/kernel_gen.py

 def generate_kernel(region: FusedRegionOp, name: str) -> str:
     """Generate a CUDA kernel by walking the region's ops directly."""

 No strategies — direct codegen from ops

 The generator doesn't classify regions into strategies. It walks the ops in topological order and emits code directly:

 1. Analyze dimensions: for each dim in the region's tensors, classify as outer (in output, parallelize), inner (reduced away, loop), or free (thread-level).
 2. Walk ops in topo order: for each op, emit the corresponding C code:
   - ElementwiseOp → inline C expression (a * b, expf(a), etc.)
   - ReduceOp → warp-shuffle reduction block
 3. Tiling emerges from dimensions: if there's a reduced dim, the generator emits a tiled loop over it. If a reduced dim feeds into another reduce (online softmax pattern), the generator emits running max/sum
  tracking + output rescaling.

 Expression map

 _EXPR = {
     "mul": "{a} * {b}",
     "add": "{a} + {b}",
     "sub": "{a} - {b}",
     "div": "{a} / {b}",
     "neg": "-{a}",
     "exp": "expf({a})",
     "rsqrt": "rsqrtf({a})",
     "recip": "1.0f / {a}",
 }

 Reduction emit

 def _emit_reduce(fn, var):
     op = "+" if fn == "sum" else "fmaxf"
     # warp shuffle + shared memory cross-warp reduction

 What naturally emerges

 - Pointwise chain (all ElementwiseOp): no reduced dims → no tile loop → each thread processes one element, ops become inline expressions
 - Reduction + pointwise (RMSNorm): one reduced dim → one tile loop, pointwise ops become prologue/epilogue around the reduction
 - Matmul (Reduce{sum}(Ewise{mul})): contraction dim → tiled loop with shared memory for two inputs
 - Online softmax (reduce_max + exp + reduce_sum between two tiled contractions): the generator detects that a reduce feeds into another tiled contraction and emits running accumulation instead of
 materializing
 - Flash attention: two contractions sharing an inner dim with online softmax between them → tiled inner loop with online max/sum tracking + output rescaling

 No "if matmul → dispatch to matmul codegen". The code structure emerges from the ops and their dimensions.

 Tile size defaults

 Fixed for simplicity (auto-tune later):

 ┌───────────────────┬─────────────┬───────────────┐
 │   Kernel shape    │    Block    │ Tile per dim  │
 ├───────────────────┼─────────────┼───────────────┤
 │ No reduction      │ (256, 1, 1) │ 1 elem/thread │
 ├───────────────────┼─────────────┼───────────────┤
 │ 1D reduction      │ (256, 1, 1) │ 1 row/block   │
 ├───────────────────┼─────────────┼───────────────┤
 │ 2D tiled (matmul) │ (16, 16, 1) │ BK=16         │
 └───────────────────┴─────────────┴───────────────┘

 Phase 3: Integration

 Changes to existing code

 compiler/ops.py: Add FusedRegionOp. Keep existing fused ops temporarily for backwards compat.

 compiler/plan.py: plan_graph handles FusedRegionOp — uses the kernel source from the op.

 compiler/backend/cuda/backend.py: _compile_fused_region handler — the kernel source is already generated, just compute grid/block from the region's dimensions and wrap in a Launch.

 compiler/pipeline.py or rewriter: call auto_fuse + generate_kernels after decomposition.

 What gets deleted (Phase 4)

 - rules/fusion/*.py — all 5 hand-written rules
 - backend/cuda/kernels/*.cu — hand-written templates (except matmul_naive.cu used by lower.py for TMA SGEMM)
 - Existing fused op types (MatmulOp, FusedRMSNormOp, etc.) — replaced by FusedRegionOp

 Implementation order

 Phase 1: auto_fuse (no codegen yet)

 1. compiler/fusion.py — UnionFind, edge scoring, greedy merge, FusedRegionOp
 2. tests/compiler/test_fusion.py — verify regions on decomposed TinyLlama
 3. Verify: correct number of regions, correct groupings

 Phase 2: kernel_gen — pointwise + reduction

 4. compiler/kernel_gen.py — pointwise and reduction strategies
 5. Wire into auto_fuse: each FusedRegionOp gets generated kernel_source
 6. Make test_kernel_gen_silu_mul and test_kernel_gen_rmsnorm pass
 7. Make test_kernel_gen_softmax pass

 Phase 3: kernel_gen — matmul

 8. compiler/kernel_gen.py — matmul strategy (tiled, shared memory)
 9. Make test_kernel_gen_matmul pass
 10. Make test_kernel_gen_matmul_residual_add pass (matmul + pointwise epilogue)
 11. Make test_kernel_gen_triple_matmul pass (shared input)

 Phase 4: kernel_gen — attention (flash)

 12. compiler/kernel_gen.py — online softmax detection + tiled attention
 13. Make test_kernel_gen_attention pass
 14. This is the milestone: flash attention from primitives

 Phase 5: Integration + cleanup

 15. Wire auto_fuse + kernel_gen into the full pipeline
 16. Delete hand-written fusion rules
 17. Delete hand-written .cu templates (keep TMA matmul for SGEMM benchmarks)
 18. All existing tests pass via new path

 Verification

 # Phase 1: regions discovered correctly
 ./venv/bin/pytest tests/compiler/test_fusion.py -v

 # Phase 2-4: generated kernels match PyTorch reference
 ./venv/bin/pytest tests/compiler/test_kernel_gen.py -v

 # Phase 5: full pipeline
 deplodock run tests/compiler/fixtures/tinyllama_layer0.json --benchmark
 make test && make lint

 Files

 ┌───────┬───────────────────────────────────┬──────────────────────────────────────────┐
 │ Phase │               File                │                  Action                  │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 1     │ compiler/fusion.py                │ NEW: auto_fuse, UnionFind, FusedRegionOp │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 1     │ compiler/ops.py                   │ ADD: FusedRegionOp                       │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 1     │ tests/compiler/test_fusion.py     │ NEW                                      │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 2-4   │ compiler/kernel_gen.py            │ NEW: generate_kernel, strategies         │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 2-4   │ tests/compiler/test_kernel_gen.py │ UPDATE: un-skip as each passes           │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 5     │ compiler/plan.py                  │ UPDATE: handle FusedRegionOp             │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 5     │ backend/cuda/backend.py           │ UPDATE: _compile_fused_region            │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 5     │ rules/fusion/*.py                 │ DELETE                                   │
 ├───────┼───────────────────────────────────┼──────────────────────────────────────────┤
 │ 5     │ backend/cuda/kernels/*.cu         │ DELETE (except TMA matmul)               │
 └───────┴───────────────────────────────────┴──────────────────────────────────────────┘
