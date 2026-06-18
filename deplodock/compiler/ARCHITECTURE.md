# Compiler Architecture

Three layers over a shared `Graph` container.

```
PyTorch module
   │  trace/              ── PyTorch → Graph IR capture
   ▼
Graph (frontend ops)                        ── Layer 1
   │  pipeline/passes/frontend/decomposition
   │  pipeline/passes/frontend/optimization
   │  pipeline/passes/loop/lifting
   │  pipeline/passes/loop/fusion
   ▼
Graph[LoopOp]  (one LoopOp = one kernel)    ── Layer 2
   │  pipeline/passes/lowering/tile         (Loop IR → Tile IR)
   │  pipeline/passes/lowering/kernel       (Tile IR → Kernel IR)
   │  pipeline/passes/lowering/cuda         (Kernel IR → CUDA source)
   ▼
Graph[CudaOp]                               ── Layer 3
   │  backend/cuda                          (cupy.RawKernel via NVRTC)
   ▼
GPU
```

`Graph` (`compiler/graph.py`) hosts nodes from every dialect; rewrite
passes swap node ops in place, so there is no separate "program" type.

`Graph.structural_key()` implements the `Structural` protocol
(`compiler/structural.py`) — a Merkle-style hex sha256 digest used for
candidate dedup in autotuning loops. Per node it folds in op kind,
op body's `Body.structural_key()` (or other dataclass fields for leaf
ops, skipping `name`), `Tensor.shape` / `dtype` (skipping `Tensor.name`),
and the recursive digests of input nodes; the top-level digest folds in
the graph's `inputs` / `outputs` sequences. Hints and graph-internal
node ids are excluded. Two graphs that compute the same dataflow
through structurally-equivalent kernels hash equal regardless of
node-id naming or inconsequential body details. Not cached — `Graph`
is mutable; callers that dedup many candidates snapshot the digest
themselves.

`Structural` is the one convention: anything we compare or cache by
structure (`Graph`, `Body`, `Context`, future fork-option payloads and
subgraph slices) implements `structural_key() -> str`. The `digest(...)`
helper in `compiler/structural.py` is the canonical fold; composite
implementers call it with child digests + their own discriminating
fields. Each implementer's docstring documents what's deliberately
excluded (names, hints, ambient I/O) — the contract is "include only
bits that affect codegen output or dataflow semantics" so the
autotuning cache doesn't bust on cosmetic edits.

## Module layout

| Path                  | Role                                    | See                          |
|-----------------------|-----------------------------------------|------------------------------|
| `graph.py`            | `Graph`, `Node`, `Tensor`, `Hints`      | —                            |
| `dim.py`              | `Dim` — shape extent backed by an `Expr` (static or symbolic) | —    |
| `ir/`                 | Op-type definitions per dialect         | `ir/ARCHITECTURE.md`         |
| `trace/`              | PyTorch/HuggingFace → Graph IR          | `trace/ARCHITECTURE.md`      |
| `pipeline/`           | Rewrite engine, passes, dump hooks      | `pipeline/ARCHITECTURE.md`   |
| `backend/`            | Execution (numpy / loop / cuda)         | `backend/ARCHITECTURE.md`    |
| `loader/`             | Bind constants (safetensors / `nn.Module` → `input_data`) | —              |
| `pipeline/search/`    | Autotune DB + MCTS tree (see below)     | `pipeline/ARCHITECTURE.md`   |
| `structural.py`       | `Structural` protocol + `digest()` fold | —                            |
| `provenance.py`       | Op provenance — map fused kernels back to original frontend ops | — (see below) |

## Per-layer rules

- **Layer 1** — no GPU, no CUDA, no backend imports. Dialect ops
  implement `infer_output_shape(input_shapes)` and a numpy `forward()`.
- **Layer 2** — operates on `Graph` + Loop IR only. Every `LoopOp`'s
  `__post_init__` canonicalizes (`ir/loop/normalize.py`) and simplifies
  (`ir/loop/simplify.py`) its body.
- **Layer 3** — backends are the only place GPU specifics live.

## Shared invariants

- **Shape lives on the graph**, not on the op — `node.output.shape`. Each shape element is a `Dim`
  (`compiler/dim.py`) that wraps an `Expr` from `ir/expr.py`: static (`Dim(32)` → `Literal(32)`), atomic
  symbolic (`Dim("seq_len")` → `Var("seq_len")`), or composite from arithmetic (`Dim("S") * Dim(2)` →
  `BinaryExpr("*", Var("S"), Literal(2))`). `Dim` overloads `+`/`-`/`*`/`//`/`%` and eager-folds via
  `Expr.simplify` — static math matches plain int math byte-for-byte; symbolic stays as `BinaryExpr`. It
  also exposes `ceil_div` (`(self + (b-1))//b`): the single masked-tile grid-extent formula for both
  regimes — it folds to the integer ceil (`-(-E//b)`) for a static dim and builds the composite ceil-div
  `Expr` for a symbolic one, so the partition planner's masked block-axis / masked-K sites need no
  static-vs-symbolic branch.
  Read sites use `d.expr` (always works), `d.as_static()` (raises on symbolic), `d.as_atom_name()` (raises
  unless `Var`-backed), or `d.value` (back-compat shim: int for `Literal`, str for `Var`, raises on
  composite). There is deliberately no `__int__` / `__index__`, so `int(d)` and `range(d)` fail loudly
  on anything but a static-int `Dim`. Symbolic dims resolve at launch via `d.expr.eval(sym_env)` —
  composite shapes (e.g. an `S * 2` concat output) resolve from input array axes without per-site
  branching. `Tensor.__post_init__` and `Axis.__post_init__` coerce bare `int` / `str` to `Dim`, so
  producer call sites need no change. An atomic symbolic `Dim` also carries a `hint` — its *expected*
  size (default `DEFAULT_SEQ_HINT=512`, set automatically so reconstruction can't lose it; an explicit
  `Dim(name, hint=...)` overrides). The hint is pure metadata (excluded from `==`/`hash`/structural keys),
  read only by the tuner / partition planner to size tiles for a dynamic axis.
- **A symbolic free axis is tiled for its hint and emitted as a *masked* tile.** The partition planner
  (`pipeline/passes/lowering/tile/010_partition_loops.py`) treats a symbolic M/N axis as size `hint`,
  always-overhang: the block axis becomes a composite ceil-div over the symbolic extent
  (`(seq_len + bf - 1)//bf`), and a boundary `Cond(decoded_coord < seq_len)` wraps the body. So one cached
  kernel runs at any runtime `seq_len` — the grid (`ir/cuda/ir.py:GridDimSpec` now accepts an `Expr`
  factor, resolved via `Expr.eval` at launch) and the guard read the runtime value; the tile shape is
  tuned for the hint. This covers the **warp/MMA tier** too: symbolic M and/or N enumerate masked
  mma.sync rows (`OVERHANG`-stamped, no hint-divisibility) — the boundary Cond gates whole tiles, the
  `RegStore` carries per-element row/col guards for tiles straddling the bound (the fragment lane offsets
  are invisible to σ), staged slab fills clamp their gmem reads to the runtime extent
  (`Source.gmem_extents` with symbolic `Expr` bounds), unstaged operands take clamped gmem-direct
  fragment loads (`LdmatrixLoad.gmem_guard`), and a symbolic output inner extent resolves its `ldm` from
  the runtime kernel arg at render. Symbolic K stays off the warp tier (flash-style attention is future
  work). SDPA-prologue matmuls with a **symbolic K** (P@V — which never stages) get masked THREAD tiles
  with `FM = FN = 1` (`mask_f1`); static-K prologue kernels (fused gated-MLP) and cooperative-reduce
  kernels keep symbolic axes degenerate (one element per thread) — their staged pipelines can't coexist
  with the per-row guard (`021`'s hoist would break the prologue's SSA ordering; it now refuses such
  lifts), and their deployment path is the structural split (`005_split_demoted`), which offers on
  symbolic ROW **and** symbolic N axes (the rotary QK^T's symbolic-N key cone materializes canonically,
  reaching the masked warp tier) — so e.g. the dynamic o_proj's collapsed attn-out splits into a
  contiguizing `xn` producer + a warp-tier consumer instead of staying fused-scalar. The symbolic-N B
  operand `xnb[…, K, N]` further reaches the **TMA + warp-specialized** tier (matching its static twin,
  not just cp.async): `_split_demoted._pad_inner_for_tma` rounds the materialized inner N up to a
  multiple of 64 so the K dim's gmem stride stays 16 B-aligned at any runtime `seq_len`, which is the
  `cuTensorMapEncodeTiled` requirement `050_use_tma` otherwise declines a symbolic innermost dim for
  (`_inner_stride_aligned`). The buffer stays runtime-sized (correct above the hint, unlike a fixed
  static width), and the padded `[seq_len, round_up)` overhang columns feed the mma only into
  store-masked output positions, so they can't contaminate a live score. The cut keeps a
  symbolic ROW/N buffer's runtime dim var (`seq_len` in a collapsed-reshape stride) as a legitimate read,
  not an unmodeled-scope bail (`_split_demoted.dim_names`). Symbolic K stays bailed: the warp tier needs a
  static reduce, so a symbolic-K matmul (P@V) is scalar fused or split — the split would only add a
  materialization with no tier upgrade.
  Cooperative-reduce kernels still regain CTA-level parallelism on symbolic-row graphs via
  **strided-cooperative rows**: their STATIC free axes thread-bind alongside the `BR` cooperative lanes
  (the symbolic axis keeps its exact whole-to-grid bind, no mask), so e.g. a per-head q/k-norm with
  symbolic seq deploys a `BN×BR` CTA instead of the v1 `BR`-thread degenerate one. The combine for such
  2D CTAs is a segmented warp shuffle over each row's BR lanes (`K_c` is the innermost THREAD axis; the
  enumerator clips those rows' BR to powers of two ≤ warp_size —
  `lowering/kernel/_combine.cooperative_combine_geometry`). The backend
  benches a symbolic graph at the hint when no real inputs are supplied
  (`backend/cuda/program.py:_symbolic_hints` / `_resolve_symbolic`), so `tune` and `compile`
  agree on a hint-sized variant.
- **`ElementwiseOp` inputs must already share the output shape.** The
  decomposition helper
  `pipeline/passes/frontend/decomposition/_broadcast.broadcast_to` wraps
  mismatched inputs in an `IndexMapOp`.
- **One `LoopOp` = one kernel.** Fusion produces `LoopOp` nodes;
  lowering turns each into `KernelOp` (AST) then `CudaOp` (rendered
  source).
- **`LoopOp.forward()` executes.** The body interpreter in
  `ir/loop/interpret.py` lets the default `Backend.run` topo-walk
  (`backend/base.py`) run post-fusion graphs on CPU — fusion
  correctness can be checked without a GPU.

## Op provenance

`provenance.py` threads a single `Node.hints["prov"]` map —
`{origin_id: {"kind": <op-class>, "pieces": [piece_id, …]}}` — from the traced frontend graph all the way to each
`CudaOp`, so a fused kernel knows which original PyTorch ops it implements. `origin_id` is the trace-time node id of an
original op (`rms_norm_0`); `pieces` are the primitives it decomposed into and that this node embodies. Coverage of an
origin is `len(pieces)` over the union of that origin's pieces across the whole graph (`totals` / `coverage`) — so the
`i/N` fraction stays correct under CSE and recursive decomposition instead of freezing `N` at the first split.

It rides on one chokepoint: `Graph.splice` calls `provenance.propagate` with a `mint_pieces` flag (set by
`Candidate.apply` from the pass namespace — `True` only for `frontend/decomposition`). Decomposition *mints* each new
fragment node as a fresh piece of the consumed origins; fusion / lifting / optimization folds *aggregate* the consumed
piece sets onto the merged node (unioning the dissolved producers so a multi-output splice drops nothing). Lowering is
in-place `Op` rebinds, so prov rides through `LoopOp → TileOp → KernelOp → CudaOp` untouched. Seeded once at
`Pipeline.tune_async` / `Pipeline.run` entry (idempotent); pure metadata, excluded from structural / cache keys. Boundary sentinels
(`InputOp`/`ConstantOp`) never carry prov: `put` refuses to stamp them and `propagate` scrubs splice outputs that land
on one (the generic hint merge would otherwise copy prov onto e.g. the ConstantOp produced by the sm_90+
weight-transpose fold, inflating `totals` so every kernel of that origin read partial coverage).

Consumers: `provenance.name_for` (called from `pipeline/passes/loop/fusion/991_stamp_loop_names`, the last
loop-dialect rule) names kernels after the ops they realize (`k_rms_norm` when full, `k_rms_norm_reduce` when partial)
and stamps the name onto `LoopOp.name`; every subsequent dialect (`TileOp`/`KernelOp`/`CudaOp`) just copies it
through. Multi-op labels sort dominant-first (descending piece count, lexical tie-break), so the name is independent
of fusion merge order — the attention kernel is `k_sdpa_linear_reduce`, its QKV-prologue twin `k_linear_sdpa_reduce`.
Layout/plumbing origins (`_WEAK_KINDS`: transpose / reshape / unsqueeze / cat / slice) label a kernel only when no
strong op is present — RoPE plumbing fused into attention doesn't pollute the name, while a standalone copy kernel
still reads `k_cat_…` instead of the node-id fallback. `pipeline/dump._dump_torch_repro` slices the pristine frontend graph by a kernel's origins into a runnable
`<kname>.torch.json`; `backend/torch_ref` runs that slice through real torch for the `run --ir` vs-torch comparison.
