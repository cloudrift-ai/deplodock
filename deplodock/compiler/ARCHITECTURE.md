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
| `ir/`                 | Op-type definitions per dialect         | `ir/ARCHITECTURE.md`         |
| `trace/`              | PyTorch/HuggingFace → Graph IR          | `trace/ARCHITECTURE.md`      |
| `pipeline/`           | Rewrite engine, passes, dump hooks      | `pipeline/ARCHITECTURE.md`   |
| `backend/`            | Execution (numpy / loop / cuda)         | `backend/ARCHITECTURE.md`    |
| `loader/`             | Bind constants (safetensors / `nn.Module` → `input_data`) | —              |

## Per-layer rules

- **Layer 1** — no GPU, no CUDA, no backend imports. Dialect ops
  implement `infer_output_shape(input_shapes)` and a numpy `forward()`.
- **Layer 2** — operates on `Graph` + Loop IR only. Every `LoopOp`'s
  `__post_init__` canonicalizes (`ir/loop/normalize.py`) and simplifies
  (`ir/loop/simplify.py`) its body.
- **Layer 3** — backends are the only place GPU specifics live.

## Shared invariants

- **Shape lives on the graph**, not on the op — `node.output.shape`.
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
