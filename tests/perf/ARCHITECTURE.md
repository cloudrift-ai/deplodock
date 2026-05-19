# Perf Suite — `tests/perf/`

Apples-to-apples kernel-level performance comparison between Deplodock and PyTorch eager. Gated by the `perf` pytest
marker — **deselected by default** so `make test` stays fast. Run explicitly with `make bench-kernels` (or
`pytest -m perf`).

## What it measures

The case list mirrors the 12 fused kernels Deplodock emits when compiling a single Qwen3-Embedding-0.6B decoder layer
(`deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0 --ir loop`). Each case is a small Python expression that
lowers to the same single fused launch, so the suite is a kernel-by-kernel view of what the compiler actually runs at
the model level. Model dims: hidden=1024, intermediate=3072, num_heads=16, num_kv_heads=8, head_dim=128. Each case
runs at `seq_len ∈ {32, 128, 512}`.

| Op          | Maps to Qwen3 kernel(s)                                | What runs (deplodock)                              | What runs (PyTorch)                            |
|-------------|--------------------------------------------------------|----------------------------------------------------|------------------------------------------------|
| `rmsnorm`   | 0 / 9 (input + post-attn layernorm), 4 / 5 (q/k norm)  | `RmsNormOp(x, w)`                                  | `F.rms_norm`                                   |
| `matmul`    | 1 / 2 / 3 (q_proj, kv_proj)                            | `MatmulOp(a, b)`                                   | `torch.matmul`                                 |
| `sdpa`      | 6 + 7 (masked QK + softmax/AV)                         | `SdpaOp(q, k, v, is_causal=True)`                  | `F.scaled_dot_product_attention(..., GQA)`     |
| `matmul_add`| 8 (o_proj+resid), 11 (down_proj+resid)                 | `MatmulOp → ElementwiseOp("add")`                  | `torch.matmul(x, w) + r`                       |
| `gated_mlp` | 10 (gate·silu·up fused, sans down-proj)                | `MatmulOp ×2 → silu → multiply`                    | `F.silu(torch.matmul(x, wg)) * torch.matmul(x, wu)` |

Primitive ops (`rmsnorm`, `matmul`) live in `test_primitives.py`. Fused signatures (`sdpa`, `matmul_add`,
`gated_mlp`) live in `test_fused.py`. `gated_mlp` is the gate+up fused launch (Qwen3 layer kernel 10); both matmuls
share the same hidden-dim reduce and the silu·multiply is folded into the epilogue. The down_proj+residual that
follows is a separate kernel (`matmul_add.down_proj`).

Deplodock currently emits FP32 only, so both sides run FP32. The `dtype` field on `Case` is kept for future FP16
support.

## How it's wired

- `cases.py` — `Case` dataclass + curated `CASES` list + `build_torch_ref` / `build_deplodock_graph`.
- `conftest.py` — `bench_pair` fixture (CUDA-event timing on the torch side, `CudaBackend.benchmark` on the deplodock
  side), session-end summary table, JSON dump to `.results/`.
- `test_primitives.py` / `test_fused.py` — one parametrized test each, ids = `case.name`.

The deplodock-side timing reuses `CudaBackend.benchmark` (per-kernel CUDA events via cupy, see
`deplodock/compiler/backend/cuda/program.py:369`) so launch counts and per-kernel latency are available; the torch
side uses `torch.cuda.Event` matching the pattern in `scripts/bench_block.py:194-202`.

## Reading the output

`pytest_terminal_summary` prints one table sorted by `ratio = torch_us / deplodock_us` ascending — losses
(ratio < 1) first:

```
case                          shape                          torch_us  depl_us  ratio  launches
----------------------------  -----------------------------  --------  -------  -----  --------
gated_mlp.qwen3emb.s512       (1,512,1024) x (1024,3072) ...   1230.0   2890.4  0.43x         1
rmsnorm.qwen3emb.layer.s512   (1,512,1024) x (1024,)             12.1      9.8  1.23x         1
...
```

The same data is dumped to `tests/perf/.results/<utc-timestamp>.json` for cross-run diffing. `DEPLODOCK_GIT_REV` is
recorded if set. A self-contained ECharts plot of `ratio` per case (sorted ascending, bars colored by op, optional
`torch.compile` overlay) is written next to the JSON as `<utc-timestamp>.html` — open it in a browser; no server
needed (the chart loads ECharts from the jsDelivr CDN).

## Adding a case

Append to the appropriate builder in `cases.py` (`_matmul_proj_cases`, `_rmsnorm_layer_cases`, …) — no test code
changes required, parametrize picks it up by name. If a new fused launch appears in the Qwen3-Embedding layer dump
that the current taxonomy doesn't cover, add a new op to `Case.op` along with matching branches in `code`,
`build_torch_ref`, and `build_deplodock_graph`.

## Running

```bash
make bench-kernels                                                  # full suite
./venv/bin/pytest tests/perf/ -m perf -v -k matmul                  # one op
./venv/bin/pytest tests/perf/ -m perf -v -k 'qwen3emb.layer'        # one kernel family
./venv/bin/pytest tests/perf/test_primitives.py -m perf -v -s       # primitives only
```

`make test` continues to skip everything in `tests/perf/`.
