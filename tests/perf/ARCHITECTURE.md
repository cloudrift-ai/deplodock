# Perf Suite — `tests/perf/`

Apples-to-apples kernel-level performance comparison between Deplodock and PyTorch
eager. Gated by the `perf` pytest marker — **deselected by default** so `make test`
stays fast. Run explicitly with `make bench-kernels` (or `pytest -m perf`).

## What it measures

Each case is one (op, shape) pair drawn from real transformer blocks (TinyLlama
hidden=2048, intermediate=5632, heads=32, head_dim=64; Qwen2.5-7B hidden=3584,
intermediate=18944, heads=28, head_dim=128) at `seq_len ∈ {32, 128, 512}`.

| Op                | What runs (deplodock)                          | What runs (PyTorch)              |
|-------------------|------------------------------------------------|----------------------------------|
| `matmul`          | `MatmulOp(a, b)`                               | `torch.matmul`                   |
| `rmsnorm`         | `RmsNormOp(x, w)`                              | `F.rms_norm`                     |
| `softmax`         | `SoftmaxOp(x)`                                 | `F.softmax`                      |
| `silu_mul`        | `ElementwiseOp("silu") → ElementwiseOp("mul")` | `F.silu(gate) * up`              |
| `sdpa`            | `SdpaOp(q, k, v, is_causal=True)` (fused)      | `F.scaled_dot_product_attention` |
| `matmul_add`      | `MatmulOp → ElementwiseOp("add")`              | `torch.matmul(x, w) + r`         |
| `silu_mul_matmul` | `silu → mul → MatmulOp`                        | `torch.matmul(F.silu(g)*u, w)`   |

Primitive ops (`matmul`, `rmsnorm`, `softmax`, `silu_mul`) live in
`test_primitives.py`. Fused signatures (`sdpa`, `matmul_add`,
`silu_mul_matmul`) live in `test_fused.py`.

`matmul_add` mirrors the `k_add_*_reduce` kernels Deplodock emits inside a
block — the residual add gets fused into the matmul epilogue for o_proj /
down_proj. `silu_mul_matmul` mirrors `k_mul_*_reduce` — the SiLU+up gating
fuses into the down_proj reduction. Both should compile to a single launch;
that's the kernel that actually dominates block latency, so measuring the
fused chain is the apples-to-apples comparison.

Deplodock currently emits FP32 only, so both sides run FP32. The `dtype` field
on `Case` is kept for future FP16 support.

## How it's wired

- `cases.py` — `Case` dataclass + curated `CASES` list + `build_torch_ref` /
  `build_deplodock_graph`.
- `conftest.py` — `bench_pair` fixture (CUDA-event timing on the torch side,
  `CudaBackend.benchmark` on the deplodock side), session-end summary table,
  JSON dump to `.results/`.
- `test_primitives.py` / `test_fused.py` — one parametrized test each, ids =
  `case.name`.

The deplodock-side timing reuses `CudaBackend.benchmark` (per-kernel CUDA events
via cupy, see `deplodock/compiler/backend/cuda/program.py:369`) so launch counts
and per-kernel latency are available; the torch side uses `torch.cuda.Event`
matching the pattern in `scripts/bench_block.py:194-202`.

## Reading the output

`pytest_terminal_summary` prints one table sorted by `ratio = torch_us /
deplodock_us` ascending — losses (ratio < 1) first:

```
op       case                       shape                   torch_us  depl_us  ratio  launches
-------  -------------------------  ----------------------  --------  -------  -----  --------
matmul   matmul.qwen.gate_proj.s512 (1,512,3584) x (3584,18944)  1230.0  2890.4  0.43x         1
rmsnorm  rmsnorm.tinyllama.s512     (1,512,2048) x (2048,)         12.1    9.8   1.23x         1
...
```

The same data is dumped to `tests/perf/.results/<utc-timestamp>.json` for
cross-run diffing. `DEPLODOCK_GIT_REV` is recorded if set.

## Adding a case

Append to the appropriate builder in `cases.py` (`_matmul_cases`,
`_rmsnorm_cases`, …) — no test code changes required, parametrize picks it up
by name.

## Running

```bash
make bench-kernels                                                # full suite
./venv/bin/pytest tests/perf/ -m perf -v -k matmul                # one op
./venv/bin/pytest tests/perf/ -m perf -v -k 'matmul.tinyllama'    # one model
./venv/bin/pytest tests/perf/test_primitives.py -m perf -v -s     # primitives only
```

`make test` continues to skip everything in `tests/perf/`.
