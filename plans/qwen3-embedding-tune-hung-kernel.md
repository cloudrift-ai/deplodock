# Qwen3-Embedding-0.6B: `k_linear_mean_reduce` generates a non-terminating -O3 kernel (OPEN)

Status: **unfixed codegen bug, escalated**. First seen as an occasional pick during the original two-layer tune; since
the 2026-06-09 layer-0 tune into the default caches (`~/.cache/deplodock/autotune.db` + prior), the hanging config is
the prior's **greedy pick** for this op — every `run` / `compile` / bench that deploys this kernel at -O3 now hangs
deterministically. The bench harness survives it (watchdog → `bench_fail`, worker SIGKILL — see "already hardened"
below), but the kernel itself still needs a codegen fix, and real deployments of Qwen3-class models would hang.

## 1. The kernel

The slice is the decoder MLP path: post-attention RMSNorm (`pow → mean → +eps → rsqrt → mul → weight-mul`) feeding the
gate/up projections (1024 → 3072, fp16) and the SwiGLU combine (`silu(gate) * up`), output `mul_12` of shape
`(1, 32, 3072)`. Healthy picks lower it to a register-tiled fused linear+mean kernel (e.g. the pre-tune greedy config
ran 88 µs; a cold-prior pick ran 185 µs); the hanging variant is in the same register-tiled family.

## 2. Symptom

- The -O1 tuning sweep benches the variant fine — the hang only manifests when the winner is **recompiled at -O3**
  (the deployable default). The kernel never returns: GPU pinned at 100 %, host spin-waiting.
- The per-launch watchdog catches it (`kernel 'k_linear_mean_reduce_23ab9c' did not complete within 1000 ms — variant
  marked bench_fail`), but the launched kernel **stays resident on the device until the owning process exits** — CUDA
  context teardown blocks behind it, and any other process creating a context meanwhile stalls too.
- Almost certainly a miscompile or a loop bound that -O3 (`cicc`) optimization turns non-terminating in the unrolled
  register-tile reduce loop. The pick came out of the -O1 ranking sweep, whose latencies don't reflect (or exercise)
  the -O3 codegen.

## 3. Reproduction (deterministic, current default caches)

```bash
DEPLODOCK_DUMP_DIR=/tmp/dd deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0   # writes the reproducer + .cu
timeout 300 deplodock run --ir /tmp/dd/07_lowering_cuda.kernels/k_linear_mean_reduce_*.torch.json --bench
# → HungKernelError within ~1 s of the first -O3 launch; GPU stays busy until the process exits
```

`deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --bench` reproduces it twice: the full-model -O3 bench fails with
the watchdog message above, and the per-kernel row for this kernel is skipped. Bench this kernel **last** and under
`timeout` when sweeping reproducers — the resident kernel outlives the bench attempt.

## 4. Already hardened (the bug no longer wedges runs)

- Per-launch watchdog + SIGKILL-able bench worker (#215): a hung kernel fails one variant/row, not the run.
- PR #216 follow-ups, found while reproducing this: the parent previously wedged forever **writing the next request**
  to a worker stuck in CUDA teardown behind the immortal kernel (the wall budget only bounded the response read — now
  the send has the same deadline), and the persistent worker silently degraded torch.compile comparisons to eager via
  dynamo's per-code-object recompile limit (now reset before every bench compile).
- Measurement fixes in #216 also corrected this doc's old per-kernel numbers: timings are CUDA-graph-captured (pure
  GPU), the torch reference runs fp16 like the real model (a dtype bug ran it fp32), and reproducer slices keep
  constant-derived boundaries (the old NaN rows now pass accuracy). The previously-reported "deplodock wins most
  per-kernel rows" conclusion was a measurement artifact; honestly measured, torch.compile wins or ties most rows and
  this kernel is deplodock's worst (eager 63 µs / torch.compile 14 µs / deplodock 88 µs on the healthy pre-tune pick —
  and a hang on the current tuned pick).

## 5. Root-cause plan

1. **Pin the hanging variant.** `compile` with a dump now picks it by default: read its knob row from the dump's
   kernel stats / `cuda_op` table and capture the exact knob set (suspect family: degenerate `BK` / `STAGE` / `FK`
   register-tile combinations).
2. **Inspect the -O3 codegen.** Diff the dumped `.cu`'s reduce loop bounds against a healthy variant; compare -O1 vs
   -O3 SASS (`cuobjdump -sass` on the cached cubins) for the loop that never exits — look for a hoisted/folded exit
   condition or an induction variable the unroller wraps.
3. **Bisect knobs.** Re-compile with `DEPLODOCK_<KNOB>` pinning (`_pinned_knobs`), flipping one knob at a time from the
   hanging set toward a healthy set to find the minimal trigger.
4. **Fix + regression-test.** Fix the emitter (or reject the degenerate combination at enumeration time); add a
   compile-only CUDA-assert test with the full knob set + `--target` pinned so GPU-less CI checks the emitted source,
   plus a GPU test that the variant completes under the watchdog.
5. **Unblock the prior.** Once the config is fixed or rejected, re-tune layer 0 so the DB/prior rows for this op stop
   pointing at it.

## 6. Fixed since the original report (for the record)

- The RMSNorm reproducer (`k_mean`) standalone-lowering failure is gone — it lowers and benches (1 µs, 37× vs eager).
- The three NaN reproducers (Q-norm / K-norm / SwiGLU) verify accuracy now: slice boundaries that are pure functions
  of constants (the pow exponent, eps) keep their constant chains instead of being fed random bench data.
- The `k_reshape_linear_mean_reduce` 113 µs / 56 µs mispicks (grid=1 single-block norm) were prior cold-start gaps;
  tuning the shape fixed them (11 / 10 µs — 5.4–5.6× faster than eager, within 1.4× of torch.compile's 8 µs).
