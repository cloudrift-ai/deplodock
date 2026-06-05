# Bug: `LayerNorm` has no decomposition path (`unknown elementwise op 'layer_norm'`)

**Status:** open — discovered 2026-06-05 while generating a multi-op tuning dataset for the learned-prior model
bake-off (`scripts/gen_prior_data.sh`). Not related to the prior work; a standalone front-end gap.

## Symptom

Tracing / compiling / tuning a `torch.nn.LayerNorm` module fails almost immediately (~2 s) with:

```
ValueError: unknown elementwise op name: 'layer_norm' (not in numpy or _NAME_TO_FN)
```

raised from `deplodock/compiler/ir/elementwise.py:74` (`Elementwise.__init__`, when neither numpy nor
`_NAME_TO_FN` resolves the op name).

## Reproduce

```bash
./venv/bin/deplodock compile --code "torch.nn.LayerNorm(2048)(torch.randn(1,128,2048))"
# or trace / run / tune --code with the same expression
```

`RMSNorm` with the same shape works fine — only `LayerNorm` is broken.

## Root cause

`LayerNorm` is missing **both halves** of the treatment its sibling `RMSNorm` gets:

1. **No frontend capture.** `trace/torch.py:757` turns the traced `rms_norm` aten op into a dedicated
   `RmsNormOp` (`ir/frontend/ir.py:281`). There is **no** `LayerNormOp` and **no** corresponding capture for
   `layer_norm` / `native_layer_norm` in `trace/torch.py`. So the aten `layer_norm` survives as a generic
   elementwise op literally named `"layer_norm"`.
2. **No decomposition rule.** `passes/frontend/decomposition/080_rms_norm.py`
   (`PATTERN = [Pattern("root", RmsNormOp)]`) decomposes `RmsNormOp` into primitives; there is no analogous
   `0xx_layer_norm.py`. Even if the op were captured, nothing would lower it.

With neither, the bare `"layer_norm"` name reaches `Elementwise.__init__`, which only accepts numpy ufuncs or the
`_NAME_TO_FN` intrinsics (`rsqrt`, `relu`, …) — and rejects it.

## Fix sketch (mirror `RMSNorm`)

1. **Frontend op + capture.** Add a `LayerNormOp(eps=...)` to `ir/frontend/ir.py` next to `RmsNormOp`, and a
   capture branch in `trace/torch.py` (near line 757) that maps the traced `layer_norm` / `native_layer_norm` to
   it, threading the `weight` and `bias` inputs (LayerNorm has both; RMSNorm has weight only).
2. **Decomposition rule.** Add `passes/frontend/decomposition/0xx_layer_norm.py` with
   `PATTERN = [Pattern("root", LayerNormOp)]` decomposing

   ```
   layer_norm(x, w, b, eps) = (x - mean(x, -1, keepdim)) * rsqrt(var(x, -1, keepdim) + eps) * w + b
   ```

   reusing the same `_helpers` (`const_bc`, `broadcast_to`, `reduction_shape`, `open_fragment`) and the existing
   `MeanOp`. Unlike `080_rms_norm.py` (which `RuleSkipped`s when weight is absent), LayerNorm's affine `w`/`b` are
   standard; decide whether to support the no-affine (`elementwise_affine=False`) case or skip it.

Once decomposed, the existing reduction + elementwise lowering handles the rest (same machinery `RMSNorm` and
`Softmax` already ride), so it should tune/run like any other row-reduction op.

## Related robustness gap (separate, noted in passing)

The same sweep showed pointwise ops (`GELU`, `add`) failing the **final assembled compile** during `tune` with:

```
LoweringError: compile: 1 node(s) left un-lowered — the chosen tile shape produced a kernel that failed
validate(ctx) and the deterministic compile had no fallback
```

This is the greedy-uses-prior path (the just-trained linear prior's argmax picking an *invalid* tile shape, no
fallback to option-0). It is part of the in-flight learned-prior work — tracked there, not here — but it argues
for a `validate(ctx)` fallback in the greedy driver regardless of the prior model chosen.
