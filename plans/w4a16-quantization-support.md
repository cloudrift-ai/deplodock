# W4A16 (AWQ-INT4) quantization support in the deplodock compiler

> Status: **Phase 0 implemented** (functional correctness). Phases 1–6 below are forward-looking.
>
> Phase 0 landed the full op + trace integration + integer/cast codegen: a compressed-tensors AWQ-INT4 model
> traces, compiles, and runs on the CUDA backend with in-kernel dequant. Verified end-to-end on
> `cyankiwi/Llama-3.1-8B-Instruct-AWQ-INT4 --layer 0` (224 linears substituted; accuracy vs eager
> `max_diff=0.0020, mean_diff=8e-6 PASS`) plus a hermetic `tests/compiler/test_dequant.py` suite (numerics oracle,
> `compressed_tensors` parity, integer/cast codegen, substitution walker, CUDA e2e). Notably the default fusion
> pipeline already pulls the dequant producer cone *into* the matmul kernel (gmem holds only packed int32 + scales +
> zp, the fp16 weight lives transiently in registers), so this build already realizes much of the Phase 3 VRAM goal —
> no separate `DEPLODOCK_W4A16_FUSE` gate was needed. One real codegen fix beyond the plan: `right_shift` /
> `bitwise_and` are forced to an i32 result in `dtype_promote` (they are integer-only and have no float form), which
> keeps the unpack correct when a staged-smem Load's dtype is unresolved at stamp time.

## Context

`deplodock serve cyankiwi/Llama-3.1-8B-Instruct-AWQ-INT4` (and equally `compile`/`run`/`tune` on any
compressed-tensors AWQ model) fails. Root cause: deplodock's compiler is **fp16/fp32-dense only** — it has zero
quantization handling. Today the only path that "works" is `serve --stock` (raw vLLM, native AWQ), which bypasses
deplodock kernels entirely.

The compressed-tensors checkpoint registers a forward **decompress pre-hook** that unpacks int4 weights on first
forward. When `torch.export` traces the model, that hook fires mid-trace and injects a data-dependent slice
(`unpack_from_int32 → int(shape[1])`, `compressed_tensors/.../pack_quantized/helpers.py:141/153`), which torch.export
cannot specialize → `GuardOnDataDependentSymNode: ...u0`.

**Goal (chosen scope):** true **W4A16 in-kernel dequant** — packed int4 weights stay packed in GPU memory and are
dequantized in registers/smem *before* the matmul, preserving the VRAM savings (not materializing a full fp16 weight in
gmem). Surfaced through **`deplodock compile` / `run` / `tune`** (the CLI trace+compile paths). `serve` is out of scope for
now — and **not** free: it has its own live-module constant-binding path (`serving/runner.py:119-128`) that
blanket-casts every parameter/buffer through `float32` → `dtype_str`, which would corrupt packed int32 weights and
zero-points. It reuses the same *trace* path, but needs the same dtype-preserving binding fix (below) applied to that
runner before it works.

This is a multi-phase compiler feature. The plan is **correctness-first, then performance**: prove numerics with a
materialized fp16 weight, then fuse, then move dequant into the kernel for the actual VRAM win.

## Verified format (this exact model)

`config.quantization_config`: `format=pack-quantized`, `num_bits=4`, `group_size=32`, `strategy=group`,
**`symmetric=false`** (asymmetric → zero-points present), `targets=["Linear"]`, `ignore=["lm_head"]`.

Per-linear on-disk tensors (e.g. `mlp.gate_proj`, out=14336, in=4096):
- `weight_packed` **I32 `[out, in/8]`** — 8 **unsigned** int4 nibbles per int32, **interleaved** along dim 1
  (`packed_dim=1`): `nibble[:, i::8] = (packed >> (4*i)) & 0xF`, i in 0..7 (values 0..15).
- `weight_scale` **F16 `[out, in/group]`** — group along the K/in axis (G=32).
- `weight_zero_point` **I32 `[out/8, in/group]`** — zp is itself **int4-packed along OUT** (different axis than the
  weight), also unsigned nibbles. **Dequant (asymmetric, this model): `w = (nibble − zp) · scale`** — unsigned nibble
  minus unsigned zp, **no −8**.
- **Offset is conditional, not unconditional.** The symmetric/no-zp case is `(nibble − 8) · scale` (the `8` is the
  implicit zero-point). compressed-tensors' `unpack_from_int32` subtracts 8 from **both** weight and zp
  (`helpers.py:158-159`), then `dequantize` does `(w−zp)·scale` — the two −8s cancel, reducing to the same
  `nibble − zp`. So the decomposition must subtract zp **xor** 8 (gated on `symmetric`), never both — applying −8 *and*
  an unsigned-zp subtract double-shifts. (Matches the vLLM W4A16 triton reference: unsigned nibbles − zp, 8 only as the
  no-zp bias.)
- `lm_head` is in `ignore` → stays dense fp16, must NOT be substituted.

## Approach (decided)

**Pre-trace submodule substitution** (the established `trace/huggingface.py` `_PassThroughRotary` /
rotary→buffer-swap pattern, lines ~53-123). Walk the loaded model, `setattr` each targeted quant linear with a
deplodock-authored `DequantLinear` (holding `weight_packed`/`weight_scale`/`weight_zero_point` as buffers + optional
bias) **before** `torch.export`. This: (1) keeps the packed int32 / fp16-scale / int32-zp tensors as graph
**constants**, so the binder + the already-dtype-generic uploader carry them packed to gmem with no new plumbing; (2)
prevents the decompress pre-hook from ever firing, so the data-dependent guard never arises; (3) emits a single opaque
custom op carrying the packed buffers + scheme metadata, which the Phase 0 decomposition (`045_dequant_linear`) expands
into the unpack/dequant/matmul cone the rest of the pipeline lowers. (Rejected: trace-through-and-fix-the-guard —
fights torch.export and still materializes a full fp16 weight; post-trace IR rewrite — must re-derive quant metadata
from shapes, strictly harder.)

**Trace integration — `DequantLinear.forward` is a registered custom op.** `torch.export` decomposes a plain
`nn.Module` into constituent ATen calls, and the tracer (`trace/torch.py`) maps only `call_function` nodes by name
(`linear`→`LinearOp` at `:526`, `mm`/`addmm`→`MatmulOp` at `:538/548`); a `DequantLinear` module would therefore
dissolve into a cone of aten ops, *not* surface as one op. Two further tracer facts make tracing that cone hostile:
`aten.to`/`type_as` are **pass-through** (`trace/torch.py:698` — the pipeline is "dtype-unified", so a dtype-changing
cast is silently dropped), and the unpack surfaces as `aten.__rshift__.Scalar` / `aten.__and__.Scalar` (verified in the
failed-trace dump), names the tracer doesn't canonicalize. **Decision:** make `DequantLinear.forward` a
`torch.library.custom_op` (`deplodock::dequant_linear`, with a `register_fake` for shape prop) so export preserves it
as a single opaque `call_function` node; map that name in `trace/torch.py` beside `linear`/`mm` to a new frontend
`DequantLinearOp`. **The buffers must be passed as explicit tensor arguments**, not closed over — the module forward
calls `dequant_linear(x, self.weight_packed, self.weight_scale, self.weight_zero_point, self.bias, ...scheme...)`; an
opaque op that reads `self.weight_packed` internally would hide those tensors from export, so they'd never surface as
graph constants for binding. The op's **non-tensor `QuantScheme`** (num_bits / group_size / packed_dim / symmetric) is
recorded as `DequantLinearOp` metadata (scalar args), not as tensors. The unpack/dequant arithmetic is then **authored
in deplodock's Phase 0 decomposition** (`045_dequant_linear`), using deplodock-internal op names and an explicit cast
op — so the tracer never has to handle aten casts or `__rshift__` name variants at all. (Fallback if the custom op is
impractical:
pattern-collapse the exported aten cone into
`DequantLinearOp`; that route *does* require canonicalizing `__rshift__`/`__and__`/`bitwise_right_shift`/`bitwise_and`
and intercepting the dtype-changing `to`.)

## Phases

### Phase 0 — Functional correctness: a quantized linear compiles + runs (no VRAM win yet)
The dequant cone lowers to its own kernel(s) materializing a full fp16 weight in gmem; the matmul reads it. Isolates
numerics from fusion. This phase carries the **op + trace integration + integer/cast codegen** — everything needed for a
`DequantLinearOp` to execute correctly.
- **New** `compiler/trace/quantized.py`: parse `quantization_config` (handle `quantization_config` and legacy
  `compression_config`; `num_bits/group_size/symmetric/strategy/packed_dim/targets/ignore`); a `DequantLinear` module
  whose `forward` is the registered `torch.library.custom_op` `deplodock::dequant_linear` (eager impl delegates to a
  clean torch dequant for the accuracy reference; `register_fake` returns `[*x.shape[:-1], out]`), holding
  `weight_packed`/`weight_scale`/`weight_zero_point` (+bias) as buffers; a module-substitution walker. Detect via
  `quant_method == "compressed-tensors"`; no-op (byte-identical) when absent. Skip `ignore` modules (`lm_head`).
- **Trace integration** (the answer to "how the op gets into the graph"): **new** frontend `DequantLinearOp` beside
  `LinearOp` (`compiler/ir/frontend/ir.py`) with a numpy `forward()` (CPU interpret + accuracy harness); map the custom
  op's name to it in `trace/torch.py` (beside the `linear`/`mm`/`addmm` cases at `:526-548`). **Extend `_op_name`**
  (`trace/torch.py:398`, today `aten.`-only) to return a name for the `deplodock::dequant_linear` target — otherwise it
  returns `None` and the op hits the `op_name is None` fallback (`:467`) that silently aliases the node to its first
  input `x`, dropping the entire dequant.
- **Custom-op arg handling — a dedicated resolver branch, not the generic one.** `_resolve_inputs`
  (`trace/torch.py:325`) walks only `fx_node.args` (ignores `kwargs`) and turns positional numeric scalars into
  `ConstantOp` graph inputs (`:335-346`). So the generic path would (a) drop any scheme passed as kwargs and (b) leave
  `num_bits`/`group_size`/`packed_dim`/`symmetric` as bogus graph inputs. The custom-op case must **peel the tensor
  inputs** (`x`, `weight_packed`, `weight_scale`, `weight_zero_point`, optional `bias`) as the op's graph inputs/
  constants from the **scalar scheme**, read `fx_node.kwargs` too, and store the scheme on `DequantLinearOp` as op
  metadata (a `QuantScheme`) — never as graph `ConstantOp`s. **Record `has_bias`** on the op (like `LinearOp.has_bias`):
  generic `_resolve_inputs` drops a `None`/non-tensor arg, so a `None` bias and a real bias-tensor would otherwise be
  indistinguishable — the decomposition's "add bias when present" needs that flag to know which.
- **New decomposition rule** `passes/frontend/decomposition/045_dequant_linear.py` (+ `dequant_decompose` in
  `_helpers.py`, mirroring `matmul_decompose`). **Two independent unsigned-nibble unpacks** for this asymmetric model:
  (1) `weight_packed` along `packed_dim` → `[out,in]` unsigned-nibble int; (2) `weight_zero_point`, itself int4-packed
  **along OUT**, → `[out, in/group]` unsigned-nibble int. Then **`(nibble − zp)`** with **zp broadcast on OUT and
  group-indexed on K** (a distinct IndexMap from the scale's) → int→fp16 cast → group-broadcast `· scale` →
  **transpose `[out,in]`→`[in,out]`** (`matmul_decompose` expects `A[…,K] @ B[K,N]`; `LinearOp` does this same
  `TransposeOp(axes=(-2,-1))` before the matmul, `040_linear.py:21-27` — without it the K/N axes and output features are
  wrong) → `matmul_decompose` → **add bias** (the captured `bias`, broadcast + added *after* the matmul exactly as
  `LinearOp` does, `040_linear.py:32`, gated on the recorded `has_bias`). **Offset gating:** `symmetric=false` subtracts the unpacked `zp`;
  `symmetric=true` subtracts the constant `8` instead (NOT "skip the subtract" — dropping it omits the implicit
  zero-point). Never both (see the format note).
- **Unpack representation — 8 fixed-shift lanes (Phase 0, no IR gap).** The reference unpack is
  `unpacked[..., i::8] = (packed >> 4i) & 0xF` for `i in 0..7`. Phase 0 emits exactly this: **eight lanes, each a
  constant shift** `lane_i = (packed >> 4i) & 0xF` (the `4i`/`0xF` are dtype-preserving int scalar literals — no
  coordinate-dependent operand, so it fits today's `Assign(args: tuple[str,...])` which carries SSA names, not Exprs).
  **Assemble `[out,in]` via a multi-source `IndexMapOp`** (8 `IndexSource`s — source selected by `k%8`, coord `k//8` —
  exactly what `150_cat.py` already lowers a cat *into*), then a copy materializes it to gmem. Do **not** use an 8-way
  `CatOp`: only the 2-tensor cat variant is lowered (`150_cat.py:4`); a multi-source `IndexMapOp` sidesteps that (the
  alternative is adding variadic / binary-tree cat). The zp unpacks the same way, 8 fixed-shift lanes assembled along
  OUT (`out%8` / `out//8`). This **avoids the index-dependent shift entirely**; Phase 1 fuses the same lanes.
- **The cast op** (today the IR has only `ElementwiseOp` + dtype-stamped `Assign`; no cast op exists). Define a
  **tensor-IR `CastOp(target_dtype)` in `ir/tensor/ir.py`** — it's a *post-decomposition* primitive (emitted by
  `045_dequant_linear`, consumed by loop lifting), so it belongs in tensor IR, not frontend: numpy `forward` =
  `.astype`; serializes `target_dtype`; output `Tensor` carries the target dtype, input stays i32.
  - **Lifting** — a loop-lifting rule for `CastOp` (the elementwise lift `010_lift_elementwise.py:45` emits
    `Assign(op=root.op.op, ...)` with **no `dtype=`**; reuse it only if it's taught to stamp `dtype=root.output.dtype`,
    else add a dedicated rule emitting a copy `Assign` with `dtype=F16`).
  - **Render — an explicit copy/cast path, not the generic `Assign` path.** `Assign.render` takes the native non-f32
    path only when *all* args are already at `result_dt` (`leaves.py:414`); a cast's i32 input never matches its f16
    result, so it falls to the **f32-promote path** (`(float)v` → `__float2half`) — numerically fine but not the direct
    `__int2half_rn` the codegen test asserts. Add a copy/cast render branch (keyed on `copy` with a dtype change) that
    calls `target.convert(arg, src_dt, result_dt)` **directly** (i32→f16 = `__int2half_rn`), one conversion, no f32
    detour.
  - CPU correctness of the cast is checked via the numpy reference, **not** the C++ Loop runner (float-only — see the
    Loop CPU-interpret bullet below, option (b)).
- Wire substitution into the single tracing chokepoint `commands/compile.py::_trace_model` (which `run`/`tune` reach
  through `load_or_trace`).
- **Fix the blanket fp32 casts** that would destroy int weights — bind each constant at its graph-declared dtype:
  - `compiler/loader/binder.py:96-97` (`bind_constants_from_module`, `.float().numpy()` — note it uses `state_dict()`
    for tied weights; the `DequantLinear` buffers surface there).
  - `commands/run.py:_bind_inputs` (~1267-1269, `.astype(np.float32)` for params/buffers). Inputs there already cast by
    `node.output.dtype.np` — extend that per-node logic to the constant `sources` dict.
- **Integer + cast codegen — the first mixed-int/fp graph in the pipeline** (the IR is otherwise "dtype-unified", so
  this is more than a one-liner). All of:
  - `compiler/ir/elementwise.py` `_NAME_TO_FN` — add `right_shift` (`np.right_shift`), `bitwise_and` (`np.bitwise_and`)
    (deplodock-internal names, emitted by our decomposition — no aten canonicalization needed via the custom-op route).
  - `compiler/ir/stmt/base.py` — add `>>`/`&` to `_BINARY_OP`/`op_to_expr`; integer rule in `dtype_promote`
    (all-i32 → i32, never silently → f32).
  - `compiler/ir/expr.py` `BinaryExpr` — add `>>`/`&` to the op set (`expr.py:274` has `+ - * / // % … ^` but not the
    bitwise pair), with `eval` (`np.right_shift`/`np.bitwise_and`), `render`, and a precedence entry. This is the layer
    `op_to_expr` builds on, and it powers CPU interpret, pretty output, and the Phase 1 coordinate path. Note `%` and
    `//` already live here, so the `k%8` / `k//8` index math the shift-lookup needs is already available.
  - `compiler/backend/cuda/render_target.py` — `type_name` `i32 → "int"` (today defaults to `"float"`, a latent bug);
    **`convert()` gains `i32↔f16` and `i32↔f32`** (today only f16↔f32 exist — without this, `Assign.render`'s
    f32-promote path at `leaves.py:441` silently no-ops the missing conversion); `has_native_op` returns True for the
    native integer ops (`+ - * >> &`) so `leaves.py:415` takes the native-i32 path instead of f32-promote.
  - **Dtype-preserving scalar-literal constants** — the unpack's immediates (shift `4`, mask `15`, offset `8`) flow as
    scalar `ConstantOp`s, but `010_lower_kernelop.py:37` coerces every inlined scalar const via `float(t.value)` and
    `Load.render` (`leaves.py:236-239`) / the literal-SSA substitution (`base.py:530`) stamp them `f32` — so `packed >>
    4` would render `packed >> 4.0f` and fail to compile. Carry integer-typed scalar constants as **int** literals
    (`int(t.value)` + `i32` ssa dtype) on all three paths.
  - **Dtype-aware `Select`** — the lane assembly lowers to a `Select` (`030_lift_indexmap.py:43` emits one for a
    multi-source `IndexMapOp`), but `Select.render` (`leaves.py:872`) hardcodes `float v = …` and `select_to_ternary`
    (`base.py:240`) casts every branch to `float` — so the i32 lanes are forced to f32 *before* the `− zp` int
    subtract. Add a result-dtype field on `Select` (like `Assign.dtype`) and render the declared type (`int` for i32),
    the ternary casting branches to the result dtype, not unconditionally `float`. **Thread the new field through every
    site that touches `Select`**, or fusion/simplify silently drops it: the type-stamp pass `030_stamp_types.py` (set it
    so downstream `Write.value_dtype` is right) and the rewrite/simplify helpers that reconstruct
    `Select(name=, branches=)` today (`ir/stmt/passes.py:161` and `:225`). Required by **both** Phase 0 (multi-source
    IndexMap assembly) and Phase 1 (lane select).
  - the `CastOp` `convert()` lowering (above) — the dequant's int→fp boundary must render here, not later.
- **Loop CPU interpret dtype preservation** — bigger than the numpy arrays. `LoopOp.forward` (`ir/loop/ir.py:212`)
  coerces every input to `np.float32`, **and** `render_loopop_cpp` (`ir/loop/runner.py:63-68`) emits `const float*`
  inputs + a `float*` output — so even dtype-correct numpy gets reinterpreted as float by the generated C++. Two
  acceptable resolutions, pick one explicitly: **(a)** make the runner signature + output dtype-aware (typed params per
  `buf.dtype`, e.g. `const int*` for i32) and stop the `float32` coercion; or **(b)** — the cheaper default — **skip
  `LoopOp.forward` CPU validation for mixed-int graphs** and validate the dequant cone with a numpy-only reference (the
  deplodock CUDA path is still covered by the GPU accuracy check). Until (a) lands, the Phase 1 `LoopOp.forward()`
  check is meaningless for the int unpack, so the plan assumes (b).
- **Verify**: unit test `DequantLinearOp.forward` (numpy) + `DequantLinear.forward` (eager) against a **tiny synthetic
  packed fixture** built in-test (network-free per `tests/ARCHITECTURE.md`: no GPU/Docker/network) — exact on the
  integer unpack, fp16-eps on the scale multiply; an **optional, network-gated** parity test vs
  `compressed_tensors.dequantize` (skip unless the package + a checked-in slice are present). Then (GPU-gated)
  `deplodock run --layer 0` accuracy vs eager passes.

### Phase 1 — Fusible unpack layout (refine the decomposition for Phase 3)
The `DequantLinearOp` + its decomposition already exist (Phase 0); here the unpack is re-expressed so reading
`unpacked[o,k]` *inside* the matmul K-loop needs no materialized `[out,in]` tensor — i.e. element `[o,k]` =
`(packed[o, k//8] >> (4*(k%8))) & 0xF`. Two coordinate-dependent pieces, and how each is represented:
- The **gather** `packed[o, k//8]` is a `Load` whose `Load.index` carries the `k//8` Expr — already supported (`%` and
  `//` exist in `ir/expr.py`, `BinaryExpr` at `expr.py:274`).
- The **index-dependent shift** `>> (4*(k%8))` is the real gap: `Assign.args` are SSA names, not Exprs, so the shift
  amount can't be a coordinate expression on the op. **Chosen representation: Phase 0's 8 fixed-shift lanes + an 8-way
  `Select`/`TernaryExpr` keyed on `k%8`** — all eight lanes `(packed[o,k//8] >> 4i) & 0xF` share the one `Load` of
  `packed[o,k//8]` (the `k//8` is a `Load.index` Expr), and the select picks lane `k%8` (`Select`/ternary carry Exprs).
  Every shift is a constant `4i`, so no `Assign` ever needs a coordinate-valued operand. **No bound tensor is needed** —
  this matters because the obvious `[8]` shift-lookup constant has no storage path: `ConstantOp.value` is scalar-only
  (`ir/base.py:127`) and synthetic small tensor-constants aren't materialized/bound, so a lookup tensor would require
  new infra. (Alternatives, both heavier: add synthetic small-tensor-constant support for the shift lookup; or extend
  `Assign` to admit expr-valued scalar operands.) This keeps the cone a strict pointwise producer of the matmul's B
  Load (the Phase 3 precondition) rather than a materialize-then-read pair.
- **Verify**: a **numpy-only reference** for the dequant cone matches Phase 0's `DequantLinearOp.forward` (Phase 0
  option (b) — the float-only `LoopOp.forward` / `render_loopop_cpp` path can't carry i32, so it's skipped for the
  mixed-int cone; if option (a) lands, use `LoopOp.forward` instead). End-to-end correctness rides the GPU accuracy
  check.

### Phase 2 — Group/zp granularity generalization
- Group broadcast reuses the existing `gqa_broadcast` IndexMap pattern (`_helpers.py:147-157`): scale indexed by
  `k // G` on the K/in axis. **zp is packed along OUT** → a *second, distinct* IndexMap (different axis-divide) — easy
  to transpose-confuse, unit-test each independently. Don't hardcode G=32 (test G=128 and per-tensor/per-channel too).
- (The int→fp16 cast op moved to Phase 0 — it's foundational, not a granularity concern.)

### Phase 3 — Dequant→matmul fusion (the VRAM requirement: no full fp16 weight in gmem)
The dequant producer chain is a strict-producer cone feeding the matmul's B Load — the same shape as
`005_split_demoted`'s computed-B cone. The splicer (`ir/loop/splicer.py`, `passes/loop/fusion/010_merge_loop_ops.py`)
fuses it into the matmul `LoopOp`, so gmem holds only packed int32 + scales + zp; the fp16 weight exists transiently
per-tile.
- Check the **stager** (`passes/lowering/tile/020_stage_inputs._classify`) admits the non-affine access (`packed[o,
  k//8]` gather + index-dependent `(>> 4*(k%8)) & 0xF`, plus group-divide `k//G` scale). Confirm
  `classify_matmul_operands` (`_atom.py:42-91`, structural/dtype-agnostic) still tags the cone as B.
- **Fallback gate** `DEPLODOCK_W4A16_FUSE` (default off until this lands): if fusion bails, keep Phase 0's separate
  dequant kernel — correct but no VRAM win — so a correctness build always exists.
- **Verify**: `compile --ir loop|cuda` shows no `[out,in]` fp16 scratch slab; measure peak VRAM vs Phase 0.

### Phase 4 — In-smem dequant via `StageBundle.compute` (scalar tier first)
Stage packed int32 (+ scales + zp) into smem, then a hoisted **`StageBundle.compute`** phase
(`ir/tile/ir.py:1315`, emitted by `030_hoist_invariant_compute`) unpacks/dequantizes cooperatively into an fp16 smem
slab the matmul body Loads from. Reuses an existing phase slot — medium invasiveness, no new tier.
- **Verify**: `compute-sanitizer` clean; accuracy vs eager; smem within `ctx.max_dynamic_smem`.

### Phase 5 — MMA tier (the real W4A16 win)
**Preferred first cut:** dequant lands in the smem slab (Phase 4) and standard `ldmatrix` reads fp16 — this already
satisfies the goal (gmem holds only packed int4) with **medium** invasiveness and no fragment-loader rewrite.
- Files: `passes/lowering/kernel/005_lower_atom_tile.py:82-223` (MMA fragment emission),
  `ir/kernel/render.py:134-220` (`dpl_ldmatrix_*` / `dpl_mma_load_b_*` templates), `ir/kernel/ir.py`.
- True register-resident dequant (skip the fp16 smem slab, assemble `__half2` fragments from nibbles in the PTX
  m16n8k16 lane layout) is the final optimization — **high** invasiveness, not required for the VRAM goal.
- Note: `RegEpilogue` (`ir/kernel/ir.py:358`) is POST-accumulate → NOT usable for pre-mma weight dequant.

### Phase 6 — Tuning / goldens / accuracy gate
- `tune <model>` reaches the same `load_or_trace`/`_trace_model` path (and `_bench_worker` rebuilds via `load_or_trace`
  in-child), so it works once Phase 0 wiring lands; dequant kernels are new op identities → tuned like any kernel.
- `S_*` structural features (`992_stamp_structural_features`: stmt/op histogram + dtypes) make a W4A16 matmul key
  distinctly from its fp16 twin in `ShapeKey`/the prior automatically — no schema change. Optionally add a dedicated
  `.w4a16` `GOLDEN_CONFIGS` shape via the `tune-golden` flow for a tuned prior reference.
- Accuracy: eager == the substituted `DequantLinear`, which is the deployed reference (deplodock-kernel ↔
  DequantLinear eager), and is **optionally cross-checked** against `compressed_tensors.dequantize` in the gated parity
  test — so the default accuracy gate stays network/package-free. int4 round-trip is exact on the integer side (error is
  only fp16 scale rounding) → keep a tight (~fp16-eps) tolerance, not a loose int one (`commands/run.py::_check_accuracy`).

### Phase 7 — Tests (per `tests/ARCHITECTURE.md`: no GPU / Docker / network by default)
- Unit (no GPU, no network): `right_shift`/`bitwise_and` numpy forward; `dequant_decompose` vs a numpy reference over a
  **tiny synthetic packed fixture built in-test** (the numerics oracle). An **optional, network-gated** test asserts
  parity vs `compressed_tensors.dequantize` (skipped unless the package + a checked-in slice are present).
- Unit (codegen): i32 local renders as `int`, `>>`/`&` emit verbatim, the `CastOp` emits `__int2half_rn`, no spurious
  `__half2float`.
- e2e (GPU-gated, `cuda` xdist group): one quantized linear via `compile --code` toy `DequantLinear` (self-contained,
  no download). A `run --layer 0` on the real HF model is a **separate GPU+network-gated** check, not part of the
  default suite; peak-VRAM assertion once Phase 3+ lands.

## Critical files
- **new** `deplodock/compiler/trace/quantized.py` — config parse + `DequantLinear` (the `deplodock::dequant_linear`
  `torch.library.custom_op` + `register_fake`) + pre-trace substitution (model on `trace/huggingface.py:53-123`)
- `deplodock/compiler/trace/torch.py` — extend `_op_name` (`:398`, aten-only → recognize `deplodock::dequant_linear`
  before the `:467` first-input-alias fallback); map it to `DequantLinearOp` (beside `linear`/`mm`/`addmm` at
  `:526-548`); **custom-op arg branch** that peels tensor inputs from the scalar `QuantScheme` and reads `kwargs`
  (`_resolve_inputs:325` walks only `args` and turns scalars into `ConstantOp`s) — scheme scalars become op metadata,
  not graph inputs
- `deplodock/compiler/ir/tensor/ir.py` `IndexMapOp` (multi-source assembly of the 8 lanes — `passes/.../150_cat.py`
  lowers only 2-tensor `CatOp`, so use a multi-source `IndexMapOp`, not an 8-way cat) and `ir/stmt/leaves.py` `Select` /
  `ir/expr.py` `TernaryExpr` (the `k%8` lane select — no bound lookup tensor, since `ConstantOp.value` is scalar-only)
- **Dtype-aware `Select`** — add a result-dtype field and thread it everywhere: `ir/stmt/leaves.py:872`
  (`Select.render` hardcodes `float`), `ir/stmt/base.py:240` (`select_to_ternary` casts branches to `float`),
  `passes/loop/lifting/030_lift_indexmap.py:43` (emit the dtype), `030_stamp_types.py` (so `Write.value_dtype` is
  right), and the `Select` reconstructions in `ir/stmt/passes.py:161` + `:225` (else simplify/fusion drops the field)
- `deplodock/compiler/ir/stmt/base.py` (`_BINARY_OP`/`op_to_expr` + `dtype_promote` integer rule) and
  `deplodock/compiler/ir/expr.py` (`BinaryExpr` `>>`/`&`: eval/render/precedence) and
  `deplodock/compiler/backend/cuda/render_target.py` — `type_name` i32→`int`, **`convert()` i32↔f16/f32**,
  `has_native_op` integer ops (NOT just `_TYPE_NAME`); + the int→fp16 cast (`__int2half_rn`)
- `deplodock/compiler/ir/elementwise.py` — `right_shift`/`bitwise_and` in `_NAME_TO_FN`
- **Integer scalar-literal lowering** — `passes/lowering/cuda/010_lower_kernelop.py:37` (`float(t.value)`),
  `ir/stmt/leaves.py:236-239` (`Load.render` scalar-const path), `ir/stmt/base.py:~530` (literal-SSA substitution):
  carry int-typed scalar consts as int literals, not `f32`
- **`CastOp`** — define in `ir/tensor/ir.py` (post-decomposition primitive); **lifting** rule
  (`passes/loop/lifting/010_lift_elementwise.py:45` emits `Assign(..., )` with no `dtype=` → stamp
  `dtype=root.output.dtype` or a dedicated rule); **explicit copy/cast render path** in `ir/stmt/leaves.py`
  (`Assign.render`'s native path needs all-args-at-result `:414`, which a cast fails → add a direct
  `target.convert(arg, src, result)` branch, not the f32-promote detour)
- **Loop CPU-interpret dtype** — `ir/loop/ir.py:212` (`np.float32` coercion) + `render_loopop_cpp`
  (`ir/loop/runner.py:63-68`, emits `const float*`/`float*`): either make signature+output dtype-aware (option a) or
  skip `LoopOp.forward` for mixed-int cones + use a numpy reference (option b, default)
- `deplodock/compiler/loader/binder.py` and `deplodock/commands/run.py` (`_bind_inputs`) — bind at declared dtype, not
  blanket fp32 (and `serving/runner.py:119-128` when serve is later in scope)
- `deplodock/compiler/ir/frontend/ir.py` (`DequantLinearOp` + `has_bias`) (+ `passes/frontend/decomposition/
  045_dequant_linear.py`, `decomposition/_helpers.py`) — `dequant_decompose`, weight + zp unpacks, transpose, bias add,
  group/zp broadcasts (emits the tensor-IR `CastOp` above)
- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` (+ `030_hoist_invariant_compute`,
  `ir/tile/ir.py` `StageBundle.compute`), `deplodock/compiler/ir/kernel/render.py`,
  `passes/lowering/kernel/005_lower_atom_tile.py` — in-smem/register dequant
- `deplodock/commands/compile.py::_trace_model` — substitution entry point
- docs to update on completion: `compiler/ARCHITECTURE.md`, `compiler/backend/cuda/ARCHITECTURE.md`, `CLAUDE.md`

## Biggest technical risk
**The interleaved `i::8` unpack as fusible in-kernel IR.** Numerics (Phase 0/1) are mechanical, but the VRAM goal
demands the unpack's non-affine index math (`packed[o, k//8]` gather + index-dependent `(>> 4*(k%8)) & 0xF`, group
`k//G` scale, OUT-axis-packed zp) survive fusion/staging without `020_stage_inputs._classify` bailing to a separate
kernel. This is the first index shape mixing a layout gather with an index-dependent compute inside a staged slab. If
the stager can't represent it, you fall back to Phase 0's materialized weight — correct, but forfeits the VRAM win.
De-risk early by prototyping the `StageBundle.compute` dequant (Phase 4) on a tiny shape before the MMA tier.

## Extensibility to other quantization formats

The design isolates all format-specific logic in **two narrow seams**; everything below them is format-agnostic, so a
new format reuses the whole pipeline:

1. **Config parser + `DequantLinear.forward`** (`compiler/trace/quantized.py`) — *how to unpack + dequant in torch ops*.
2. **The `045_dequant_linear` decomposition rule** — *how `DequantLinearOp` lowers to unpack → dequant → matmul*.

The bitwise/integer IR ops, `dtype_promote`, the dequant→matmul fusion (Phase 3), the in-smem/register dequant
(Phases 4–5), and tuning (Phase 6) never know it's AWQ. **Design choice that locks this in:** build `DequantLinear` /
`DequantLinearOp` around a small **`QuantScheme` descriptor** (`num_bits`, `pack_layout` enum, `group_size`,
`symmetric`, `scale_dtype`, `zp_packing`) and keep the dequant arithmetic *data-driven by that descriptor* — never bake
AWQ-specific constants into the IR ops or kernel codegen. The `i::8` interleave is already expressed as an IndexMap
coordinate function, so a different packing is just a different coordinate function feeding the same pipeline.

| Tier | Formats | What changes | Effort |
| --- | --- | --- | --- |
| **Easy** (parametric) | other bit widths (INT8/3/2); symmetric (no zp); per-tensor / per-channel / per-group scale | `num_bits`, drop the zp node, or the scale-broadcast IndexMap divisor (`k//G` / `k//in` / const) | trivial |
| **Easy–medium** | GPTQ / autoawq packing | new config parse + a new unpack-layout IndexMap; unpack ops + dequant chain + fusion + kernel dequant all reused | small–medium |
| **Moderate** (different dequant *shape*, same fusion) | bitsandbytes NF4/FP4 (non-uniform → 16-entry **codebook lookup**, needs a gather-from-constant-table op); FP8 weight-only e4m3 (*simpler* — cast + scale, no bit unpack) | a new dequant primitive, but still rides the dequant-producer → matmul fusion | medium |
| **Hard** (out of scope — separate project) | activation quantization: W8A8, FP8 W8A8 | runtime activation-quant kernel (per-token scale from input) + **integer/fp8 MMA tier** (int8→int32 dp4a/mma, or Hopper/Ada fp8 mma) + accumulator dequant — NOT the "dequant a constant operand" pattern | new kernel tiers |

This W4A16 plan keeps the matmul fp16-accumulate and only dequantizes the *weight* operand. Anything that quantizes
**activations** needs new accumulate tiers and is explicitly not covered here.

## End-to-end verification
1. `pytest tests/compiler/.../test_dequant*.py` — unit numerics over a synthetic in-test fixture, no GPU/network
   (the `compressed_tensors` parity assertion is an optional, package-gated test).
2. *(GPU + network gated, manual)* `deplodock run cyankiwi/Llama-3.1-8B-Instruct-AWQ-INT4 --layer 0` — accuracy vs
   eager passes (Phase 0+).
3. *(GPU + network gated, manual)* `deplodock run ... --layer 0 --bench` — kernel runs; peak VRAM shows only packed
   weights resident (Phase 3+).
4. *(GPU + network gated, manual)* `deplodock tune ... --layer 0` then `eval golden` — tuning reaches the dequant
   kernels.
5. `make test && make lint`.
