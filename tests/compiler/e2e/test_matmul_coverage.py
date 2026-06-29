"""Matmul (SEMIRING) coverage — the contraction across every tier, one file.

The scalar contraction's free-axis output tile (the ``TILE`` codec) and the tensor-core warp
fragment (the ``WARP`` codec) are the two materializers for the same SEMIRING ``project ∘ contract``
algebra; this file pins each and checks accuracy vs numpy/torch AND the matching lowering
structure, over static AND symbolic (dynamic-grid) shapes. Sections:

- **scalar TILE tier** — register-tile variants, fused epilogues, operand staging (the ``STAGE``
  codec), and the FN>1 blocked-GEMM / fused-prologue regression reproducers.
- **warp MMA tier** — ``mma.sync`` plain / transposed-B / epilogue coverage, static-vs-dynamic
  parity across cp.async + TMA transports, the staging invariants (bit-identical, bf16), and
  atomic-free split-K.
- **masked symbolic warp tier** — off-hint straddling sizes for symbolic M / N / K (the
  boundary-guard + clamp + zero-fill interplay the tile-divisor sweeps can't reach), the demoted
  B-cone, the batched / softmax-P@V split-consumers, cp.async AND TMA.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane. The CPU-render
structure tests (forced sm_120) need no GPU; warp-tier accuracy needs sm_90+.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import BF16, F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.knob import mma_atom

from ..conftest import dyn_M, requires_cuda, requires_sm90


def _has_cuda() -> bool:
    try:
        import cupy as cp  # noqa: PLC0415

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_tma() -> bool:
    """TMA (``cp.async.bulk.tensor``) needs sm_90+ (Hopper / Blackwell)."""
    if not _has_cuda():
        return False
    import cupy as cp  # noqa: PLC0415

    return int(cp.cuda.Device().compute_capability) >= 90


def _dtype(name: str):
    return {"f16": F16, "f32": F32}[name]


# =========================================================================== #
# Scalar TILE tier — register-tile variants, epilogues, staging, regressions.
# =========================================================================== #

# Square base shape; divisible by every variant's parallel·register product so the static
# column is exact-cover and the dynamic column runs at an off-divisor length (masked tail).
_M = _K = _N = 64
_DYN_M = 70  # off the 64 base → a partial last register-row when M is register-tiled
_SHAPES = ("static", "dynamic")


def _matmul_graph(mode: str):
    """``(1, M, K) @ (K, N)``; ``mode='dynamic'`` makes the M (row) axis symbolic."""
    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Mg, _N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _run(mode: str, tile: str, monkeypatch) -> tuple[np.ndarray, np.ndarray, str]:
    """Compile the matmul under the pinned ``DEPLODOCK_TILE`` codec, run on seeded inputs at the
    mode's runtime M, and return ``(output, reference, kernel_source)``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_TILE", tile)
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, m, _K), dtype=np.float32)
    b = rng.standard_normal((_K, _N), dtype=np.float32)
    be = CudaBackend()
    compiled = be.compile(_matmul_graph(mode))
    got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, (a @ b), src


# (label, TILE codec, expects-register-replication, expected __launch_bounds__ or None).
#   none        ("")            — one thread per cell, no register replication / unroll
#   reg_inner   (f4)            — 4 register cells along N, B-load shared across them
#   reg_2d      (f2xf2)         — full 2×2 register block, both operands reused
#   single_cta  (n32xm16/f2xf4) — par·reg == 64×64 ⇒ one 512-thread CTA (static)
_VARIANTS = {
    "none": ("", False, None),
    "reg_inner": ("f4", True, None),
    "reg_2d": ("f2xf2", True, None),
    "single_cta": ("n32xm16/f2xf4", True, 512),
}


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_tile_coverage(variant, mode, monkeypatch):
    tile, has_reg, launch_bounds = _VARIANTS[variant]
    got, ref, src = _run(mode, tile, monkeypatch)

    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{variant}/{mode}: matmul mismatch (max abs err {diff})"

    has_copy = "__c0_1" in src or "__c1_0" in src  # a replicated register-cell binding
    if has_reg:
        assert has_copy, f"{variant}/{mode}: expected replicated register cells (__c*)"
        assert "#pragma unroll" in src, f"{variant}/{mode}: the small inner reduce must be unrolled"
    else:
        assert not has_copy, f"{variant}/{mode}: per-cell tier must not replicate register cells"
    if launch_bounds is not None:
        assert f"__launch_bounds__({launch_bounds})" in src, f"{variant}/{mode}: expected a {launch_bounds}-thread CTA"

    if mode == "dynamic":
        # The dynamic-grid tier: the launch sizes from the runtime extent (the symbolic ``Dim``
        # threaded as an ``int`` arg), and a register-tiled symbolic axis guards its tail store.
        assert "int seq_len" in src, f"{variant}/dynamic: symbolic grid must carry the runtime extent arg"


# Fused epilogues — a projection ``Map`` over the ``Semiring`` (``project ∘ contract``): the
# pointwise op folds into the contraction kernel's tail, replicated per register cell. Each is a
# distinct tail shape: a broadcast scalar, a per-``n`` bias (shared across the ``m`` cells), a
# pure activation, and a full ``(m, n)`` residual (no sharing). Pinned to a 2×2 register tile.
_EPILOGUE_TILE = "n16xm16/f2xf2"
_EPILOGUES = ("scale", "bias", "relu", "residual")


def _epilogue_graph(mode: str, epilogue: str):
    """``(1, M, K) @ (K, N)`` with a fused pointwise ``epilogue`` on the contraction output."""
    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (1, Mg, _N)), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "scale":
        g.add_node(InputOp(), [], Tensor("s", (1,)), node_id="s")
        g.add_node(ElementwiseOp("multiply"), ["ab", "s"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("s")
    elif epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (_N,)), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("o", (1, Mg, _N)), node_id="o")
    else:  # residual — a full (1, M, N) add (depends on both cell axes, no load sharing)
        g.add_node(InputOp(), [], Tensor("r", (1, Mg, _N)), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["o"]
    return g


def _epilogue_ref(epilogue: str, feed: dict) -> np.ndarray:
    base = feed["a"] @ feed["b"]
    if epilogue == "scale":
        return base * feed["s"]
    if epilogue == "bias":
        return base + feed["bias"]
    if epilogue == "relu":
        return np.maximum(base, 0.0)
    return base + feed["r"]


@pytest.mark.parametrize("epilogue", _EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_reg_tile_epilogue(epilogue, mode, monkeypatch):
    """A register-tiled contraction with a fused pointwise epilogue stays accurate AND folds the
    epilogue into the ONE contraction kernel (no separate elementwise launch), over static and
    symbolic M."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_TILE", _EPILOGUE_TILE)
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    feed = {"a": rng.standard_normal((1, m, _K), dtype=np.float32), "b": rng.standard_normal((_K, _N), dtype=np.float32)}
    if epilogue == "scale":
        feed["s"] = np.array([1.5], dtype=np.float32)
    elif epilogue == "bias":
        feed["bias"] = rng.standard_normal((_N,), dtype=np.float32)
    elif epilogue == "residual":
        feed["r"] = rng.standard_normal((1, m, _N), dtype=np.float32)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(mode, epilogue))
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["o"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))

    ref = _epilogue_ref(epilogue, feed)
    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{epilogue}/{mode}: fused-epilogue mismatch (max abs err {diff})"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one contraction kernel"
    assert "__c0_1" in src or "__c1_0" in src, f"{epilogue}/{mode}: expected the register-tiled tail (__c*)"


# --- scalar operand staging (the orthogonal STAGE codec) --------------------
# The ``STAGE`` codec (``d<depth>/sync|cp|tma``) annotates the typed ``Stage`` schedule struct on
# a Semiring contraction; the materializer assembles the smem slab + cooperative producer from it.


def _scalar_stage_graph(M: int = 64, N: int = 64, K: int = 64) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def test_scalar_matmul_stages_through_pipeline(monkeypatch) -> None:
    """The ``TILE_PASSES`` chain stamps the ``STAGE`` codec onto the scalar matmul's typed
    ``Stage`` schedule struct (the orthogonal codec on the ``SemiringKernel`` arm). ``d1/sync`` is
    the single-buffer plain-``__syncthreads`` staging point."""
    from deplodock.compiler.ir.tile import TileOp  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_STAGE", "d1/sync")
    out = Pipeline.build(TILE_PASSES).run(_scalar_stage_graph(), ctx=Context.from_target((8, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.knobs.get("STAGE") == "d1/sync", tile_op.knobs.get("STAGE")
    stage = tile_op.kernel.schedule.stage
    assert stage is not None and stage.transport == "sync" and stage.depth == 1, stage


@requires_cuda
@pytest.mark.parametrize("stage_mask", ["11", "10", "01", "00"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (64, 47, 64), (128, 128, 128)])
def test_staged_scalar_matmul_matches_reference(monkeypatch, stage_mask, shape) -> None:
    """Every stage subset (both / A-only / B-only / none) lowers to a kernel that
    matches a numpy matmul, including a masked (non-divisor) output axis."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_STAGE", stage_mask)
    # Pin a small in-budget scalar tile so the deep-BK emission default can't overflow the staged
    # smem slab on the larger shape. Legacy env pins route through the ingest mapper.
    for k, v in (("BN", "16"), ("BM", "16"), ("FN", "2"), ("FM", "2"), ("BK", "16")):
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    M, N, K = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(_scalar_stage_graph(M, N, K)), input_data={"a": a, "b": b})[0].outputs
    got = list(out.values())[0].reshape(M, N)
    np.testing.assert_allclose(got, a @ b, atol=1e-3, rtol=1e-3)


# --- FN>1 blocked-GEMM / fused-prologue regressions -------------------------
# The per-cell + replicator + ``dedup_replicated`` pipeline reproduces register-blocked GEMM at
# the autotune knob bundles that used to fault: smem-vectorize misalignment hangs (FN=3), and the
# fused-prologue duplication that blew the nvcc compile budget (FN=32 / FN=64).


def _random(shape: tuple[int, ...], *, seed: int = 0, scale: float = 1.0, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(dtype)


def _numpy_reference(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from deplodock.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

    be = NumpyBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs


def _assert_close(out: np.ndarray, ref: np.ndarray, *, atol_rel: float = 0.05, atol_min: float = 1e-3) -> None:
    """Tolerance scales with the reference peak — matmul reductions over K elements drift by
    ~K·eps on f32, the blocked-vs-per-cell difference is well below that floor."""
    assert out.shape == ref.shape, f"shape mismatch {out.shape} vs {ref.shape}"
    assert np.all(np.isfinite(out)), "output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(atol_min, atol_rel * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=atol_rel)


def _pin_knobs(monkeypatch, **knobs) -> None:
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))


@requires_cuda
@pytest.mark.parametrize("dt", ["f32", "f16"])
def test_blocked_matmul_vectorize_misalign(monkeypatch, dt):
    """``BK=64 BM=32 BN=32 BR=1 FM=1 FN=3 SPLITK=1 STAGE=111`` on ``matmul(a@b)*scalar`` (N=96 =
    BN·FN) previously hung / faulted: ``050_vectorize_loads`` packed two cells into one
    ``float2`` / ``__half2`` reinterpret without seeing the stride-3 (FN=3) base address, so half
    the threads read off-alignment (byte 12 for float2, byte 6 for __half2). The fix walks back
    into the Source's innermost cache-axis extent and refuses to vectorize when it isn't a
    multiple of the pack count. Only the matmul-chained-into-mul gives the planner a STAGE=111
    enumeration with all three buffers stage-able."""
    from deplodock.compiler import dtype as _dt  # noqa: PLC0415

    _pin_knobs(monkeypatch, BK=64, BM=32, BN=32, BR=1, FM=1, FN=3, SPLITK=1, STAGE=111)
    npd = np.float16 if dt == "f16" else np.float32
    td = _dt.get("f16") if dt == "f16" else _dt.get("f32")
    scale = 0.1 if dt == "f16" else 1.0
    atol_min = 5e-3 if dt == "f16" else 1e-3

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (32, 64), td), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (64, 96), td), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("s", (1,), td), node_id="s")  # broadcast scalar
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("ab", (32, 96), td), node_id="ab")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["ab", "s"], output=Tensor("o", (32, 96), td), node_id="o")
    g.inputs, g.outputs = ["a", "b", "s"], ["o"]

    inputs = {
        "a": _random((32, 64), seed=1, scale=scale, dtype=npd),
        "b": _random((64, 96), seed=2, scale=scale, dtype=npd),
        "s": np.array([1.0 if dt == "f16" else 1.5], dtype=npd),
    }
    ref = _numpy_reference(g, inputs)["o"]
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    out = CudaBackend().run(CudaBackend().compile(g), input_data=inputs)[0].outputs["o"]
    _assert_close(out, ref, atol_min=atol_min)


# (label, knobs, N, line-budget, assert-single-x-smem). The fused RMSNorm+Linear prologue must
# fold to ONE body-level chain (+ short per-cell guarded multiplies) rather than duplicate per
# register cell — else the rendered kernel blows the ~2 s nvcc budget.
_FUSED_PROLOGUE = {
    "rmsnorm_linear_fn32": {
        "knobs": dict(BK=64, BM=1, BN=128, BR=1, FM=1, FN=32, SPLITK=1),
        "N": 4096,
        "lines": 360,
        "one_smem": True,
    },
    "qwen_lmhead_fn64": {
        "knobs": dict(BK=64, BM=1, BN=64, BR=1, FM=1, FN=64, SPLITK=1, STAGE=1),
        "N": 4099,
        "lines": 850,
        "one_smem": False,
    },
}


@requires_cuda
@pytest.mark.parametrize("case", list(_FUSED_PROLOGUE))
def test_fused_prologue_compiles_in_budget(monkeypatch, case):
    """A fused RMSNorm→Linear at lm_head-style knobs (FN=32 / FN=64) folds the N-invariant
    prologue chain (mean reduce + rsqrt + ``norm_weight·v``) back into one body-level copy with
    per-cell guarded multiplies, so the rendered kernel stays under the nvcc budget (line count
    below threshold) and matches the numpy reference. The duplicated-prologue regression inflates
    the body well past these thresholds (and, for the FN=32 case, opens a SECOND ``x_smem`` slab)."""
    import re as _re  # noqa: PLC0415

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    spec = _FUSED_PROLOGUE[case]
    _pin_knobs(monkeypatch, **spec["knobs"])
    M, K, N = 2, 1024, spec["N"]
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (K,)), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wl", (N, K)), node_id="wl")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "wn"], output=Tensor("xn", (M, K)), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wl"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wl"], ["o"]

    inputs = {
        "x": _random((M, K), seed=5),
        "wn": _random((K,), seed=6, scale=0.1),
        "wl": _random((N, K), seed=7, scale=0.02),  # scaled so output stays bounded
    }
    # Reference BEFORE backend.compile (which mutates ops in place).
    ref = _numpy_reference(g, inputs)["o"]

    backend = CudaBackend()
    compiled = backend.compile(g)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops, "expected a CudaOp in the lowered graph"
    cuda_src = "\n".join(op.kernel_source for op in cuda_ops)
    n_lines = cuda_src.count("\n")
    assert n_lines < spec["lines"], (
        f"{case}: rendered kernel is {n_lines} lines (budget {spec['lines']}) — a regression that "
        f"fails to dedup the N-invariant prologue chain inflates it."
    )
    if spec["one_smem"]:
        # ONE smem allocation for the RMSNorm input shared by the mean reduce + matmul body; a
        # regression opening its own RegisterTile staging context forces a second ``x_smem``.
        x_smem_decls = len(_re.findall(r"__shared__\s+(?:__align__\([0-9]+\)\s+)?float\s+x_smem\b", cuda_src))
        assert x_smem_decls == 1, f"{case}: expected 1 ``__shared__ float x_smem`` decl (per-cell shares staging); got {x_smem_decls}"

    out = backend.run(compiled, input_data=inputs)[0].outputs["o"]
    _assert_close(out, ref, atol_min=1e-3)


# =========================================================================== #
# Warp MMA tier (tensor-core mma.sync) — the WARP codec materializer.
# =========================================================================== #
# The contraction tiles onto ``WM·WN`` warps of ``mma_m16n8k16`` atom cells: f16/bf16 operands,
# f32 accumulate, f16|f32 store. Requires sm_90+ (the warp tier is pin-only / non-functional
# below: ldmatrix host fault + ``sm_NNa`` TMA compile).
_WARP_PIN = "a:mma_m16n8k16_f16/w2xw2/f4xf8/k2"  # WM·FM·atom_m = WN·FN·atom_n = 128 tile, 128 threads
_F16 = "f16"


def _mma_matmul_graph(mode: str, M: int, N: int, K: int, out: str, trans: bool):
    """A hand-built ``C[i,j] = Σ_k A[i,k]·B[…]`` over f16 operands; ``trans`` makes B ``[j,k]``
    (Q@Kᵀ, K last). ``mode='dynamic'`` makes the M (row) axis symbolic (the dynamic-grid tier)."""
    from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415
    from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
    from deplodock.compiler.ir.stmt import Accum, Assign  # noqa: PLC0415

    Mg = dyn_M(mode, M)
    i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
    b_index = (Var("j"), Var("k")) if trans else (Var("k"), Var("j"))
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=b_index),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (N, K) if trans else (K, N), dtype=F16), node_id="b")
    g.add_node(op, ["a", "b"], Tensor("c", (Mg, N), dtype=_dtype(out)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _compile_run_mma(graph, feed: dict) -> tuple[np.ndarray, str]:
    """Compile the (already WARP-pinned) graph and run it on the seeded f16 operands in ``feed``;
    return ``(output, kernel_source)``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, src


# (M, N, K, out_dtype, transposed-B). 128 / 256 are tile (128) multiples (exact-cover static);
# the dynamic column runs at M+2 to straddle the tile and exercise the masked store / clamp.
_MMA_CASES = [
    (128, 128, 128, "f32", False),
    (256, 256, 128, "f32", False),
    (128, 128, 128, "f16", False),
    (128, 256, 128, "f16", False),
    (128, 128, 128, "f32", True),  # transposed-B (Q@Kᵀ)
    (128, 128, 128, "f16", True),
]


@pytest.mark.parametrize(("M", "N", "K", "out", "trans"), _MMA_CASES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_coverage(M, N, K, out, trans, mode, monkeypatch):
    """An f16×f16 matmul under the pinned ``DEPLODOCK_WARP`` codec lowers to ``mma.sync`` and
    agrees with the f32 reference (within fp16 tolerance) — for canonical AND transposed-B
    operands, f16 AND f32 output, over a STATIC M (tile divisor) and a SYMBOLIC M run at a
    straddling length (the dynamic-grid tier + the masked-tile store)."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_PIN)
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((N, K) if trans else (K, N)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, out, trans), {"a": a, "b": b})

    ref = a.astype(np.float32) @ (b.T if trans else b).astype(np.float32)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    tol = 5e-2 if out == _F16 else 1e-2
    assert diff < tol, f"{M}x{N}x{K} out={out} trans={trans}/{mode}: mma mismatch (max abs err {diff})"

    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "dpl_mma_load_a_gmem" in src, "operands must load via the gmem-direct fragment helper"
    assert "wmma::" not in src, "the mma.sync path must not mix in legacy wmma intrinsics"
    assert ("__floats2half2_rn" in src) == (out == _F16), "f16 output needs the fp32→fp16 __half2 downconvert"
    if trans:
        assert "dpl_mma_load_b_gmem_trans" in src, "transposed-B must use the gmem-direct trans helper"
    if mode == "dynamic":
        assert "int seq_len" in src, "the symbolic-M grid must carry the runtime extent arg"


# Fused epilogues over the warp tier — a projection ``Map`` (or a causal ``Select``) folds into
# the ``RegStore`` per fragment element.
_MMA_EPILOGUES = ("bias", "relu", "residual", "causal")


def _mma_epilogue_graph(mode: str, epilogue: str):
    """An f16 ``128³`` matmul with a fused ``epilogue`` (causal uses a hand-built coord ``Select``;
    the rest fold a frontend ``ElementwiseOp``)."""
    M = N = K = 128
    Mg = dyn_M(mode, M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype=F16), node_id="b")
    if epilogue == "causal":
        from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
        from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write  # noqa: PLC0415
        from deplodock.compiler.ir.stmt import Accum, Assign, Select, SelectBranch  # noqa: PLC0415

        i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
        op = LoopOp(
            body=(
                Loop(
                    axis=i,
                    body=(
                        Loop(
                            axis=j,
                            body=(
                                Load(name="mz", input="mz", index=(Literal(0, "int"),)),
                                Load(name="mf", input="mf", index=(Literal(0, "int"),)),
                                Loop(
                                    axis=k,
                                    body=(
                                        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                        Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                        Accum(name="acc", value="p"),
                                    ),
                                ),
                                Select(
                                    name="out",
                                    branches=(
                                        SelectBranch(value="acc", select=BinaryExpr("<=", Var("j"), Var("i"))),
                                        SelectBranch(value="mf", select=BinaryExpr(">", Var("j"), Var("i"))),
                                    ),
                                ),
                                Write(output="c", index=(Var("i"), Var("j")), value="out"),
                            ),
                        ),
                    ),
                ),
            ),
        )
        g.add_node(InputOp(), [], Tensor("mz", (1,), dtype=F32), node_id="mz")
        g.add_node(InputOp(), [], Tensor("mf", (1,), dtype=F32), node_id="mf")
        g.add_node(op, ["a", "b", "mz", "mf"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        g.inputs, g.outputs = ["a", "b", "mz", "mf"], ["c"]
        return g

    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (Mg, N), dtype=F32), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (N,), dtype=F32), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
    else:  # residual
        g.add_node(InputOp(), [], Tensor("r", (Mg, N), dtype=F32), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["c"]
    return g


@pytest.mark.parametrize("epilogue", _MMA_EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_epilogue_coverage(epilogue, mode, monkeypatch):
    """A warp-tier matmul with a fused pointwise / causal epilogue stays accurate AND folds the
    epilogue into the ONE mma.sync kernel (the per-element ``RegStore`` chain), over static and
    symbolic M."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_PIN)
    M = N = K = 128
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    base = a.astype(np.float32) @ b.astype(np.float32)
    feed = {"a": a, "b": b}
    if epilogue == "bias":
        bias = rng.standard_normal((N,)).astype(np.float32)
        feed["bias"] = bias
        ref = base + bias
    elif epilogue == "relu":
        ref = np.maximum(base, 0.0)
    elif epilogue == "residual":
        r = rng.standard_normal((run_m, N)).astype(np.float32)
        feed["r"] = r
        ref = base + r
    else:  # causal
        feed["mz"] = np.array([0.0], np.float32)
        feed["mf"] = np.array([-1e30], np.float32)
        keep = np.arange(N)[None, :] <= np.arange(run_m)[:, None]
        ref = np.where(keep, base, -1e30)

    got, src = _compile_run_mma(_mma_epilogue_graph(mode, epilogue), feed)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    assert diff < 1e-2, f"{epilogue}/{mode}: fused-epilogue mma mismatch (max abs err {diff})"
    assert "mma.sync.aligned.m16n8k16" in src, f"{epilogue}/{mode}: must reach the warp tier"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one mma kernel"
    if epilogue == "causal":
        assert "?" in src, "the causal mask must render as a per-element ternary"


# --- static-vs-dynamic parity across cp.async + TMA transports --------------
# One matmul op compiled BOTH shape-specialised (static M) and dynamic (Dim('seq_len')), run at
# the SAME runtime M, across cp.async AND TMA (pinned). K static so the source innermost dim stays
# static — TMA-eligible. The ``shape_mode`` × ``transport`` fixtures fan one body over the matrix.
_PN, _PK = 1024, 512
_WARP_CODEC = "a:mma_m16n8k16_f16/w2xw2/f2xf2/k2"  # WM=WN=FM=FN=2, BK=2 — the 64-row tile


def _parity_mma_graph(mode: str, *, M: int):
    """``a @ b`` with the M axis static (``mode='static'``) or symbolic (``Dim('seq_len')``)."""
    m_dim = dyn_M(mode, M)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m_dim, _PK), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_PK, _PN), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m_dim, _PN), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@pytest.fixture(params=["cp.async", "tma"])
def transport(request, monkeypatch) -> str:
    """Pin the warp tile (``WARP`` codec) + the operand-staging transport (``STAGE`` codec —
    ``d2/cp`` = cp.async, ``d2/tma`` = cp.async.bulk.tensor). The "pinned knobs" fixture."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
    monkeypatch.setenv("DEPLODOCK_STAGE", "d2/tma" if request.param == "tma" else "d2/cp")
    return request.param


def test_pinned_transport_and_shape_fire(shape_mode, transport):
    """CPU render (forced sm_120): the pinned knobs select the intended transport and the symbolic
    M threads a runtime ``seq_len`` arg — so the accuracy test below exercises the path it claims."""
    lowered = Pipeline.build(CUDA_PASSES).run(_parity_mma_graph(shape_mode, M=512), ctx=Context(compute_capability=(12, 0)))
    src = lowered.nodes["o"].op.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src, "must be on the s16816 tensor-core tier"
    if transport == "tma":
        assert "cp.async.bulk.tensor" in src, f"{shape_mode}/tma: TMA must fire"
        assert "CUtensorMap" in src, "TMA kernel must take the descriptor param"
    else:
        assert "cp.async.bulk.tensor" not in src, f"{shape_mode}/cp.async: TMA must NOT fire"
        assert "cp.async" in src, f"{shape_mode}/cp.async: operands must stage via cp.async"
        assert "__shared__" in src, f"{shape_mode}/cp.async: staged operands need an smem slab"
    if shape_mode == "dynamic":
        assert "int seq_len" in src, "dynamic kernel must carry the runtime extent arg"
    else:
        assert "int seq_len" not in src, "static kernel bakes M — no runtime extent arg"


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("M", [256, 512])
def test_static_dynamic_mma_parity(shape_mode, transport, M):
    """The SAME matmul, compiled static and dynamic, on cp.async and TMA, is accurate vs torch —
    so the four paths agree. ``M`` is a multiple of the WM·FM·16 = 64-row tile, where ALL four
    combos fire. Dynamic robustness at off-hint sizes is covered by the masked sweep below."""
    if transport == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_parity_mma_graph(shape_mode, M=M))
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert got.shape == (M, _PN)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"{shape_mode}/{transport} M={M}: max abs err {diff}"


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("M", [128, 256])
def test_staged_matches_gmem_direct_bit_for_bit(monkeypatch, M):
    """cp.async operand staging is a PURE perf transform: the staged kernel (``STAGE=d2/cp``) must
    produce **bit-identical** output to the gmem-direct baseline (same ``WARP`` tile, no ``STAGE``)
    on the same inputs — and actually stage (cp.async + smem slab) where the baseline does not."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
        if stage:
            monkeypatch.setenv("DEPLODOCK_STAGE", stage)
        else:
            monkeypatch.delenv("DEPLODOCK_STAGE", raising=False)
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    staged, staged_src = _go("d2/cp")
    gmem, gmem_src = _go(None)
    assert "cp.async" in staged_src and "__shared__" in staged_src, "STAGE=d2/cp must stage via a cp.async smem slab"
    assert "cp.async" not in gmem_src, "the gmem-direct baseline must not stage"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert np.abs(staged.astype(np.float32).reshape(M, _PN) - want).max() < 5e-2


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("tr", ["cp", "tma"])
@pytest.mark.parametrize("M", [128, 256])
def test_register_double_buffer_matches_single_buffer_bit_for_bit(monkeypatch, tr, M):
    """The smem→register double-buffer (``STAGE=d2/<tr>/p2``) is a PURE perf transform over the
    single-buffer staged kernel (``d2/<tr>``): same loads, same mmas, only ldmatrix-prefetched
    onto alternate fragment slots — so the output is **bit-identical**, and the source actually
    ping-pongs (``_a0_s0``/``_a0_s1`` fragments the single-buffer kernel does not emit)."""
    if tr == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
        monkeypatch.setenv("DEPLODOCK_STAGE", stage)
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    single, single_src = _go(f"d2/{tr}")
    double, double_src = _go(f"d2/{tr}/p2")
    assert "_a0_s0" in double_src and "_a0_s1" in double_src, "p2 must declare per-slot ping-pong fragments"
    assert "_s0" not in single_src, "the single-buffer kernel must not slot its fragments"
    np.testing.assert_array_equal(double, single)  # bit-identical: prefetch reordering perturbs nothing


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("M", [128, 256])
def test_cp_async_deep_ring_matches_gmem_direct_bit_for_bit(monkeypatch, depth, M):
    """The gmem→smem ring (``STAGE=d<depth>/cp``, depth≥2) prefetches ``depth-1`` K-chunks ahead so
    the cp.async copy overlaps the mma. It is a PURE perf transform: bit-identical to the gmem-direct
    baseline, and the kernel allocates ``depth`` ring slots (``depth`` cp.async ``commit_group``\\ s —
    the prologue primes ``depth-1``, the steady loop commits one prefetch per chunk)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(1)
    a = (rng.standard_normal((M, _PK)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_PK, _PN)) * 0.1).astype(np.float16)

    def _go(stage: str | None) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
        if stage:
            monkeypatch.setenv("DEPLODOCK_STAGE", stage)
        else:
            monkeypatch.delenv("DEPLODOCK_STAGE", raising=False)
        be = CudaBackend()
        compiled = be.compile(_parity_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    ring, ring_src = _go(f"d{depth}/cp")
    gmem, _ = _go(None)
    np.testing.assert_array_equal(ring, gmem)  # bit-identical: prefetch perturbs nothing
    assert ring_src.count("commit_group") == depth, f"a depth-{depth} ring must issue {depth} cp.async commit groups"


@requires_sm90
@requires_cuda
def test_bf16_operands_stage_via_cp_async(monkeypatch):
    """The bf16 MMA atom (``mma_m16n8k16_bf16``) stages through cp.async and stays accurate vs
    torch — the cp.async byte-width fill must handle the 2-byte bf16 operand. (No native numpy
    bf16: feed the bits as uint16 and reinterpret the uint16 output.)"""
    import torch  # noqa: PLC0415

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_WARP", "a:mma_m16n8k16_bf16/w2xw2/f2xf2/k2")
    monkeypatch.setenv("DEPLODOCK_STAGE", "d2/cp")
    M = 256
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, _PK), dtype=BF16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_PK, _PN), dtype=BF16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, _PN), dtype=BF16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    be = CudaBackend()
    compiled = be.compile(g)
    src = compiled.nodes["o"].op.kernel_source
    assert "cp.async" in src and "mma.sync.aligned.m16n8k16" in src, "bf16 operands must stage on the mma tier"
    assert "cp.async.bulk.tensor" not in src, "cp.async transport must not emit TMA"
    torch.manual_seed(0)
    qa = (torch.randn(M, _PK) * 0.1).to(torch.bfloat16)
    qb = (torch.randn(_PK, _PN) * 0.1).to(torch.bfloat16)
    data = {"a": qa.view(torch.uint16).numpy(), "b": qb.view(torch.uint16).numpy()}
    got_bits = np.asarray(be.run(compiled, input_data=data)[0].outputs["o"]).astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy().reshape(M, _PN)
    want = (qa.float() @ qb.float()).numpy()
    diff = float(np.abs(got - want).max())
    assert diff < 1e-1, f"bf16 staged mma mismatch (max abs err {diff})"


# --- atomic-free split-K on the warp tier -----------------------------------
# Dropping the ``is_warp`` early-out lets an MMA split-K route its C-fragment store into a
# ``workspace[K_s, M, N]`` and reuse the additive ``Accum``-sum reduce kernel instead of a codegen
# ``atomicAdd``. The finalize is the native ``REDUCE`` codec ``c``-letter (``c2k`` deferred / ``c2a``
# atomic), pinned through the native move-knob set (``ATOM`` / ``SPLIT`` / ``REDUCE``).
_WARP_NATIVE = {"DEPLODOCK_ATOM": "mma_m16n8k16_f16", "DEPLODOCK_SPLIT": "2x2"}


def _splitk_mma_graph(m: int, k: int, n: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("finalize", ["deferred", "atomic"])
def test_mma_splitk_finalize(monkeypatch, finalize):
    """fp16 MMA split-K is accurate vs numpy under both finalize folds: the deferred combine
    kernel (``s2/c2k``) emits NO ``atomicAdd`` (the atomic-free path), and the in-place atomic
    finalize (``s2/c2a``) stays selectable and accurate."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    for key, val in _WARP_NATIVE.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv("DEPLODOCK_REDUCE", "s2/c2k" if finalize == "deferred" else "s2/c2a")
    m, k, n = 128, 512, 128
    rng = np.random.default_rng(4 if finalize == "deferred" else 5)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
    be = CudaBackend()
    compiled = be.compile(_splitk_mma_graph(m, k, n))
    src = "\n".join(node.op.kernel_source for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None))
    if finalize == "deferred":
        # The deferred path retargets the C-fragment store into a workspace and reuses the additive
        # reduce kernel on the tensor-core tier — no codegen ``atomicAdd``.
        assert "mma.sync.aligned.m16n8k16" in src, "must be on the tensor-core tier"
        assert "atomicAdd" not in src, "atomic-free split-K must not emit atomicAdd"
    out = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"]).reshape(m, n)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


# =========================================================================== #
# Compile-time schedule guards — pins that would silently lower to a wrong / un-launchable
# kernel. Run the TILE pass only (no GPU): the schedule rejects the pin with a clear
# ``ValueError`` instead of corrupting numerics (warp static-K tail) or failing the launch
# (oversized TILE parallel block).
# =========================================================================== #


def _guard_mm_graph(M, N, K, *, dtype=F16) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=dtype), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=dtype), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _run_tile_pass(graph: Graph):
    return Pipeline.build(TILE_PASSES).run(graph, ctx=Context.from_target((12, 0)))


def test_warp_static_k_indivisible_rejected(monkeypatch) -> None:
    """A WARP pin whose static K is not a multiple of ``atom_k·bk`` is rejected — the warp
    K-loop has no static-K tail masking, so lowering it would silently corrupt the result (the
    error the accuracy gate's mean-error escape clause misses)."""
    monkeypatch.setenv("DEPLODOCK_WARP", "a:mma_m16n8k16_f16/w1xw1/f1xf1/k1")  # K-step 16
    with pytest.raises(ValueError, match="does not divide the static contraction K=100"):
        _run_tile_pass(_guard_mm_graph(128, 128, 100))


def test_warp_static_k_divisible_ok(monkeypatch) -> None:
    """The same pin on a K that IS a multiple of the K-step lowers without the guard firing."""
    monkeypatch.setenv("DEPLODOCK_WARP", "a:mma_m16n8k16_f16/w1xw1/f1xf1/k1")
    _run_tile_pass(_guard_mm_graph(128, 128, 128))  # 128 % 16 == 0 — no raise


def test_warp_symbolic_k_not_guarded(monkeypatch) -> None:
    """A symbolic K reaches the masked tier (ceil-div grid + zero-filled partial slab), so the
    static-K guard does not fire even when the hint is not a K-step multiple."""
    monkeypatch.setenv("DEPLODOCK_WARP", "a:mma_m16n8k16_f16/w1xw1/f1xf1/k2")  # K-step 32
    _run_tile_pass(_guard_mm_graph(64, 128, Dim("seq_len")))  # symbolic K — no raise


def test_tile_block_over_thread_limit_rejected(monkeypatch) -> None:
    """A TILE parallel tile over the 1024-thread/CTA limit is rejected at compile time instead
    of failing the launch with an opaque ``CUDA_ERROR_INVALID_VALUE``."""
    monkeypatch.setenv("DEPLODOCK_TILE", "n128xm128")  # 16384 threads
    with pytest.raises(ValueError, match="exceeds the 1024-thread/CTA limit"):
        _run_tile_pass(_guard_mm_graph(256, 256, 256, dtype=F32))


def test_tile_block_within_limit_ok(monkeypatch) -> None:
    """A TILE parallel tile within the thread limit lowers without the guard firing."""
    monkeypatch.setenv("DEPLODOCK_TILE", "n8xm8")  # 64 threads
    _run_tile_pass(_guard_mm_graph(256, 256, 256, dtype=F32))


# =========================================================================== #
# Masked symbolic warp tier — off-hint straddling sizes, every symbolic axis.
# =========================================================================== #
# A matmul whose M (and/or N, K) axis is symbolic reaches the mma.sync warp tier as a MASKED tile:
# the planner ceil-divs the grid, hoists the K-pipeline above a boundary ``Cond`` (clamped slab
# fill), stamps per-element row/col guards onto the ``RegStore``, and zero-fills the partial final
# K slab past a symbolic reduce extent. One cached kernel serves every runtime size. The point is
# off-hint / straddling sizes (1, 31, 130, 700 — NOT tile-divisor multiples), which exercise the
# boundary-guard + clamp + zero-fill interplay the tile-divisor parity sweep cannot reach.
_MASK_WARP = "a:mma_m16n8k16_f16/w2xw2/f2xf2/k2"
_CP_KNOBS = {"WARP": _MASK_WARP, "STAGE": "d2/cp"}
_TMA_KNOBS = {"WARP": _MASK_WARP, "STAGE": "d2/tma"}


def _symbolic_m_graph(*, K: int = 512, N: int = 1024) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _symbolic_k_graph(*, M: int = 64, N: int = 128) -> Graph:
    """A @ B with the REDUCE axis symbolic — the SDPA P@V shape after the demoted-matmul split."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, Dim("seq_len")), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (Dim("seq_len"), N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _batched_symbolic_mk_graph(*, H: int = 16, N: int = 128) -> Graph:
    """The SDPA P@V split-consumer: a BATCHED matmul (``H`` heads) whose M (query) AND K (key)
    axes are both symbolic ``seq_len`` — ``xna[H, seq, seq] @ xnb[H, seq, N]``."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xna", (H, Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="xna")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (H, Dim("seq_len"), N), dtype=F16), node_id="xnb")
    g.add_node(op=MatmulOp(), inputs=["xna", "xnb"], output=Tensor("o", (H, Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["xna", "xnb"], ["o"]
    return g


def _demoted_symbolic_n_graph(K: int = 128) -> Graph:
    """Computed-B-cone matmul (the rotary QK^T shape): an elementwise scale on BOTH operands feeds
    a transposed-``[N, K]`` Linear, so fusion demotes the matmul and ``010_split_demoted``
    materializes the canonical ``xnb[K, N]`` producer. M = N = ``Dim('seq_len')``."""
    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (s, K), dtype=F16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sx", (s, K), dtype=F16), node_id="sx")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (s, K), dtype=F16), node_id="w")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sw", (s, K), dtype=F16), node_id="sw")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["x", "sx"], output=Tensor("xs", (s, K), dtype=F16), node_id="xs")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["w", "sw"], output=Tensor("ws", (s, K), dtype=F16), node_id="ws")
    g.add_node(op=LinearOp(), inputs=["xs", "ws"], output=Tensor("o", (s, s), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "sx", "w", "sw"], ["o"]
    return g


def _pv_softmax_graph(H: int = 16, N: int = 128) -> Graph:
    """Softmax(scores) @ V with the reduce K = ``seq_len`` symbolic (the SDPA P@V shape). Fusion
    demotes the matmul; ``010_split_demoted`` materializes the softmax-prob A cone + symbolic-K gemm."""
    from deplodock.compiler.ir.frontend.ir import SoftmaxOp  # noqa: PLC0415

    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("scores", (H, s, s), dtype=F16), node_id="scores")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("v", (H, s, N), dtype=F16), node_id="v")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["scores"], output=Tensor("probs", (H, s, s), dtype=F16), node_id="probs")
    g.add_node(op=MatmulOp(), inputs=["probs", "v"], output=Tensor("o", (H, s, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


@pytest.mark.parametrize("transport", ["cp", "tma"])
def test_masked_symbolic_m_structure(transport, monkeypatch):
    """End-to-end render (CPU, forced sm_120): the symbolic-M masked warp kernel carries the
    runtime ``seq_len`` arg + the mma.sync pipeline. cp.async stages a clamped A-slab fill with
    per-element row guards on the fragment store; TMA takes a ``CUtensorMap`` descriptor param and
    stages the A operand with ``cp.async.bulk.tensor`` (TMA zero-fills the masked overhang)."""
    knobs = _CP_KNOBS if transport == "cp" else _TMA_KNOBS
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16"
    src = kop.kernel_source
    assert "int seq_len" in src, "runtime extent must be a kernel arg"
    assert "mma.sync.aligned.m16n8k16" in src
    if transport == "cp":
        assert kop.knobs.get("S_ext_n_symbolic_axis"), "symbolic-M warp row must carry a symbolic axis"
        assert "ldmatrix" in src
        # Clamp on the hoisted cooperative A fill: bound by the runtime extent, fall back to last row.
        assert "< seq_len) ?" in src and "seq_len - 1" in src, "A-slab fill must clamp to the runtime extent"
        # Per-element row guards from the RegStore (both fragment row blocks).
        assert "+ _g < (seq_len)" in src and "+ _g + 8 < (seq_len)" in src, "fragment store must row-guard against seq_len"
    else:
        stage = kop.knobs.get("STAGE", "")
        assert stage.endswith("/tma"), f"symbolic-M with static innermost dim must stage via TMA: {stage!r}"
        assert "cp.async.bulk.tensor" in src, "A operand must stage via TMA"
        assert "CUtensorMap" in src, "kernel must take the TMA descriptor param"


def test_batched_symbolic_mk_reaches_warp(monkeypatch):
    """The batched masked-M + masked-K P@V consumer must reach the mma.sync tier (the
    ``classify_matmul_operands`` batch-aware B test), not stay a LoopOp."""
    for k, v in _CP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_batched_symbolic_mk_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16", "batched symbolic M+K matmul must reach the warp tier"
    src = kop.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src
    assert "int seq_len" in src, "runtime extent must be a kernel arg"


# (label, env, seqs, make). ``make(seq)`` builds (graph, feed, want) for one off-hint runtime
# size; the driver compiles once per case and runs at each straddling size. ``env`` is the full
# ``DEPLODOCK_*`` pin set (some cases route to the scalar tier with no WARP pin).


def _make_symbolic_m(seq):
    g = _symbolic_m_graph()
    rng = np.random.default_rng(0)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    return g, {"a": a, "b": b}, a.astype(np.float32) @ b.astype(np.float32)


def _make_symbolic_mn(seq):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("q", (Dim("seq_len"), 128), dtype=F16), node_id="q")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("kT", (128, Dim("seq_len")), dtype=F16), node_id="kT")
    g.add_node(op=MatmulOp(), inputs=["q", "kT"], output=Tensor("o", (Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["q", "kT"], ["o"]
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    kt = (rng.standard_normal((128, seq)) * 0.1).astype(np.float16)
    return g, {"q": q, "kT": kt}, q.astype(np.float32) @ kt.astype(np.float32)


def _make_symbolic_m_residual(seq):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), 512), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (512, 1024), dtype=F16), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("r", (Dim("seq_len"), 1024), dtype=F16), node_id="r")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("mm", (Dim("seq_len"), 1024), dtype=F16), node_id="mm")
    g.add_node(op=ElementwiseOp("add"), inputs=["mm", "r"], output=Tensor("o", (Dim("seq_len"), 1024), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b", "r"], ["o"]
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    r = (rng.standard_normal((seq, 1024)) * 0.1).astype(np.float16)
    want = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32)
    return g, {"a": a, "b": b, "r": r}, want


def _make_symbolic_k(seq):
    g = _symbolic_k_graph()
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, seq)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    return g, {"a": a, "b": b}, a.astype(np.float32) @ b.astype(np.float32)


def _make_demoted_n(seq):
    g = _demoted_symbolic_n_graph()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sx = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sw = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    want = (x * sx).astype(np.float32) @ (w * sw).astype(np.float32).T
    return g, {"x": x, "sx": sx, "w": w, "sw": sw}, want


def _make_batched_mk(seq):
    g = _batched_symbolic_mk_graph()
    rng = np.random.default_rng(0)
    xna = (rng.standard_normal((16, seq, seq)) * 0.1).astype(np.float16)
    xnb = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    return g, {"xna": xna, "xnb": xnb}, np.matmul(xna.astype(np.float32), xnb.astype(np.float32))


def _make_pv_softmax(seq):
    g = _pv_softmax_graph()
    rng = np.random.default_rng(seq)
    scores = (rng.standard_normal((16, seq, seq)) * 2).astype(np.float16)
    v = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    sc = scores.astype(np.float32)
    e = np.exp(sc - sc.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    want = np.matmul(probs, v.astype(np.float32))
    return g, {"scores": scores, "v": v}, want


# Each case: (env, delenv, seqs, make). The demoted / batched cases route to the scalar tier or
# greedy lower (no WARP pin); the rest pin the cp.async (or TMA) warp staging.
_MASKED_CASES = {
    "symbolic_m_cp": (_CP_KNOBS, (), [1, 31, 512, 700], _make_symbolic_m),
    "symbolic_m_tma": (_TMA_KNOBS, (), [1, 31, 512, 700], _make_symbolic_m),
    "symbolic_mn_cp": (_CP_KNOBS, (), [31, 512, 700], _make_symbolic_mn),
    "residual_cp": (_CP_KNOBS, (), [100], _make_symbolic_m_residual),
    "symbolic_k_cp": (_CP_KNOBS, (), [16, 31, 130, 512, 700], _make_symbolic_k),
    # Routed through the SCALAR tier (no WARP pin): the batched-warp masked-M+K fragment codegen
    # at runtime is a separate gap, so accuracy rides the scalar tier (the structure render
    # reaches the warp tier — see ``test_batched_symbolic_mk_reaches_warp``).
    "batched_mk": ({}, (), [16, 31, 130, 512, 700], _make_batched_mk),
    # The demoted B-cone / softmax-P@V splits run under GREEDY (the multi-kernel producer rejects
    # a global warp pin under ``validate_pins``); ``SPLIT_CONE`` forces the demotion split.
    "demoted_n": ({"SPLIT_CONE": "1"}, (), [31, 130, 512, 700], _make_demoted_n),
    "demoted_pv_tma": ({**_CP_KNOBS, "SPLIT_CONE": "1"}, ("TMA",), [16, 31, 130, 512, 700], _make_pv_softmax),
}
_MASKED_PARAMS = [(label, seq) for label, (_e, _d, seqs, _m) in _MASKED_CASES.items() for seq in seqs]


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("label,seq", _MASKED_PARAMS)
def test_masked_symbolic_accuracy(label, seq, monkeypatch):
    """One compiled symbolic kernel is accurate at runtime sizes below / at / above the 512 hint —
    including the straddling cases (1, 31, 130, 700 are not tile-divisor multiples), which exercise
    the masked-M row guard, the masked-N column store, the zero-filled partial-K slab, the demoted
    B-cone overhang, and the TMA-staged P@V — each fed as a synthetic standalone graph."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    env, delenv, _seqs, make = _MASKED_CASES[label]
    for k, v in env.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    for k in delenv:
        monkeypatch.delenv(f"DEPLODOCK_{k}", raising=False)

    graph, feed, want = make(seq)
    be = CudaBackend()
    compiled = be.compile(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]]).astype(np.float32)
    assert got.shape == want.shape, f"{label}/seq={seq}: shape {got.shape} vs {want.shape}"
    diff = float(np.abs(got - want).max())
    assert diff < 5e-2, f"{label}/seq={seq}: masked symbolic mma mismatch (max abs err {diff})"
