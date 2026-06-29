"""Matmul register-tile coverage — one accuracy + structure matrix over (TILE variant × shape).

The scalar contraction's free-axis output tile (the ``TILE`` codec — ``n<N>[xm<M>]`` parallel
thread-tile / ``f<fn>[xf<fm>]`` register sub-tile) lowers each thread to a ``reg_m × reg_n``
block of output cells, the reduce-loop body replicated per cell with its operand loads deduped
(``A[m,k]`` reused across the ``n`` cells, ``B[k,n]`` across the ``m`` cells) — the
arithmetic-intensity lever for scalar SGEMM. This test pins each TILE variant and checks the
contraction stays accurate vs numpy AND emits the matching lowering structure (register
replication, the ``#pragma unroll``'d inner reduce, the per-CTA thread count), over BOTH a
static M and a SYMBOLIC M (the dynamic-grid tier: the launch sizes from the runtime extent, the
overhang cell clamp-reads + skips its store). All pins are the ``DEPLODOCK_TILE`` codec — the
register-tile analogue of ``test_reduction_combine_coverage``'s ``DEPLODOCK_REDUCE`` matrix; no
legacy ``BN`` / ``BM`` / ``FM`` / ``FN``.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import dyn_M, requires_cuda

# Square base shape; divisible by every variant's parallel·register product so the static
# column is exact-cover (one CTA where the variant asks for it) and the dynamic column runs at
# an off-divisor length to exercise the masked tail.
_M = _K = _N = 64
_DYN_M = 70  # off the 64 base → a partial last register-row when M is register-tiled


def _matmul_graph(mode: str):
    """``(1, M, K) @ (K, N)``; ``mode='dynamic'`` makes the M (row) axis symbolic."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

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
    from deplodock.compiler.backend.cuda.backend import CudaBackend

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
_SHAPES = ("static", "dynamic")


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
# pure activation, and a full ``(m, n)`` residual (no sharing). Pinned to a 2×2 register tile so
# the reg-tile tail-replication + load-dedup is exercised by every epilogue.
_EPILOGUE_TILE = "n16xm16/f2xf2"
_EPILOGUES = ("scale", "bias", "relu", "residual")


def _epilogue_graph(mode: str, epilogue: str):
    """``(1, M, K) @ (K, N)`` with a fused pointwise ``epilogue`` on the contraction output."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

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
    symbolic M. The epilogue is replicated per register cell in the tail (a per-``n`` bias shared
    across the ``m`` cells, a full residual not shared)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

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
