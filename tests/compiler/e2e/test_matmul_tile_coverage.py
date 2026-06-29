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
