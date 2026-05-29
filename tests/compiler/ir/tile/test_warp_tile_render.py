"""Render + launch-bounds tests for the WarpTile primitive (M2).

Hand-builds a minimal ``GridTile > WarpTile > Write`` KernelOp and runs
it through ``render_kernelop`` — verifies the CUDA source carries
``__launch_bounds__(N * 32)``, the ``warp_id`` row-major decode over
the warp axes, and the unconditional ``int lane = threadIdx.x & 31;``.

No upstream pass emits ``WarpTile`` yet (consumer plans land separately);
this exercises the primitive's render + launch-bounds wiring against a
synthetic kernel.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import GridTile, WarpTile

# A scalar literal-constant input ``one`` resolves the Load to a 1.0f
# literal at render time (``Load.is_literal`` path); the Write then
# stores it into ``C[m_b, m_w]``. Keeps the kernel signature minimal —
# one output, no real inputs.
_LITERALS = {"one_const": 1.0}


def _warp_kernel(m_b: int = 4, m_w: int = 2) -> KernelOp:
    """``GridTile(M_b) > WarpTile(M_w) > Load+Write 1.0f``.

    The Write index is intentionally trivial (one element per warp) so
    the kernel renders without per-thread cooperation scaffolding —
    M2's surface is the warp-id/lane decode + launch bounds, not the
    cooperative-load / Accum-combine path.
    """
    M_b = Axis("m_b", m_b)
    M_w = Axis("m_w", m_w)
    load = Load(name="one", input="one_const", index=())
    write = Write(output="C", index=(Var("m_b"), Var("m_w")), value="one")
    warp = WarpTile(axes=(M_w,), body=Body((load, write)))
    grid = GridTile(axes=(M_b,), body=Body((warp,)))
    return KernelOp(body=(grid,), name="k_warp_smoke")


def test_warp_kernel_launch_bounds_matches_prod_extents_times_32():
    src = render_kernelop(_warp_kernel(m_b=4, m_w=2), shapes={"C": (4, 2)}, literal_constants=_LITERALS)
    # 2 warps × 32 lanes = 64 threads per CTA.
    assert "__launch_bounds__(64)" in src, src


def test_warp_kernel_launch_bounds_single_warp_case():
    src = render_kernelop(_warp_kernel(m_b=8, m_w=1), shapes={"C": (8, 1)}, literal_constants=_LITERALS)
    # One warp × 32 lanes = 32 threads per CTA.
    assert "__launch_bounds__(32)" in src, src


def test_warp_kernel_renders_warp_id_decode_and_lane_decl():
    src = render_kernelop(_warp_kernel(m_b=4, m_w=2), shapes={"C": (4, 2)}, literal_constants=_LITERALS)
    # warp_id is decoded from threadIdx.x.
    assert "int warp_id = threadIdx.x / 32;" in src, src
    # The single warp axis decodes from warp_id directly (single-axis
    # case in _render_grid_axis_decode).
    assert "int m_w = warp_id;" in src, src
    # Lane is exposed unconditionally — the body presumes it's available.
    assert "int lane = threadIdx.x & 31;" in src, src


def test_warp_kernel_renders_block_axis_decode():
    src = render_kernelop(_warp_kernel(m_b=4, m_w=2), shapes={"C": (4, 2)}, literal_constants=_LITERALS)
    # The outer GridTile emits the standard blockIdx.x decode.
    assert "int m_b = blockIdx.x;" in src, src


def test_warp_kernel_multi_axis_warp_decode():
    """Two warp axes → row-major decode of warp_id into (m_w, n_w)."""
    M_b = Axis("m_b", 4)
    M_w = Axis("m_w", 2)
    N_w = Axis("n_w", 4)
    load = Load(name="one", input="one_const", index=())
    write = Write(output="C", index=(Var("m_b"), Var("m_w"), Var("n_w")), value="one")
    warp = WarpTile(axes=(M_w, N_w), body=Body((load, write)))
    grid = GridTile(axes=(M_b,), body=Body((warp,)))
    src = render_kernelop(KernelOp(body=(grid,), name="k_warp_2d"), shapes={"C": (4, 2, 4)}, literal_constants=_LITERALS)
    # 2 × 4 = 8 warps × 32 = 256 threads.
    assert "__launch_bounds__(256)" in src, src
    # n_w (inner) decodes as warp_id % 4; m_w (outer) decodes as warp_id / 4.
    assert "int n_w = warp_id % 4;" in src, src
    assert "int m_w = warp_id / (4);" in src, src


def test_warp_kernel_standalone_render_raises():
    """Top-level standalone WarpTile (no GridTile wrapper) is not
    supported in v1 — kernel render should raise NotImplementedError."""
    M_w = Axis("m_w", 2)
    load = Load(name="one", input="one_const", index=())
    write = Write(output="C", index=(Var("m_w"),), value="one")
    warp = WarpTile(axes=(M_w,), body=Body((load, write)))
    kop = KernelOp(body=(warp,), name="k_warp_alone")
    with pytest.raises(NotImplementedError, match="WarpTile outside GridTile"):
        render_kernelop(kop, shapes={"C": (2,)}, literal_constants=_LITERALS)


# ---------------------------------------------------------------------------
# NVRTC compile check — gated on GPU. Confirms the rendered CUDA actually
# compiles (catches subtle render bugs that pass the string-level asserts
# but produce malformed source).
# ---------------------------------------------------------------------------

from tests.compiler.conftest import requires_cuda  # noqa: E402


@requires_cuda
def test_warp_kernel_nvrtc_compiles():
    import cupy as cp  # noqa: PLC0415

    src = render_kernelop(_warp_kernel(m_b=4, m_w=2), shapes={"C": (4, 2)}, literal_constants=_LITERALS)
    # NVRTC drives the lazy JIT inside cp.RawKernel; touching ``kernel``
    # below triggers PTX compilation and surfaces any syntax error.
    raw = cp.RawKernel(src, "k_warp_smoke")
    _ = raw.kernel  # forces compile
