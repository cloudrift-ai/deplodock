"""``020_stage_inputs._masked_k_mma_pad`` — alignment pad for the masked-K MMA slab.

The masked-K (symbolic-``seq_len``) SDPA P@V consumer stages its softmaxed ``P``
operand in a flat ``[…, M, K]`` smem slab read by ``ldmatrix.x4``. When the M-row
stride (the innermost block-scaled alloc-extent) is a multiple of 128 bytes the
ldmatrix row-lanes all alias one bank — the 3.67M-load-conflict storm in the
Qwen3-Embedding dynamic P@V. ``_masked_k_mma_pad`` returns an alignment-
preserving inner pad (one 16-byte ldmatrix chunk) that steps the stride off the
alias while keeping every row 16-byte aligned. It is stamped intrinsically on the
Source (not via the ``070_pad_smem`` autotune fork) so greedy deploys it; this
pins the gate + the pad value.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "020_stage_inputs.py"
    spec = importlib.util.spec_from_file_location("stage_inputs", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The shape the dynamic SDPA P@V consumer stages: 3 cache axes (K-tile, M, K)
# with an MMA atom block, K innermost. With block=(1,16,16) the innermost
# alloc-extent is 32*16 = 512 elems → 1024 B (fp16) = the 128-B bank alias.
_CACHE_AXES = (Axis("a3", 1), Axis("a1", 2), Axis("a7", 32))
_BLOCK = (1, 16, 16)


def test_pads_inner_axis_on_128b_alias_fp16():
    mod = _load_pass()
    pad = mod._masked_k_mma_pad((2, object()), _CACHE_AXES, _BLOCK, 2)
    # fp16 → one 16-byte ldmatrix chunk = 8 elements, on the innermost axis only.
    assert pad == (0, 0, 8)
    # The pad keeps the row 16-byte aligned: (inner_extent + pad) * elem_bytes % 16 == 0.
    inner = _CACHE_AXES[-1].extent.as_static() * _BLOCK[-1]
    assert ((inner + pad[-1]) * 2) % 16 == 0


def test_pads_inner_axis_fp32_chunk_is_four_elems():
    mod = _load_pass()
    # 4-byte operand: 16 B chunk = 4 elements; inner stride 512*4 = 2048 B is a
    # 128-B multiple, so the alias still fires.
    pad = mod._masked_k_mma_pad((2, object()), _CACHE_AXES, _BLOCK, 4)
    assert pad == (0, 0, 4)


def test_no_pad_without_kmask():
    mod = _load_pass()
    assert mod._masked_k_mma_pad(None, _CACHE_AXES, _BLOCK, 2) == ()


def test_no_pad_without_block():
    # A scalar masked-K reduce (no MMA atom block) isn't read by ldmatrix.
    mod = _load_pass()
    assert mod._masked_k_mma_pad((2, object()), _CACHE_AXES, (), 2) == ()


def test_no_pad_when_stride_not_aliased():
    mod = _load_pass()
    # Innermost alloc-extent 30*16 = 480 elems → 960 B (fp16), not a 128-B
    # multiple, so the slab is already conflict-light → leave it dense.
    axes = (Axis("a3", 1), Axis("a1", 2), Axis("a7", 30))
    assert mod._masked_k_mma_pad((2, object()), axes, _BLOCK, 2) == ()


def test_no_pad_single_axis():
    # A rank-1 slab has no outer M dim → no M-row stride alias to break.
    mod = _load_pass()
    assert mod._masked_k_mma_pad((0, object()), (Axis("a7", 32),), (16,), 2) == ()
