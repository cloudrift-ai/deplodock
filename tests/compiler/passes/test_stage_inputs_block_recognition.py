"""M3 of ``plans/mma-smem-staging.md`` — ``_classify`` block stamping.

Drives ``020_stage_inputs._classify`` on a hand-built atom-strided Load
(σ output: ``m_w · 1024 + m_r · 16`` for ``WM=64, atom_M=16``) and
verifies the derived ``AffineAddressing.block`` matches the per-axis
atom factor. Pairs with the existing
``test_stage_inputs_classify.py`` which covers the unit-stride scalar
path — together they pin both shapes of the affine recognizer.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "020_stage_inputs.py"
    spec = importlib.util.spec_from_file_location("stage_inputs", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_classify_unit_stride_load_emits_empty_block():
    """Single cache axis with σ coefficient 1 → ``block=()`` (the
    no-multiplier sentinel). Matches scalar matmul / RMSNorm; M2's
    byte-clean assumption depends on this case staying empty."""
    mod = _load_pass()
    tid_fan = Axis("tid_fan", 32)
    tid_cache = Axis("tid_cache", 16)
    k = Axis("k", 8)
    load = Load(name="v", input="x", index=(Var("tid_cache"), Var("k")))
    slab = mod._classify(
        load,
        thread_axes=(tid_fan, tid_cache),
        reduce_axis=k,
        scope_axes=(tid_fan, tid_cache, k),
        slab_cap=1 << 20,
    )
    assert slab is not None
    assert slab.block == ()


def test_classify_atom_strided_load_stamps_block():
    """σ output ``m_w·1024 + m_r·16`` (an MMA ``mma_m16n8k16_f16``
    A-side stride for ``WM=1, FM=4, atom_M=16`` packed as multi-axis-
    per-dim) → block ``(16, 16)`` on cache axes ``(m_w, m_r)``. Walks
    right-to-left: coef(m_r)=16 / suffix=1 → block_m_r=16, suffix
    becomes 4·16=64; coef(m_w)=1024 / 64 = 16 → block_m_w=16.

    Gated on ``atom_kind`` being set — scalar paths with a non-1 σ coef
    fall to template instead (see
    ``test_classify_strided_load_without_atom_kind_falls_to_template``).
    """
    mod = _load_pass()
    tid_fan = Axis("tid_fan", 32)
    # m_w extent 1, m_r extent 4 — both share source dim 0. σ output:
    # m_w · 1024 + m_r · 16  (consistent with FM=4, atom_M=16, WM=1).
    m_w = Axis("m_w", 1)
    m_r = Axis("m_r", 4)
    k = Axis("k", 8)
    load = Load(
        name="v",
        input="x",
        index=(Var("m_w") * Literal(1024, "int") + Var("m_r") * Literal(16, "int"),),
    )
    slab = mod._classify(
        load,
        thread_axes=(tid_fan, m_w, m_r),
        reduce_axis=k,
        scope_axes=(tid_fan, m_w, m_r, k),
        slab_cap=1 << 30,
        atom_kind="mma_m16n8k16_f16",
    )
    assert slab is not None
    # cache_axes are sorted by source-dim; both m_w and m_r are on dim 0,
    # so they appear in their σ-stride order (MSB first → m_w then m_r).
    assert tuple(ax.name for ax in slab.cache_axes) == ("m_w", "m_r")
    assert slab.block == (16, 16)
    # The block-scaled smem budget kicks in: slab is 1×16 × 4×16 = 16 × 64
    # = 1024 elements × 4 bytes = 4096 bytes (≪ slab_cap).
    assert slab.n_bytes == 4 * 16 * 64


def test_classify_strided_load_without_atom_kind_falls_to_template():
    """The same atom-strided σ as the test above, but ``atom_kind=None``
    (scalar path). The legacy coef-1 composite check fails (because the
    σ has non-unit literal multipliers), block extraction is gated off,
    so the slab falls back to template addressing — preserving the
    scalar matmul / attention behavior the e2e tests depend on."""
    mod = _load_pass()
    tid_fan = Axis("tid_fan", 32)
    m_w = Axis("m_w", 1)
    m_r = Axis("m_r", 4)
    k = Axis("k", 8)
    load = Load(
        name="v",
        input="x",
        index=(Var("m_w") * Literal(1024, "int") + Var("m_r") * Literal(16, "int"),),
    )
    slab = mod._classify(
        load,
        thread_axes=(tid_fan, m_w, m_r),
        reduce_axis=k,
        scope_axes=(tid_fan, m_w, m_r, k),
        slab_cap=1 << 30,
    )
    assert slab is not None
    assert slab.block == ()
    # template is the verbatim load.index when the affine path bails.
    assert slab.template is not None


def test_classify_block_exceeds_budget_drops_slab():
    """A block-scaled slab that exceeds the per-scope smem cap is rejected
    by the post-derivation re-check — without it, the σ-strided Source
    would be admitted at the unblocked extent, then 005 / TMA passes
    would read garbage from the 255/256 of the slab nobody wrote."""
    mod = _load_pass()
    tid_fan = Axis("tid_fan", 32)
    # A wide pre-block slab (16 m_r × 32 k_i × 4 B = 2 KB) that would
    # blow past slab_cap once the 16× atom block scales it up.
    m_r = Axis("m_r", 16)
    k_i = Axis("k_i", 32)
    load = Load(
        name="v",
        input="x",
        index=(Var("m_r") * Literal(16, "int"), Var("k_i") * Literal(16, "int")),
    )
    slab = mod._classify(
        load,
        thread_axes=(tid_fan, m_r),
        reduce_axis=k_i,
        scope_axes=(tid_fan, m_r, k_i),
        # Generous enough for unblocked but tight against the 16× × 16×
        # scaling: unblocked footprint 2 KB × 256 = 512 KB.
        slab_cap=64 * 1024,
        atom_kind="mma_m16n8k16_f16",
    )
    assert slab is None
