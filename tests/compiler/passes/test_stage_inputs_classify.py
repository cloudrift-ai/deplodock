"""Scoped unit tests for ``020_stage_inputs._classify``.

Exercises the helper directly (bypassing the rule's outer scope walk and
Body rewriting) so we can pin down the slab-eligibility contract on a
single Load. Two paired cases differ only in whether one cache axis has
a static or symbolic extent — the symbolic case must drop the slab.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "020_stage_inputs.py"
    spec = importlib.util.spec_from_file_location("stage_inputs", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(index: tuple) -> Load:
    """Two-name vector Load on buffer ``x`` so the helper's fan-in /
    cache-axis logic has something concrete to walk over."""
    return Load(name="v", input="x", index=index)


def test_classify_static_cache_axes_returns_slab():
    """Baseline: a fan-in thread axis plus two static cache axes (one
    thread, one reduce) → ``_classify`` returns a non-None slab whose
    byte-count matches ``BYTES_PER_ELEM * extent_t * extent_k``."""
    mod = _load_pass()

    tid_fan = Axis("tid_fan", 32)
    tid_cache = Axis("tid_cache", 16)
    k = Axis("k", 8)
    load = _load((Var("tid_cache"), Var("k")))

    slab = mod._classify(
        load,
        thread_axes=(tid_fan, tid_cache),
        reduce_axis=k,
        scope_axes=(tid_fan, tid_cache, k),
        slab_cap=1 << 20,
    )

    assert slab is not None
    # 4 (BYTES_PER_ELEM) * 16 * 8 = 512.
    assert slab.n_bytes == 4 * 16 * 8


def test_classify_symbolic_cache_axis_drops_slab():
    """A cache axis with a symbolic extent (``Dim("seq_len")``) makes the
    slab size unbounded at compile time, so ``_classify`` must skip the
    candidate. Same scope shape as the static case, only ``tid_cache``'s
    extent differs — proves the symbolic-extent guard is the deciding
    factor, not some unrelated bail-out."""
    mod = _load_pass()

    tid_fan = Axis("tid_fan", 32)
    tid_cache = Axis("tid_cache", Dim("seq_len"))
    k = Axis("k", 8)
    load = _load((Var("tid_cache"), Var("k")))

    slab = mod._classify(
        load,
        thread_axes=(tid_fan, tid_cache),
        reduce_axis=k,
        scope_axes=(tid_fan, tid_cache, k),
        slab_cap=1 << 20,
    )

    assert slab is None
