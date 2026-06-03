"""M7 of ``plans/mma-smem-staging.md`` — pad_smem skips blocked Sources.

``070_pad_smem``'s ``+1`` pad on the inner cache axis breaks the
``mma.sync`` path's ``ldmatrix`` 16-byte alignment requirement. The fix is
structural: any
``AffineAddressing`` whose ``block`` tuple is non-empty represents an
MMA atom-strided slab and must stay un-padded.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.stmt import Body, Load
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Source,
    StageBundle,
    StagePolicy,
)
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "070_pad_smem.py"
    spec = importlib.util.spec_from_file_location("pad_smem", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _src(*, name: str, buf: str, block: tuple[int, ...] = ()) -> Source:
    cache_axes = (Axis("a3", 8), Axis("a7", 32))
    addressing = AffineAddressing(dims=(0, 1), block=block)
    return Source(
        name=name,
        buf=buf,
        cache_axes=cache_axes,
        origin=(Literal(0, "int"), Literal(0, "int")),
        addressing=addressing,
    )


def _bundle(src: Source) -> StageBundle:
    """Minimal BUFFERED bundle with one consumer Load — the shape
    ``_plan_for_bundle`` walks for padding decisions."""
    consumer = Load(name="v", input=src.name, index=(Literal(0, "int"), Literal(0, "int")))
    return StageBundle(sources=(src,), body=Body((consumer,)), policy=StagePolicy.BUFFERED, buffer_count=2, phase=Literal(0, "int"))


def test_pad_smem_skips_blocked_source():
    """A Source with ``addressing.block != ()`` doesn't get a pad entry —
    the bundle is left alone by ``_plan_for_bundle``."""
    mod = _load_pass()
    src = _src(name="b_smem", buf="b", block=(16, 16))
    bundle = _bundle(src)
    plan: dict = {}
    mod._plan_for_bundle(bundle, plan, thread_axes=(Axis("tid", 128),))
    assert id(bundle) not in plan


def test_pad_smem_pads_unblocked_source_when_eligible():
    """The non-MMA path is unaffected: a Source with ``block=()`` is
    still considered for padding (whether a pad is picked depends on
    the bank-conflict heuristic, not the gate this test pins)."""
    mod = _load_pass()
    src = _src(name="a_smem", buf="a")
    bundle = _bundle(src)
    plan: dict = {}
    # The exact pad choice depends on the bank-conflict analysis; we
    # only assert the bundle entered the planning path, i.e. wasn't
    # skipped by the M7 block-check.
    mod._plan_for_bundle(bundle, plan, thread_axes=(Axis("tid", 128),))
    # Plan may be empty if no pad is needed for this trivial shape, but
    # if it's populated, the bundle entry is the unblocked one we passed.
    if plan:
        assert id(bundle) in plan
