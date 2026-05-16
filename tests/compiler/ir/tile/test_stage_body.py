"""Phase-2 tests for ``Stage.body``: body is the single source of truth;
``buf`` / ``origin`` / ``addressing`` are derived properties.

Stage stays opaque to generic body walkers (no ``nested()`` /
``with_bodies()`` override) because exposing the cooperative-load Loads
to ``008_register_tile``'s F-axis replication corrupts their semantics.
σ-substitution through the body is handled by the Stage-specific
``rewrite`` / ``simplify`` handlers in ``deplodock/compiler/ir/tile/passes.py``.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, Stage, TemplateAddressing, trivial_stage_body


def _affine_stage() -> Stage:
    name = "x_smem"
    axes = (Axis("c0", 16), Axis("c1", 8))
    origin = (Var("k_outer") * Literal(16, "int"), Literal(0, "int"))
    body = trivial_stage_body(name, "x", origin, axes, AffineAddressing(dims=(0, 1)))
    return Stage(name=name, axes=axes, body=body)


def test_trivial_body_builder_matches_documented_shape():
    name = "x_smem"
    axes = (Axis("c0", 16), Axis("c1", 8))
    origin = (Var("k_outer") * Literal(16, "int"), Literal(0, "int"))
    body = trivial_stage_body(name, "x", origin, axes, AffineAddressing(dims=(0, 1)))

    assert len(body) == 2
    load, write = body
    assert isinstance(load, Load)
    assert load.input == "x"
    assert load.index[0] == origin[0] + Var("c0")
    assert load.index[1] == origin[1] + Var("c1")
    assert isinstance(write, Write)
    assert write.output == "x_smem"
    assert write.index == (Var("c0"), Var("c1"))
    assert write.value == load.name


def test_derived_buf_origin_addressing_affine():
    stage = _affine_stage()
    assert stage.buf == "x"
    # origin: cache-axis Vars substituted to 0 + simplified
    assert stage.origin[0].pretty() == (Var("k_outer") * Literal(16, "int")).pretty()
    assert stage.origin[1].pretty() == Literal(0, "int").pretty()
    addr = stage.addressing
    assert isinstance(addr, AffineAddressing)
    assert addr.dims == (0, 1)


def test_derived_addressing_template_for_nonaffine_index():
    # Build a body whose Load index is non-affine in the cache axes —
    # property derivation should fall back to TemplateAddressing.
    name = "x_smem"
    axes = (Axis("c0", 8),)
    nonlinear_index = (Var("c0") * Literal(2, "int"),)  # 2·c0 — coef-1 probe rejects
    body = Body(
        (
            Load(name=f"{name}__src", input="x", index=nonlinear_index),
            Write(output=name, index=(Var("c0"),), value=f"{name}__src"),
        )
    )
    stage = Stage(name=name, axes=axes, body=body)
    addr = stage.addressing
    assert isinstance(addr, TemplateAddressing)
    assert addr.exprs == nonlinear_index


def test_primary_load_raises_for_multi_load_body():
    name = "x_smem"
    axes = (Axis("c0", 4),)
    body = Body(
        (
            Load(name="l0", input="a", index=(Var("c0"),)),
            Load(name="l1", input="b", index=(Var("c0"),)),
            Write(output=name, index=(Var("c0"),), value="l0"),
        )
    )
    stage = Stage(name=name, axes=axes, body=body)
    with pytest.raises(ValueError, match="primary_load undefined"):
        _ = stage.primary_load
    # source_loads still works
    assert len(stage.source_loads) == 2


def test_rewrite_threads_sigma_through_body():
    stage = _affine_stage()
    # k_outer → k_outer + 1 — touches body Load's index. ``origin`` is
    # derived, so it picks up the σ-substitution transparently.
    new_k = Var("k_outer") + Literal(1, "int")
    sigma = Sigma({"k_outer": new_k})
    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten, Stage)
    # body Load index reflects the substitution
    assert rewritten.body[0].index[0] == new_k * Literal(16, "int") + Var("c0")
    # derived origin also reflects it (no double-substitution because
    # cache-axis Var → 0 happens at property access on the post-σ body).
    assert rewritten.origin[0].pretty() == (new_k * Literal(16, "int")).pretty()


def test_body_coerced_to_body_subclass_after_rewrite():
    stage = _affine_stage()
    sigma = Sigma({"k_outer": Literal(7, "int")})
    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten.body, Body)
