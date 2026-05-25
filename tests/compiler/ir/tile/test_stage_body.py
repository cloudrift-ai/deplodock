"""Tests for the wrap-body ``Stage`` IR.

``Stage.body`` is the consumer subtree; producer cooperative-load is
synthesized at materialize time from ``Stage.sources``. Each Source
carries ``buf`` / ``origin`` / ``cache_dims`` / ``addressing``
explicitly; σ-substitution walks the consumer body via
``deplodock/compiler/ir/tile/passes.py``.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load
from deplodock.compiler.ir.tile.ir import AffineAddressing, CacheDim, Source, Stage, TemplateAddressing


def _affine_source() -> Source:
    name = "x_smem"
    cache_dims = (CacheDim(axis=Axis("c0", 16), source_dim=0), CacheDim(axis=Axis("c1", 8), source_dim=1))
    origin = (Var("k_outer") * Literal(16, "int"), Literal(0, "int"))
    return Source(name=name, buf="x", cache_dims=cache_dims, origin=origin)


def _affine_stage() -> Stage:
    """Build a Stage wrapping a trivial consumer body (one Load from staged smem)."""
    src = _affine_source()
    consumer = Body((Load(name="v", input=src.name, index=(Var("c0"), Var("c1"))),))
    return Stage(sources=(src,), body=consumer)


def test_source_carries_buf_origin_addressing():
    src = _affine_source()
    assert src.buf == "x"
    assert src.origin[0].pretty() == (Var("k_outer") * Literal(16, "int")).pretty()
    assert src.origin[1].pretty() == Literal(0, "int").pretty()
    addr = src.addressing
    assert isinstance(addr, AffineAddressing)
    assert addr.dims == (0, 1)


def test_source_template_addressing_for_explicit_template_index():
    cache_dims = (CacheDim(axis=Axis("c0", 8), source_dim=0),)
    nonlinear_index = (Var("c0") * Literal(2, "int"),)
    src = Source(
        name="x_smem",
        buf="x",
        cache_dims=cache_dims,
        origin=(Literal(0, "int"),),
        template_index=nonlinear_index,
    )
    addr = src.addressing
    assert isinstance(addr, TemplateAddressing)
    assert addr.exprs == nonlinear_index


def test_multi_source_stage():
    """Multi-source Stages are the typical matmul case (A + B)."""
    a = _affine_source()
    b = Source(
        name="b_smem",
        buf="b",
        cache_dims=(CacheDim(axis=Axis("k", 8), source_dim=0), CacheDim(axis=Axis("n", 16), source_dim=1)),
        origin=(Literal(0, "int"), Var("n_b") * Literal(16, "int")),
    )
    stage = Stage(sources=(a, b), body=Body(()))
    assert stage.external_reads() == ("x", "b")
    assert stage.local_decls() == ("x_smem", "b_smem")


def test_stage_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one Source"):
        Stage(sources=(), body=Body(()))


def test_rewrite_threads_sigma_through_body_and_sources():
    """σ-rewrite must touch the consumer body's Loads AND each Source's origin."""
    stage = _affine_stage()
    # k_outer → k_outer + 1 — touches sources[0].origin[0]; body has no k_outer.
    new_k = Var("k_outer") + Literal(1, "int")
    sigma = Sigma({"k_outer": new_k})
    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten, Stage)
    assert rewritten.sources[0].origin[0].pretty() == (new_k * Literal(16, "int")).pretty()


def test_body_coerced_to_body_subclass_after_rewrite():
    stage = _affine_stage()
    sigma = Sigma({"k_outer": Literal(7, "int")})
    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten.body, Body)


def test_nested_returns_consumer_body():
    stage = _affine_stage()
    (body,) = stage.nested()
    assert body is stage.body


def test_with_bodies_round_trips():
    stage = _affine_stage()
    new_body = Body((Load(name="v2", input=stage.sources[0].name, index=(Var("c0"), Var("c1"))),))
    rebuilt = stage.with_bodies((new_body,))
    assert isinstance(rebuilt, Stage)
    assert rebuilt.body == new_body
    assert rebuilt.sources == stage.sources
