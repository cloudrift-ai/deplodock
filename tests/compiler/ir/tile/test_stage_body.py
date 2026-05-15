"""Phase-1 tests for ``Stage.body``: auto-synthesis from legacy
``(buf, origin, axes, addressing)`` fields and σ-substitution through
the explicit ``rewrite`` handler.

Phase 1 keeps ``Stage`` as a *leaf* stmt (no ``nested()`` / ``with_bodies``
override) so generic body walkers don't descend into the cooperative-load
program — that breaks ``008_register_tile``'s F-axis replication. The body
field exists, is populated, and is threaded by ``Stage.rewrite`` /
``Stage.simplify``; everything else routes through legacy fields until
phase 2.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, Stage, TemplateAddressing


def _make_stage(addressing) -> Stage:
    return Stage(
        name="x_smem",
        buf="x",
        origin=(Var("k_outer") * Literal(16, "int"), Literal(0, "int")),
        axes=(Axis("c0", 16), Axis("c1", 8)),
        addressing=addressing,
    )


def test_trivial_body_synthesized_for_affine_addressing():
    stage = _make_stage(AffineAddressing(dims=(0, 1)))

    assert isinstance(stage.body, Body)
    assert len(stage.body) == 2

    load, write = stage.body
    assert isinstance(load, Load)
    assert load.input == "x"
    # source index = origin[d] + cache_var (per affine dim mapping)
    assert load.index[0] == stage.origin[0] + Var("c0")
    assert load.index[1] == stage.origin[1] + Var("c1")

    assert isinstance(write, Write)
    assert write.output == "x_smem"
    assert write.index == (Var("c0"), Var("c1"))
    assert write.value == load.name


def test_trivial_body_synthesized_for_template_addressing():
    # Template carries the original symbolic Load index verbatim.
    template = (Var("c0") * Literal(8, "int") + Var("c1"), Var("c1"))
    stage = _make_stage(TemplateAddressing(exprs=template))

    load, write = stage.body
    assert load.index == template
    assert write.index == (Var("c0"), Var("c1"))


def test_explicit_body_passed_through():
    custom_load = Load(name="raw", input="x", index=(Var("c0"), Var("c1")))
    custom_write = Write(output="x_smem", index=(Var("c0"), Var("c1")), value="raw")
    explicit_body = Body((custom_load, custom_write))
    stage = Stage(
        name="x_smem",
        buf="x",
        origin=(Literal(0, "int"), Literal(0, "int")),
        axes=(Axis("c0", 16), Axis("c1", 8)),
        addressing=AffineAddressing(dims=(0, 1)),
        body=explicit_body,
    )

    assert stage.body is explicit_body or tuple(stage.body) == tuple(explicit_body)


def test_rewrite_threads_sigma_through_body():
    stage = _make_stage(AffineAddressing(dims=(0, 1)))
    # Substitute k_outer → k_outer + 1 — touches both origin and the body
    # Load's index, since both reference k_outer.
    new_k = Var("k_outer") + Literal(1, "int")
    sigma = Sigma({"k_outer": new_k})

    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten, Stage)

    # Legacy origin is σ-substituted.
    assert rewritten.origin[0] == new_k * Literal(16, "int")

    # Body's Load index is σ-substituted consistently.
    body_load = rewritten.body[0]
    assert isinstance(body_load, Load)
    assert body_load.index[0] == new_k * Literal(16, "int") + Var("c0")


def test_body_is_body_subclass_after_rewrite():
    # rewrite handler builds body as a plain tuple comprehension; Stage's
    # __post_init__ must coerce it back to Body so downstream walkers
    # (Body.map / Body.iter) keep working.
    stage = _make_stage(AffineAddressing(dims=(0, 1)))
    sigma = Sigma({"k_outer": Literal(7, "int")})
    rewritten = stage.rewrite(lambda n: n, sigma=sigma)
    assert isinstance(rewritten.body, Body)
