"""Pre-trace substitution for compressed-tensors weight-only quantized models.

A compressed-tensors checkpoint registers a forward *decompress pre-hook* that
unpacks the int4 weights on first forward. When ``torch.export`` traces the
model that hook fires mid-trace and injects a data-dependent slice
(``unpack_from_int32`` → ``int(shape[1])``) that export can't specialize, so
the whole model fails to trace.

This module walks the loaded model *before* export and swaps each quantized
linear for a deplodock-authored :class:`DequantLinear` whose ``forward`` is the
single opaque custom op ``deplodock::dequant_linear``. That (1) keeps the packed
int32 / fp16-scale / int32-zp tensors as graph constants for the binder, (2)
prevents the decompress hook from ever firing, and (3) emits one
``DequantLinearOp`` node the ``045_dequant_linear`` decomposition expands into
the unpack → dequant → matmul cone the rest of the pipeline lowers.

The format-specific logic (config parse + the torch dequant the custom op's
eager impl runs) is isolated here; everything below the ``DequantLinearOp`` is
format-agnostic. See ``plans/w4a16-quantization-support.md``.
"""

# NOTE: deliberately NO ``from __future__ import annotations`` — ``torch.library``
# infers the custom-op schema from the eager impl's real type annotations, and
# stringized annotations break its schema inference (it can't resolve ``torch``).

import logging
from typing import Optional

from deplodock.compiler.ir.frontend.ir import QuantScheme

logger = logging.getLogger(__name__)

_CUSTOM_OP_REGISTERED = False


def _ensure_custom_op() -> None:
    """Register ``deplodock::dequant_linear`` once per process.

    The op is opaque to ``torch.export`` (preserved as one ``call_function``
    node via ``register_fake``); its eager impl runs the real dequant so a
    substituted ``DequantLinear`` is the deployed accuracy reference. Buffers
    are passed as explicit tensor args (not closed over) so they surface as
    graph constants for binding.
    """
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return
    import torch

    @torch.library.custom_op("deplodock::dequant_linear", mutates_args=())
    def dequant_linear(  # noqa: F811 — registered into torch.ops, never called by name
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        bias: Optional[torch.Tensor],  # noqa: UP045 — torch schema infer needs typing.Optional, not ``| None``
        num_bits: int,
        group_size: int,
        packed_dim: int,
        symmetric: bool,
    ) -> torch.Tensor:
        weight = _dequant_weight_torch(weight_packed, weight_scale, weight_zero_point, num_bits, group_size, symmetric)
        return torch.nn.functional.linear(x, weight, bias)

    @dequant_linear.register_fake
    def _(x, weight_packed, weight_scale, weight_zero_point, bias, num_bits, group_size, packed_dim, symmetric):
        out_features = weight_packed.shape[0]
        return x.new_empty((*x.shape[:-1], out_features))

    _CUSTOM_OP_REGISTERED = True


def _dequant_weight_torch(weight_packed, weight_scale, weight_zero_point, num_bits, group_size, symmetric):
    """Torch twin of ``ir.frontend.ir.dequant_weight_numpy`` — returns the
    ``[out, in]`` weight in the scale's dtype (fp16). Kept in lock-step with
    the numpy oracle (and cross-checked against ``compressed_tensors.dequantize``
    in the gated parity test)."""
    import torch

    per = 32 // num_bits
    mask = (1 << num_bits) - 1
    out_f, in_packed = weight_packed.shape
    in_f = in_packed * per

    wp = weight_packed.to(torch.int64)
    nib = torch.empty((out_f, in_f), dtype=torch.int64, device=wp.device)
    for i in range(per):
        nib[:, i::per] = (wp >> (num_bits * i)) & mask

    if symmetric:
        deq_int = nib - (1 << (num_bits - 1))
    else:
        zpp = weight_zero_point.to(torch.int64)
        out_packed, zp_groups = zpp.shape
        zp = torch.empty((out_packed * per, zp_groups), dtype=torch.int64, device=zpp.device)
        for i in range(per):
            zp[i::per, :] = (zpp >> (num_bits * i)) & mask
        zp = zp[:out_f, :]
        zp_bc = zp.repeat_interleave(group_size, dim=1)[:, :in_f]
        deq_int = nib - zp_bc

    scale_bc = weight_scale.repeat_interleave(group_size, dim=1)[:, :in_f]
    return deq_int.to(scale_bc.dtype) * scale_bc


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _config_get(cfg, key, default=None):
    """Read ``key`` off a config that may be a dict or a config object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def parse_quant_config(model_config) -> tuple[QuantScheme, list[str]] | None:
    """Return ``(scheme, ignore)`` for a compressed-tensors model, else ``None``.

    Handles both ``quantization_config`` and the legacy ``compression_config``,
    and a dict- or object-shaped config. Transformers wraps the compressed-tensors
    ``QuantizationConfig`` one level deep (``CompressedTensorsConfig`` →
    ``.quantization_config``), so descend when ``config_groups`` isn't at the top.
    Reads the first weight config group (uniform across linears for the verified
    format). The ``quant_method`` enum compares equal to its ``"compressed-tensors"``
    string value.
    """
    qc = getattr(model_config, "quantization_config", None) or getattr(model_config, "compression_config", None)
    if qc is None:
        return None
    if _config_get(qc, "quant_method") != "compressed-tensors":
        return None

    # The transformers wrapper holds the real groups under a nested config.
    inner = qc
    if _config_get(inner, "config_groups") is None and _config_get(inner, "quantization_config") is not None:
        inner = _config_get(inner, "quantization_config")

    groups = _config_get(inner, "config_groups") or {}
    weights = None
    for group in groups.values() if isinstance(groups, dict) else groups:
        weights = _config_get(group, "weights")
        if weights is not None:
            break
    if weights is None:
        logger.warning("compressed-tensors config has no weight group; skipping quant substitution")
        return None

    scheme = QuantScheme(
        num_bits=int(_config_get(weights, "num_bits", 4)),
        group_size=int(_config_get(weights, "group_size", 32)),
        packed_dim=1,  # pack-quantized interleaves nibbles along the in/K axis
        symmetric=bool(_config_get(weights, "symmetric", False)),
    )
    ignore = list(_config_get(inner, "ignore", []) or [])
    return scheme, ignore


# ---------------------------------------------------------------------------
# DequantLinear module + substitution walker
# ---------------------------------------------------------------------------


def _build_dequant_linear(module, scheme: QuantScheme):
    """Wrap a quantized linear's packed buffers in a ``DequantLinear``."""
    import torch.nn as nn

    _ensure_custom_op()

    class DequantLinear(nn.Module):
        """Trace-friendly replacement: ``forward`` is the opaque custom op.

        Holds the packed int32 weight / fp16 scale / int32 zero-point (and
        optional bias) as buffers so they trace as graph constants. The format
        numbers ride as scalar args to the custom op (recorded as
        ``DequantLinearOp`` metadata by the tracer, never as graph constants)."""

        def __init__(self, weight_packed, weight_scale, weight_zero_point, bias):
            super().__init__()
            self.register_buffer("weight_packed", weight_packed)
            self.register_buffer("weight_scale", weight_scale)
            self.register_buffer("weight_zero_point", weight_zero_point)
            self.register_buffer("bias", bias)  # None registers a None buffer
            self._num_bits = scheme.num_bits
            self._group_size = scheme.group_size
            self._packed_dim = scheme.packed_dim
            self._symmetric = scheme.symmetric

        def forward(self, x):
            import torch  # noqa: PLC0415

            return torch.ops.deplodock.dequant_linear(
                x,
                self.weight_packed,
                self.weight_scale,
                self.weight_zero_point,
                self.bias,
                self._num_bits,
                self._group_size,
                self._packed_dim,
                self._symmetric,
            )

    bias = getattr(module, "bias", None)
    return DequantLinear(module.weight_packed, module.weight_scale, module.weight_zero_point, bias)


def _is_quantized_linear(module) -> bool:
    """A compressed-tensors packed linear carries these three buffers."""
    return all(hasattr(module, attr) for attr in ("weight_packed", "weight_scale", "weight_zero_point"))


def _is_ignored(name: str, ignore: list[str]) -> bool:
    """Match a module name against the config ``ignore`` list (final-component
    or substring match — covers ``"lm_head"`` and ``"re:.*lm_head"`` forms)."""
    leaf = name.rsplit(".", 1)[-1]
    for ig in ignore:
        pat = ig[3:] if ig.startswith("re:") else ig
        if pat == leaf or pat == name or pat in name:
            return True
    return False


def apply_quant_substitution(model) -> int:
    """Swap every targeted quantized linear for a ``DequantLinear``, in place.

    No-op (returns 0) for a non-compressed-tensors model. Skips ``ignore``
    modules (e.g. ``lm_head``, which stays dense fp16). Returns the number of
    modules substituted so callers can log / assert.
    """
    parsed = parse_quant_config(getattr(model, "config", None))
    if parsed is None:
        return 0
    scheme, ignore = parsed

    targets: list[tuple[str, object]] = [
        (name, m) for name, m in model.named_modules() if _is_quantized_linear(m) and not _is_ignored(name, ignore)
    ]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        setattr(parent, attr, _build_dequant_linear(module, scheme))

    if targets:
        logger.info(
            "W4A16: substituted %d quantized linears (num_bits=%d group_size=%d symmetric=%s)",
            len(targets),
            scheme.num_bits,
            scheme.group_size,
            scheme.symmetric,
        )
    return len(targets)
