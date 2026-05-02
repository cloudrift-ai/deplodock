"""Bind constants from external storage (safetensors, live nn.Module)
into the per-node ``input_data`` dict the backends consume."""

from deplodock.compiler.loader.binder import bind_constants
from deplodock.compiler.loader.safetensors import load_constants_from_safetensors

__all__ = ["bind_constants", "load_constants_from_safetensors"]
