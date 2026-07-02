"""Bind constants from external storage (safetensors, live nn.Module)
into the per-node ``input_data`` dict the backends consume."""

from emmy.compiler.loader.binder import bind_constants, bind_constants_from_module
from emmy.compiler.loader.safetensors import load_constants_from_safetensors

__all__ = ["bind_constants", "bind_constants_from_module", "load_constants_from_safetensors"]
