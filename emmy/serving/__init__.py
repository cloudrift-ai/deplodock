"""vLLM out-of-tree model plugin: deplodock-compiled kernels behind vLLM's serving shell.

``register`` is the ``vllm.general_plugins`` entry point (see pyproject.toml);
vLLM calls it in every process at engine start. The model class itself lives in
``vllm_model.py`` and is registered by lazy string path so importing this
package never pulls in vllm (or CUDA) by itself.
"""


def register() -> None:
    from vllm import ModelRegistry

    if "DeplodockEmbedModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("DeplodockEmbedModel", "deplodock.serving.vllm_model:DeplodockEmbedModel")
    if "DeplodockGenModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("DeplodockGenModel", "deplodock.serving.vllm_model_gen:DeplodockGenModel")
