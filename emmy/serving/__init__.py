"""vLLM out-of-tree model plugin: emmy-compiled kernels behind vLLM's serving shell.

``register`` is the ``vllm.general_plugins`` entry point (see pyproject.toml);
vLLM calls it in every process at engine start. The model class itself lives in
``vllm_model.py`` and is registered by lazy string path so importing this
package never pulls in vllm (or CUDA) by itself.
"""


def register() -> None:
    from vllm import ModelRegistry

    if "EmmyEmbedModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyEmbedModel", "emmy.serving.vllm_model:EmmyEmbedModel")
    if "EmmyGenModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("EmmyGenModel", "emmy.serving.vllm_model_gen:EmmyGenModel")
