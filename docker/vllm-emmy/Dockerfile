# vLLM serving image with the emmy embedding plugin baked in.
#
# Build from the repo root (the wheel must exist first — `make wheel`):
#   make vllm-emmy-image          # builds cloudriftai/vllm-emmy:<vllm-ver>-<git-sha>
#   make vllm-emmy-push
#
# Serve (same entrypoint as the stock image, plus the plugin flags):
#   docker run --gpus all -p 8000:8000 cloudriftai/vllm-emmy:TAG \
#     --model Qwen/Qwen3-Embedding-0.6B --runner pooling --enforce-eager \
#     --max-model-len 4096 --hf-overrides '{"architectures":["EmmyEmbedModel"]}'
ARG VLLM_VERSION=v0.22.1
FROM vllm/vllm-openai:${VLLM_VERSION}

# The emmy wheel installs --no-deps so the image's pinned torch /
# transformers / numpy (vLLM's own dependency set) are never touched; the
# runtime deps the serving path actually uses are added explicitly. cppyy is
# deliberately absent — the CUDA serving path compiles via nvcc/NVRTC only.
# The cupy wheel must match the image's CUDA major (vllm-openai:v0.22.1
# ships CUDA 13.0 + nvcc; transformers 5.10.2 satisfies the compile pin).
ARG CUPY_PACKAGE=cupy-cuda13x
COPY dist/emmy_llm-*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/emmy_llm-*.whl \
    && pip install --no-cache-dir "${CUPY_PACKAGE}" catboost pyyaml httpx "huggingface_hub[hf_transfer]" safetensors \
    && rm -rf /tmp/wheels

# Compiled-cubin cache under the HF cache mount (compose mounts the model dir
# there), so container restarts on the same VM skip recompiles. Recipes can
# override via extra_env.
ENV EMMY_CUBIN_CACHE=/root/.cache/huggingface/emmy-cubins
