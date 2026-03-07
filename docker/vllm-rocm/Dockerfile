ARG BASE_IMAGE=rocm/vllm-dev:base_gfx950_therock_base_20260123
FROM ${BASE_IMAGE}

ENV PYTORCH_ROCM_ARCH=gfx950
ENV CMAKE_PREFIX_PATH="/opt/python/lib/python3.12/site-packages/torch/share/cmake"
# Skip runtime GPU detection in rocm_sdk (no GPU during docker build).
# Without this, offload-arch fails and its warning pollutes compiler flags.
ENV ROCM_SDK_TARGET_FAMILY=gfx950-dcgpu

# Clone latest vLLM
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main
RUN git clone --depth 1 --branch ${VLLM_REF} ${VLLM_REPO} /vllm

WORKDIR /vllm
RUN pip install -r requirements/rocm.txt \
    && pip install --no-build-isolation .

# Install amd-quark for MXFP4 support
RUN pip install amd-quark

ENV SAFETENSORS_FAST_GPU=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["vllm", "serve"]
