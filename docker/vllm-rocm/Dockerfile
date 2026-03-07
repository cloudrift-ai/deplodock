ARG BASE_IMAGE=rocm/vllm-dev:base_gfx950_therock_base_20260123
FROM ${BASE_IMAGE} AS base

ENV PYTORCH_ROCM_ARCH=gfx950

# Clone latest vLLM
FROM base AS build_vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main
RUN git clone --depth 1 --branch ${VLLM_REF} ${VLLM_REPO} /vllm

WORKDIR /vllm
RUN pip install -r requirements/rocm.txt \
    && python3 setup.py bdist_wheel --dist-dir=dist

FROM base AS final
COPY --from=build_vllm /vllm/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Install amd-quark for MXFP4 support
RUN pip install amd-quark

ENV SAFETENSORS_FAST_GPU=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["vllm", "serve"]
