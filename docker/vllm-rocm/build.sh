#!/bin/bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cloudriftai/vllm-rocm}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

if [ "${PUSH:-0}" = "1" ]; then
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"
fi
