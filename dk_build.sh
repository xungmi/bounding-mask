#!/bin/bash

# Detect nvcc
if ! command -v nvcc &> /dev/null; then
  echo "CUDA not installed. Building without CUDA support."
  USE_CUDA=0
  TORCH_CUDA_ARCH_LIST=""
else
  NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  if (( $(echo "$NVCC_VERSION > 11.0" | bc -l) )); then
    echo "CUDA version $NVCC_VERSION detected. Building with CUDA support."
    USE_CUDA=1
    TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
  else
    echo "CUDA version $NVCC_VERSION is not supported. Building without CUDA."
    USE_CUDA=0
    TORCH_CUDA_ARCH_LIST=""
  fi
fi

# Build Docker image
docker build \
  --build-arg USE_CUDA=${USE_CUDA} \
  --build-arg TORCH_ARCH="${TORCH_CUDA_ARCH_LIST}" \
  -t gsa:v2 .
