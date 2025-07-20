FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ARG USE_CUDA=0
ARG TORCH_ARCH=
ENV DEBIAN_FRONTEND=noninteractive

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=${USE_CUDA}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_ARCH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

# 🔧 Install dependencies + CUDA components + TensorRT + cleanup in 1 RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ninja-build wget gnupg python3-dev python3-pip build-essential \
        cuda-nvcc-11-8 \
        cuda-compiler-11-8 \
        cuda-cudart-dev-11-8 \
        cuda-driver-dev-11-8 \
        cuda-libraries-dev-11-8 \
        cuda-command-line-tools-11-8 \
        cuda-profiler-api-11-8 \
        cuda-nvtx-11-8 \
        libcublas-11-8 && \
    rm -f /etc/apt/sources.list.d/cuda* && \
    rm -f /etc/apt/sources.list.d/nvidia* && \
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tensorrt python3-libnvinfer-dev libnvinfer-bin && \
    rm -rf /var/lib/apt/lists/*

# 🔧 Link CUDA stub + update environment
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    mkdir -p /usr/local/cuda/bin && \
    ln -sf /usr/local/cuda-11.8/bin/nvcc /usr/local/cuda/bin/nvcc

ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV PATH=$PATH:/usr/src/tensorrt/bin

# 🗂️ Set workdir and copy source code
WORKDIR /app
COPY . /app

# 🐍 Install python dependencies in one RUN to keep layers minimal
RUN python3 -m pip install --no-cache-dir -e segment_anything && \
    python3 -m pip install --no-cache-dir wheel && \
    pip install --no-cache-dir --force-reinstall numpy==1.24.4 opencv-python==4.7.0.72 && \
    pip install --no-cache-dir matplotlib==3.7.1 && \
    pip install --no-cache-dir \
        supervision==0.3.0 \
        diffusers[torch]==0.15.1 \
        pycocotools==2.0.6 \
        onnxruntime==1.14.1 \
        onnx==1.13.1 \
        ipykernel==6.16.2 \
        scipy \
        gradio \
        openai \
        onnxruntime-gpu && \
    CUDA_INC_DIR=/usr/local/cuda/include pip install --no-cache-dir pycuda

CMD ["tail", "-f", "/dev/null"]