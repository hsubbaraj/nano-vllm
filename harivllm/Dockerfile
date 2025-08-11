# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    nvtop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Create workspace
WORKDIR /workspace

# Install build dependencies first
RUN pip install packaging wheel setuptools --break-system-packages

# Install torch first (required for flash-attn build)
RUN pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu126 --break-system-packages

# Install packages that depend on torch
RUN pip install flash-attn>=2.4.0 xformers>=0.0.23 triton>=2.1.0 --break-system-packages

# Install remaining Python packages
RUN pip install transformers>=4.36.0 accelerate>=0.25.0 tokenizers>=0.15.0 \
    numpy>=1.24.0 tqdm>=4.66.0 pyyaml>=6.0 omegaconf>=2.3.0 \
    fastapi>=0.104.0 uvicorn>=0.24.0 pydantic>=2.5.0 httpx>=0.25.0 \
    prometheus-client>=0.19.0 psutil>=5.9.0 \
    pytest>=7.4.0 pytest-asyncio>=0.21.0 pytest-benchmark>=4.0.0 \
    ipdb>=0.13.0 black>=23.0.0 isort>=5.12.0 flake8>=6.0.0 --break-system-packages

# Install additional tools for development
RUN pip install ipython jupyter notebook tensorboard --break-system-packages

# Create directories
RUN mkdir -p /workspace/models /workspace/data /workspace/logs /data/modelpreload

# Set up Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Verify GPU is accessible
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command
CMD ["/bin/bash"] 