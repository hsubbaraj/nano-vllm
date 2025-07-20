# Module 0: Environment Setup with Docker

## 🎯 Module Overview

In this module, we'll set up a complete development environment for building our LLM inference engine. We'll use Docker to ensure everyone has the same environment, regardless of their operating system.

**Time Required**: 30-45 minutes

## 📋 Learning Objectives

By the end of this module, you will:
- ✅ Have a fully configured Docker environment with GPU support
- ✅ Understand the project structure and dependencies
- ✅ Run your first LLM inference
- ✅ Verify GPU acceleration is working

## 🛠️ Prerequisites

- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
- NVIDIA GPU with 8GB+ VRAM (or access to cloud GPU)
- 50GB+ free disk space for models and Docker images
- Basic familiarity with command line

## 📁 Project Structure

```
llm-inference-tutorial/
├── docker/
│   ├── Dockerfile              # Main development environment
│   ├── docker-compose.yml      # Multi-container setup
│   └── requirements.txt        # Python dependencies
├── module-00-setup/            # This module
├── module-01-minimal/          # Minimal engine
├── module-02-batching/         # Batching implementation
├── module-03-kvcache/          # KV cache
├── module-04-paged/            # PagedAttention
├── module-05-parallel/         # Tensor parallelism
├── module-06-production/       # Production deployment
├── models/                     # Model storage
├── data/                       # Test data
└── shared/                     # Shared utilities
```

## 🚀 Step 1: Create the Docker Environment

First, let's create our development environment. Create the following files:

### `docker/Dockerfile`

```dockerfile
# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    nvtop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR /workspace

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install -r /tmp/requirements.txt

# Install additional tools for development
RUN pip install ipython jupyter notebook tensorboard

# Create directories
RUN mkdir -p /workspace/models /workspace/data /workspace/logs

# Set up Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Verify GPU is accessible
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command
CMD ["/bin/bash"]
```

### `docker/requirements.txt`

```txt
# Core dependencies
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.25.0
tokenizers>=0.15.0

# Performance libraries
triton>=2.1.0
flash-attn>=2.4.0
xformers>=0.0.23

# Utilities
numpy>=1.24.0
tqdm>=4.66.0
pyyaml>=6.0
omegaconf>=2.3.0

# API and serving
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
httpx>=0.25.0

# Monitoring
prometheus-client>=0.19.0
psutil>=5.9.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0

# Development
ipdb>=0.13.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
```

### `docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  llm-tutorial:
    build:
      context: .
      dockerfile: Dockerfile
    image: llm-inference-tutorial:latest
    container_name: llm-tutorial
    hostname: llm-tutorial
    runtime: nvidia
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/workspace
      - HF_HOME=/workspace/models/huggingface
      
    volumes:
      # Mount the entire project
      - ../:/workspace
      # Separate volume for models (persists between containers)
      - llm-models:/workspace/models
      # Jupyter notebooks
      - jupyter-data:/root/.jupyter
      
    ports:
      # Jupyter
      - "8888:8888"
      # TensorBoard
      - "6006:6006"
      # API server (for later modules)
      - "8000:8000"
      # Debugging
      - "5678:5678"
      
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
              
    stdin_open: true
    tty: true
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
      interval: 30s
      timeout: 10s
      retries: 3
      
volumes:
  llm-models:
    driver: local
  jupyter-data:
    driver: local
```

## 🏗️ Step 2: Build and Start the Environment

1. **Navigate to the project root** and create the Docker directory:
```bash
mkdir -p llm-inference-tutorial/docker
cd llm-inference-tutorial/docker
```

2. **Create the files above** (Dockerfile, requirements.txt, docker-compose.yml)

3. **Build the Docker image**:
```bash
docker-compose build
```

This will take 5-10 minutes the first time as it downloads all dependencies.

4. **Start the container**:
```bash
docker-compose up -d
```

5. **Enter the container**:
```bash
docker exec -it llm-tutorial bash
```

## ✅ Step 3: Verify the Environment

Once inside the container, let's verify everything is working:

### Check GPU Access

```python
# verify_gpu.py
import torch
import subprocess

def check_environment():
    print("🔍 Checking Environment...\n")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation successful!")
    else:
        print("❌ No GPU detected. Some modules will run slowly.")
    
    # Check NVIDIA-SMI
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n📊 NVIDIA-SMI Output:")
        print(result.stdout)
    except:
        print("❌ nvidia-smi not available")

if __name__ == "__main__":
    check_environment()
```

Run the verification:
```bash
python verify_gpu.py
```

### Test Model Loading

Let's test loading a small model to ensure everything works:

```python
# test_inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def test_model_loading():
    print("🤖 Testing model loading and inference...\n")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Test inference
    prompt = "Hello, how are you?"
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"\n⏱️  Generation time: {generation_time:.2f} seconds")
    print(f"📊 Tokens generated: {outputs.shape[1] - inputs.shape[1]}")
    print(f"⚡ Tokens/second: {(outputs.shape[1] - inputs.shape[1]) / generation_time:.2f}")

if __name__ == "__main__":
    test_model_loading()
```

Run the test:
```bash
python test_inference.py
```

## 📚 Step 4: Understand the Development Workflow

### Working with the Container

1. **Start a development session**:
```bash
# From your host machine
docker-compose up -d
docker exec -it llm-tutorial bash
```

2. **Use multiple terminals**:
```bash
# Terminal 1: Main development
docker exec -it llm-tutorial bash

# Terminal 2: Running tests
docker exec -it llm-tutorial bash

# Terminal 3: Monitoring
docker exec -it llm-tutorial watch nvidia-smi
```

3. **Use Jupyter notebooks** (optional):
```bash
# Inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then open http://localhost:8888 in your browser.

### Managing Models

Models will be stored in `/workspace/models` which is persisted across container restarts:

```bash
# Download a model (inside container)
cd /workspace/models
git clone https://huggingface.co/microsoft/DialoGPT-small

# Or use huggingface-cli
pip install huggingface-hub
huggingface-cli download microsoft/DialoGPT-small --local-dir ./DialoGPT-small
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **"No GPU detected"**
   - Ensure NVIDIA drivers are installed on host
   - Check Docker has GPU support: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
   - On WSL2: Ensure WSL2 is updated and CUDA toolkit is installed

2. **"Out of memory" errors**
   - Reduce batch size in later modules
   - Use smaller models for testing
   - Monitor GPU memory: `watch -n 1 nvidia-smi`

3. **"Connection refused" on ports**
   - Check if ports are already in use: `sudo lsof -i :8888`
   - Modify port mappings in docker-compose.yml

4. **Slow model downloads**
   - Models are cached in the persistent volume
   - Use a faster mirror: `export HF_ENDPOINT=https://hf-mirror.com`

## 📝 Summary

You now have a complete development environment for building an LLM inference engine! 

✅ **What we accomplished:**
- Set up Docker with GPU support
- Installed all necessary dependencies
- Verified GPU acceleration works
- Tested model loading and inference
- Learned the development workflow

📚 **Key files created:**
- `docker/Dockerfile` - Development environment definition
- `docker/requirements.txt` - Python dependencies
- `docker/docker-compose.yml` - Container orchestration
- `verify_gpu.py` - GPU verification script
- `test_inference.py` - Model testing script

## 🎯 Exercises

Before moving to Module 1, try these exercises:

1. **Modify the test script** to use a different model (e.g., "gpt2")
2. **Monitor GPU usage** while running inference
3. **Experiment with generation parameters** (temperature, max_tokens)
4. **Set up Jupyter** and create a notebook for experimentation

## ⏭️ Next Steps

Ready to build your first inference engine? Continue to [Module 1: Building a Minimal Engine](../module-01-minimal/README.md)

---

**Need Help?** 
- Check the [Troubleshooting Guide](./troubleshooting.md)
- Join our [Discord Community](https://discord.gg/llm-inference)
- Open an issue on [GitHub](https://github.com/yourusername/llm-inference-tutorial)