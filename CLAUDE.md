# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation built from scratch in ~1,200 lines of Python code. It provides fast offline inference comparable to vLLM with optimizations like prefix caching, tensor parallelism, torch compilation, and CUDA graphs.

## Installation and Setup

Install the package directly from the repository:
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

For development work, install in editable mode:
```bash
pip install -e .
```

## Model Setup

Download model weights manually (example with Qwen3-0.6B):
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Core Architecture

### Engine Components
- **LLMEngine** (`nanovllm/engine/llm_engine.py`): Main orchestrator that manages multiprocess tensor parallelism
- **Scheduler** (`nanovllm/engine/scheduler.py`): Handles request scheduling and batching
- **ModelRunner** (`nanovllm/engine/model_runner.py`): Executes model inference across tensor parallel processes
- **BlockManager** (`nanovllm/engine/block_manager.py`): Manages KV cache memory allocation

### Model Implementation
- **Models** (`nanovllm/models/`): Currently supports Qwen3 architecture
- **Layers** (`nanovllm/layers/`): Modular transformer components with tensor parallelism support
- **Attention** (`nanovllm/layers/attention.py`): Flash attention and KV caching implementation

### Key Classes
- **LLM** (`nanovllm/llm.py`): High-level API interface that wraps LLMEngine
- **SamplingParams** (`nanovllm/sampling_params.py`): Configuration for generation parameters
- **Sequence** (`nanovllm/engine/sequence.py`): Represents individual inference requests

## Usage Examples

Run the provided examples:
```bash
python example.py    # Basic usage demonstration
python bench.py      # Performance benchmark against vLLM
```

## Key Differences from vLLM

- API differences in `LLM.generate()` method signature
- Uses multiprocessing with spawn context for tensor parallelism
- Simplified codebase focusing on core inference functionality
- No serving infrastructure (offline inference only)

## Dependencies

- torch>=2.4.0
- triton>=3.0.0  
- transformers>=4.51.0
- flash-attn
- xxhash
- Python 3.10-3.12

## Development Notes

- No formal test suite - validation is done through example.py and bench.py
- Architecture follows vLLM patterns but simplified for readability
- Model path is typically `~/huggingface/MODEL_NAME/` format
- Uses distributed tensor parallelism across GPU processes