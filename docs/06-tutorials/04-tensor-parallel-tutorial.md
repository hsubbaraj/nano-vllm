# Tutorial: Implementing Multi-GPU Tensor Parallelism

## Overview

In this tutorial, we'll add tensor parallelism to our inference engine, enabling it to run models larger than a single GPU's memory and achieve higher throughput. We'll start with a single-GPU implementation and progressively add multi-GPU support.

## Prerequisites

- Completed previous tutorials (minimal engine, batching, KV cache)
- Multiple GPUs available (or ability to simulate)
- Understanding of distributed computing basics
- PyTorch with NCCL support

## Starting Point

We'll begin with our cached transformer model from the previous tutorial:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Tuple
import os

# Our existing model (single GPU)
class SingleGPUTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

## Step 1: Understanding Tensor Parallelism

Tensor parallelism splits model layers across GPUs:
- **Column Parallel**: Split output dimension (e.g., QKV projection)
- **Row Parallel**: Split input dimension (e.g., output projection)

```
Single GPU:           Tensor Parallel (2 GPUs):
[W] → [Y]            [W1] → [Y1]  
                     [W2] → [Y2]  → AllReduce → [Y]
```

## Step 2: Setting Up Distributed Environment

First, let's create utilities for distributed setup:

```python
class DistributedConfig:
    def __init__(self):
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.rank = int(os.getenv('RANK', '0'))
        self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.master_addr = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '29500')
        
def init_distributed():
    """Initialize distributed process group."""
    config = DistributedConfig()
    
    if config.world_size > 1:
        # Set environment variables
        os.environ['MASTER_ADDR'] = config.master_addr
        os.environ['MASTER_PORT'] = config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            world_size=config.world_size,
            rank=config.rank
        )
        
        # Set CUDA device
        torch.cuda.set_device(config.local_rank)
        
        print(f"Initialized rank {config.rank}/{config.world_size} on GPU {config.local_rank}")
    
    return config

# Helper functions for communication
def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across all processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor

def all_gather_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return tensors
```

## Step 3: Implementing Parallel Linear Layers

### Column Parallel Linear

```python
class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    
    The weight matrix is split along the output dimension:
    A = [A_1, A_2, ..., A_p]
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = True, init_std: float = 0.02):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        # Get parallel config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate per-partition size
        assert out_features % self.world_size == 0
        self.out_features_per_partition = out_features // self.world_size
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(
            self.out_features_per_partition, in_features
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute local output
        output = F.linear(input, self.weight, self.bias)
        
        # Gather output from all partitions if needed
        if self.gather_output and self.world_size > 1:
            # All-gather along last dimension
            output_list = [torch.empty_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)
        
        return output
```

### Row Parallel Linear

```python
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    
    The weight matrix is split along the input dimension:
    A = [A_1; A_2; ...; A_p]
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_is_parallel: bool = False, init_std: float = 0.02):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        # Get parallel config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate per-partition size
        assert in_features % self.world_size == 0
        self.in_features_per_partition = in_features // self.world_size
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(
            out_features, self.in_features_per_partition
        ))
        
        if bias:
            # Bias is not partitioned
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Split input along last dimension if not already parallel
        if not self.input_is_parallel and self.world_size > 1:
            # Split input
            input_list = input.chunk(self.world_size, dim=-1)
            input = input_list[self.rank]
        
        # Compute local output
        output = F.linear(input, self.weight, None)  # No bias in local computation
        
        # All-reduce across partitions
        if self.world_size > 1:
            dist.all_reduce(output)
        
        # Add bias after all-reduce
        if self.bias is not None:
            output = output + self.bias
        
        return output
```

## Step 4: Parallel Attention Implementation

Now let's implement attention with tensor parallelism:

```python
class ParallelMultiHeadAttention(nn.Module):
    """Multi-head attention with tensor parallelism."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Get parallel config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Ensure heads are divisible by world size
        assert self.num_heads % self.world_size == 0
        self.num_heads_per_partition = self.num_heads // self.world_size
        
        # QKV projection - column parallel
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=False, gather_output=False
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=False, gather_output=False  
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=False, gather_output=False
        )
        
        # Output projection - row parallel
        self.o_proj = RowParallelLinear(
            self.hidden_size, self.hidden_size, bias=False, input_is_parallel=True
        )
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections (outputs are already partitioned)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        # Note: we only have num_heads_per_partition heads
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads_per_partition, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update_and_get(k, v, self.rank)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads_per_partition * self.head_dim)
        
        # Output projection (includes all-reduce)
        output = self.o_proj(attn_output)
        
        return output
```

## Step 5: Parallel MLP Implementation

```python
class ParallelMLP(nn.Module):
    """MLP with tensor parallelism."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate and up projections - column parallel
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False, gather_output=False
        )
        
        # Down projection - row parallel
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size, bias=False, input_is_parallel=True
        )
        
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute gate and up projections
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
        # Apply activation and combine
        intermediate = self.act_fn(gate) * up
        
        # Down projection (includes all-reduce)
        output = self.down_proj(intermediate)
        
        return output
```

## Step 6: Building the Complete Parallel Model

```python
class ParallelTransformerLayer(nn.Module):
    """Transformer layer with tensor parallelism."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = ParallelMultiHeadAttention(config)
        self.mlp = ParallelMLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attention(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class ParallelTransformerModel(nn.Module):
    """Complete transformer model with tensor parallelism."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get parallel info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Embeddings (replicated across all GPUs)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ParallelTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm (replicated)
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # LM head - can be parallelized or replicated
        if config.parallel_lm_head:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False, gather_output=True
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return logits
```

## Step 7: Distributed Weight Loading

Loading weights correctly is crucial for tensor parallelism:

```python
class DistributedWeightLoader:
    """Load and distribute weights across tensor parallel ranks."""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
    
    def load_parallel_weights(self, model: nn.Module, checkpoint_path: str):
        """Load checkpoint and distribute weights."""
        # Only rank 0 loads the checkpoint
        if self.rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            state_dict = None
        
        # Broadcast checkpoint existence
        checkpoint_exists = torch.tensor([1 if state_dict is not None else 0], device='cuda')
        dist.broadcast(checkpoint_exists, src=0)
        
        if checkpoint_exists.item() == 0:
            raise ValueError("Checkpoint not found")
        
        # Load weights with proper partitioning
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self.rank == 0:
                    # Get full parameter from checkpoint
                    if name in state_dict:
                        full_param = state_dict[name]
                    else:
                        print(f"Warning: {name} not found in checkpoint")
                        full_param = param.new_zeros(param.shape)
                    
                    # Determine how to partition
                    if isinstance(model.get_submodule(name.rsplit('.', 1)[0]), ColumnParallelLinear):
                        # Split along dimension 0 (output dimension)
                        chunks = full_param.chunk(self.world_size, dim=0)
                        local_param = chunks[self.rank]
                    elif isinstance(model.get_submodule(name.rsplit('.', 1)[0]), RowParallelLinear):
                        # Split along dimension 1 (input dimension)
                        chunks = full_param.chunk(self.world_size, dim=1)
                        local_param = chunks[self.rank]
                    else:
                        # Replicate parameter
                        local_param = full_param
                else:
                    # Other ranks allocate buffer
                    local_param = param.new_empty(param.shape)
                
                # Broadcast from rank 0
                dist.broadcast(local_param, src=0)
                
                # Copy to parameter
                param.data.copy_(local_param)
        
        print(f"Rank {self.rank}: Weights loaded successfully")
```

## Step 8: Parallel KV Cache

We need to adapt our KV cache for tensor parallelism:

```python
class ParallelKVCache:
    """KV cache for tensor parallel models."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int,
                 head_dim: int, world_size: int, rank: int):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.world_size = world_size
        self.rank = rank
        
        # Each rank stores KV for its partition of heads
        assert num_heads % world_size == 0
        self.num_heads_per_partition = num_heads // world_size
        
        # Allocate cache
        self.k_cache = torch.zeros(
            num_layers, max_seq_len, self.num_heads_per_partition, head_dim,
            dtype=torch.float16, device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        self.seq_len = 0
    
    def update_and_get(self, k: torch.Tensor, v: torch.Tensor, 
                      layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return all cached KV."""
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Update cache
        if seq_len == 1:  # Generating
            pos = self.seq_len
            self.k_cache[layer_idx, pos] = k[0, :, 0]
            self.v_cache[layer_idx, pos] = v[0, :, 0]
            self.seq_len += 1
            
            # Return all cached values
            k_all = self.k_cache[layer_idx, :self.seq_len].unsqueeze(0)
            v_all = self.v_cache[layer_idx, :self.seq_len].unsqueeze(0)
            
            return k_all.transpose(1, 2), v_all.transpose(1, 2)
        else:  # Prefill
            # Store all positions
            self.k_cache[layer_idx, :seq_len] = k[0].transpose(0, 1)
            self.v_cache[layer_idx, :seq_len] = v[0].transpose(0, 1)
            self.seq_len = seq_len
            
            return k, v
```

## Step 9: Distributed Generation

Let's implement generation with tensor parallelism:

```python
class ParallelGenerator:
    """Text generation with tensor parallel models."""
    
    def __init__(self, model: ParallelTransformerModel, tokenizer, config: DistributedConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 1.0) -> str:
        """Generate text using tensor parallel model."""
        self.model.eval()
        
        # Only rank 0 handles tokenization
        if self.config.rank == 0:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
            prompt_len = input_ids.shape[1]
        else:
            input_ids = torch.empty(1, 1, dtype=torch.long, device='cuda')
            prompt_len = 0
        
        # Broadcast prompt length
        prompt_len_tensor = torch.tensor([prompt_len], device='cuda')
        dist.broadcast(prompt_len_tensor, src=0)
        prompt_len = prompt_len_tensor.item()
        
        # Broadcast input_ids
        if self.config.rank != 0:
            input_ids = torch.empty(1, prompt_len, dtype=torch.long, device='cuda')
        dist.broadcast(input_ids, src=0)
        
        # Create KV cache
        kv_cache = ParallelKVCache(
            max_seq_len=prompt_len + max_new_tokens,
            num_layers=self.model.config.num_layers,
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Prefill phase
        logits = self.model(input_ids, kv_cache=kv_cache)
        next_token_logits = logits[0, -1, :]
        
        # Generation loop
        generated_tokens = []
        for _ in range(max_new_tokens):
            # Sample next token (only on rank 0)
            if self.config.rank == 0:
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = torch.empty(1, dtype=torch.long, device='cuda')
            
            # Broadcast next token
            dist.broadcast(next_token, src=0)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            if self.config.rank == 0:
                generated_tokens.append(next_token.item())
            
            # Forward pass with new token
            logits = self.model(next_token.unsqueeze(0), kv_cache=kv_cache)
            next_token_logits = logits[0, 0, :]
        
        # Decode on rank 0
        if self.config.rank == 0:
            generated_text = self.tokenizer.decode(generated_tokens)
            return prompt + generated_text
        else:
            return ""
```

## Step 10: Optimizations

### Communication Optimization

```python
class OptimizedParallelLinear(nn.Module):
    """Optimized parallel linear with overlapped communication."""
    
    def __init__(self, in_features: int, out_features: int, 
                 is_column_parallel: bool = True):
        super().__init__()
        self.is_column_parallel = is_column_parallel
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if is_column_parallel:
            self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features // world_size))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_column_parallel:
            # Compute local GEMM
            output = F.linear(input, self.weight)
            
            # All-gather with computation overlap
            if dist.is_initialized():
                # Start async all-gather
                output_list = [torch.empty_like(output) for _ in range(dist.get_world_size())]
                handle = dist.all_gather(output_list, output, async_op=True)
                
                # Do other work here if possible
                
                # Wait for communication
                handle.wait()
                output = torch.cat(output_list, dim=-1)
            
            return output
        else:
            # Row parallel logic with overlapped all-reduce
            output = F.linear(input, self.weight)
            
            if dist.is_initialized():
                # Async all-reduce
                handle = dist.all_reduce(output, async_op=True)
                
                # Could do bias addition or other ops here
                
                handle.wait()
            
            return output
```

### Memory Optimization

```python
def checkpoint_parallel_layer(layer: nn.Module, *args, **kwargs):
    """Apply gradient checkpointing to parallel layers."""
    from torch.utils.checkpoint import checkpoint
    
    # Only checkpoint during training
    if layer.training:
        return checkpoint(layer, *args, **kwargs)
    else:
        return layer(*args, **kwargs)
```

## Step 11: Testing and Benchmarking

```python
def test_tensor_parallel_correctness():
    """Verify tensor parallel model produces same outputs as single GPU."""
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    # Single GPU model
    single_model = SingleGPUTransformer(config).cuda()
    
    # Tensor parallel model
    dist_config = init_distributed()
    parallel_model = ParallelTransformerModel(config).cuda()
    
    # Load same weights
    if dist_config.rank == 0:
        # Save single GPU weights
        torch.save(single_model.state_dict(), 'temp_weights.pt')
    
    dist.barrier()
    
    # Load into parallel model
    loader = DistributedWeightLoader(dist_config.world_size, dist_config.rank)
    loader.load_parallel_weights(parallel_model, 'temp_weights.pt')
    
    # Test input
    test_input = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    dist.broadcast(test_input, src=0)
    
    # Forward pass
    with torch.no_grad():
        if dist_config.rank == 0:
            single_output = single_model(test_input)
        
        parallel_output = parallel_model(test_input)
    
    # Compare outputs (only on rank 0)
    if dist_config.rank == 0:
        max_diff = (single_output - parallel_output).abs().max().item()
        print(f"Maximum difference: {max_diff}")
        assert max_diff < 1e-3, f"Outputs differ by {max_diff}"
        print("✓ Tensor parallel correctness test passed!")

def benchmark_tensor_parallel_performance():
    """Benchmark speedup from tensor parallelism."""
    import time
    
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=11008
    )
    
    dist_config = init_distributed()
    model = ParallelTransformerModel(config).cuda().eval()
    
    # Warmup
    for _ in range(10):
        input_ids = torch.randint(0, config.vocab_size, (1, 512)).cuda()
        _ = model(input_ids)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        input_ids = torch.randint(0, config.vocab_size, (1, 512)).cuda()
        _ = model(input_ids)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Gather times from all ranks
    times = torch.tensor([elapsed], device='cuda')
    dist.all_reduce(times)
    avg_time = times.item() / dist.get_world_size()
    
    if dist_config.rank == 0:
        throughput = num_iterations * 512 / avg_time
        print(f"Average time per iteration: {avg_time/num_iterations*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Theoretical speedup: {dist_config.world_size}x")
```

## Step 12: Production Deployment

### Launch Script

```python
# launch_parallel.py
import subprocess
import sys

def launch_tensor_parallel(script_path: str, num_gpus: int, *args):
    """Launch tensor parallel training/inference."""
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",
        "--use_env",
        script_path
    ] + list(args)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    # Example: python launch_parallel.py inference.py 4
    script = sys.argv[1]
    num_gpus = int(sys.argv[2])
    remaining_args = sys.argv[3:]
    
    launch_tensor_parallel(script, num_gpus, *remaining_args)
```

### Production-Ready Inference Server

```python
# inference_server.py
class TensorParallelInferenceServer:
    def __init__(self, model_path: str, config_path: str):
        # Initialize distributed
        self.dist_config = init_distributed()
        
        # Load config
        with open(config_path) as f:
            self.model_config = ModelConfig(**json.load(f))
        
        # Create model
        self.model = ParallelTransformerModel(self.model_config).cuda()
        
        # Load weights
        loader = DistributedWeightLoader(
            self.dist_config.world_size, 
            self.dist_config.rank
        )
        loader.load_parallel_weights(self.model, model_path)
        
        # Initialize tokenizer (only on rank 0)
        if self.dist_config.rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = None
        
        self.model.eval()
    
    def process_request(self, request: Dict) -> Dict:
        """Process inference request."""
        prompt = request['prompt']
        max_tokens = request.get('max_tokens', 100)
        temperature = request.get('temperature', 1.0)
        
        # Generate
        generator = ParallelGenerator(self.model, self.tokenizer, self.dist_config)
        output = generator.generate(prompt, max_tokens, temperature)
        
        # Only rank 0 returns response
        if self.dist_config.rank == 0:
            return {
                'generated_text': output,
                'model': self.model_config.name,
                'num_gpus': self.dist_config.world_size
            }
        else:
            return {}
    
    def benchmark_latency(self, batch_size: int = 1, seq_len: int = 512):
        """Benchmark inference latency."""
        input_ids = torch.randint(
            0, self.model_config.vocab_size, 
            (batch_size, seq_len)
        ).cuda()
        
        # Warmup
        for _ in range(10):
            _ = self.model(input_ids)
        
        # Measure
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = self.model(input_ids)
        end.record()
        
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)
        
        if self.dist_config.rank == 0:
            print(f"Latency: {latency:.2f} ms")
            print(f"Throughput: {batch_size * seq_len / (latency / 1000):.2f} tokens/sec")
```

## Best Practices

### 1. Communication Optimization

```python
# Minimize communication by fusing operations
class FusedParallelLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fuse multiple projections
        self.fused_qkv = ColumnParallelLinear(
            config.hidden_size, 
            3 * config.hidden_size,
            gather_output=False
        )
```

### 2. Load Balancing

```python
# Ensure even distribution of computation
def verify_load_balance(model: nn.Module):
    """Check if model is well-balanced across GPUs."""
    local_params = sum(p.numel() for p in model.parameters())
    total_params = torch.tensor([local_params], device='cuda')
    
    if dist.is_initialized():
        all_params = [torch.empty_like(total_params) for _ in range(dist.get_world_size())]
        dist.all_gather(all_params, total_params)
        
        if dist.get_rank() == 0:
            params_per_gpu = [p.item() for p in all_params]
            imbalance = max(params_per_gpu) / min(params_per_gpu) - 1
            print(f"Parameter distribution: {params_per_gpu}")
            print(f"Load imbalance: {imbalance:.1%}")
```

### 3. Error Handling

```python
def safe_distributed_call(func, *args, **kwargs):
    """Safely execute distributed operations."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Ensure all ranks exit together
        error_occurred = torch.tensor([1], device='cuda')
        dist.all_reduce(error_occurred)
        
        if dist.get_rank() == 0:
            print(f"Error in distributed operation: {e}")
        
        # Clean exit
        dist.destroy_process_group()
        raise
```

## Conclusion

In this tutorial, we've implemented a complete tensor parallel inference engine:

1. **Parallel Linear Layers**: Column and row parallelism
2. **Distributed Communication**: Efficient all-reduce and all-gather
3. **Parallel Attention/MLP**: Splitting computation across GPUs
4. **Weight Loading**: Correct distribution of model weights
5. **Generation**: Coordinated text generation
6. **Optimization**: Communication overlap and memory efficiency

Key takeaways:
- Tensor parallelism enables larger models and higher throughput
- Careful attention to communication patterns is crucial
- Load balancing ensures efficient GPU utilization
- Testing correctness against single-GPU implementation is essential

Next steps:
- Implement pipeline parallelism for even larger models
- Add sequence parallelism for long contexts
- Optimize for specific hardware (A100, H100)
- Integrate with production serving frameworks

Your inference engine now supports multi-GPU scaling!