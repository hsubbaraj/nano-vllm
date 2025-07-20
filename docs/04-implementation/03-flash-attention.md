# Integrating Flash Attention with KV Caching

## Overview

Flash Attention is a revolutionary algorithm that computes exact attention with O(1) memory complexity instead of O(n²), enabling efficient processing of long sequences. This guide covers integrating Flash Attention into an inference engine with support for KV caching, covering both the algorithm understanding and practical implementation.

## Understanding Flash Attention

### The Memory Bottleneck

Standard attention has quadratic memory complexity:
```python
# Standard attention pseudocode
scores = Q @ K.T  # O(n²) memory
attn_weights = softmax(scores)  # O(n²) memory  
output = attn_weights @ V  # O(n²) intermediate
```

Flash Attention solves this by:
1. Tiling computation into blocks
2. Fusing operations in SRAM
3. Recomputing instead of storing intermediates

### Algorithm Overview

```python
# Flash Attention conceptual algorithm
def flash_attention(Q, K, V, block_size=256):
    """
    Q, K, V: [batch, heads, seq_len, head_dim]
    Returns: [batch, heads, seq_len, head_dim]
    """
    seq_len = Q.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size
    
    output = torch.zeros_like(Q)
    
    # Process in blocks to fit in SRAM
    for i in range(num_blocks):
        q_block = Q[:, :, i*block_size:(i+1)*block_size]
        
        # Accumulate attention from all K,V blocks
        block_output = torch.zeros_like(q_block)
        
        for j in range(num_blocks):
            k_block = K[:, :, j*block_size:(j+1)*block_size]
            v_block = V[:, :, j*block_size:(j+1)*block_size]
            
            # Compute attention for this block pair
            scores = q_block @ k_block.transpose(-2, -1)
            
            # Causal mask if needed
            if i < j:
                scores.fill_(-float('inf'))
            
            # Stable softmax with online normalization
            block_output += softmax(scores) @ v_block
        
        output[:, :, i*block_size:(i+1)*block_size] = block_output
    
    return output
```

## Setting Up Flash Attention

### Installation and Dependencies

```python
# Install Flash Attention
# pip install flash-attn --no-build-isolation

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    flash_attn_with_kvcache,
    flash_attn_varlen_func
)
```

### Basic Flash Attention Usage

```python
class FlashAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, 
                 dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, L, H, D]
        q, k, v = qkv.unbind(0)
        
        # Flash attention
        # Note: flash_attn expects [B, L, H, D] format
        output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.causal
        )
        
        # Reshape and project
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(output)
        
        return output
```

## Integrating with KV Cache

### KV Cache-Aware Flash Attention

```python
class FlashAttentionWithCache(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary embeddings if used
        if config.use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                cache_position: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(v, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        if kv_cache is not None:
            # Use Flash Attention with KV cache
            output = self._flash_attention_with_cache(
                q, k, v, kv_cache, cache_position
            )
        else:
            # Standard Flash Attention for prefill
            output = flash_attn_func(q, k, v, causal=True)
        
        # Reshape and project output
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output
    
    def _flash_attention_with_cache(self, q, k, v, kv_cache, cache_position):
        """Flash attention with KV cache support."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Store new KV pairs in cache
        kv_cache.update(k, v, self.layer_idx, cache_position)
        
        # Get cache for this layer
        k_cache, v_cache = kv_cache.get_cache(self.layer_idx)
        
        # Use flash_attn_with_kvcache for generation
        output = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,  # New keys to append
            v=v,  # New values to append
            cache_seqlens=cache_position,  # Current cache lengths
            causal=True
        )
        
        return output
```

### Efficient KV Cache Storage

```python
import triton
import triton.language as tl

@triton.jit
def flash_kvcache_store_kernel(
    keys, values,           # New KV to store
    key_cache, value_cache, # Cache storage
    cache_positions,        # Where to store in cache
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_kc_b, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_b, stride_vc_h, stride_vc_s, stride_vc_d,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Optimized kernel for storing KV pairs in cache."""
    # Get program ID
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Load cache position for this sequence element
    cache_pos = tl.load(cache_positions + batch_idx * seq_idx)
    
    # Calculate offsets for head dimension
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load K and V
    k_ptrs = keys + batch_idx * stride_k_b + head_idx * stride_k_h + \
             seq_idx * stride_k_s + offs_d * stride_k_d
    v_ptrs = values + batch_idx * stride_v_b + head_idx * stride_v_h + \
             seq_idx * stride_v_s + offs_d * stride_v_d
    
    k_vals = tl.load(k_ptrs)
    v_vals = tl.load(v_ptrs)
    
    # Store in cache
    kc_ptrs = key_cache + batch_idx * stride_kc_b + head_idx * stride_kc_h + \
              cache_pos * stride_kc_s + offs_d * stride_kc_d
    vc_ptrs = value_cache + batch_idx * stride_vc_b + head_idx * stride_vc_h + \
              cache_pos * stride_vc_s + offs_d * stride_vc_d
    
    tl.store(kc_ptrs, k_vals)
    tl.store(vc_ptrs, v_vals)

class OptimizedKVCache:
    def __init__(self, num_blocks: int, num_layers: int, num_heads: int,
                 head_dim: int, block_size: int, dtype: torch.dtype):
        # Allocate cache as contiguous memory
        self.key_cache = torch.zeros(
            num_layers, num_blocks * block_size, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.value_cache = torch.zeros(
            num_layers, num_blocks * block_size, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def update(self, keys: torch.Tensor, values: torch.Tensor,
               layer_idx: int, cache_positions: torch.Tensor):
        """Update cache using optimized Triton kernel."""
        batch_size, seq_len, num_heads, head_dim = keys.shape
        
        # Launch Triton kernel
        grid = (batch_size, num_heads, seq_len)
        
        flash_kvcache_store_kernel[grid](
            keys, values,
            self.key_cache[layer_idx], self.value_cache[layer_idx],
            cache_positions,
            keys.stride(0), keys.stride(2), keys.stride(1), keys.stride(3),
            values.stride(0), values.stride(2), values.stride(1), values.stride(3),
            self.key_cache.stride(1), self.key_cache.stride(2), 
            self.key_cache.stride(0), self.key_cache.stride(3),
            self.value_cache.stride(1), self.value_cache.stride(2),
            self.value_cache.stride(0), self.value_cache.stride(3),
            BLOCK_SIZE=self.block_size,
            HEAD_DIM=head_dim
        )
```

## Variable Length Sequences

### Handling Multiple Sequences Efficiently

```python
class FlashAttentionVarlen(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Standard projections
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                max_seqlen: int) -> torch.Tensor:
        """
        Forward pass for variable length sequences.
        
        Args:
            hidden_states: [total_tokens, hidden_size] - packed sequences
            cu_seqlens: [batch_size + 1] - cumulative sequence lengths
            max_seqlen: Maximum sequence length in the batch
        """
        total_tokens = hidden_states.shape[0]
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # [3, total_tokens, H, D]
        q, k, v = qkv.unbind(0)
        
        # Flash attention for variable length
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=True
        )
        
        # Reshape and project
        output = output.reshape(total_tokens, self.hidden_size)
        output = self.o_proj(output)
        
        return output

def pack_sequences(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pack variable length sequences for efficient processing."""
    # Concatenate all sequences
    packed = torch.cat(sequences, dim=0)
    
    # Calculate cumulative sequence lengths
    lengths = [seq.shape[0] for seq in sequences]
    cu_seqlens = torch.tensor([0] + list(accumulate(lengths)), 
                             dtype=torch.int32, device=packed.device)
    
    max_seqlen = max(lengths)
    
    return packed, cu_seqlens, max_seqlen
```

## Advanced Flash Attention Features

### Sliding Window Attention

```python
class SlidingWindowFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projections
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV computation
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        
        # Flash attention with sliding window
        output = flash_attn_func(
            q, k, v,
            causal=True,
            window_size=(self.window_size, -1)  # Local attention window
        )
        
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)
```

### Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)

```python
class GroupedQueryFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads  # Can be < num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Separate projections for Q and KV
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 
                                2 * self.num_kv_heads * self.head_dim, 
                                bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query projection
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Key-Value projection (fewer heads)
        kv = self.kv_proj(hidden_states)
        kv = kv.view(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        
        # Repeat KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)
        
        # Flash attention
        output = flash_attn_func(q, k, v, causal=True)
        
        output = output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of Q heads."""
    if n_rep == 1:
        return hidden_states
    
    batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seq_len, num_kv_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)
```

## Performance Optimization

### Benchmarking Flash Attention

```python
def benchmark_attention_implementations():
    """Compare standard attention vs Flash Attention."""
    batch_size = 8
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    
    # Create random inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Warmup
    for _ in range(10):
        _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
    
    # Benchmark Flash Attention
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        output_flash = flash_attn_func(q, k, v, causal=True)
    
    torch.cuda.synchronize()
    flash_time = time.time() - start
    
    # Benchmark standard attention (for comparison)
    def standard_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    # Compare memory usage
    print(f"Flash Attention time: {flash_time:.3f}s")
    print(f"Flash Attention memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Optimizing for Different Hardware

```python
class HardwareAwareFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Detect hardware capabilities
        self.gpu_type = torch.cuda.get_device_name()
        self.sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        
        # Set parameters based on hardware
        if "A100" in self.gpu_type:
            self.optimal_block_size = 128
            self.use_tf32 = True
        elif "V100" in self.gpu_type:
            self.optimal_block_size = 64
            self.use_tf32 = False
        else:
            self.optimal_block_size = 32
            self.use_tf32 = False
        
        # Initialize layers
        self._setup_layers()
    
    def _setup_layers(self):
        # Configure based on hardware capabilities
        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
```

## Debugging Flash Attention

### Verification Against Standard Attention

```python
def verify_flash_attention_correctness():
    """Verify Flash Attention produces correct results."""
    torch.manual_seed(42)
    
    # Small test case
    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Flash Attention
    output_flash = flash_attn_func(q, k, v, causal=True)
    
    # Reference implementation
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(head_dim)
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1)
    scores.masked_fill_(mask.bool(), float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output_ref = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
    
    # Compare outputs
    max_diff = (output_flash - output_ref).abs().max().item()
    mean_diff = (output_flash - output_ref).abs().mean().item()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    assert max_diff < 1e-3, f"Flash Attention output differs significantly: {max_diff}"
```

### Common Issues and Solutions

```python
class FlashAttentionTroubleshooting:
    @staticmethod
    def check_input_requirements(q, k, v):
        """Verify inputs meet Flash Attention requirements."""
        issues = []
        
        # Check dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            issues.append("Flash Attention requires fp16 or bf16 inputs")
        
        # Check device
        if not q.is_cuda:
            issues.append("Flash Attention requires CUDA tensors")
        
        # Check contiguity
        if not q.is_contiguous():
            issues.append("Query tensor must be contiguous")
        
        # Check shapes
        if q.shape != k.shape or q.shape != v.shape:
            issues.append("Q, K, V must have the same shape")
        
        # Check head dimension
        if q.shape[-1] > 256:
            issues.append("Head dimension must be <= 256")
        
        return issues
    
    @staticmethod
    def fix_common_issues(q, k, v):
        """Fix common input issues."""
        # Ensure correct dtype
        if q.dtype == torch.float32:
            q = q.half()
            k = k.half()
            v = v.half()
        
        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        return q, k, v
```

## Production Integration

### Complete Flash Attention Module

```python
class ProductionFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, 
                               self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, 
                               self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Optional features
        self.use_rotary = getattr(config, 'use_rotary_embeddings', True)
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Flash Attention settings
        self.dropout = config.attention_dropout if self.training else 0.0
        self.window_size = getattr(config, 'sliding_window_size', None)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if self.use_rotary and position_ids is not None:
            cos, sin = self.rotary_emb(v, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            # Concatenate with past KV
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Update cache if requested
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
        
        # Repeat KV heads if using MQA/GQA
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)
        
        # Flash Attention
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout,
            causal=True,
            window_size=(self.window_size, -1) if self.window_size else (-1, -1)
        )
        
        # Reshape and project
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value
```

## Best Practices

### 1. Memory Management

```python
# Pre-allocate output buffer to avoid allocation during forward pass
output_buffer = torch.empty_like(q)
output = flash_attn_func(q, k, v, causal=True, out=output_buffer)
```

### 2. Batch Processing

```python
# Process multiple sequences efficiently
def batch_flash_attention(sequences: List[Dict[str, torch.Tensor]]):
    # Pack sequences
    all_q = torch.cat([s['q'] for s in sequences], dim=0)
    all_k = torch.cat([s['k'] for s in sequences], dim=0)
    all_v = torch.cat([s['v'] for s in sequences], dim=0)
    
    # Create cumulative sequence lengths
    lengths = [s['q'].shape[0] for s in sequences]
    cu_seqlens = torch.tensor([0] + list(accumulate(lengths)))
    
    # Single Flash Attention call
    output = flash_attn_varlen_func(
        all_q, all_k, all_v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths)
    )
    
    return output
```

### 3. Error Handling

```python
def safe_flash_attention(q, k, v, **kwargs):
    try:
        return flash_attn_func(q, k, v, **kwargs)
    except RuntimeError as e:
        if "head_size" in str(e):
            # Fall back to standard attention for unsupported head sizes
            return standard_attention(q, k, v, **kwargs)
        else:
            raise
```

## Conclusion

Flash Attention with KV caching is essential for efficient LLM inference:

1. **Memory efficiency**: O(1) memory complexity enables long sequences
2. **Speed**: Fused kernels provide significant speedup
3. **KV cache integration**: Seamless support for incremental generation
4. **Hardware optimization**: Achieves near-peak GPU utilization
5. **Flexibility**: Supports various attention patterns and optimizations

Proper integration of Flash Attention can provide 2-4x speedup in attention computation while enabling much longer context lengths. Next, we'll explore CUDA graphs for further optimization.