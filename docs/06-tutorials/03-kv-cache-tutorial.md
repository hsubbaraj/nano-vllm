# Tutorial: Adding KV Cache to Your Inference Engine

## Overview

In this hands-on tutorial, we'll add KV (key-value) caching to a basic inference engine. KV caching is essential for efficient autoregressive generation, reducing computation from O(n²) to O(n) by caching attention keys and values from previous tokens.

## Prerequisites

Before starting, you should have:
- A basic transformer model implementation
- Understanding of attention mechanisms
- Completed the previous tutorials (minimal engine, batching)

## Starting Point

Let's begin with a simple model without KV caching:

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)
```

## Step 1: Understanding the Problem

Without KV caching, during generation:
- Token 1: Compute K,V for token 1
- Token 2: Recompute K,V for tokens 1-2
- Token 3: Recompute K,V for tokens 1-3
- ...
- Token n: Recompute K,V for tokens 1-n

This leads to O(n²) complexity. With KV caching, we compute each token's K,V only once.

## Step 2: Basic KV Cache Implementation

Let's create a simple KV cache:

```python
class KVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, 
                 head_dim: int, dtype: torch.dtype = torch.float16):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache tensors
        self.k_cache = torch.zeros(
            num_layers, max_seq_len, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros(
            num_layers, max_seq_len, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        
        # Track current sequence length
        self.seq_len = 0
    
    def update(self, key: torch.Tensor, value: torch.Tensor, 
               layer_idx: int, position: int):
        """Update cache with new key/value pair."""
        # key, value shape: [batch_size=1, seq_len=1, num_heads, head_dim]
        self.k_cache[layer_idx, position] = key[0, 0]
        self.v_cache[layer_idx, position] = value[0, 0]
        self.seq_len = max(self.seq_len, position + 1)
    
    def get(self, layer_idx: int, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a layer."""
        if seq_len is None:
            seq_len = self.seq_len
        
        k = self.k_cache[layer_idx, :seq_len]
        v = self.v_cache[layer_idx, :seq_len]
        
        # Add batch dimension
        return k.unsqueeze(0), v.unsqueeze(0)
    
    def clear(self):
        """Clear the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0
```

## Step 3: Modifying Attention for KV Cache

Now let's modify our attention layer to use the cache:

```python
class CachedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                position: int,
                kv_cache: Optional[KVCache] = None,
                use_cache: bool = True) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q for current token(s)
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if kv_cache is None or not use_cache:
            # No cache: compute K, V for all tokens
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Standard attention
            scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim)
            
            # Causal mask
            if seq_len > 1:
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            
        else:
            # Using cache
            if seq_len == 1:  # Generating single token
                # Compute K, V for current token only
                k_new = self.k_proj(hidden_states)
                v_new = self.v_proj(hidden_states)
                k_new = k_new.view(batch_size, 1, self.num_heads, self.head_dim)
                v_new = v_new.view(batch_size, 1, self.num_heads, self.head_dim)
                
                # Update cache
                kv_cache.update(k_new, v_new, self.layer_idx, position)
                
                # Get all cached K, V
                k_all, v_all = kv_cache.get(self.layer_idx, position + 1)
                
                # Attention with cached values
                scores = torch.einsum('bqhd,bkhd->bhqk', q, k_all) / math.sqrt(self.head_dim)
                v = v_all
                
            else:  # Prefill phase
                # Compute K, V for all tokens
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
                v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                # Store in cache
                for i in range(seq_len):
                    kv_cache.update(
                        k[:, i:i+1], v[:, i:i+1], 
                        self.layer_idx, position + i
                    )
                
                # Standard attention
                scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim)
                
                # Causal mask
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax and compute output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)
```

## Step 4: Building a Complete Model with KV Cache

Let's create a transformer model that uses our cached attention:

```python
class CachedTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, 
                 layer_idx: int):
        super().__init__()
        self.attention = CachedAttention(hidden_size, num_heads, layer_idx)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, position: int,
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, position, kv_cache)
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class CachedTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 num_heads: int, intermediate_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            CachedTransformerLayer(hidden_size, num_heads, intermediate_size, i)
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, position: int = 0,
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position, kv_cache)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
```

## Step 5: Implementing Generation with KV Cache

Now let's implement text generation using our KV cache:

```python
class CachedGenerator:
    def __init__(self, model: CachedTransformerModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1.0) -> str:
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = input_ids.shape[1]
        
        # Create KV cache
        kv_cache = KVCache(
            max_seq_len=prompt_len + max_new_tokens,
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            head_dim=self.model.hidden_size // self.model.num_heads
        )
        
        # Prefill phase: process entire prompt
        print(f"Prefill phase: processing {prompt_len} tokens...")
        logits = self.model(input_ids, position=0, kv_cache=kv_cache)
        
        # Get last token logits for first generation
        next_token_logits = logits[0, -1, :]
        
        # Generation phase: generate new tokens one by one
        generated_tokens = []
        for i in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.item())
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Generate next token using cache
            position = prompt_len + i
            logits = self.model(next_token.unsqueeze(0), position=position, kv_cache=kv_cache)
            next_token_logits = logits[0, 0, :]
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        return prompt + generated_text
```

## Step 6: Advanced KV Cache Features

### 6.1 Multi-Batch KV Cache

Let's extend our cache to handle multiple sequences:

```python
class MultiBatchKVCache:
    def __init__(self, batch_size: int, max_seq_len: int, num_layers: int,
                 num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # Cache per sequence
        self.k_cache = torch.zeros(
            batch_size, num_layers, max_seq_len, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros(
            batch_size, num_layers, max_seq_len, num_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        
        # Track sequence lengths
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long, device='cuda')
    
    def update(self, key: torch.Tensor, value: torch.Tensor, 
               layer_idx: int, batch_indices: torch.Tensor, positions: torch.Tensor):
        """Update cache for multiple sequences."""
        # key, value shape: [num_updates, 1, num_heads, head_dim]
        # batch_indices: [num_updates] - which sequence each update belongs to
        # positions: [num_updates] - position in each sequence
        
        for i, (batch_idx, pos) in enumerate(zip(batch_indices, positions)):
            self.k_cache[batch_idx, layer_idx, pos] = key[i, 0]
            self.v_cache[batch_idx, layer_idx, pos] = value[i, 0]
            self.seq_lens[batch_idx] = max(self.seq_lens[batch_idx], pos + 1)
    
    def get_for_sequence(self, batch_idx: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cache for a specific sequence."""
        seq_len = self.seq_lens[batch_idx].item()
        k = self.k_cache[batch_idx, layer_idx, :seq_len]
        v = self.v_cache[batch_idx, layer_idx, :seq_len]
        return k, v
```

### 6.2 Block-Based KV Cache

For production systems, we need block-based allocation:

```python
class BlockBasedKVCache:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        
        # Physical cache storage
        self.k_blocks = torch.zeros(
            num_blocks, num_layers, block_size, num_heads, head_dim,
            dtype=torch.float16, device='cuda'
        )
        self.v_blocks = torch.zeros_like(self.k_blocks)
        
        # Block management
        self.free_blocks = list(range(num_blocks))
        self.seq_to_blocks = {}  # seq_id -> list of block indices
    
    def allocate_for_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Not enough free blocks: need {num_blocks_needed}, have {len(self.free_blocks)}")
        
        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            allocated.append(block_idx)
        
        self.seq_to_blocks[seq_id] = allocated
        return allocated
    
    def update(self, seq_id: int, layer_idx: int, position: int,
               key: torch.Tensor, value: torch.Tensor):
        """Update cache for a specific position."""
        block_idx = position // self.block_size
        block_offset = position % self.block_size
        
        physical_block = self.seq_to_blocks[seq_id][block_idx]
        
        self.k_blocks[physical_block, layer_idx, block_offset] = key
        self.v_blocks[physical_block, layer_idx, block_offset] = value
    
    def get(self, seq_id: int, layer_idx: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV for a sequence."""
        blocks = self.seq_to_blocks[seq_id]
        
        # Gather from blocks
        keys = []
        values = []
        
        for i, block_idx in enumerate(blocks):
            start_pos = i * self.block_size
            end_pos = min(start_pos + self.block_size, seq_len)
            block_len = end_pos - start_pos
            
            if block_len > 0:
                keys.append(self.k_blocks[block_idx, layer_idx, :block_len])
                values.append(self.v_blocks[block_idx, layer_idx, :block_len])
        
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)
    
    def free_sequence(self, seq_id: int):
        """Free blocks allocated to a sequence."""
        if seq_id in self.seq_to_blocks:
            blocks = self.seq_to_blocks[seq_id]
            self.free_blocks.extend(blocks)
            del self.seq_to_blocks[seq_id]
```

## Step 7: Performance Optimization

### 7.1 Efficient KV Cache Kernels

Let's implement a custom CUDA kernel for faster cache updates:

```python
import triton
import triton.language as tl

@triton.jit
def kv_cache_update_kernel(
    keys, values,  # New KV to store [seq_len, num_heads, head_dim]
    k_cache, v_cache,  # Cache storage
    positions,  # Where to store each KV pair
    layer_idx,
    stride_k_s, stride_k_h, stride_k_d,
    stride_cache_l, stride_cache_p, stride_cache_h, stride_cache_d,
    HEAD_DIM: tl.constexpr,
):
    # Get position and head indices
    pos_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Load position for this token
    cache_pos = tl.load(positions + pos_idx)
    
    # Calculate offsets
    k_offset = pos_idx * stride_k_s + head_idx * stride_k_h
    cache_offset = (layer_idx * stride_cache_l + 
                   cache_pos * stride_cache_p + 
                   head_idx * stride_cache_h)
    
    # Copy head_dim elements
    for d in range(HEAD_DIM):
        k_val = tl.load(keys + k_offset + d * stride_k_d)
        v_val = tl.load(values + k_offset + d * stride_k_d)
        
        tl.store(k_cache + cache_offset + d * stride_cache_d, k_val)
        tl.store(v_cache + cache_offset + d * stride_cache_d, v_val)

class OptimizedKVCache(KVCache):
    def update_triton(self, keys: torch.Tensor, values: torch.Tensor,
                     layer_idx: int, positions: torch.Tensor):
        """Update cache using Triton kernel."""
        seq_len, num_heads, head_dim = keys.shape
        
        # Launch kernel
        grid = (seq_len, num_heads)
        
        kv_cache_update_kernel[grid](
            keys, values,
            self.k_cache, self.v_cache,
            positions,
            layer_idx,
            keys.stride(0), keys.stride(1), keys.stride(2),
            self.k_cache.stride(0), self.k_cache.stride(1), 
            self.k_cache.stride(2), self.k_cache.stride(3),
            HEAD_DIM=head_dim
        )
```

### 7.2 Memory-Efficient Attention with KV Cache

```python
def efficient_cached_attention(q: torch.Tensor, k_cache: torch.Tensor, 
                              v_cache: torch.Tensor, scale: float) -> torch.Tensor:
    """Memory-efficient attention computation with cached KV."""
    batch_size, num_heads, q_len, head_dim = q.shape
    _, _, kv_len, _ = k_cache.shape
    
    # Process in chunks to save memory
    chunk_size = 512
    output = torch.zeros_like(q)
    
    for i in range(0, kv_len, chunk_size):
        end_i = min(i + chunk_size, kv_len)
        
        # Compute attention scores for chunk
        k_chunk = k_cache[:, :, i:end_i]
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if q_len > 1:  # Prefill phase
            mask = torch.ones(q_len, end_i - i, device=scores.device)
            for j in range(q_len):
                if i + j < kv_len:
                    mask[j, :max(0, j - (kv_len - end_i))] = 0
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Compute attention weights
        if i == 0:
            attn_weights = torch.softmax(scores, dim=-1)
        else:
            # Stable softmax across chunks
            prev_max = output_max
            output_max = torch.maximum(output_max, scores.max(dim=-1, keepdim=True)[0])
            
            correction = torch.exp(prev_max - output_max)
            attn_weights = torch.exp(scores - output_max) + correction * attn_weights.sum(dim=-1, keepdim=True)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        
        # Apply attention to values
        v_chunk = v_cache[:, :, i:end_i]
        output += torch.matmul(attn_weights, v_chunk)
    
    return output
```

## Step 8: Testing and Benchmarking

Let's create comprehensive tests for our KV cache implementation:

```python
def test_kv_cache_correctness():
    """Test that cached attention produces same results as non-cached."""
    torch.manual_seed(42)
    
    # Model parameters
    batch_size = 1
    seq_len = 100
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    
    # Create model
    model = CachedTransformerModel(
        vocab_size=50000,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=3072
    ).cuda()
    
    # Test input
    input_ids = torch.randint(0, 50000, (batch_size, seq_len)).cuda()
    
    # Forward pass without cache
    with torch.no_grad():
        output_no_cache = model(input_ids, position=0, kv_cache=None)
    
    # Forward pass with cache (prefill + generation)
    kv_cache = KVCache(
        max_seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads
    )
    
    with torch.no_grad():
        # Prefill
        output_prefill = model(input_ids[:, :50], position=0, kv_cache=kv_cache)
        
        # Generate remaining tokens one by one
        generated_logits = [output_prefill]
        
        for i in range(50, seq_len):
            token = input_ids[:, i:i+1]
            logits = model(token, position=i, kv_cache=kv_cache)
            generated_logits.append(logits)
        
        # Concatenate all logits
        output_with_cache = torch.cat(generated_logits, dim=1)
    
    # Compare outputs
    max_diff = (output_no_cache - output_with_cache).abs().max().item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 1e-3, f"Outputs differ by {max_diff}"
    print("✓ KV cache correctness test passed!")

def benchmark_kv_cache_performance():
    """Benchmark performance improvement from KV caching."""
    import time
    
    # Model setup
    model = CachedTransformerModel(
        vocab_size=50000,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096
    ).cuda().eval()
    
    seq_lengths = [100, 500, 1000, 2000]
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 50000, (1, seq_len)).cuda()
        
        # Benchmark without cache
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for i in range(1, seq_len):
                _ = model(input_ids[:, :i], position=0, kv_cache=None)
        
        torch.cuda.synchronize()
        time_no_cache = time.time() - start
        
        # Benchmark with cache
        kv_cache = KVCache(
            max_seq_len=seq_len,
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            head_dim=model.hidden_size // model.num_heads
        )
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            # Prefill first token
            _ = model(input_ids[:, :1], position=0, kv_cache=kv_cache)
            
            # Generate remaining tokens
            for i in range(1, seq_len):
                _ = model(input_ids[:, i:i+1], position=i, kv_cache=kv_cache)
        
        torch.cuda.synchronize()
        time_with_cache = time.time() - start
        
        speedup = time_no_cache / time_with_cache
        print(f"Sequence length {seq_len}:")
        print(f"  Without cache: {time_no_cache:.2f}s")
        print(f"  With cache: {time_with_cache:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print()
```

## Step 9: Production Considerations

### 9.1 Dynamic Memory Management

```python
class DynamicKVCache:
    """KV cache that grows dynamically as needed."""
    
    def __init__(self, initial_size: int = 128, growth_factor: float = 1.5):
        self.current_size = initial_size
        self.growth_factor = growth_factor
        self.cache = None
        self._allocate(initial_size)
    
    def _allocate(self, size: int):
        """Allocate or reallocate cache."""
        new_cache = {
            'keys': torch.zeros(size, device='cuda'),
            'values': torch.zeros(size, device='cuda')
        }
        
        # Copy existing data if reallocating
        if self.cache is not None:
            old_size = self.cache['keys'].shape[0]
            new_cache['keys'][:old_size] = self.cache['keys']
            new_cache['values'][:old_size] = self.cache['values']
        
        self.cache = new_cache
    
    def ensure_capacity(self, required_size: int):
        """Ensure cache has enough capacity."""
        if required_size > self.current_size:
            new_size = int(required_size * self.growth_factor)
            self._allocate(new_size)
            self.current_size = new_size
```

### 9.2 Cache Persistence

```python
def save_kv_cache(kv_cache: KVCache, filepath: str):
    """Save KV cache to disk."""
    torch.save({
        'k_cache': kv_cache.k_cache,
        'v_cache': kv_cache.v_cache,
        'seq_len': kv_cache.seq_len,
        'config': {
            'max_seq_len': kv_cache.max_seq_len,
            'num_layers': kv_cache.num_layers,
            'num_heads': kv_cache.num_heads,
            'head_dim': kv_cache.head_dim
        }
    }, filepath)

def load_kv_cache(filepath: str) -> KVCache:
    """Load KV cache from disk."""
    data = torch.load(filepath)
    
    cache = KVCache(**data['config'])
    cache.k_cache = data['k_cache']
    cache.v_cache = data['v_cache']
    cache.seq_len = data['seq_len']
    
    return cache
```

## Conclusion

In this tutorial, we've built a complete KV caching system:

1. **Basic Implementation**: Simple cache for single sequences
2. **Integration**: Modified attention to use cache
3. **Multi-batch Support**: Handle multiple sequences
4. **Block-based Storage**: Production-ready memory management
5. **Optimization**: Custom kernels and efficient algorithms
6. **Testing**: Correctness verification and benchmarking

Key takeaways:
- KV caching is essential for efficient autoregressive generation
- Careful attention to memory layout improves performance
- Block-based allocation enables production scaling
- Testing both correctness and performance is critical

Next steps:
- Implement prefix caching for shared prompts
- Add support for different attention patterns
- Integrate with tensor parallelism
- Optimize for specific hardware

With KV caching implemented, your inference engine is now ready for efficient text generation!