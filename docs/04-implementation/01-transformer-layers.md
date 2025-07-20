# Implementing Transformer Layers for Inference

## Overview

This guide walks through implementing efficient transformer layers optimized for inference. We'll cover attention mechanisms, MLP layers, normalization, and how to integrate them into a complete transformer model with a focus on memory efficiency and performance.

## Core Transformer Architecture

### The Basic Building Block

A transformer layer consists of:
1. Multi-head self-attention
2. Layer normalization  
3. Feed-forward network (MLP)
4. Residual connections

```python
class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ln2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
```

## Implementing Efficient Attention

### Multi-Head Attention Basics

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # QKV projection
        self.qkv_proj = nn.Linear(
            self.hidden_size, 
            3 * self.hidden_size, 
            bias=config.attention_bias
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.hidden_size, 
            self.hidden_size, 
            bias=config.attention_bias
        )
        
        # Rotary embeddings if used
        if config.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
```

### Optimized Attention Forward Pass

```python
def forward(self, hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    
    # QKV projection
    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply rotary embeddings
    if hasattr(self, 'rotary_emb'):
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    
    # KV cache handling
    if kv_cache is not None:
        k, v = kv_cache.update(k, v)
    
    # Attention computation
    attn_output = self.compute_attention(q, k, v, attention_mask)
    
    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    
    return attn_output
```

### Flash Attention Integration

For maximum performance:

```python
def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.use_flash_attention and attention_mask is None:
        # Use Flash Attention 2
        return flash_attn_func(q, k, v, causal=True)
    else:
        # Fallback to standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
```

## Rotary Position Embeddings (RoPE)

### Implementation

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, 
                 base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for maximum length
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

### Applying Rotary Embeddings

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor,
                        position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Gather position embeddings
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

## MLP Implementation

### Standard MLP

```python
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = get_activation(config.hidden_act)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        down = self.down_proj(intermediate)
        return down
```

### Activation Functions

```python
def get_activation(act_fn: str):
    """Get activation function by name."""
    if act_fn == "silu":
        return nn.SiLU()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {act_fn}")

# Optimized SiLU for inference
class FastSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x, inplace=True)  # In-place for memory efficiency
```

## Layer Normalization

### RMSNorm Implementation

RMSNorm is often preferred for its simplicity and efficiency:

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

### LayerNorm with Optimizations

```python
class OptimizedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's optimized implementation
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
```

## KV Cache Integration

### KV Cache Structure

```python
class KVCache:
    def __init__(self, num_blocks: int, num_layers: int, num_heads: int,
                 head_dim: int, block_size: int, dtype: torch.dtype):
        self.cache = torch.zeros(
            num_blocks, 2, num_layers, num_heads, block_size, head_dim,
            dtype=dtype, device="cuda"
        )
        self.block_size = block_size
    
    def update(self, key: torch.Tensor, value: torch.Tensor, 
               layer_idx: int, slot_mapping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store new KV pairs
        self.store_kv(key, value, layer_idx, slot_mapping)
        
        # Retrieve all KV pairs for attention
        return self.fetch_kv(layer_idx, slot_mapping)
```

### Efficient KV Storage with Triton

```python
import triton
import triton.language as tl

@triton.jit
def store_kvcache_kernel(
    key, value, key_cache, value_cache,
    slot_mapping, 
    stride_k_bs, stride_k_h, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_d,
    stride_kc_bl, stride_kc_h, stride_kc_bs, stride_kc_d,
    stride_vc_bl, stride_vc_h, stride_vc_bs, stride_vc_d,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Efficient kernel to store KV pairs in cache
    cur_seq_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    
    # Load slot for this sequence position
    slot = tl.load(slot_mapping + cur_seq_idx)
    
    # Calculate offsets
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load and store key
    k = tl.load(key + cur_seq_idx * stride_k_bs + cur_head_idx * stride_k_h + offs_d)
    kc_ptrs = key_cache + slot * stride_kc_bl + cur_head_idx * stride_kc_h + offs_d
    tl.store(kc_ptrs, k)
    
    # Load and store value
    v = tl.load(value + cur_seq_idx * stride_v_bs + cur_head_idx * stride_v_h + offs_d)
    vc_ptrs = value_cache + slot * stride_vc_bl + cur_head_idx * stride_vc_h + offs_d
    tl.store(vc_ptrs, v)
```

## Complete Model Assembly

### Building the Full Model

```python
class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (can be tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    
    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache.get_layer_cache(idx) if kv_cache else None
            )
        
        # Final norm and output projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
```

## Tensor Parallelism Support

### Parallel Linear Layers

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 tp_size: int, tp_rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Each rank gets a slice of output features
        self.out_features_per_rank = out_features // tp_size
        
        self.weight = nn.Parameter(torch.empty(
            self.out_features_per_rank, in_features
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local computation
        output = F.linear(x, self.weight, self.bias)
        
        # No communication needed for column parallel
        return output
```

### Parallel Attention

```python
class ParallelAttention(MultiHeadAttention):
    def __init__(self, config: ModelConfig, tp_size: int, tp_rank: int):
        super().__init__(config)
        
        # Split heads across tensor parallel ranks
        assert config.num_attention_heads % tp_size == 0
        self.num_heads = config.num_attention_heads // tp_size
        
        # Replace projections with parallel versions
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.attention_bias
        )
        
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.attention_bias
        )
```

## Optimization Techniques

### 1. Fused Operations

```python
class FusedMLP(nn.Module):
    """MLP with fused operations for better performance."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Fuse gate and up projections
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 
            2 * config.intermediate_size, 
            bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, 
            config.hidden_size, 
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matrix multiply instead of two
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Fused activation and multiply
        intermediate = F.silu(gate) * up
        
        return self.down_proj(intermediate)
```

### 2. Kernel Fusion with Triton

```python
@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Fused attention computation
    # Combines QK^T computation, softmax, and QK^T @ V in one kernel
    # Implementation details omitted for brevity
    pass
```

### 3. Memory-Efficient Attention

```python
def memory_efficient_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             chunk_size: int = 512) -> torch.Tensor:
    """Compute attention in chunks to save memory."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Process attention in chunks
    output = torch.zeros_like(q)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[:, :, i:end_i]
        
        # Compute attention scores for this chunk
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Causal mask
        if i == 0:
            mask = torch.triu(torch.ones(end_i - i, seq_len), diagonal=1)
            scores.masked_fill_(mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output[:, :, i:end_i] = torch.matmul(attn_weights, v)
    
    return output
```

## Testing and Validation

### Unit Tests

```python
def test_attention_layer():
    config = ModelConfig(
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=2048
    )
    
    attention = MultiHeadAttention(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    output = attention(x)
    
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(output).any()
    
    # Test with KV cache
    kv_cache = KVCache(...)
    output_with_cache = attention(x[:, -1:], kv_cache=kv_cache)
    assert output_with_cache.shape == (batch_size, 1, config.hidden_size)
```

### Performance Benchmarks

```python
def benchmark_transformer_layer():
    config = ModelConfig(hidden_size=4096, num_attention_heads=32)
    layer = TransformerLayer(config).cuda()
    
    # Warmup
    x = torch.randn(1, 2048, 4096, device='cuda')
    for _ in range(10):
        _ = layer(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        _ = layer(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Time per forward pass: {elapsed / 100 * 1000:.2f} ms")
    print(f"Throughput: {100 * 2048 / elapsed:.2f} tokens/s")
```

## Best Practices

### 1. Numerical Stability

```python
def stable_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract max for numerical stability
    scores_max = scores.max(dim=dim, keepdim=True)[0]
    scores = scores - scores_max
    
    # Compute exponentials
    scores_exp = torch.exp(scores)
    
    # Normalize
    return scores_exp / scores_exp.sum(dim=dim, keepdim=True)
```

### 2. Gradient Checkpointing Support

```python
class CheckpointedTransformerLayer(TransformerLayer):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(super().forward, x, **kwargs)
        else:
            return super().forward(x, **kwargs)
```

### 3. Mixed Precision

```python
class MixedPrecisionTransformer(TransformerModel):
    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return super().forward(*args, **kwargs)
```

## Conclusion

Implementing efficient transformer layers requires careful attention to:

1. **Memory efficiency**: KV cache integration and fused operations
2. **Computational efficiency**: Flash attention and kernel fusion
3. **Scalability**: Tensor parallelism support
4. **Numerical stability**: Proper normalization and precision handling
5. **Flexibility**: Support for different architectures and optimizations

These implementations form the foundation of high-performance inference engines. Next, we'll explore how to implement tensor parallelism to scale across multiple GPUs.