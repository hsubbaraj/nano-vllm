# Memory, Attention, and KV Caching

## The Attention Mechanism Revisited

In transformers, attention allows each token to attend to all previous tokens. For a sequence of length `N`, the attention computation involves:

```python
Q = input @ W_q  # Query: [N, d_model] @ [d_model, d_head] = [N, d_head]
K = input @ W_k  # Key: [N, d_model] @ [d_model, d_head] = [N, d_head]  
V = input @ W_v  # Value: [N, d_model] @ [d_model, d_head] = [N, d_head]

attention_scores = Q @ K.T / sqrt(d_head)  # [N, N]
attention_weights = softmax(attention_scores)  # [N, N]
output = attention_weights @ V  # [N, N] @ [N, d_head] = [N, d_head]
```

## The KV Cache Problem

### Without KV Cache (Naive Approach)
For each new token, we must:
1. Recompute K and V for ALL previous tokens
2. Store and process the full attention matrix
3. Computational complexity: O(N²) for each new token

```
Token 1: Compute K₁, V₁
Token 2: Recompute K₁, V₁, compute K₂, V₂  
Token 3: Recompute K₁, V₁, K₂, V₂, compute K₃, V₃
...
```

This is extremely inefficient - we're recomputing the same K, V values repeatedly!

### With KV Cache (Optimized Approach)
Store previously computed K, V values and only compute new ones:

```
Token 1: Compute K₁, V₁ → Cache[K₁, V₁]
Token 2: Load K₁, V₁ from cache, compute K₂, V₂ → Cache[K₁, V₁, K₂, V₂]
Token 3: Load K₁, V₁, K₂, V₂ from cache, compute K₃, V₃ → Cache[K₁, V₁, K₂, V₂, K₃, V₃]
```

## KV Cache Memory Layout

### Per-Layer Storage
For each transformer layer, we store:
```python
# Shape: [num_tokens, num_kv_heads, head_dim]
key_cache = torch.zeros(max_tokens, num_kv_heads, head_dim)
value_cache = torch.zeros(max_tokens, num_kv_heads, head_dim)
```

### Memory Requirements
For a model with:
- `L` layers
- `H` attention heads (KV heads may be less due to Grouped Query Attention)
- `D` head dimension
- `T` sequence length
- `dtype` precision (e.g., float16 = 2 bytes)

**Total KV Cache Memory = 2 × L × H × D × T × sizeof(dtype)**

Example for Llama-7B with 2048 tokens:
- 32 layers × 32 heads × 128 head_dim × 2048 tokens × 2 bytes × 2 (K+V) = **1.07 GB per sequence**

## Memory Management Challenges

### 1. Variable Sequence Lengths
Different requests have different sequence lengths, making memory allocation complex:
```
Request A: 100 tokens → needs 52MB
Request B: 1500 tokens → needs 781MB  
Request C: 300 tokens → needs 156MB
```

Simple allocation would waste enormous amounts of memory with padding.

### 2. Dynamic Growth
Sequences grow during generation:
```
Initial: "What is the capital" (5 tokens)
After generation: "What is the capital of France? The capital of France is Paris." (15 tokens)
```

Memory must be allocated/reallocated as sequences grow.

### 3. Memory Fragmentation
As sequences finish and new ones start, memory becomes fragmented:
```
Memory: [Seq A: 100 tokens][FREE: 200][Seq B: 300 tokens][FREE: 150][Seq C: 250 tokens]
```

## Block-Based KV Cache Management

Modern inference engines use **block-based memory management** similar to virtual memory in operating systems.

### Key Concepts

#### Memory Blocks
- Fixed-size chunks (typically 256 tokens worth of KV cache)
- Only allocate blocks as needed
- Share blocks between sequences when possible

#### Block Table  
Each sequence maintains a block table mapping logical positions to physical blocks:
```python
# Sequence with 800 tokens needs 4 blocks (800 ÷ 256 = 3.125 → 4 blocks)
block_table = [physical_block_7, physical_block_12, physical_block_3, physical_block_19]
```

#### Block Pool
Global pool of available blocks:
```python
free_blocks = [0, 1, 2, 5, 6, 8, 9, 11, ...]  # Available block IDs
allocated_blocks = {3: seq_A, 7: seq_B, 12: seq_C, ...}  # Block ID → Sequence mapping
```

### Block-Based Access Pattern

```python
def get_kv_cache(sequence_id, token_position):
    block_table = sequence_block_tables[sequence_id]
    block_index = token_position // BLOCK_SIZE  # Which block?
    within_block_offset = token_position % BLOCK_SIZE  # Position within block
    
    physical_block = block_table[block_index]
    return kv_cache[physical_block][within_block_offset]
```

### Advantages of Block-Based Management

1. **No Fragmentation**: Blocks are uniform size, no wasted space
2. **Dynamic Allocation**: Allocate blocks only as sequences grow
3. **Efficient Sharing**: Multiple sequences can share identical blocks (prefix caching)
4. **Simple Bookkeeping**: Block tables are small and efficient

## Attention Computation with KV Cache

### Prefill Phase
Process entire input sequence, populate KV cache:

```python
def prefill_attention(input_tokens, layer):
    # Compute Q, K, V for all input tokens
    Q = input_tokens @ layer.W_q  # [seq_len, num_heads, head_dim]
    K = input_tokens @ layer.W_k  # [seq_len, num_kv_heads, head_dim] 
    V = input_tokens @ layer.W_v  # [seq_len, num_kv_heads, head_dim]
    
    # Store K, V in cache
    store_kv_cache(K, V, sequence_id, start_pos=0)
    
    # Compute attention over full sequence
    return flash_attention(Q, K, V)  # Uses Flash Attention for efficiency
```

### Decode Phase
Generate one token, append to KV cache:

```python
def decode_attention(new_token, layer, sequence_id):
    # Compute Q, K, V only for the new token
    q = new_token @ layer.W_q  # [1, num_heads, head_dim]
    k = new_token @ layer.W_k  # [1, num_kv_heads, head_dim]
    v = new_token @ layer.W_v  # [1, num_kv_heads, head_dim]
    
    # Append new K, V to cache
    append_kv_cache(k, v, sequence_id)
    
    # Load all previous K, V from cache
    all_K, all_V = load_kv_cache(sequence_id)  # [current_len, num_kv_heads, head_dim]
    
    # Compute attention with new query against all keys/values
    return flash_attention_decode(q, all_K, all_V)
```

## Grouped Query Attention (GQA)

Many modern models use Grouped Query Attention to reduce KV cache size:

### Multi-Head Attention (MHA)
- Query heads: 32
- Key heads: 32  
- Value heads: 32
- **KV Cache**: Full size

### Grouped Query Attention (GQA)
- Query heads: 32
- Key heads: 8 (groups of 4 queries share same K, V)
- Value heads: 8
- **KV Cache**: 4× smaller!

```python
# In GQA, multiple query heads share the same key/value heads
num_query_heads = 32
num_kv_heads = 8  # 32 ÷ 8 = 4 queries per KV head
head_groups = num_query_heads // num_kv_heads  # 4

# During attention, replicate K, V for each group
K_expanded = K.repeat_interleave(head_groups, dim=1)  # [seq_len, 32, head_dim]  
V_expanded = V.repeat_interleave(head_groups, dim=1)  # [seq_len, 32, head_dim]
```

## Memory Access Patterns

### Sequential Access (Prefill)
During prefill, we access KV cache sequentially:
```
Access pattern: K₁, K₂, K₃, ..., Kₙ
Cache behavior: Good spatial locality, efficient memory bandwidth usage
```

### Growing Access (Decode)
During decode, we access all previous tokens:
```
Step 1: K₁
Step 2: K₁, K₂  
Step 3: K₁, K₂, K₃
Step n: K₁, K₂, K₃, ..., Kₙ
```

This growing access pattern becomes memory-bound as sequences get longer.

## Flash Attention Integration

Flash Attention optimizes memory access patterns:

### Standard Attention Problem
```python
# Memory usage: O(N²) for attention matrix
attention_scores = Q @ K.T  # [N, N] matrix - huge memory!
attention_weights = softmax(attention_scores)  # Still [N, N]
output = attention_weights @ V  # [N, N] @ [N, d] = [N, d]
```

### Flash Attention Solution
- **Tile-based computation**: Process attention in smaller tiles
- **Online softmax**: Compute softmax incrementally without storing full matrix
- **Fused kernels**: Combine multiple operations to reduce memory traffic

```python
def flash_attention_tiled(Q, K, V, tile_size=64):
    output = torch.zeros_like(Q)
    max_vals = torch.full((Q.size(0),), float('-inf'))
    sum_exp = torch.zeros(Q.size(0))
    
    # Process in tiles to avoid storing full attention matrix
    for k_start in range(0, K.size(0), tile_size):
        k_tile = K[k_start:k_start+tile_size]
        v_tile = V[k_start:k_start+tile_size]
        
        # Compute attention scores for this tile only
        scores_tile = Q @ k_tile.T  # [N, tile_size] instead of [N, N]
        
        # Update running softmax statistics
        # ... (online softmax computation)
```

## Prefix Caching: Advanced KV Sharing

When multiple requests share common prefixes, we can share KV cache blocks:

### Common Prefix Example
```
Request A: "Translate to French: Hello world"
Request B: "Translate to French: Good morning"  
Request C: "Translate to French: How are you?"

Shared prefix: "Translate to French:" → Can share KV cache blocks!
```

### Content-Based Block Sharing
```python
# Each block has a content hash based on the tokens it contains
def compute_block_hash(tokens):
    return xxhash.xxh64(' '.join(map(str, tokens))).hexdigest()

# Example hashes
prefix_tokens = [1234, 5678, 9012]  # "Translate to French:"
block_hash = compute_block_hash(prefix_tokens)  # "abc123def456"

# Multiple sequences can reference the same physical block
shared_block_table = {
    "abc123def456": physical_block_5  # Shared by multiple sequences
}
```

### Memory Savings
With 100 concurrent requests sharing a 50-token system prompt:
- **Without sharing**: 100 × 50 = 5,000 tokens of KV cache
- **With sharing**: 50 tokens of KV cache (99% reduction!)

## Implementation in nano-vLLM

nano-vLLM's block manager (`nanovllm/engine/block_manager.py`) implements sophisticated KV cache management:

```python
class BlockManager:
    def __init__(self, block_size=256):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))  # Pool of available blocks
        self.block_table = {}  # sequence_id → [block_ids]
        self.content_hash_to_block = {}  # content_hash → block_id (for sharing)
        self.block_ref_count = {}  # block_id → reference_count
```

Key features:
1. **Content-based sharing**: Identical token sequences share physical blocks
2. **Reference counting**: Automatic cleanup when sequences finish
3. **Hash chaining**: Efficient lookup for prefix matching
4. **Memory pool management**: Efficient allocation and deallocation

## Key Takeaways

1. **KV Cache is Essential**: Without it, inference would be O(N²) per token
2. **Memory is the Bottleneck**: KV cache memory often limits batch size more than compute
3. **Block-based Management**: Modern engines use virtual memory techniques
4. **Attention Variants Matter**: GQA significantly reduces memory requirements
5. **Sharing Opportunities**: Prefix caching can dramatically reduce memory usage
6. **Access Patterns Drive Design**: Different phases have different memory access patterns

Understanding KV cache management is crucial for building efficient inference engines. The next section will explore how to leverage parallelism to scale beyond single-GPU limitations.