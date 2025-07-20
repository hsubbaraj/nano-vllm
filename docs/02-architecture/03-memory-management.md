# Memory Management in LLM Inference Engines

## Overview

Memory management is one of the most critical aspects of building a high-performance LLM inference engine. Unlike training, where memory usage is relatively predictable, inference must handle:

- Variable-length prompts and outputs
- Multiple concurrent requests with different memory requirements
- Dynamic memory allocation and deallocation as sequences progress
- Efficient sharing of common prefixes across requests

This guide explores how modern inference engines manage memory, with a focus on the KV cache which typically consumes 80-90% of GPU memory during inference.

## The Memory Challenge

### Memory Components in LLM Inference

1. **Model Weights**: Fixed size, loaded once
   - For a 7B parameter model with FP16: ~14GB
   - Can be quantized to reduce size

2. **Activations**: Temporary buffers for computation
   - Relatively small compared to other components
   - Can be reused across batches

3. **KV Cache**: The dominant memory consumer
   - Grows with sequence length and batch size
   - For a 7B model: ~2GB per 2048 tokens at batch size 1
   - Formula: `2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_size`

### Why KV Cache Dominates

During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, we'd need to recompute key and value vectors for all previous tokens at each step, making generation O(n²) in computation.

## Block-Based Memory Management

### The Problem with Continuous Allocation

Naive approaches allocate continuous memory buffers for each sequence:
```python
# Naive approach - leads to fragmentation
kv_cache[seq_id] = torch.zeros(max_seq_len, ...)
```

Problems:
- Internal fragmentation: Most sequences don't use full allocated length
- External fragmentation: Gaps between allocations can't be used
- No sharing: Duplicate prefixes waste memory

### Block-Based Solution

Modern engines like vLLM and nano-vLLM use block-based allocation:

```python
class Block:
    def __init__(self, block_size: int = 256):
        self.block_size = block_size
        self.ref_count = 0
        self.last_accessed = time.time()
```

Benefits:
- **Fixed-size blocks**: Eliminates external fragmentation
- **Reference counting**: Enables sharing between sequences
- **Fine-grained allocation**: Only allocate what's needed

### Implementation Details

1. **Block Size Selection**:
   - Common sizes: 16, 32, 64, 128, 256 tokens
   - Trade-off: Smaller blocks reduce waste but increase metadata overhead
   - nano-vLLM uses 256 tokens per block

2. **Block Table Structure**:
   ```python
   # Each sequence maintains a list of block IDs
   sequence.block_table = [block_id_0, block_id_1, ...]
   
   # Physical blocks store actual KV data
   physical_blocks[block_id] = kv_tensor
   ```

3. **Address Translation**:
   ```python
   def get_physical_address(seq_id, token_idx):
       block_idx = token_idx // block_size
       block_offset = token_idx % block_size
       block_id = block_tables[seq_id][block_idx]
       return physical_blocks[block_id], block_offset
   ```

## Prefix Caching and Deduplication

### Content-Based Deduplication

When multiple requests share common prefixes (e.g., system prompts), we can share KV cache blocks:

```python
def allocate_block(content_hash):
    if content_hash in hash_to_block:
        block = hash_to_block[content_hash]
        block.ref_count += 1
        return block.id
    else:
        # Allocate new block
        block = allocate_new_block()
        hash_to_block[content_hash] = block
        return block.id
```

### Hashing Strategy

nano-vLLM uses xxhash for fast content hashing:
```python
def compute_block_hash(tokens, kv_data):
    hasher = xxhash.xxh64()
    hasher.update(tokens.numpy().tobytes())
    # Include KV data for exact matching
    hasher.update(kv_data.cpu().numpy().tobytes())
    return hasher.intdigest()
```

### Reference Counting

Blocks are shared using reference counting:
```python
def free_block(block_id):
    block = blocks[block_id]
    block.ref_count -= 1
    if block.ref_count == 0:
        free_list.append(block_id)
        del hash_to_block[block.content_hash]
```

## Memory Pool Management

### GPU Memory Calculation

```python
def calculate_available_kv_cache_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Reserve memory for model weights
    model_memory = sum(p.nbytes for p in model.parameters())
    
    # Reserve memory for activations (typically 1-2GB)
    activation_memory = 2 * 1024**3
    
    # Leave headroom for fragmentation
    headroom = 0.1 * total_memory
    
    available = total_memory - model_memory - activation_memory - headroom
    return max(0, available)
```

### Block Pool Initialization

```python
class BlockPool:
    def __init__(self, num_blocks, block_size, ...):
        # Pre-allocate all blocks
        self.kv_cache = torch.zeros(
            num_blocks, 2, num_layers, num_heads, 
            block_size, head_dim, dtype=dtype
        )
        
        # Free list starts with all blocks
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}
```

## Preemption and Eviction

### When Memory Runs Out

When the free block list is empty, the scheduler must:

1. **Preempt Running Sequences**:
   ```python
   def preempt_sequences(needed_blocks):
       # Sort by priority (e.g., least recently used)
       candidates = sorted(running_sequences, 
                          key=lambda s: s.last_access_time)
       
       freed_blocks = 0
       preempted = []
       
       for seq in candidates:
           if freed_blocks >= needed_blocks:
               break
           freed_blocks += len(seq.block_table)
           preempted.append(seq)
           free_sequence_blocks(seq)
       
       return preempted
   ```

2. **Swap to CPU** (Optional):
   - Some engines support swapping KV cache to CPU memory
   - Adds complexity but allows oversubscription

### Recomputation vs Swapping

Trade-offs:
- **Recomputation**: Higher latency but simpler
- **Swapping**: Lower latency but requires CPU-GPU synchronization

## Memory Allocation Strategies

### 1. Greedy Allocation
Allocate maximum possible blocks upfront:
```python
num_blocks = available_memory // block_memory_size
```

### 2. Conservative Allocation
Leave room for other operations:
```python
num_blocks = int(0.9 * available_memory // block_memory_size)
```

### 3. Dynamic Allocation
Grow pool as needed (more complex):
```python
if len(free_blocks) < threshold:
    allocate_additional_blocks()
```

## Best Practices

### 1. Profile Memory Usage
```python
def profile_memory():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free = total_memory - allocated
    
    print(f"Allocated: {allocated / 1e9:.2f} GB")
    print(f"Reserved: {reserved / 1e9:.2f} GB")
    print(f"Free: {free / 1e9:.2f} GB")
```

### 2. Monitor Fragmentation
Track internal fragmentation:
```python
utilization = used_tokens_in_blocks / (num_allocated_blocks * block_size)
```

### 3. Tune Block Size
Experiment with different block sizes for your workload:
- Smaller blocks: Better utilization, more metadata
- Larger blocks: Less metadata, potential waste

### 4. Implement Metrics
```python
class MemoryMetrics:
    def __init__(self):
        self.allocation_count = 0
        self.deallocation_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fragmentation_ratio = 0.0
```

## Advanced Techniques

### 1. NUMA Awareness
For multi-GPU systems, consider NUMA topology:
```python
def allocate_on_numa_node(gpu_id):
    numa_node = get_numa_node(gpu_id)
    with torch.cuda.device(gpu_id):
        # Allocate on correct NUMA node
        return torch.zeros(...).pin_memory()
```

### 2. Memory Defragmentation
Periodically compact memory:
```python
def defragment_memory():
    # Move all allocated blocks to beginning
    # Update block mappings
    # Free contiguous region at end
```

### 3. Predictive Preallocation
Use request patterns to predict memory needs:
```python
def predict_memory_needs(request_history):
    # Analyze patterns
    avg_seq_length = calculate_average_length(request_history)
    peak_batch_size = calculate_peak_batch(request_history)
    
    # Preallocate accordingly
    return estimate_blocks_needed(avg_seq_length, peak_batch_size)
```

## Debugging Memory Issues

### Common Problems

1. **OOM Errors**: 
   - Reduce batch size
   - Decrease max sequence length
   - Enable preemption

2. **Memory Leaks**:
   - Check reference counting
   - Verify cleanup on sequence completion

3. **Fragmentation**:
   - Monitor utilization metrics
   - Adjust block size

### Debugging Tools

```python
# Memory snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Track allocations
torch.cuda.memory._record_memory_history()

# Get detailed stats
print(torch.cuda.memory_stats())
```

## Conclusion

Effective memory management is crucial for LLM inference performance. Key takeaways:

1. **Block-based allocation** prevents fragmentation
2. **Prefix caching** dramatically reduces memory usage
3. **Reference counting** enables safe sharing
4. **Preemption** allows oversubscription
5. **Monitoring** helps identify optimization opportunities

The techniques covered here form the foundation of production inference engines. In the next section, we'll explore how scheduling strategies interact with memory management to maximize throughput.