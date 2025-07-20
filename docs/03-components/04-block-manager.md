# Block Manager: Advanced KV Cache Management

## Overview

The Block Manager is a critical component that manages the allocation, deallocation, and sharing of KV cache memory blocks. It implements sophisticated memory management techniques including reference counting, content-based deduplication (prefix caching), and efficient block allocation strategies.

This guide explores the Block Manager's design and implementation, showing how it enables efficient memory usage in production LLM inference engines.

## The Need for Block Management

### Problems with Naive KV Cache Management

Without proper management, KV cache faces several issues:

1. **Memory Fragmentation**: Variable-length sequences create gaps
2. **Memory Waste**: Over-allocation for maximum sequence length
3. **No Sharing**: Duplicate content wastes memory
4. **Complex Allocation**: Hard to track what memory is free

### Block-Based Solution

The Block Manager solves these by:
- Dividing memory into fixed-size blocks
- Tracking block usage with reference counting
- Enabling sharing through content hashing
- Maintaining efficient free lists

## Core Architecture

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 256):
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Block tracking
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks
        
        # Content-based deduplication
        self.cached_blocks = {}  # hash -> block_id
        self.block_hashes = {}   # block_id -> hash
        
        # Statistics
        self.stats = BlockManagerStats()
```

## Block Allocation Strategies

### Basic Allocation

```python
def allocate_block(self) -> Optional[int]:
    """Allocate a single block from the free list."""
    if not self.free_blocks:
        return None
    
    block_id = self.free_blocks.pop()
    self.ref_counts[block_id] = 1
    self.stats.allocated_blocks += 1
    
    return block_id

def allocate_blocks(self, num_blocks: int) -> List[int]:
    """Allocate multiple blocks, returning None if not enough available."""
    if len(self.free_blocks) < num_blocks:
        return None
    
    blocks = []
    for _ in range(num_blocks):
        block = self.allocate_block()
        if block is None:
            # Rollback on failure
            for b in blocks:
                self.free_block(b)
            return None
        blocks.append(block)
    
    return blocks
```

### Sequence Allocation

For allocating blocks for an entire sequence:

```python
def allocate_for_sequence(self, sequence_length: int) -> List[int]:
    """Allocate blocks for a sequence of given length."""
    num_blocks_needed = (sequence_length + self.block_size - 1) // self.block_size
    
    blocks = self.allocate_blocks(num_blocks_needed)
    if blocks is None:
        raise OutOfMemoryError(f"Cannot allocate {num_blocks_needed} blocks")
    
    return blocks
```

## Reference Counting

### Managing Block References

```python
def add_reference(self, block_id: int):
    """Increment reference count for a block."""
    assert 0 <= block_id < self.num_blocks
    assert self.ref_counts[block_id] > 0, "Adding reference to freed block"
    
    self.ref_counts[block_id] += 1
    self.stats.reference_added()

def remove_reference(self, block_id: int):
    """Decrement reference count, freeing if it reaches zero."""
    assert 0 <= block_id < self.num_blocks
    assert self.ref_counts[block_id] > 0
    
    self.ref_counts[block_id] -= 1
    
    if self.ref_counts[block_id] == 0:
        self.free_block(block_id)
```

### Safe Block Sharing

```python
def share_blocks(self, source_blocks: List[int], 
                 start_idx: int = 0) -> List[int]:
    """Share blocks from another sequence, incrementing ref counts."""
    shared_blocks = []
    
    for block_id in source_blocks[start_idx:]:
        self.add_reference(block_id)
        shared_blocks.append(block_id)
    
    return shared_blocks
```

## Content-Based Deduplication (Prefix Caching)

### Hash-Based Block Identification

```python
def compute_block_hash(self, tokens: List[int], 
                      kv_data: Optional[torch.Tensor] = None) -> int:
    """Compute hash for block content."""
    hasher = xxhash.xxh64()
    
    # Hash tokens
    token_bytes = struct.pack(f'{len(tokens)}I', *tokens)
    hasher.update(token_bytes)
    
    # Optionally hash KV data for exact matching
    if kv_data is not None:
        hasher.update(kv_data.cpu().numpy().tobytes())
    
    return hasher.intdigest()

def find_cached_blocks(self, token_blocks: List[List[int]]) -> List[Optional[int]]:
    """Find cached blocks matching the given token blocks."""
    cached_blocks = []
    
    for tokens in token_blocks:
        block_hash = self.compute_block_hash(tokens)
        
        if block_hash in self.cached_blocks:
            block_id = self.cached_blocks[block_hash]
            # Verify block is still valid
            if self.ref_counts[block_id] > 0:
                cached_blocks.append(block_id)
                self.stats.cache_hit()
            else:
                # Stale cache entry
                del self.cached_blocks[block_hash]
                cached_blocks.append(None)
                self.stats.cache_miss()
        else:
            cached_blocks.append(None)
            self.stats.cache_miss()
    
    return cached_blocks
```

### Allocating with Prefix Caching

```python
def allocate_with_caching(self, token_ids: List[int]) -> Tuple[List[int], int]:
    """
    Allocate blocks for tokens, reusing cached blocks where possible.
    Returns (block_ids, num_cached_tokens).
    """
    # Divide tokens into blocks
    token_blocks = []
    for i in range(0, len(token_ids), self.block_size):
        token_blocks.append(token_ids[i:i + self.block_size])
    
    # Find cached blocks
    cached_blocks = self.find_cached_blocks(token_blocks)
    
    # Allocate blocks
    allocated_blocks = []
    num_cached_tokens = 0
    
    for i, (tokens, cached_block) in enumerate(zip(token_blocks, cached_blocks)):
        if cached_block is not None:
            # Reuse cached block
            self.add_reference(cached_block)
            allocated_blocks.append(cached_block)
            num_cached_tokens += len(tokens)
        else:
            # Allocate new block
            new_block = self.allocate_block()
            if new_block is None:
                # Rollback and fail
                for block in allocated_blocks[:i]:
                    self.remove_reference(block)
                raise OutOfMemoryError("Cannot allocate block")
            
            allocated_blocks.append(new_block)
            
            # Add to cache
            block_hash = self.compute_block_hash(tokens)
            self.cached_blocks[block_hash] = new_block
            self.block_hashes[new_block] = block_hash
    
    return allocated_blocks, num_cached_tokens
```

### Cache Invalidation

```python
def invalidate_block_cache(self, block_id: int):
    """Remove a block from the content cache."""
    if block_id in self.block_hashes:
        block_hash = self.block_hashes[block_id]
        
        # Remove from cache maps
        if block_hash in self.cached_blocks:
            del self.cached_blocks[block_hash]
        del self.block_hashes[block_id]

def free_block(self, block_id: int):
    """Free a block and return it to the pool."""
    assert self.ref_counts[block_id] == 0
    
    # Remove from cache
    self.invalidate_block_cache(block_id)
    
    # Return to free list
    self.free_blocks.append(block_id)
    self.stats.freed_blocks += 1
```

## Memory Preemption Support

### Eviction Strategies

```python
def get_eviction_candidates(self, num_blocks_needed: int) -> List[int]:
    """Find sequences that can be evicted to free memory."""
    # This would typically be implemented by the scheduler
    # The block manager provides support functions
    
    blocks_by_sequence = self.get_blocks_by_sequence()
    eviction_candidates = []
    blocks_freed = 0
    
    # Sort by eviction priority (e.g., LRU, size)
    for seq_id, blocks in sorted(blocks_by_sequence.items(), 
                                key=lambda x: len(x[1])):
        eviction_candidates.append(seq_id)
        blocks_freed += len(blocks)
        
        if blocks_freed >= num_blocks_needed:
            break
    
    return eviction_candidates

def free_sequence_blocks(self, block_table: List[int]):
    """Free all blocks belonging to a sequence."""
    for block_id in block_table:
        self.remove_reference(block_id)
```

### Supporting Swapping

For engines that support CPU-GPU swapping:

```python
class SwappableBlockManager(BlockManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_blocks = {}  # block_id -> cpu_tensor
        self.swap_out_queue = []
        self.swap_in_queue = []
    
    def swap_out_blocks(self, block_ids: List[int]) -> List[CPUBlock]:
        """Move blocks from GPU to CPU memory."""
        cpu_blocks = []
        
        for block_id in block_ids:
            # Copy to CPU
            gpu_data = self.get_block_data(block_id)
            cpu_data = gpu_data.cpu()
            
            # Store CPU block
            cpu_block = CPUBlock(block_id, cpu_data)
            self.cpu_blocks[block_id] = cpu_block
            cpu_blocks.append(cpu_block)
            
            # Free GPU block
            self.free_block(block_id)
        
        return cpu_blocks
    
    def swap_in_blocks(self, cpu_blocks: List[CPUBlock]) -> List[int]:
        """Move blocks from CPU back to GPU."""
        gpu_blocks = []
        
        for cpu_block in cpu_blocks:
            # Allocate GPU block
            gpu_block_id = self.allocate_block()
            if gpu_block_id is None:
                # Need to evict something
                raise OutOfMemoryError("Cannot allocate for swap-in")
            
            # Copy to GPU
            gpu_data = self.get_block_data(gpu_block_id)
            gpu_data.copy_(cpu_block.data.cuda())
            
            gpu_blocks.append(gpu_block_id)
            
            # Clean up CPU block
            del self.cpu_blocks[cpu_block.original_id]
        
        return gpu_blocks
```

## Advanced Features

### Block Coalescing

Merge adjacent free blocks to reduce fragmentation:

```python
def coalesce_free_blocks(self):
    """Merge adjacent free blocks for better allocation."""
    if not self.supports_coalescing:
        return
    
    # Sort free blocks
    self.free_blocks.sort()
    
    # Find consecutive runs
    coalesced = []
    current_run = [self.free_blocks[0]]
    
    for block in self.free_blocks[1:]:
        if block == current_run[-1] + 1:
            current_run.append(block)
        else:
            coalesced.append(current_run)
            current_run = [block]
    
    coalesced.append(current_run)
    
    # Create super-blocks from runs
    self.create_superblocks(coalesced)
```

### Multi-Level Block Sizes

Support different block sizes for different use cases:

```python
class MultiLevelBlockManager:
    def __init__(self):
        self.levels = {
            'small': BlockPool(block_size=16, num_blocks=1000),
            'medium': BlockPool(block_size=64, num_blocks=500),
            'large': BlockPool(block_size=256, num_blocks=200)
        }
    
    def allocate_adaptive(self, sequence_length: int) -> List[Block]:
        """Allocate using appropriate block sizes."""
        blocks = []
        remaining = sequence_length
        
        # Use large blocks first
        for level in ['large', 'medium', 'small']:
            pool = self.levels[level]
            block_size = pool.block_size
            
            while remaining >= block_size and pool.has_free_blocks():
                block = pool.allocate()
                blocks.append(block)
                remaining -= block_size
        
        if remaining > 0:
            # Try smallest size for remainder
            if self.levels['small'].has_free_blocks():
                blocks.append(self.levels['small'].allocate())
            else:
                # Rollback
                for block in blocks:
                    block.free()
                raise OutOfMemoryError()
        
        return blocks
```

### NUMA-Aware Allocation

For multi-socket systems:

```python
class NUMABlockManager(BlockManager):
    def __init__(self, numa_node: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numa_node = numa_node
        
        # Allocate on specific NUMA node
        with set_numa_affinity(numa_node):
            self.initialize_blocks()
    
    def allocate_block(self, preferred_numa: Optional[int] = None) -> int:
        if preferred_numa == self.numa_node:
            # Fast path - local allocation
            return super().allocate_block()
        else:
            # Cross-NUMA allocation - track for migration
            block = super().allocate_block()
            if block is not None:
                self.mark_for_migration(block, preferred_numa)
            return block
```

## Monitoring and Debugging

### Statistics Collection

```python
class BlockManagerStats:
    def __init__(self):
        self.allocated_blocks = 0
        self.freed_blocks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.peak_usage = 0
        self.allocation_failures = 0
        
    def update_peak_usage(self, current_usage: int):
        self.peak_usage = max(self.peak_usage, current_usage)
    
    def get_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def get_fragmentation(self, block_manager) -> float:
        # Internal fragmentation within allocated blocks
        total_allocated = block_manager.get_num_allocated_blocks()
        total_used = block_manager.get_num_used_tokens()
        total_capacity = total_allocated * block_manager.block_size
        
        return 1.0 - (total_used / total_capacity) if total_capacity > 0 else 0.0
```

### Debug Visualization

```python
def visualize_memory_state(self):
    """Create visual representation of memory state."""
    rows = []
    blocks_per_row = 32
    
    for i in range(0, self.num_blocks, blocks_per_row):
        row = []
        for j in range(blocks_per_row):
            block_id = i + j
            if block_id >= self.num_blocks:
                break
                
            if self.ref_counts[block_id] == 0:
                row.append('.')  # Free
            elif self.ref_counts[block_id] == 1:
                row.append('#')  # Allocated
            else:
                row.append(str(min(9, self.ref_counts[block_id])))  # Shared
        
        rows.append(''.join(row))
    
    print("\nMemory State Visualization:")
    print("Legend: . = free, # = allocated, 2-9 = shared (ref count)")
    for row in rows:
        print(row)
    
    print(f"\nStats: {self.get_num_free_blocks()}/{self.num_blocks} free")
    print(f"Cache hit rate: {self.stats.get_cache_hit_rate():.2%}")
```

### Memory Leak Detection

```python
class LeakDetector:
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.sequence_blocks = {}  # seq_id -> set(block_ids)
    
    def register_sequence(self, seq_id: str, blocks: List[int]):
        self.sequence_blocks[seq_id] = set(blocks)
    
    def unregister_sequence(self, seq_id: str):
        if seq_id in self.sequence_blocks:
            blocks = self.sequence_blocks[seq_id]
            
            # Check all blocks were freed
            for block_id in blocks:
                if self.block_manager.ref_counts[block_id] > 0:
                    print(f"LEAK: Block {block_id} still has {self.block_manager.ref_counts[block_id]} refs")
            
            del self.sequence_blocks[seq_id]
    
    def check_consistency(self):
        # Verify ref counts match tracked sequences
        actual_refs = defaultdict(int)
        
        for blocks in self.sequence_blocks.values():
            for block_id in blocks:
                actual_refs[block_id] += 1
        
        for block_id in range(self.block_manager.num_blocks):
            expected = self.block_manager.ref_counts[block_id]
            actual = actual_refs.get(block_id, 0)
            
            if expected != actual:
                print(f"INCONSISTENCY: Block {block_id} has {expected} refs but tracked {actual}")
```

## Best Practices

### 1. Choose Appropriate Block Size

```python
def calculate_optimal_block_size(model_config, typical_sequence_length):
    # Balance between granularity and overhead
    kv_per_token = 2 * model_config.num_layers * model_config.hidden_size
    
    # Aim for blocks that are 1-10MB
    target_block_memory = 5 * 1024 * 1024  # 5MB
    tokens_per_block = target_block_memory // (kv_per_token * 2)  # FP16
    
    # Round to power of 2
    block_size = 2 ** int(math.log2(tokens_per_block))
    
    # Constrain to reasonable range
    return max(16, min(512, block_size))
```

### 2. Implement Proper Cleanup

```python
def cleanup_sequence(self, sequence):
    try:
        # Free blocks
        for block_id in sequence.block_table:
            self.remove_reference(block_id)
        
        # Clear sequence state
        sequence.block_table.clear()
        
        # Update statistics
        self.stats.sequences_completed += 1
        
    except Exception as e:
        logger.error(f"Error cleaning up sequence {sequence.id}: {e}")
        # Don't re-raise - avoid memory leaks
```

### 3. Monitor Memory Health

```python
def health_check(self):
    issues = []
    
    # Check fragmentation
    frag = self.stats.get_fragmentation(self)
    if frag > 0.5:
        issues.append(f"High fragmentation: {frag:.2%}")
    
    # Check cache effectiveness
    hit_rate = self.stats.get_cache_hit_rate()
    if hit_rate < 0.1:
        issues.append(f"Low cache hit rate: {hit_rate:.2%}")
    
    # Check for leaks
    if self.stats.allocated_blocks - self.stats.freed_blocks > self.num_blocks * 0.9:
        issues.append("Possible memory leak detected")
    
    return issues
```

### 4. Test Edge Cases

```python
def test_block_manager():
    bm = BlockManager(num_blocks=10, block_size=4)
    
    # Test allocation
    blocks1 = bm.allocate_blocks(3)
    assert len(blocks1) == 3
    assert bm.get_num_free_blocks() == 7
    
    # Test sharing
    bm.add_reference(blocks1[0])
    assert bm.ref_counts[blocks1[0]] == 2
    
    # Test deallocation
    bm.remove_reference(blocks1[0])
    assert bm.ref_counts[blocks1[0]] == 1
    bm.remove_reference(blocks1[0])
    assert bm.ref_counts[blocks1[0]] == 0
    assert bm.get_num_free_blocks() == 8
    
    # Test OOM
    blocks2 = bm.allocate_blocks(10)
    assert blocks2 is None
    
    # Test prefix caching
    # ... more tests
```

## Integration with Other Components

### Working with Scheduler

```python
class SchedulerBlockInterface:
    def __init__(self, block_manager):
        self.block_manager = block_manager
    
    def can_allocate_sequence(self, prompt_tokens: int, max_tokens: int) -> bool:
        total_tokens = prompt_tokens + max_tokens
        required_blocks = (total_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
        
        return self.block_manager.get_num_free_blocks() >= required_blocks
    
    def allocate_sequence(self, sequence) -> bool:
        try:
            blocks, cached_tokens = self.block_manager.allocate_with_caching(
                sequence.prompt_tokens
            )
            sequence.block_table = blocks
            sequence.cached_tokens = cached_tokens
            return True
        except OutOfMemoryError:
            return False
```

### Working with Model Runner

```python
class ModelRunnerBlockInterface:
    def __init__(self, block_manager):
        self.block_manager = block_manager
    
    def get_block_addresses(self, sequence) -> List[int]:
        """Convert logical blocks to physical addresses."""
        addresses = []
        
        for block_id in sequence.block_table:
            base_addr = block_id * self.block_manager.block_size
            addresses.append(base_addr)
        
        return addresses
```

## Conclusion

The Block Manager is a sophisticated component that enables efficient KV cache usage through:

1. **Fixed-size blocks** eliminate fragmentation
2. **Reference counting** enables safe sharing
3. **Content hashing** provides automatic deduplication  
4. **Flexible allocation** supports various strategies
5. **Rich monitoring** helps identify issues

A well-designed Block Manager is essential for production LLM inference engines, enabling them to handle more concurrent requests with less memory. Next, we'll explore sequence management, which builds on the Block Manager to track request state throughout their lifecycle.