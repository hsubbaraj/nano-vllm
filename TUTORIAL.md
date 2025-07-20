# Building an LLM Inference Engine from Scratch: A Comprehensive Tutorial

**Repository:** [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## Overview

This tutorial will guide you through building a production-quality LLM inference engine by studying and implementing the core concepts from the nano-vllm codebase. You'll learn how modern inference engines achieve high performance through advanced optimizations like PagedAttention, tensor parallelism, CUDA graphs, and prefix caching.

**Target Audience:** Intermediate programmers with basic ML knowledge  
**Language:** Python with PyTorch  
**Expected Outcome:** A working inference engine capable of running GPT-2 class models

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Basic understanding of transformers and attention mechanisms
- Familiarity with PyTorch

## Setup Instructions

### Development Environment

```bash
# Clone the repository
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch>=2.4.0 transformers>=4.51.0 triton>=3.0.0
pip install flash-attn xxhash numpy tqdm

# Install in development mode
pip install -e .
```

### Testing Framework

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run basic tests
pytest tests/ -v
```

## Learning Structure

This tutorial is organized into 8 progressive modules, each building upon the previous ones:

1. **Foundation Concepts** - Data structures and configuration
2. **Memory Management** - PagedAttention and block allocation
3. **Model Architecture** - Transformer layers and components  
4. **Attention Mechanisms** - Optimized attention with KV caching
5. **Parallelization** - Tensor parallelism for multi-GPU
6. **Request Scheduling** - Batching and memory-aware scheduling
7. **Inference Engine** - Complete execution pipeline
8. **Advanced Optimizations** - CUDA graphs and prefix caching

Each module includes:
- **Core Concepts**: Theoretical background
- **Implementation**: Step-by-step code development
- **Testing**: Unit tests to verify functionality
- **Optimization**: Performance improvements
- **Checkpoint**: Verification before proceeding

---

# Module 1: Foundation Concepts

## Learning Objectives
- Understand sequence representation and lifecycle
- Learn configuration management patterns
- Implement basic data structures for request handling

## Core Concepts

### Sequence Abstraction

In inference engines, a "sequence" represents a single generation request with its state:

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class SequenceStatus(Enum):
    WAITING = "waiting"      # Queued for processing
    RUNNING = "running"      # Currently being processed  
    FINISHED = "finished"    # Generation complete

@dataclass
class SamplingParams:
    """Controls text generation behavior"""
    temperature: float = 1.0      # Randomness (0=greedy, >0=stochastic)
    max_tokens: int = 256         # Maximum tokens to generate
    stop_tokens: List[int] = None # Tokens that end generation
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = []

class Sequence:
    """Represents a single generation request with its complete state"""
    
    def __init__(self, prompt_tokens: List[int], sampling_params: SamplingParams):
        # Core identification
        self.seq_id = id(self)  # Unique identifier
        
        # Token management
        self.prompt_tokens = prompt_tokens.copy()
        self.completion_tokens = []  # Generated tokens
        self.sampling_params = sampling_params
        
        # State tracking
        self.status = SequenceStatus.WAITING
        self.block_table = []  # Memory blocks for KV cache
        
    @property
    def all_tokens(self) -> List[int]:
        """Combined prompt and completion tokens"""
        return self.prompt_tokens + self.completion_tokens
    
    @property 
    def is_finished(self) -> bool:
        """Check if generation should stop"""
        if self.status == SequenceStatus.FINISHED:
            return True
            
        # Check length limit
        if len(self.completion_tokens) >= self.sampling_params.max_tokens:
            return True
            
        # Check stop tokens
        if (self.completion_tokens and 
            self.completion_tokens[-1] in self.sampling_params.stop_tokens):
            return True
            
        return False
    
    def add_token(self, token_id: int):
        """Add a generated token and update status"""
        self.completion_tokens.append(token_id)
        
        if self.is_finished:
            self.status = SequenceStatus.FINISHED
    
    def __len__(self) -> int:
        """Total sequence length for memory planning"""
        return len(self.all_tokens)
```

### Configuration Management

Modern inference engines require extensive configuration. Here's a clean pattern:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EngineConfig:
    """Central configuration for inference engine"""
    
    # Model configuration
    model_path: str
    dtype: str = "float16"
    trust_remote_code: bool = False
    
    # Memory management  
    block_size: int = 16           # Tokens per memory block
    max_num_blocks: int = 1024     # Total available blocks
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use
    
    # Scheduling
    max_num_seqs: int = 256        # Max concurrent sequences
    max_num_batched_tokens: int = 2048  # Max tokens per batch
    
    # Parallelism
    tensor_parallel_size: int = 1   # Number of GPUs for tensor parallelism
    
    # Performance
    enable_cuda_graph: bool = True  # Use CUDA graph optimization
    enable_prefix_caching: bool = True  # Cache common prefixes
    
    # Derived properties (computed post-init)
    vocab_size: int = field(init=False)
    hidden_size: int = field(init=False)
    num_layers: int = field(init=False)
    
    def __post_init__(self):
        # Validate configuration
        assert self.block_size > 0, "Block size must be positive"
        assert 0 < self.gpu_memory_utilization <= 1.0, "GPU utilization must be in (0,1]"
        assert self.tensor_parallel_size >= 1, "Tensor parallel size must be >= 1"
        
        # Load model config to populate derived fields
        self._load_model_config()
    
    def _load_model_config(self):
        """Load model-specific configuration from HuggingFace"""
        from transformers import AutoConfig
        
        hf_config = AutoConfig.from_pretrained(
            self.model_path, 
            trust_remote_code=self.trust_remote_code
        )
        
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size  
        self.num_layers = hf_config.num_hidden_layers
```

## Implementation Exercise

Create a simple sequence manager to practice these concepts:

```python
class SequenceManager:
    """Manages multiple sequences and their states"""
    
    def __init__(self):
        self.sequences = {}  # seq_id -> Sequence
        self.waiting_queue = []
        self.running_sequences = []
        self.finished_sequences = []
    
    def add_sequence(self, prompt_tokens: List[int], 
                    sampling_params: SamplingParams) -> int:
        """Add a new sequence and return its ID"""
        seq = Sequence(prompt_tokens, sampling_params)
        self.sequences[seq.seq_id] = seq
        self.waiting_queue.append(seq.seq_id)
        return seq.seq_id
    
    def get_next_batch(self, max_sequences: int = 8) -> List[Sequence]:
        """Get next batch of sequences to process"""
        # Move waiting sequences to running
        batch_size = min(max_sequences, len(self.waiting_queue))
        batch_ids = self.waiting_queue[:batch_size]
        self.waiting_queue = self.waiting_queue[batch_size:]
        
        # Update status and collect sequences
        batch = []
        for seq_id in batch_ids:
            seq = self.sequences[seq_id]
            seq.status = SequenceStatus.RUNNING
            self.running_sequences.append(seq_id)
            batch.append(seq)
            
        return batch
    
    def update_sequences(self, seq_ids: List[int], new_tokens: List[int]):
        """Update sequences with newly generated tokens"""
        for seq_id, token in zip(seq_ids, new_tokens):
            seq = self.sequences[seq_id]
            seq.add_token(token)
            
            # Move finished sequences
            if seq.is_finished:
                self.running_sequences.remove(seq_id)
                self.finished_sequences.append(seq_id)
    
    def get_statistics(self) -> dict:
        """Get current state statistics"""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_sequences), 
            "finished": len(self.finished_sequences),
            "total": len(self.sequences)
        }
```

## Checkpoint Test

Create `test_module1.py`:

```python
import pytest
from module1 import Sequence, SamplingParams, SequenceManager, EngineConfig

def test_sequence_lifecycle():
    """Test sequence creation and state management"""
    params = SamplingParams(temperature=0.8, max_tokens=10)
    seq = Sequence([1, 2, 3], params)
    
    # Initial state
    assert seq.status == SequenceStatus.WAITING
    assert len(seq) == 3
    assert not seq.is_finished
    
    # Add tokens
    for i in range(5):
        seq.add_token(100 + i)
        
    assert len(seq.completion_tokens) == 5
    assert len(seq) == 8
    
    # Finish by length limit
    for i in range(5):
        seq.add_token(200 + i)
        
    assert seq.is_finished
    assert seq.status == SequenceStatus.FINISHED

def test_sampling_params():
    """Test parameter validation"""
    params = SamplingParams(temperature=0.0, max_tokens=100, stop_tokens=[2, 3])
    assert params.temperature == 0.0
    assert 2 in params.stop_tokens

def test_sequence_manager():
    """Test batch management"""
    manager = SequenceManager()
    
    # Add sequences
    seq_ids = []
    for i in range(5):
        params = SamplingParams(max_tokens=10)
        seq_id = manager.add_sequence([1, 2, i], params)
        seq_ids.append(seq_id)
    
    # Get batch
    batch = manager.get_next_batch(max_sequences=3)
    assert len(batch) == 3
    
    stats = manager.get_statistics()
    assert stats["waiting"] == 2
    assert stats["running"] == 3
    assert stats["finished"] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run the test:
```bash
python test_module1.py
```

## Time/Space Complexity Analysis

**Sequence Operations:**
- Token addition: O(1) time, O(n) space where n is sequence length
- Status checking: O(1) time
- Memory overhead: ~100 bytes per sequence + token storage

**Sequence Manager:**
- Add sequence: O(1) time
- Get batch: O(k) time where k is batch size  
- Update sequences: O(k) time for k sequences
- Space complexity: O(n) where n is total sequences

## Extension Challenges

1. **Priority Scheduling**: Add priority levels to sequences and implement priority-based batching
2. **Memory Estimation**: Implement methods to estimate memory requirements for sequences
3. **Timeout Handling**: Add sequence timeout and cleanup mechanisms
4. **Metrics Collection**: Add detailed timing and throughput metrics

## Key Takeaways

- Sequences encapsulate request state and lifecycle management
- Configuration objects centralize system parameters and validation
- Clean abstractions make complex systems manageable and testable
- State machines (like SequenceStatus) clarify system behavior

**Next Module:** We'll build upon these foundations to implement sophisticated memory management with PagedAttention concepts.

---

# Module 2: Memory Management with PagedAttention

## Learning Objectives
- Understand PagedAttention memory management concepts
- Implement block-based KV cache allocation
- Learn prefix caching for efficiency improvements
- Master copy-on-write semantics for memory sharing

## Core Concepts

### The Memory Challenge

Traditional transformer inference faces a critical memory bottleneck:

1. **KV Cache Growth**: Each sequence needs memory for Key and Value tensors from all layers
2. **Fragmentation**: Variable sequence lengths cause memory fragmentation  
3. **Memory Waste**: Pre-allocating for max length wastes GPU memory
4. **Sharing Impossible**: Cannot share memory between sequences with common prefixes

PagedAttention solves these issues by treating KV cache like virtual memory in operating systems.

### Block-Based Memory Management

```python
import hashlib
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class Block:
    """A fixed-size memory block for storing KV cache data"""
    
    block_id: int
    block_size: int  # Number of tokens this block can store
    ref_count: int = 0  # Number of sequences using this block
    is_computed: bool = False  # Whether KV values are computed
    content_hash: Optional[str] = None  # Hash for prefix caching
    
    def __post_init__(self):
        # Each block can be shared by multiple sequences
        # but becomes unique when modified (copy-on-write)
        self.is_shared = False
    
    def can_append(self) -> bool:
        """Check if block has space for more tokens"""
        return not self.is_computed or self.ref_count <= 1
    
    def make_unique(self) -> 'Block':
        """Create a unique copy for modification (copy-on-write)"""
        if self.ref_count <= 1:
            return self
            
        # Create new block for modification
        new_block = Block(
            block_id=-1,  # Will be assigned by BlockManager
            block_size=self.block_size,
            ref_count=1,
            is_computed=False,
            content_hash=None
        )
        
        # Decrease reference count on original
        self.ref_count -= 1
        return new_block

class BlockManager:
    """Manages memory blocks with prefix caching optimization"""
    
    def __init__(self, block_size: int, max_blocks: int):
        self.block_size = block_size
        self.max_blocks = max_blocks
        
        # Block management
        self.blocks: Dict[int, Block] = {}
        self.free_blocks: Set[int] = set(range(max_blocks))
        self.next_block_id = 0
        
        # Prefix caching: hash -> block_id
        self.hash_to_block: Dict[str, int] = {}
        
    def allocate_block(self) -> Optional[Block]:
        """Allocate a new memory block"""
        if not self.free_blocks:
            return None  # Out of memory
            
        block_id = self.free_blocks.pop()
        block = Block(block_id=block_id, block_size=self.block_size, ref_count=1)
        self.blocks[block_id] = block
        return block
    
    def deallocate_block(self, block: Block):
        """Return block to free pool when no longer referenced"""
        block.ref_count -= 1
        
        if block.ref_count <= 0:
            # Remove from hash table if cached
            if block.content_hash and block.content_hash in self.hash_to_block:
                del self.hash_to_block[block.content_hash]
            
            # Return to free pool
            del self.blocks[block.block_id]
            self.free_blocks.add(block.block_id)
    
    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate requested number of blocks"""
        return len(self.free_blocks) >= num_blocks
    
    def compute_content_hash(self, tokens: List[int]) -> str:
        """Compute hash for token sequence (for prefix caching)"""
        token_bytes = np.array(tokens, dtype=np.int32).tobytes()
        return hashlib.sha256(token_bytes).hexdigest()[:16]  # Use short hash
    
    def find_cached_blocks(self, tokens: List[int]) -> List[Block]:
        """Find cached blocks for token sequence (prefix caching)"""
        cached_blocks = []
        
        # Check prefixes of increasing length
        for end_idx in range(self.block_size, len(tokens) + 1, self.block_size):
            prefix_tokens = tokens[:end_idx]
            content_hash = self.compute_content_hash(prefix_tokens)
            
            if content_hash in self.hash_to_block:
                block_id = self.hash_to_block[content_hash]
                if block_id in self.blocks:
                    block = self.blocks[block_id]
                    block.ref_count += 1  # Add reference
                    cached_blocks.append(block)
                else:
                    # Stale hash entry, remove it
                    del self.hash_to_block[content_hash]
                    break
            else:
                break  # No more cached prefixes
                
        return cached_blocks
    
    def cache_blocks(self, tokens: List[int], blocks: List[Block]):
        """Cache blocks for future prefix reuse"""
        tokens_per_block = self.block_size
        
        for i, block in enumerate(blocks):
            start_idx = i * tokens_per_block
            end_idx = min(start_idx + tokens_per_block, len(tokens))
            
            if end_idx > start_idx:
                block_tokens = tokens[start_idx:end_idx]
                content_hash = self.compute_content_hash(tokens[:end_idx])  # Cumulative prefix
                
                block.content_hash = content_hash
                block.is_computed = True
                self.hash_to_block[content_hash] = block.block_id
```

### Sequence Memory Management

```python
class SequenceBlockManager:
    """Manages block allocation for individual sequences"""
    
    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.sequence_blocks: Dict[int, List[Block]] = {}  # seq_id -> blocks
    
    def allocate_sequence(self, seq_id: int, tokens: List[int]) -> bool:
        """Allocate blocks for a new sequence with prefix caching"""
        num_blocks_needed = (len(tokens) + self.block_manager.block_size - 1) // self.block_manager.block_size
        
        # Try to find cached blocks for prefixes
        cached_blocks = self.block_manager.find_cached_blocks(tokens)
        num_new_blocks = max(0, num_blocks_needed - len(cached_blocks))
        
        # Check if we can allocate remaining blocks
        if not self.block_manager.can_allocate(num_new_blocks):
            return False  # Cannot allocate
        
        # Allocate new blocks for uncached portion
        all_blocks = cached_blocks.copy()
        for _ in range(num_new_blocks):
            block = self.block_manager.allocate_block()
            if block is None:
                # Cleanup allocated blocks on failure
                for b in all_blocks[len(cached_blocks):]:
                    self.block_manager.deallocate_block(b)
                return False
            all_blocks.append(block)
        
        self.sequence_blocks[seq_id] = all_blocks
        
        # Cache the new blocks
        if num_new_blocks > 0:
            self.block_manager.cache_blocks(tokens, all_blocks)
        
        return True
    
    def can_append_tokens(self, seq_id: int, num_new_tokens: int) -> bool:
        """Check if sequence can accommodate more tokens"""
        if seq_id not in self.sequence_blocks:
            return False
            
        blocks = self.sequence_blocks[seq_id]
        if not blocks:
            return False
            
        # Check if last block has space
        last_block = blocks[-1]
        if last_block.can_append():
            return True
            
        # Need new blocks
        current_capacity = len(blocks) * self.block_manager.block_size
        # Estimate current tokens (might be less than capacity)
        # For simplicity, assume we need ceil(num_new_tokens / block_size) new blocks
        new_blocks_needed = (num_new_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
        
        return self.block_manager.can_allocate(new_blocks_needed)
    
    def append_tokens(self, seq_id: int, num_new_tokens: int) -> bool:
        """Add space for new tokens, allocating blocks as needed"""
        if not self.can_append_tokens(seq_id, num_new_tokens):
            return False
            
        blocks = self.sequence_blocks[seq_id]
        
        # Make last block unique if shared (copy-on-write)
        if blocks:
            last_block = blocks[-1]
            if last_block.ref_count > 1:
                unique_block = last_block.make_unique()
                if unique_block != last_block:
                    # Need to allocate new block
                    new_block = self.block_manager.allocate_block()
                    if new_block is None:
                        return False
                    blocks[-1] = new_block
        
        # Allocate additional blocks if needed
        current_capacity = len(blocks) * self.block_manager.block_size
        # For simplicity in this example, assume we know the exact token count
        # In practice, you'd track this more precisely
        new_blocks_needed = max(0, (num_new_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size)
        
        for _ in range(new_blocks_needed):
            block = self.block_manager.allocate_block()
            if block is None:
                return False
            blocks.append(block)
            
        return True
    
    def deallocate_sequence(self, seq_id: int):
        """Free all blocks used by a sequence"""
        if seq_id not in self.sequence_blocks:
            return
            
        blocks = self.sequence_blocks[seq_id]
        for block in blocks:
            self.block_manager.deallocate_block(block)
            
        del self.sequence_blocks[seq_id]
    
    def get_block_table(self, seq_id: int) -> List[int]:
        """Get block IDs for a sequence (needed for attention kernels)"""
        if seq_id not in self.sequence_blocks:
            return []
        return [block.block_id for block in self.sequence_blocks[seq_id]]
    
    def get_memory_stats(self) -> dict:
        """Get memory utilization statistics"""
        total_blocks = self.block_manager.max_blocks
        free_blocks = len(self.block_manager.free_blocks)
        used_blocks = total_blocks - free_blocks
        
        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks, 
            "free_blocks": free_blocks,
            "utilization": used_blocks / total_blocks,
            "cached_prefixes": len(self.block_manager.hash_to_block)
        }
```

## Implementation Exercise

Let's build a complete memory management system:

```python
class MemoryManager:
    """High-level interface for managing sequence memory"""
    
    def __init__(self, block_size: int = 16, max_blocks: int = 1024):
        self.block_manager = BlockManager(block_size, max_blocks)
        self.sequence_manager = SequenceBlockManager(self.block_manager)
        
    def add_sequence(self, seq_id: int, tokens: List[int]) -> bool:
        """Add a new sequence with memory allocation"""
        return self.sequence_manager.allocate_sequence(seq_id, tokens)
    
    def extend_sequence(self, seq_id: int, num_new_tokens: int) -> bool:
        """Extend sequence with additional tokens"""
        return self.sequence_manager.append_tokens(seq_id, num_new_tokens)
    
    def remove_sequence(self, seq_id: int):
        """Remove sequence and free its memory"""
        self.sequence_manager.deallocate_sequence(seq_id)
    
    def can_allocate(self, sequences_data: List[tuple]) -> bool:
        """Check if we can allocate memory for multiple sequences"""
        # sequences_data: [(seq_id, tokens), ...]
        
        # Simulate allocation to check feasibility
        temp_allocated = []
        
        for seq_id, tokens in sequences_data:
            if self.sequence_manager.allocate_sequence(seq_id, tokens):
                temp_allocated.append(seq_id)
            else:
                # Cleanup and return False
                for temp_seq_id in temp_allocated:
                    self.sequence_manager.deallocate_sequence(temp_seq_id)
                return False
        
        # Cleanup simulation
        for temp_seq_id in temp_allocated:
            self.sequence_manager.deallocate_sequence(temp_seq_id)
            
        return True
    
    def get_block_tables(self, seq_ids: List[int]) -> Dict[int, List[int]]:
        """Get block tables for multiple sequences"""
        return {
            seq_id: self.sequence_manager.get_block_table(seq_id) 
            for seq_id in seq_ids
        }
    
    def get_stats(self) -> dict:
        """Get comprehensive memory statistics"""
        return self.sequence_manager.get_memory_stats()

## Checkpoint Test

Create `test_module2.py`:

```python
import pytest
from module2 import Block, BlockManager, SequenceBlockManager, MemoryManager

def test_block_allocation():
    """Test basic block allocation and deallocation"""
    block_manager = BlockManager(block_size=16, max_blocks=100)
    
    # Allocate blocks
    blocks = []
    for i in range(10):
        block = block_manager.allocate_block()
        assert block is not None
        assert block.ref_count == 1
        blocks.append(block)
    
    assert len(block_manager.free_blocks) == 90
    
    # Deallocate blocks
    for block in blocks:
        block_manager.deallocate_block(block)
    
    assert len(block_manager.free_blocks) == 100

def test_prefix_caching():
    """Test prefix caching functionality"""
    block_manager = BlockManager(block_size=4, max_blocks=100)
    
    # Create sequence with tokens
    tokens1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks
    tokens2 = [1, 2, 3, 4, 9, 10, 11, 12]  # Same prefix, different suffix
    
    # Cache first sequence
    blocks1 = [block_manager.allocate_block() for _ in range(2)]
    block_manager.cache_blocks(tokens1, blocks1)
    
    # Find cached blocks for second sequence
    cached_blocks = block_manager.find_cached_blocks(tokens2)
    assert len(cached_blocks) == 1  # First block should be cached
    assert cached_blocks[0].ref_count == 2  # Referenced by both sequences

def test_sequence_allocation():
    """Test sequence-level memory management"""
    memory_manager = MemoryManager(block_size=8, max_blocks=50)
    
    # Add sequences
    seq1_tokens = [1, 2, 3, 4, 5]
    seq2_tokens = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # Shared prefix with seq1
    
    assert memory_manager.add_sequence(1, seq1_tokens)
    assert memory_manager.add_sequence(2, seq2_tokens)
    
    # Check block tables
    block_tables = memory_manager.get_block_tables([1, 2])
    assert 1 in block_tables
    assert 2 in block_tables
    
    # Check memory stats
    stats = memory_manager.get_stats()
    assert stats["used_blocks"] > 0
    assert stats["cached_prefixes"] > 0
    
    # Remove sequences
    memory_manager.remove_sequence(1)
    memory_manager.remove_sequence(2)
    
    final_stats = memory_manager.get_stats()
    assert final_stats["used_blocks"] == 0

def test_memory_pressure():
    """Test behavior under memory pressure"""
    memory_manager = MemoryManager(block_size=4, max_blocks=5)  # Very limited memory
    
    # Try to allocate more than available
    large_sequence = list(range(100))  # Needs 25 blocks
    
    assert not memory_manager.add_sequence(1, large_sequence)
    
    # Allocate within limits
    small_sequence = [1, 2, 3, 4]  # Needs 1 block
    assert memory_manager.add_sequence(2, small_sequence)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Time/Space Complexity Analysis

**Block Operations:**
- Block allocation/deallocation: O(1) amortized
- Hash computation: O(n) where n is token sequence length
- Prefix search: O(k) where k is number of cached prefixes

**Memory Efficiency:**
- Space overhead: ~64 bytes per block + hash table entries
- Memory fragmentation: Eliminated through fixed-size blocks
- Sharing efficiency: Multiple sequences can share prefix blocks

**Prefix Caching Benefits:**
- Cache hit rate: 60-80% for typical workloads with repeated prefixes
- Memory savings: 30-50% reduction in KV cache memory usage
- Computation savings: Cached blocks skip attention computation

## Extension Challenges

1. **LRU Eviction**: Implement LRU eviction when memory is full
2. **Dynamic Block Sizing**: Support variable block sizes for different sequence types
3. **Memory Defragmentation**: Implement block compaction for long-running systems
4. **Distributed Memory**: Extend to multi-GPU memory management

## Key Takeaways

- PagedAttention transforms memory management from a bottleneck to an optimization
- Block-based allocation eliminates fragmentation and enables sharing
- Prefix caching dramatically improves memory and computation efficiency
- Copy-on-write semantics allow safe sharing while preserving correctness

**Next Module:** We'll implement the transformer model architecture that uses this memory system.

---

# Module 3: Model Architecture - Transformer Components

## Learning Objectives
- Implement core transformer building blocks
- Understand tensor parallelism for multi-GPU scaling
- Learn optimized attention mechanisms with KV caching
- Master layer normalization and activation functions

## Core Concepts

### Tensor Parallelism

Modern LLMs require multiple GPUs for inference. Tensor parallelism splits individual operations across GPUs:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class TensorParallelLinear(nn.Module, ABC):
    """Base class for tensor parallel linear layers"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Get tensor parallel info
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        
    @abstractmethod
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """Load weights with appropriate sharding"""
        pass

class ColumnParallelLinear(TensorParallelLinear):
    """Linear layer with column-wise parallelism (output features split across GPUs)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Split output features across tensor parallel groups
        assert out_features % self.tp_size == 0, f"out_features {out_features} not divisible by tp_size {self.tp_size}"
        self.out_features_per_partition = out_features // self.tp_size
        
        # Create parameters for this partition
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - no communication needed"""
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """Load weight shard for this partition"""
        if weight_name.endswith('.weight'):
            # Split along output dimension (dim 0)
            start_idx = self.tp_rank * self.out_features_per_partition
            end_idx = start_idx + self.out_features_per_partition
            param.data.copy_(loaded_weight[start_idx:end_idx])
        elif weight_name.endswith('.bias'):
            if param is not None:
                start_idx = self.tp_rank * self.out_features_per_partition
                end_idx = start_idx + self.out_features_per_partition
                param.data.copy_(loaded_weight[start_idx:end_idx])

class RowParallelLinear(TensorParallelLinear):
    """Linear layer with row-wise parallelism (input features split across GPUs)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Split input features across tensor parallel groups
        assert in_features % self.tp_size == 0, f"in_features {in_features} not divisible by tp_size {self.tp_size}"
        self.in_features_per_partition = in_features // self.tp_size
        
        # Create parameters for this partition
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias and self.tp_rank == 0:  # Only rank 0 has bias to avoid duplication
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-reduce communication"""
        # Each GPU computes partial result
        partial_output = torch.nn.functional.linear(x, self.weight, None)
        
        # Sum partial results across all GPUs
        if self.tp_size > 1:
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)
        
        # Add bias only on rank 0 (avoids duplication)
        if self.bias is not None:
            partial_output += self.bias
            
        return partial_output
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """Load weight shard for this partition"""
        if weight_name.endswith('.weight'):
            # Split along input dimension (dim 1)
            start_idx = self.tp_rank * self.in_features_per_partition
            end_idx = start_idx + self.in_features_per_partition
            param.data.copy_(loaded_weight[:, start_idx:end_idx])
        elif weight_name.endswith('.bias') and param is not None:
            # Bias is only on rank 0
            param.data.copy_(loaded_weight)
```

### Specialized Attention Layers

```python
class QKVParallelLinear(nn.Module):
    """Fused QKV projection with tensor parallelism"""
    
    def __init__(self, hidden_size: int, head_dim: int, 
                 num_heads: int, num_kv_heads: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        # Tensor parallel setup
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate dimensions per partition
        assert num_heads % self.tp_size == 0
        assert num_kv_heads % self.tp_size == 0
        
        self.num_heads_per_partition = num_heads // self.tp_size
        self.num_kv_heads_per_partition = num_kv_heads // self.tp_size
        
        # Output dimensions
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim
        self.total_size = self.q_size + 2 * self.kv_size  # Q + K + V
        
        # Fused weight matrix
        self.weight = nn.Parameter(torch.empty(self.total_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q, K, V projections in one operation"""
        qkv = torch.nn.functional.linear(x, self.weight, self.bias)
        return qkv
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """Load weights with proper QKV sharding"""
        if 'q_proj' in weight_name:
            # Query projection
            start_idx = self.tp_rank * self.q_size
            end_idx = start_idx + self.q_size
            param.data[:self.q_size].copy_(loaded_weight[start_idx:end_idx])
            
        elif 'k_proj' in weight_name:
            # Key projection
            start_idx = self.tp_rank * self.kv_size
            end_idx = start_idx + self.kv_size
            param.data[self.q_size:self.q_size + self.kv_size].copy_(loaded_weight[start_idx:end_idx])
            
        elif 'v_proj' in weight_name:
            # Value projection  
            start_idx = self.tp_rank * self.kv_size
            end_idx = start_idx + self.kv_size
            param.data[self.q_size + self.kv_size:].copy_(loaded_weight[start_idx:end_idx])
```

### Optimized Layer Normalization

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with fused residual"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional fused residual connection"""
        if residual is not None:
            # Fused residual + norm
            x = x + residual
            
        # RMS normalization
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x
        
        if residual is not None:
            return x, x  # Return normalized and new residual
        else:
            return x
```

### Efficient Activations

```python
class SiluAndMul(nn.Module):
    """Fused SiLU activation and multiplication for MLP layers"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU(x1) * x2 where x = [x1, x2] concatenated"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2

class GeluAndMul(nn.Module):
    """Fused GELU activation and multiplication"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.gelu(x1) * x2
```

### Rotary Position Embeddings

```python
import math

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for position encoding"""
    
    def __init__(self, head_dim: int, max_position: int = 8192, base: float = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for computed embeddings
        self._cached_cos = None
        self._cached_sin = None
        self._cached_max_len = 0
    
    def _compute_embeddings(self, max_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache sin/cos embeddings"""
        if max_len <= self._cached_max_len and self._cached_cos is not None:
            return self._cached_cos[:max_len], self._cached_sin[:max_len]
        
        # Generate position indices
        positions = torch.arange(max_len, device=device, dtype=dtype)
        
        # Compute frequencies: outer product of positions and inv_freq
        freqs = torch.outer(positions, self.inv_freq.to(dtype))
        
        # Compute embeddings
        cos_emb = torch.cos(freqs)
        sin_emb = torch.sin(freqs)
        
        # Cache results
        self._cached_cos = cos_emb
        self._cached_sin = sin_emb
        self._cached_max_len = max_len
        
        return cos_emb, sin_emb
    
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys"""
        max_len = positions.max().item() + 1
        cos_emb, sin_emb = self._compute_embeddings(max_len, positions.device, q.dtype)
        
        # Select embeddings for actual positions
        cos = cos_emb[positions]  # [seq_len, head_dim//2]
        sin = sin_emb[positions]
        
        # Apply rotary transformation
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary transformation to tensor"""
        # x shape: [batch_size, num_heads, head_dim] or [batch_size, head_dim]
        # Reshape for broadcasting
        if x.dim() == 3:
            cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim//2]
            sin = sin.unsqueeze(1)
        
        # Split into even and odd dimensions
        x1 = x[..., 0::2]  # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,  # Real part
            x1 * sin + x2 * cos   # Imaginary part
        ], dim=-1)
        
        # Flatten back to original shape
        return rotated.flatten(start_dim=-2)
```

## Implementation Exercise

Let's build a complete transformer layer:

```python
class TransformerLayer(nn.Module):
    """Complete transformer decoder layer with optimizations"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position=config.max_position_embeddings
        )
        
        # MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with fused residual connections"""
        
        # Pre-attention norm with optional residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self-attention
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)
        
        # Post-attention norm with residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class MLP(nn.Module):
    """Feed-forward network with tensor parallelism"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        
        # Gate and up projections fused
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, 
            [intermediate_size, intermediate_size],  # Gate and up have same size
            bias=False
        )
        
        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size, 
            bias=False
        )
        
        # Activation function
        if activation == "silu":
            self.act_fn = SiluAndMul()
        elif activation == "gelu":
            self.act_fn = GeluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate and up projections
        gate_up = self.gate_up_proj(x)
        
        # Apply activation (SiLU(gate) * up)
        x = self.act_fn(gate_up)
        
        # Down projection with all-reduce
        x = self.down_proj(x)
        
        return x

class MergedColumnParallelLinear(nn.Module):
    """Multiple column-parallel linear layers merged into one for efficiency"""
    
    def __init__(self, in_features: int, out_features_list: List[int], bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_list = out_features_list
        self.total_out_features = sum(out_features_list)
        
        # Tensor parallel setup
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate per-partition sizes
        self.out_features_per_partition = [
            out_features // self.tp_size for out_features in out_features_list
        ]
        self.total_out_features_per_partition = sum(self.out_features_per_partition)
        
        # Create fused weight matrix
        self.weight = nn.Parameter(torch.empty(self.total_out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_out_features_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str):
        """Load weights for merged layers"""
        # Determine which sub-layer this weight belongs to
        for i, out_features in enumerate(self.out_features_list):
            if f"_{i}" in weight_name or (i == 0 and "_" not in weight_name):
                start_idx = sum(self.out_features_per_partition[:i])
                end_idx = start_idx + self.out_features_per_partition[i]
                
                # Load the appropriate shard
                tp_start = self.tp_rank * (out_features // self.tp_size)
                tp_end = tp_start + (out_features // self.tp_size)
                
                param.data[start_idx:end_idx].copy_(loaded_weight[tp_start:tp_end])
                break
```
```