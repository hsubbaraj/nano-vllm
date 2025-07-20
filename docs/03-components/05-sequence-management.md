# Sequence Management: Request Lifecycle and State Tracking

## Overview

Sequence management is the backbone of request handling in an LLM inference engine. A Sequence represents a single generation request from arrival to completion, tracking its state, tokens, memory allocation, and generation progress. This guide explores how sequences are managed throughout their lifecycle.

## The Sequence Abstraction

### Core Sequence Components

```python
class Sequence:
    def __init__(self, 
                 request_id: str,
                 prompt: str,
                 sampling_params: SamplingParams):
        # Identification
        self.request_id = request_id
        self.arrival_time = time.time()
        
        # Tokens
        self.prompt = prompt
        self.prompt_tokens: List[int] = []
        self.output_tokens: List[int] = []
        
        # Memory management
        self.block_table: List[int] = []
        self.cached_tokens: int = 0
        
        # Generation parameters
        self.sampling_params = sampling_params
        
        # State tracking
        self.status = SequenceStatus.WAITING
        self.cumulative_logprobs: float = 0.0
        
        # Metrics
        self.metrics = SequenceMetrics()
```

### Sequence Status Lifecycle

```python
class SequenceStatus(Enum):
    WAITING = "waiting"      # In scheduler queue
    RUNNING = "running"      # Actively generating
    PREEMPTED = "preempted"  # Evicted, needs restart  
    FINISHED = "finished"    # Generation complete
    ABORTED = "aborted"      # Cancelled or errored
```

## Token Management

### Prompt Processing

```python
class Sequence:
    def set_prompt_tokens(self, tokenizer):
        """Tokenize prompt and prepare for processing."""
        # Basic tokenization
        self.prompt_tokens = tokenizer.encode(self.prompt)
        
        # Add special tokens if needed
        if self.sampling_params.add_special_tokens:
            self.prompt_tokens = tokenizer.build_inputs_with_special_tokens(
                self.prompt_tokens
            )
        
        # Validate length
        if len(self.prompt_tokens) > self.sampling_params.max_model_len:
            raise ValueError(f"Prompt too long: {len(self.prompt_tokens)} > {self.sampling_params.max_model_len}")
        
        self.metrics.prompt_tokens = len(self.prompt_tokens)
```

### Output Token Handling

```python
class Sequence:
    def add_token(self, token_id: int, logprob: float = 0.0):
        """Add a generated token to the sequence."""
        self.output_tokens.append(token_id)
        self.cumulative_logprobs += logprob
        
        # Update metrics
        self.metrics.output_tokens += 1
        self.metrics.last_token_time = time.time()
        
        # Check completion conditions
        if self.is_finished():
            self.status = SequenceStatus.FINISHED
            self.metrics.completion_time = time.time()
    
    def is_finished(self) -> bool:
        """Check if generation should stop."""
        # Maximum length reached
        if len(self.output_tokens) >= self.sampling_params.max_tokens:
            return True
        
        # EOS token generated
        if (self.output_tokens and 
            self.output_tokens[-1] == self.sampling_params.eos_token_id and
            not self.sampling_params.ignore_eos):
            return True
        
        # Stop sequences matched
        if self.sampling_params.stop_sequences:
            output_text = self.get_output_text()
            for stop_seq in self.sampling_params.stop_sequences:
                if output_text.endswith(stop_seq):
                    return True
        
        return False
```

### Token Buffer Management

For efficient processing:

```python
class TokenBuffer:
    """Efficient token storage with minimal allocations."""
    def __init__(self, initial_capacity: int = 2048):
        self.tokens = torch.zeros(initial_capacity, dtype=torch.long)
        self.length = 0
        
    def append(self, token: int):
        if self.length >= len(self.tokens):
            # Grow buffer
            new_capacity = len(self.tokens) * 2
            new_tokens = torch.zeros(new_capacity, dtype=torch.long)
            new_tokens[:self.length] = self.tokens[:self.length]
            self.tokens = new_tokens
        
        self.tokens[self.length] = token
        self.length += 1
    
    def get_tokens(self) -> torch.Tensor:
        return self.tokens[:self.length]
```

## Memory Lifecycle Management

### Block Allocation

```python
class Sequence:
    def allocate_blocks(self, block_manager: BlockManager):
        """Allocate KV cache blocks for the sequence."""
        # Calculate required blocks
        total_length = len(self.prompt_tokens) + self.sampling_params.max_tokens
        num_blocks = (total_length + block_manager.block_size - 1) // block_manager.block_size
        
        # Try allocation with caching
        self.block_table, self.cached_tokens = block_manager.allocate_with_caching(
            self.prompt_tokens
        )
        
        # Track allocation in metrics
        self.metrics.blocks_allocated = len(self.block_table)
        self.metrics.cached_blocks = self.cached_tokens // block_manager.block_size
```

### Progressive Block Allocation

For memory efficiency:

```python
class Sequence:
    def allocate_blocks_incremental(self, block_manager: BlockManager):
        """Allocate blocks as needed during generation."""
        current_length = self.get_len()
        allocated_length = len(self.block_table) * block_manager.block_size
        
        if current_length > allocated_length:
            # Need more blocks
            additional_blocks = (current_length - allocated_length + 
                               block_manager.block_size - 1) // block_manager.block_size
            
            new_blocks = block_manager.allocate_blocks(additional_blocks)
            if new_blocks is None:
                raise OutOfMemoryError("Cannot allocate additional blocks")
            
            self.block_table.extend(new_blocks)
```

### Block Deallocation

```python
class Sequence:
    def free_blocks(self, block_manager: BlockManager):
        """Free all allocated blocks."""
        for block_id in self.block_table:
            block_manager.remove_reference(block_id)
        
        self.block_table.clear()
        self.metrics.blocks_freed = self.metrics.blocks_allocated
```

## Preemption and Resumption

### Handling Preemption

```python
class PreemptibleSequence(Sequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preemption_state = None
        
    def preempt(self, block_manager: BlockManager) -> PreemptionState:
        """Preempt sequence and save state for resumption."""
        # Save current state
        self.preemption_state = PreemptionState(
            output_tokens=self.output_tokens.copy(),
            cumulative_logprobs=self.cumulative_logprobs,
            block_table=self.block_table.copy(),
            cached_tokens=self.cached_tokens,
            last_position=self.get_len()
        )
        
        # Free blocks
        self.free_blocks(block_manager)
        
        # Update status
        self.status = SequenceStatus.PREEMPTED
        self.metrics.preemption_count += 1
        
        return self.preemption_state
    
    def resume(self, block_manager: BlockManager) -> bool:
        """Resume from preempted state."""
        if self.preemption_state is None:
            return False
        
        # Reallocate blocks
        try:
            # Try to reuse cached blocks
            self.block_table, self.cached_tokens = block_manager.allocate_with_caching(
                self.prompt_tokens + self.preemption_state.output_tokens
            )
        except OutOfMemoryError:
            return False
        
        # Restore state
        self.output_tokens = self.preemption_state.output_tokens
        self.cumulative_logprobs = self.preemption_state.cumulative_logprobs
        
        # Update status
        self.status = SequenceStatus.RUNNING
        self.preemption_state = None
        
        return True
```

### Swapping Support

For CPU-GPU swapping:

```python
class SwappableSequence(Sequence):
    def swap_out(self, block_manager: SwappableBlockManager) -> SwapState:
        """Swap KV cache blocks to CPU memory."""
        cpu_blocks = block_manager.swap_out_blocks(self.block_table)
        
        swap_state = SwapState(
            cpu_blocks=cpu_blocks,
            original_block_table=self.block_table.copy()
        )
        
        self.block_table.clear()
        self.status = SequenceStatus.PREEMPTED
        
        return swap_state
    
    def swap_in(self, block_manager: SwappableBlockManager, 
                swap_state: SwapState) -> bool:
        """Swap KV cache blocks back to GPU."""
        try:
            self.block_table = block_manager.swap_in_blocks(swap_state.cpu_blocks)
            self.status = SequenceStatus.RUNNING
            return True
        except OutOfMemoryError:
            return False
```

## Sequence Groups and Beam Search

### Managing Multiple Sequences

For beam search and parallel sampling:

```python
class SequenceGroup:
    def __init__(self, 
                 request_id: str,
                 prompt: str,
                 sampling_params: SamplingParams):
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        
        # Create initial sequence(s)
        self.sequences = []
        num_seqs = sampling_params.best_of or sampling_params.beam_width or 1
        
        for i in range(num_seqs):
            seq = Sequence(
                request_id=f"{request_id}_seq{i}",
                prompt=prompt,
                sampling_params=sampling_params
            )
            self.sequences.append(seq)
        
        self.finished_sequences = []
    
    def is_finished(self) -> bool:
        """Check if group generation is complete."""
        if self.sampling_params.use_beam_search:
            # Need beam_width finished sequences
            return len(self.finished_sequences) >= self.sampling_params.beam_width
        else:
            # All sequences finished
            return all(seq.is_finished() for seq in self.sequences)
```

### Beam Search Management

```python
class BeamSearchSequenceGroup(SequenceGroup):
    def update_beams(self, tokens: List[int], scores: List[float]):
        """Update beam sequences with new tokens."""
        # Create candidates
        candidates = []
        for i, (seq, token, score) in enumerate(zip(self.sequences, tokens, scores)):
            if not seq.is_finished():
                new_score = seq.cumulative_logprobs + score
                candidates.append((new_score, i, token))
        
        # Select top beams
        candidates.sort(reverse=True)
        top_beams = candidates[:self.sampling_params.beam_width]
        
        # Update sequences
        new_sequences = []
        for score, seq_idx, token in top_beams:
            seq = self.sequences[seq_idx]
            # Fork sequence if needed
            if token != tokens[seq_idx]:
                seq = self.fork_sequence(seq)
            seq.add_token(token, score - seq.cumulative_logprobs)
            new_sequences.append(seq)
        
        self.sequences = new_sequences
    
    def fork_sequence(self, original: Sequence) -> Sequence:
        """Create a copy of sequence with shared KV cache."""
        forked = Sequence(
            request_id=f"{original.request_id}_fork{time.time()}",
            prompt=original.prompt,
            sampling_params=original.sampling_params
        )
        
        # Copy state
        forked.prompt_tokens = original.prompt_tokens
        forked.output_tokens = original.output_tokens.copy()
        forked.cumulative_logprobs = original.cumulative_logprobs
        
        # Share KV cache blocks
        forked.block_table = original.block_table.copy()
        # Increment reference counts
        for block_id in forked.block_table:
            self.block_manager.add_reference(block_id)
        
        return forked
```

## Metrics and Monitoring

### Comprehensive Metrics

```python
class SequenceMetrics:
    def __init__(self):
        # Timing
        self.arrival_time: float = time.time()
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        
        # Tokens
        self.prompt_tokens: int = 0
        self.output_tokens: int = 0
        
        # Memory
        self.blocks_allocated: int = 0
        self.blocks_freed: int = 0
        self.cached_blocks: int = 0
        
        # Events
        self.preemption_count: int = 0
        self.scheduling_attempts: int = 0
    
    def get_time_to_first_token(self) -> Optional[float]:
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None
    
    def get_generation_throughput(self) -> float:
        if self.output_tokens > 0 and self.last_token_time and self.first_token_time:
            duration = self.last_token_time - self.first_token_time
            return self.output_tokens / duration if duration > 0 else 0.0
        return 0.0
    
    def get_e2e_latency(self) -> Optional[float]:
        if self.completion_time:
            return self.completion_time - self.arrival_time
        return None
```

### Sequence Tracking

```python
class SequenceTracker:
    def __init__(self):
        self.active_sequences: Dict[str, Sequence] = {}
        self.completed_sequences: List[Sequence] = []
        self.metrics_aggregator = MetricsAggregator()
    
    def register_sequence(self, sequence: Sequence):
        self.active_sequences[sequence.request_id] = sequence
        self.metrics_aggregator.record_arrival()
    
    def update_sequence_status(self, request_id: str, status: SequenceStatus):
        if request_id in self.active_sequences:
            seq = self.active_sequences[request_id]
            old_status = seq.status
            seq.status = status
            
            # Track transitions
            self.metrics_aggregator.record_transition(old_status, status)
            
            # Move to completed if finished
            if status in [SequenceStatus.FINISHED, SequenceStatus.ABORTED]:
                self.completed_sequences.append(seq)
                del self.active_sequences[request_id]
                self.metrics_aggregator.record_completion(seq.metrics)
    
    def get_system_metrics(self) -> Dict[str, float]:
        return {
            'active_sequences': len(self.active_sequences),
            'completed_sequences': len(self.completed_sequences),
            'avg_prompt_tokens': self.metrics_aggregator.get_avg_prompt_tokens(),
            'avg_output_tokens': self.metrics_aggregator.get_avg_output_tokens(),
            'avg_time_to_first_token': self.metrics_aggregator.get_avg_ttft(),
            'avg_generation_throughput': self.metrics_aggregator.get_avg_throughput(),
        }
```

## Advanced Features

### Sequence Priorities

```python
class PrioritizedSequence(Sequence):
    def __init__(self, *args, priority: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority = priority
        self.priority_boost = 0.0
        
    def update_priority(self):
        """Dynamic priority adjustment based on wait time."""
        wait_time = time.time() - self.arrival_time
        
        # Boost priority for sequences waiting too long
        if wait_time > 10.0:  # 10 seconds
            self.priority_boost = min(wait_time / 10.0, 5.0)
    
    def get_effective_priority(self) -> float:
        return self.priority + self.priority_boost
```

### Sequence Persistence

For fault tolerance:

```python
class PersistentSequence(Sequence):
    def serialize(self) -> bytes:
        """Serialize sequence state for persistence."""
        state = {
            'request_id': self.request_id,
            'prompt': self.prompt,
            'prompt_tokens': self.prompt_tokens,
            'output_tokens': self.output_tokens,
            'sampling_params': self.sampling_params.dict(),
            'status': self.status.value,
            'cumulative_logprobs': self.cumulative_logprobs,
            'metrics': asdict(self.metrics)
        }
        return pickle.dumps(state)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'PersistentSequence':
        """Restore sequence from serialized state."""
        state = pickle.loads(data)
        
        seq = cls(
            request_id=state['request_id'],
            prompt=state['prompt'],
            sampling_params=SamplingParams(**state['sampling_params'])
        )
        
        # Restore state
        seq.prompt_tokens = state['prompt_tokens']
        seq.output_tokens = state['output_tokens']
        seq.status = SequenceStatus(state['status'])
        seq.cumulative_logprobs = state['cumulative_logprobs']
        # Note: block_table needs re-allocation
        
        return seq
```

### Sequence Validation

```python
class SequenceValidator:
    @staticmethod
    def validate_tokens(sequence: Sequence, vocab_size: int):
        """Validate all tokens are within vocabulary."""
        all_tokens = sequence.prompt_tokens + sequence.output_tokens
        
        for i, token in enumerate(all_tokens):
            if not 0 <= token < vocab_size:
                raise ValueError(f"Invalid token {token} at position {i}")
    
    @staticmethod
    def validate_block_table(sequence: Sequence, block_manager: BlockManager):
        """Validate block table consistency."""
        expected_blocks = (sequence.get_len() + block_manager.block_size - 1) // block_manager.block_size
        
        if len(sequence.block_table) < expected_blocks:
            raise ValueError(f"Insufficient blocks: {len(sequence.block_table)} < {expected_blocks}")
        
        for block_id in sequence.block_table:
            if block_manager.ref_counts[block_id] == 0:
                raise ValueError(f"Sequence references freed block {block_id}")
```

## Best Practices

### 1. Efficient State Management

```python
class Sequence:
    def __init__(self, ...):
        # Use __slots__ to reduce memory overhead
        __slots__ = ['request_id', 'prompt', 'prompt_tokens', ...]
        
        # Lazy initialization for optional fields
        self._logprobs = None
        self._token_logprobs = None
    
    @property
    def logprobs(self):
        if self._logprobs is None:
            self._logprobs = []
        return self._logprobs
```

### 2. Proper Cleanup

```python
class Sequence:
    def cleanup(self, block_manager: BlockManager):
        """Comprehensive cleanup on sequence completion."""
        try:
            # Free memory
            self.free_blocks(block_manager)
            
            # Clear large data structures
            self.prompt_tokens.clear()
            if hasattr(self, '_attention_mask'):
                del self._attention_mask
            
            # Record final metrics
            self.metrics.completion_time = time.time()
            
        except Exception as e:
            logger.error(f"Error cleaning up sequence {self.request_id}: {e}")
```

### 3. Thread Safety

```python
class ThreadSafeSequence(Sequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
    
    def add_token(self, token_id: int, logprob: float = 0.0):
        with self._lock:
            super().add_token(token_id, logprob)
    
    def get_output_tokens(self) -> List[int]:
        with self._lock:
            return self.output_tokens.copy()
```

### 4. Testing

```python
def test_sequence_lifecycle():
    # Create sequence
    seq = Sequence(
        request_id="test_1",
        prompt="Hello world",
        sampling_params=SamplingParams(max_tokens=10)
    )
    
    # Test tokenization
    seq.set_prompt_tokens(tokenizer)
    assert len(seq.prompt_tokens) > 0
    
    # Test block allocation
    block_manager = BlockManager(num_blocks=100)
    seq.allocate_blocks(block_manager)
    assert len(seq.block_table) > 0
    
    # Test token addition
    seq.add_token(100, -0.5)
    assert len(seq.output_tokens) == 1
    assert seq.cumulative_logprobs == -0.5
    
    # Test completion
    for i in range(9):
        seq.add_token(101 + i)
    assert seq.is_finished()
    assert seq.status == SequenceStatus.FINISHED
    
    # Test cleanup
    seq.cleanup(block_manager)
    assert len(seq.block_table) == 0
```

## Integration Examples

### With Scheduler

```python
class SchedulerSequenceInterface:
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def add_sequence(self, sequence: Sequence):
        # Validate
        if sequence.status != SequenceStatus.WAITING:
            raise ValueError("Sequence must be in WAITING status")
        
        # Add to scheduler
        self.scheduler.add_to_waiting_queue(sequence)
        
        # Update metrics
        sequence.metrics.scheduling_attempts += 1
```

### With Model Runner

```python
class ModelRunnerSequenceInterface:
    def prepare_sequence_inputs(self, sequences: List[Sequence]):
        # Group by phase
        prefill_seqs = [s for s in sequences if s.get_num_output_tokens() == 0]
        decode_seqs = [s for s in sequences if s.get_num_output_tokens() > 0]
        
        return {
            'prefill': self.prepare_prefill_inputs(prefill_seqs),
            'decode': self.prepare_decode_inputs(decode_seqs)
        }
```

## Conclusion

Effective sequence management is crucial for LLM inference engines. Key principles:

1. **Complete lifecycle tracking** from arrival to completion
2. **Efficient token management** with proper buffering
3. **Flexible memory handling** supporting allocation, sharing, and preemption
4. **Rich metrics** for monitoring and optimization
5. **Extensibility** for features like beam search and priorities

The Sequence abstraction ties together all components of the inference engine, providing a clean interface for request handling. With the core components covered, we can now move on to implementation details in the next section.