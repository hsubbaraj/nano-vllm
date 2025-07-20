# Continuous Batching: Dynamic Request Batching and Scheduling

## Overview

Continuous batching is a revolutionary technique that allows requests to join and leave batches dynamically during generation, dramatically improving GPU utilization and throughput compared to static batching. This guide covers the theory, implementation, and optimization of continuous batching systems.

## The Problem with Static Batching

Traditional static batching has severe limitations:

```python
# Static batching - all sequences must complete before starting new ones
def static_batch_inference(requests, max_batch_size=8):
    batches = [requests[i:i+max_batch_size] for i in range(0, len(requests), max_batch_size)]
    
    for batch in batches:
        # All sequences in batch generate tokens together
        for step in range(max_tokens):
            outputs = model.forward(batch)
            
            # Problem: Even if some sequences finish early, 
            # they occupy batch slots until all complete
            if all(seq.is_finished() for seq in batch):
                break
    
    # New requests must wait for entire batch to complete
```

Issues:
- **GPU underutilization**: Finished sequences waste batch slots
- **High latency**: New requests wait for entire batches to complete
- **Memory waste**: Allocated memory for finished sequences

## Continuous Batching Solution

Continuous batching allows:
- Sequences to join batches immediately when slots are available
- Finished sequences to leave, freeing slots for new requests
- Different phases (prefill/decode) to be processed together

```python
# Continuous batching - dynamic batch composition
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int, max_batch_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.running_sequences = []
        self.waiting_sequences = deque()
    
    def schedule_iteration(self):
        # Remove finished sequences
        self.running_sequences = [seq for seq in self.running_sequences if not seq.is_finished()]
        
        # Add new sequences if space available
        while (len(self.running_sequences) < self.max_batch_size and 
               self.waiting_sequences and
               self._can_fit_sequence(self.waiting_sequences[0])):
            new_seq = self.waiting_sequences.popleft()
            self.running_sequences.append(new_seq)
        
        # Create batch for this iteration
        return self.create_iteration_batch()
```

## Implementation Details

### Core Scheduler Architecture

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum
import heapq

class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"

@dataclass
class SchedulingBudget:
    """Resources available for scheduling."""
    token_budget: int
    max_num_seqs: int
    num_batched_tokens: int = 0
    num_curr_seqs: int = 0
    
    def can_schedule(self, num_tokens: int) -> bool:
        return (self.num_curr_seqs < self.max_num_seqs and
                self.num_batched_tokens + num_tokens <= self.token_budget)
    
    def schedule(self, num_tokens: int):
        self.num_batched_tokens += num_tokens
        self.num_curr_seqs += 1
    
    @property
    def remaining_token_budget(self) -> int:
        return self.token_budget - self.num_batched_tokens

class ContinuousBatchingScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        
        # Sequence queues
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []
        
        # Memory management
        self.block_manager = BlockManager(config.num_gpu_blocks, config.block_size)
        
        # Scheduling policies
        self.prompt_limit = config.max_model_len
        self.batch_token_limit = config.max_num_batched_tokens
        
    def schedule(self) -> SchedulerOutput:
        """Main scheduling function called every iteration."""
        # Phase 1: Preemption and swapping
        self._preempt_sequences_if_needed()
        
        # Phase 2: Schedule swapped sequences
        self._schedule_swapped_sequences()
        
        # Phase 3: Schedule new sequences
        scheduled = self._schedule_new_sequences()
        
        # Phase 4: Schedule running sequences
        scheduled.update(self._schedule_running_sequences())
        
        return scheduled
```

### Iteration-Level Scheduling

```python
class IterationScheduler:
    """Handles scheduling within a single iteration."""
    
    def __init__(self, scheduler: ContinuousBatchingScheduler):
        self.scheduler = scheduler
        self.scheduled_sequences: Dict[str, List[Sequence]] = {
            'prefill': [],
            'decode': [],
            'preempted': []
        }
        
    def schedule_iteration(self) -> SchedulerOutput:
        # Calculate available budget
        budget = self._calculate_budget()
        
        # Schedule prefills first (higher priority for first token)
        self._schedule_prefills(budget)
        
        # Schedule decodes with remaining budget
        self._schedule_decodes(budget)
        
        # Handle preemptions if needed
        if budget.num_batched_tokens > budget.token_budget:
            self._handle_overbudget()
        
        return self._create_output()
    
    def _calculate_budget(self) -> SchedulingBudget:
        """Calculate token budget for this iteration."""
        # Account for already running sequences
        running_tokens = sum(
            seq.get_len() for seq in self.scheduler.running
        )
        
        return SchedulingBudget(
            token_budget=self.scheduler.batch_token_limit,
            max_num_seqs=self.scheduler.config.max_num_seqs,
            num_batched_tokens=running_tokens,
            num_curr_seqs=len(self.scheduler.running)
        )
    
    def _schedule_prefills(self, budget: SchedulingBudget):
        """Schedule prefill sequences (new prompts)."""
        # Sort by priority/arrival time
        waiting_sorted = sorted(
            self.scheduler.waiting,
            key=lambda s: (s.priority, s.arrival_time)
        )
        
        scheduled = []
        for seq in waiting_sorted:
            prompt_tokens = len(seq.prompt_token_ids)
            
            # Check if we can fit this sequence
            if budget.can_schedule(prompt_tokens):
                # Allocate blocks
                blocks = self.scheduler.block_manager.allocate_for_sequence(seq)
                if blocks is not None:
                    seq.set_block_table(blocks)
                    scheduled.append(seq)
                    budget.schedule(prompt_tokens)
                else:
                    # Need preemption
                    break
            else:
                # Can't fit more sequences
                break
        
        # Move scheduled sequences to running
        for seq in scheduled:
            self.scheduler.waiting.remove(seq)
            self.scheduler.running.append(seq)
            seq.status = SequenceStatus.RUNNING
            
        self.scheduled_sequences['prefill'] = scheduled
    
    def _schedule_decodes(self, budget: SchedulingBudget):
        """Schedule decode sequences (generating tokens)."""
        decode_sequences = []
        
        for seq in self.scheduler.running:
            if seq.get_output_len() > 0:  # Has generated at least one token
                # Each decode only processes one new token
                if budget.can_schedule(1):
                    decode_sequences.append(seq)
                    budget.schedule(1)
        
        self.scheduled_sequences['decode'] = decode_sequences
```

### Advanced Scheduling Policies

```python
class PriorityScheduler(ContinuousBatchingScheduler):
    """Scheduler with priority support."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        # Use priority queue for waiting sequences
        self.waiting_queue = []  # heap queue
        
    def add_sequence(self, sequence: Sequence):
        # Priority queue: (priority, arrival_time, sequence)
        heapq.heappush(
            self.waiting_queue,
            (-sequence.priority, sequence.arrival_time, sequence)
        )
    
    def _schedule_new_sequences(self) -> List[Sequence]:
        scheduled = []
        temp_queue = []
        
        while self.waiting_queue:
            priority, arrival_time, seq = heapq.heappop(self.waiting_queue)
            
            if self._can_schedule_sequence(seq):
                scheduled.append(seq)
                seq.status = SequenceStatus.RUNNING
            else:
                # Put back sequences we can't schedule
                temp_queue.append((priority, arrival_time, seq))
        
        # Restore unscheduled sequences
        for item in temp_queue:
            heapq.heappush(self.waiting_queue, item)
        
        return scheduled

class FairnessScheduler(ContinuousBatchingScheduler):
    """Ensures fair scheduling across requests."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.sequence_wait_times = {}
        self.max_wait_time = config.max_wait_time
        
    def _calculate_sequence_priority(self, seq: Sequence) -> float:
        """Calculate priority based on wait time."""
        wait_time = time.time() - seq.arrival_time
        
        # Boost priority for sequences waiting too long
        if wait_time > self.max_wait_time:
            return float('inf')  # Highest priority
        
        # Linear increase in priority with wait time
        return seq.base_priority + (wait_time / self.max_wait_time)
```

### Memory-Aware Scheduling

```python
class MemoryAwareScheduler(ContinuousBatchingScheduler):
    """Scheduler that considers memory availability."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.memory_monitor = GPUMemoryMonitor()
        
    def _can_schedule_sequence(self, seq: Sequence) -> bool:
        """Check if sequence can be scheduled given memory constraints."""
        # Estimate memory requirement
        required_blocks = self._estimate_blocks_needed(seq)
        available_blocks = self.block_manager.get_num_free_blocks()
        
        if required_blocks > available_blocks:
            # Check if we can free memory through preemption
            return self._can_free_blocks(required_blocks - available_blocks)
        
        return True
    
    def _estimate_blocks_needed(self, seq: Sequence) -> int:
        """Estimate KV cache blocks needed for sequence."""
        # Account for prompt and maximum generation length
        total_tokens = len(seq.prompt_token_ids) + seq.sampling_params.max_tokens
        blocks_needed = (total_tokens + self.config.block_size - 1) // self.config.block_size
        
        # Check for prefix caching opportunities
        cached_blocks = self.block_manager.get_cached_blocks_for_tokens(
            seq.prompt_token_ids
        )
        
        return blocks_needed - len(cached_blocks)
    
    def _can_free_blocks(self, num_blocks: int) -> bool:
        """Check if we can free enough blocks through preemption."""
        # Find preemption candidates
        candidates = self._find_preemption_candidates()
        
        freed_blocks = 0
        for candidate in candidates:
            freed_blocks += len(candidate.block_table)
            if freed_blocks >= num_blocks:
                return True
        
        return False
```

### Chunked Prefill Implementation

```python
class ChunkedPrefillScheduler(ContinuousBatchingScheduler):
    """Breaks long prefills into chunks for better batching."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.prefill_chunk_size = config.prefill_chunk_size
        self.chunked_prefill_states = {}  # seq_id -> chunk_state
    
    def _schedule_prefill_chunk(self, seq: Sequence) -> Optional[ScheduledSequence]:
        """Schedule a chunk of prefill for a sequence."""
        seq_id = seq.seq_id
        
        # Initialize chunk state if needed
        if seq_id not in self.chunked_prefill_states:
            self.chunked_prefill_states[seq_id] = ChunkState(
                total_tokens=len(seq.prompt_token_ids),
                processed_tokens=0
            )
        
        state = self.chunked_prefill_states[seq_id]
        
        # Calculate chunk boundaries
        chunk_start = state.processed_tokens
        chunk_end = min(
            chunk_start + self.prefill_chunk_size,
            state.total_tokens
        )
        chunk_size = chunk_end - chunk_start
        
        # Create scheduled sequence for this chunk
        scheduled = ScheduledSequence(
            sequence=seq,
            token_chunk_size=chunk_size,
            is_prefill_chunk=True,
            chunk_start=chunk_start,
            chunk_end=chunk_end
        )
        
        # Update state
        state.processed_tokens = chunk_end
        
        # Check if prefill is complete
        if state.processed_tokens >= state.total_tokens:
            del self.chunked_prefill_states[seq_id]
            seq.prefill_complete = True
        
        return scheduled
```

### Mixed Prefill-Decode Batching

```python
class MixedBatchScheduler(ContinuousBatchingScheduler):
    """Allows mixing prefill and decode in same batch."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.enable_mixed_batching = config.enable_mixed_batching
        self.prefill_token_ratio = config.prefill_token_ratio  # e.g., 0.5
        
    def create_mixed_batch(self) -> SchedulerOutput:
        """Create a batch mixing prefill and decode sequences."""
        budget = SchedulingBudget(
            token_budget=self.batch_token_limit,
            max_num_seqs=self.config.max_num_seqs
        )
        
        # Allocate portion of budget to prefill
        prefill_budget = int(budget.token_budget * self.prefill_token_ratio)
        decode_budget = budget.token_budget - prefill_budget
        
        # Schedule prefills up to budget
        prefill_sequences = self._schedule_with_budget(
            self.waiting, prefill_budget, is_prefill=True
        )
        
        # Schedule decodes with remaining budget
        decode_sequences = self._schedule_with_budget(
            self.running, decode_budget, is_prefill=False
        )
        
        # Merge into single batch
        return self._merge_batches(prefill_sequences, decode_sequences)
    
    def _merge_batches(self, prefill_seqs: List[Sequence], 
                      decode_seqs: List[Sequence]) -> SchedulerOutput:
        """Merge prefill and decode sequences into single batch."""
        # Create attention mask that handles both types
        max_seq_len = max(
            max(seq.get_len() for seq in prefill_seqs) if prefill_seqs else 0,
            max(seq.get_len() for seq in decode_seqs) if decode_seqs else 0
        )
        
        # Build combined input tensors
        all_sequences = prefill_seqs + decode_seqs
        input_ids = self._build_mixed_input_ids(all_sequences)
        positions = self._build_mixed_positions(all_sequences)
        attn_metadata = self._build_mixed_attention_metadata(
            prefill_seqs, decode_seqs, max_seq_len
        )
        
        return SchedulerOutput(
            scheduled_sequences=all_sequences,
            num_prefill_tokens=sum(seq.get_len() for seq in prefill_seqs),
            num_decode_tokens=len(decode_seqs),
            blocks_to_swap_in=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            attn_metadata=attn_metadata
        )
```

## Performance Optimizations

### Batch Formation Optimization

```python
class OptimizedBatchFormer:
    """Optimizes batch formation for maximum throughput."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.batch_stats = BatchStatistics()
        
    def form_optimal_batch(self, candidates: List[Sequence]) -> List[Sequence]:
        """Form batch optimizing for various factors."""
        # Sort by multiple criteria
        scored_sequences = []
        
        for seq in candidates:
            score = self._calculate_sequence_score(seq)
            scored_sequences.append((score, seq))
        
        # Sort by score (higher is better)
        scored_sequences.sort(reverse=True, key=lambda x: x[0])
        
        # Greedily select sequences
        batch = []
        total_tokens = 0
        
        for score, seq in scored_sequences:
            seq_tokens = self._get_sequence_tokens(seq)
            
            if (len(batch) < self.config.max_num_seqs and
                total_tokens + seq_tokens <= self.config.max_num_batched_tokens):
                batch.append(seq)
                total_tokens += seq_tokens
        
        return batch
    
    def _calculate_sequence_score(self, seq: Sequence) -> float:
        """Calculate scheduling score for sequence."""
        score = 0.0
        
        # Prioritize sequences close to completion
        if seq.get_output_len() > 0:
            progress = seq.get_output_len() / seq.sampling_params.max_tokens
            score += progress * 10
        
        # Prioritize high-priority requests
        score += seq.priority * 5
        
        # Consider wait time
        wait_time = time.time() - seq.arrival_time
        score += min(wait_time / 10, 5)  # Cap at 5 points
        
        # Penalize very long sequences
        if seq.get_len() > 1024:
            score -= (seq.get_len() - 1024) / 1000
        
        return score
```

### Padding and Packing Optimization

```python
class PaddingOptimizer:
    """Optimizes padding for better GPU utilization."""
    
    def __init__(self):
        self.padding_stats = defaultdict(list)
        
    def optimize_sequence_packing(self, sequences: List[Sequence]) -> List[List[Sequence]]:
        """Pack sequences to minimize padding waste."""
        # Group sequences by similar lengths
        length_groups = defaultdict(list)
        
        for seq in sequences:
            # Round to nearest power of 2 for grouping
            bucket = 2 ** int(math.log2(seq.get_len()) + 0.5)
            length_groups[bucket].append(seq)
        
        # Create micro-batches with similar-length sequences
        micro_batches = []
        
        for bucket, bucket_seqs in length_groups.items():
            # Further optimize within bucket
            bucket_seqs.sort(key=lambda s: s.get_len())
            
            current_batch = []
            current_max_len = 0
            
            for seq in bucket_seqs:
                seq_len = seq.get_len()
                
                # Check if adding this sequence increases padding too much
                if current_batch:
                    new_max_len = max(current_max_len, seq_len)
                    padding_waste = len(current_batch) * (new_max_len - current_max_len)
                    
                    if padding_waste > self._padding_threshold(len(current_batch)):
                        # Start new micro-batch
                        micro_batches.append(current_batch)
                        current_batch = [seq]
                        current_max_len = seq_len
                    else:
                        current_batch.append(seq)
                        current_max_len = new_max_len
                else:
                    current_batch = [seq]
                    current_max_len = seq_len
            
            if current_batch:
                micro_batches.append(current_batch)
        
        return micro_batches
    
    def _padding_threshold(self, batch_size: int) -> int:
        """Dynamic padding threshold based on batch size."""
        # Allow more padding for larger batches
        return batch_size * 64  # 64 tokens per sequence
```

### Scheduling Lookahead

```python
class LookaheadScheduler(ContinuousBatchingScheduler):
    """Uses lookahead to make better scheduling decisions."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.lookahead_steps = config.lookahead_steps
        
    def schedule_with_lookahead(self) -> SchedulerOutput:
        """Schedule considering future states."""
        # Simulate multiple future scheduling steps
        best_schedule = None
        best_score = float('-inf')
        
        # Try different scheduling strategies
        strategies = [
            self._greedy_schedule,
            self._balanced_schedule,
            self._memory_first_schedule
        ]
        
        for strategy in strategies:
            # Simulate scheduling
            simulated_state = self._copy_state()
            schedule = strategy(simulated_state)
            
            # Simulate future steps
            future_score = self._simulate_future(simulated_state, schedule)
            
            if future_score > best_score:
                best_score = future_score
                best_schedule = schedule
        
        return best_schedule
    
    def _simulate_future(self, state: SchedulerState, 
                        initial_schedule: SchedulerOutput) -> float:
        """Simulate future scheduling steps and calculate score."""
        total_score = 0.0
        
        # Apply initial schedule
        state.apply_schedule(initial_schedule)
        total_score += self._calculate_state_score(state)
        
        # Simulate future steps
        for step in range(self.lookahead_steps):
            # Simple greedy scheduling for simulation
            future_schedule = self._greedy_schedule(state)
            state.apply_schedule(future_schedule)
            
            # Discount future rewards
            discount = 0.9 ** (step + 1)
            total_score += discount * self._calculate_state_score(state)
        
        return total_score
    
    def _calculate_state_score(self, state: SchedulerState) -> float:
        """Score a scheduler state."""
        score = 0.0
        
        # Throughput component
        score += state.num_running_sequences * 10
        
        # Utilization component
        utilization = state.num_batched_tokens / self.batch_token_limit
        score += utilization * 50
        
        # Fairness component (negative score for long waits)
        max_wait = max((time.time() - seq.arrival_time for seq in state.waiting), default=0)
        score -= max_wait * 2
        
        return score
```

## Monitoring and Debugging

### Scheduler Metrics

```python
@dataclass
class SchedulerMetrics:
    """Comprehensive scheduler metrics."""
    # Throughput metrics
    scheduled_tokens: int = 0
    scheduled_sequences: int = 0
    completed_sequences: int = 0
    
    # Latency metrics
    time_to_first_token: List[float] = field(default_factory=list)
    time_per_token: List[float] = field(default_factory=list)
    queue_wait_times: List[float] = field(default_factory=list)
    
    # Utilization metrics
    gpu_utilization_history: List[float] = field(default_factory=list)
    batch_size_history: List[int] = field(default_factory=list)
    padding_waste_history: List[float] = field(default_factory=list)
    
    # Preemption metrics
    num_preemptions: int = 0
    num_swaps_out: int = 0
    num_swaps_in: int = 0
    
    def record_iteration(self, output: SchedulerOutput, iteration_time: float):
        """Record metrics for one scheduling iteration."""
        # Update throughput
        num_tokens = output.num_prefill_tokens + output.num_decode_tokens
        self.scheduled_tokens += num_tokens
        self.scheduled_sequences += len(output.scheduled_sequences)
        
        # Update utilization
        if num_tokens > 0:
            utilization = num_tokens / output.token_budget
            self.gpu_utilization_history.append(utilization)
        
        # Batch size
        self.batch_size_history.append(len(output.scheduled_sequences))
        
        # Calculate padding waste
        if output.scheduled_sequences:
            max_len = max(seq.get_len() for seq in output.scheduled_sequences)
            total_len = sum(seq.get_len() for seq in output.scheduled_sequences)
            padding_waste = (len(output.scheduled_sequences) * max_len - total_len) / total_len
            self.padding_waste_history.append(padding_waste)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            'avg_throughput_tokens_per_sec': self.scheduled_tokens / sum(self.time_per_token),
            'avg_batch_size': np.mean(self.batch_size_history),
            'avg_gpu_utilization': np.mean(self.gpu_utilization_history),
            'avg_padding_waste': np.mean(self.padding_waste_history),
            'p50_time_to_first_token': np.percentile(self.time_to_first_token, 50),
            'p99_time_to_first_token': np.percentile(self.time_to_first_token, 99),
            'preemption_rate': self.num_preemptions / self.completed_sequences
        }
```

### Debugging Tools

```python
class SchedulerDebugger:
    """Tools for debugging scheduler behavior."""
    
    def __init__(self, scheduler: ContinuousBatchingScheduler):
        self.scheduler = scheduler
        self.trace = []
        
    def trace_scheduling_decision(self, iteration: int):
        """Capture detailed scheduling decision."""
        decision = {
            'iteration': iteration,
            'timestamp': time.time(),
            'waiting_sequences': len(self.scheduler.waiting),
            'running_sequences': len(self.scheduler.running),
            'swapped_sequences': len(self.scheduler.swapped),
            'free_gpu_blocks': self.scheduler.block_manager.get_num_free_blocks(),
            'decisions': []
        }
        
        # Trace why each sequence was scheduled or not
        for seq in self.scheduler.waiting[:10]:  # Top 10 waiting
            scheduled, reason = self._analyze_sequence_scheduling(seq)
            decision['decisions'].append({
                'seq_id': seq.seq_id,
                'scheduled': scheduled,
                'reason': reason,
                'wait_time': time.time() - seq.arrival_time,
                'priority': seq.priority
            })
        
        self.trace.append(decision)
        
        # Keep trace bounded
        if len(self.trace) > 1000:
            self.trace = self.trace[-1000:]
    
    def _analyze_sequence_scheduling(self, seq: Sequence) -> Tuple[bool, str]:
        """Analyze why a sequence was or wasn't scheduled."""
        # Check various conditions
        if len(self.scheduler.running) >= self.scheduler.config.max_num_seqs:
            return False, "max_sequences_reached"
        
        required_blocks = self._estimate_blocks(seq)
        free_blocks = self.scheduler.block_manager.get_num_free_blocks()
        
        if required_blocks > free_blocks:
            return False, f"insufficient_memory_need_{required_blocks}_have_{free_blocks}"
        
        token_budget = self.scheduler.batch_token_limit
        current_tokens = sum(s.get_len() for s in self.scheduler.running)
        
        if current_tokens + len(seq.prompt_token_ids) > token_budget:
            return False, f"token_budget_exceeded_{current_tokens}+{len(seq.prompt_token_ids)}>{token_budget}"
        
        return True, "scheduled"
    
    def visualize_schedule(self, iteration: int):
        """Create visual representation of current schedule."""
        print(f"\n=== Iteration {iteration} ===")
        print(f"Waiting: {len(self.scheduler.waiting)}")
        print(f"Running: {len(self.scheduler.running)}")
        print(f"Memory: {self.scheduler.block_manager.get_num_free_blocks()}/{self.scheduler.block_manager.num_blocks} blocks free")
        
        # Show batch composition
        if self.scheduler.running:
            print("\nBatch composition:")
            for seq in self.scheduler.running[:10]:
                status = "P" if seq.is_prefill() else "D"
                print(f"  [{status}] Seq {seq.seq_id}: {seq.get_len()} tokens, priority={seq.priority}")
```

## Best Practices

### 1. Tuning Batch Parameters

```python
def auto_tune_batch_parameters(workload: List[Sequence]) -> Dict[str, int]:
    """Automatically tune batching parameters for workload."""
    # Analyze workload characteristics
    prompt_lengths = [len(seq.prompt_token_ids) for seq in workload]
    max_tokens = [seq.sampling_params.max_tokens for seq in workload]
    
    # Calculate statistics
    avg_prompt_len = np.mean(prompt_lengths)
    p95_prompt_len = np.percentile(prompt_lengths, 95)
    avg_generation_len = np.mean(max_tokens)
    
    # Recommend parameters
    recommendations = {
        'max_num_batched_tokens': int(p95_prompt_len + avg_generation_len * 8),
        'max_num_seqs': min(256, int(40000 / avg_prompt_len)),  # Based on memory
        'prefill_chunk_size': int(p95_prompt_len / 4),  # Quarter of P95
        'enable_mixed_batching': avg_prompt_len < 1024  # Enable for shorter prompts
    }
    
    return recommendations
```

### 2. Handling Edge Cases

```python
class RobustScheduler(ContinuousBatchingScheduler):
    """Scheduler with robust edge case handling."""
    
    def handle_extreme_sequences(self, seq: Sequence) -> bool:
        """Handle sequences with extreme characteristics."""
        # Very long prompts
        if len(seq.prompt_token_ids) > self.prompt_limit:
            # Use chunked prefill
            return self._schedule_chunked_prefill(seq)
        
        # Very high priority
        if seq.priority > 1000:
            # Preempt if necessary
            return self._emergency_schedule(seq)
        
        # Infinite generation
        if seq.sampling_params.max_tokens == float('inf'):
            # Set reasonable limit
            seq.sampling_params.max_tokens = 2048
            
        return True
```

### 3. Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor and alert on performance issues."""
    
    def __init__(self, scheduler: ContinuousBatchingScheduler):
        self.scheduler = scheduler
        self.alerts = []
        
    def check_health(self):
        """Check scheduler health and performance."""
        # Check queue growth
        if len(self.scheduler.waiting) > 100:
            self.alerts.append({
                'type': 'high_queue_depth',
                'value': len(self.scheduler.waiting),
                'threshold': 100
            })
        
        # Check preemption rate
        recent_preemptions = self._get_recent_preemption_rate()
        if recent_preemptions > 0.1:  # More than 10%
            self.alerts.append({
                'type': 'high_preemption_rate',
                'value': recent_preemptions,
                'threshold': 0.1
            })
        
        # Check GPU utilization
        gpu_util = self._get_gpu_utilization()
        if gpu_util < 0.7:  # Less than 70%
            self.alerts.append({
                'type': 'low_gpu_utilization',
                'value': gpu_util,
                'threshold': 0.7
            })
```

## Conclusion

Continuous batching is essential for production LLM serving:

1. **Dynamic scheduling** maximizes GPU utilization
2. **Memory-aware scheduling** prevents OOM and enables smart preemption  
3. **Mixed batching** handles diverse workloads efficiently
4. **Monitoring and debugging** tools ensure reliable operation
5. **Automatic tuning** adapts to workload characteristics

Key insights:
- Balance between throughput and latency
- Consider memory constraints in scheduling decisions
- Use chunked prefill for long prompts
- Monitor and adapt to workload patterns
- Implement robust error handling

With continuous batching, modern inference engines achieve 2-3x higher throughput compared to static batching while maintaining low latency.