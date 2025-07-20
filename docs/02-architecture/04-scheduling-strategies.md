# Scheduling Strategies for LLM Inference

## Overview

The scheduler is the brain of an LLM inference engine, responsible for deciding which requests to process, when to process them, and how to batch them together. Good scheduling can dramatically improve throughput while maintaining acceptable latency.

This guide explores various scheduling strategies, from simple FIFO to advanced techniques like continuous batching and priority-based scheduling.

## The Scheduling Challenge

### Unique Constraints in LLM Inference

1. **Variable Sequence Lengths**: Requests have different prompt and output lengths
2. **Memory Limitations**: KV cache size limits concurrent requests
3. **Iterative Generation**: Each request generates tokens one at a time
4. **Different Phases**: Prefill (parallel) vs decode (sequential) have different characteristics
5. **Preemption Needs**: May need to evict requests when memory is tight

### Key Metrics to Optimize

- **Throughput**: Tokens generated per second
- **Latency**: Time to first token (TTFT) and time per output token (TPOT)  
- **Fairness**: Preventing starvation of requests
- **Memory Utilization**: Maximizing useful work per GPU memory byte

## Basic Scheduling Strategies

### 1. First-In-First-Out (FIFO)

The simplest approach - process requests in arrival order:

```python
class FIFOScheduler:
    def __init__(self):
        self.queue = deque()
    
    def add_request(self, request):
        self.queue.append(request)
    
    def get_next_batch(self, max_batch_size):
        batch = []
        while len(batch) < max_batch_size and self.queue:
            batch.append(self.queue.popleft())
        return batch
```

Pros:
- Simple to implement
- Fair in terms of arrival time
- Predictable behavior

Cons:
- No optimization for throughput
- Can't handle different priorities
- Inefficient memory usage

### 2. Shortest-Job-First (SJF)

Prioritize requests with shorter expected lengths:

```python
class SJFScheduler:
    def __init__(self):
        self.queue = []
    
    def add_request(self, request):
        # Estimate total tokens (prompt + expected generation)
        estimated_length = len(request.prompt_tokens) + request.max_new_tokens
        heapq.heappush(self.queue, (estimated_length, request))
    
    def get_next_batch(self, max_batch_size):
        batch = []
        while len(batch) < max_batch_size and self.queue:
            _, request = heapq.heappop(self.queue)
            batch.append(request)
        return batch
```

Pros:
- Minimizes average completion time
- Good for mixed workloads

Cons:
- Can starve long requests
- Requires length estimation

## Continuous Batching

### The Revolution in LLM Scheduling

Traditional batching waits for all sequences in a batch to complete before starting new ones. Continuous batching allows sequences to join and leave dynamically:

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size, max_num_tokens):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting = deque()
        self.running = []
        
    def schedule(self):
        scheduled = ScheduledBatch()
        
        # Continue running sequences (decode)
        for seq in self.running:
            if not seq.is_finished():
                scheduled.decode_sequences.append(seq)
        
        # Add new sequences (prefill) if room available
        available_slots = self.max_batch_size - len(scheduled.decode_sequences)
        available_tokens = self.max_num_tokens - self._count_tokens(scheduled)
        
        while self.waiting and available_slots > 0 and available_tokens > 0:
            seq = self.waiting[0]
            if len(seq.prompt_tokens) <= available_tokens:
                self.waiting.popleft()
                scheduled.prefill_sequences.append(seq)
                available_slots -= 1
                available_tokens -= len(seq.prompt_tokens)
            else:
                break  # Can't fit this sequence
                
        return scheduled
```

### Benefits of Continuous Batching

1. **Higher GPU Utilization**: No idle time waiting for stragglers
2. **Lower Latency**: New requests start immediately
3. **Better Memory Usage**: Can pack more efficiently

## Memory-Aware Scheduling

### Integration with Block Manager

The scheduler must work closely with memory management:

```python
class MemoryAwareScheduler:
    def __init__(self, block_manager, ...):
        self.block_manager = block_manager
        
    def can_schedule_sequence(self, seq):
        # Check if we have enough free blocks
        required_blocks = math.ceil(seq.get_len() / self.block_size)
        available_blocks = self.block_manager.get_num_free_blocks()
        
        # Account for potential prefix caching
        cached_blocks = self.block_manager.get_cached_blocks(seq)
        needed_blocks = required_blocks - len(cached_blocks)
        
        return needed_blocks <= available_blocks
    
    def schedule_with_preemption(self):
        scheduled = []
        
        # Try to schedule waiting sequences
        for seq in self.waiting[:]:
            if self.can_schedule_sequence(seq):
                scheduled.append(seq)
                self.waiting.remove(seq)
            else:
                # Try preemption
                preempted = self.try_preempt_for(seq)
                if preempted:
                    scheduled.append(seq)
                    self.waiting.remove(seq)
```

### Preemption Strategies

1. **Least Recently Used (LRU)**:
```python
def select_victim_lru(self, running_sequences):
    return min(running_sequences, key=lambda s: s.last_token_time)
```

2. **Longest Running First**:
```python
def select_victim_longest(self, running_sequences):
    return max(running_sequences, key=lambda s: s.get_output_len())
```

3. **Lowest Priority**:
```python
def select_victim_priority(self, running_sequences):
    return min(running_sequences, key=lambda s: s.priority)
```

## Advanced Scheduling Techniques

### 1. Priority-Based Scheduling

Support different service levels:

```python
class PriorityScheduler:
    def __init__(self):
        self.priority_queues = {
            'high': deque(),
            'medium': deque(),
            'low': deque()
        }
        
    def schedule(self):
        scheduled = []
        
        # Process in priority order
        for priority in ['high', 'medium', 'low']:
            queue = self.priority_queues[priority]
            while queue and self.has_capacity(scheduled):
                scheduled.append(queue.popleft())
                
        return scheduled
```

### 2. Deadline-Aware Scheduling

Meet SLA requirements:

```python
class DeadlineScheduler:
    def __init__(self):
        self.requests = []
        
    def add_request(self, request, deadline):
        heapq.heappush(self.requests, (deadline, request))
        
    def schedule(self):
        now = time.time()
        scheduled = []
        
        # Process requests closest to deadline
        while self.requests and self.has_capacity(scheduled):
            deadline, request = heapq.heappop(self.requests)
            
            # Check if we can still meet deadline
            estimated_completion = now + self.estimate_time(request)
            if estimated_completion <= deadline:
                scheduled.append(request)
            else:
                # Log SLA violation
                self.log_missed_deadline(request)
```

### 3. Fairness-Aware Scheduling

Prevent starvation:

```python
class FairScheduler:
    def __init__(self, max_wait_time=30.0):
        self.max_wait_time = max_wait_time
        
    def schedule(self):
        scheduled = []
        now = time.time()
        
        # First, schedule any requests that have waited too long
        for request in self.waiting:
            wait_time = now - request.arrival_time
            if wait_time > self.max_wait_time:
                scheduled.append(request)
                
        # Then use normal scheduling for remaining capacity
        # ...
```

## Batching Strategies

### 1. Naive Batching

Process fixed-size batches:
```python
def naive_batch(requests, batch_size=8):
    batches = []
    for i in range(0, len(requests), batch_size):
        batches.append(requests[i:i+batch_size])
    return batches
```

### 2. Dynamic Batching

Adjust batch size based on workload:
```python
class DynamicBatcher:
    def __init__(self, min_batch=1, max_batch=32, wait_time=0.01):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.wait_time = wait_time
        
    def collect_batch(self):
        batch = []
        deadline = time.time() + self.wait_time
        
        while len(batch) < self.max_batch and time.time() < deadline:
            if self.queue:
                batch.append(self.queue.popleft())
            else:
                time.sleep(0.001)  # Small sleep to avoid busy waiting
                
        return batch if len(batch) >= self.min_batch else []
```

### 3. Token-Aware Batching

Consider total tokens, not just request count:
```python
def token_aware_batch(requests, max_tokens=2048):
    batch = []
    total_tokens = 0
    
    for request in requests:
        request_tokens = len(request.prompt_tokens)
        if total_tokens + request_tokens <= max_tokens:
            batch.append(request)
            total_tokens += request_tokens
        else:
            break  # Can't fit more
            
    return batch
```

## Scheduling for Different Phases

### Prefill vs Decode Optimization

The two phases have different characteristics:

```python
class PhaseAwareScheduler:
    def __init__(self):
        self.prefill_batch_size = 4  # Smaller due to memory
        self.decode_batch_size = 32  # Larger for efficiency
        
    def schedule(self):
        # Separate prefill and decode scheduling
        prefill_batch = self.schedule_prefill()
        decode_batch = self.schedule_decode()
        
        # Can mix if total tokens allow
        if self.can_mix_phases(prefill_batch, decode_batch):
            return self.merge_batches(prefill_batch, decode_batch)
        else:
            # Process separately
            return prefill_batch if prefill_batch else decode_batch
```

### Chunked Prefill

For very long prompts, break into chunks:

```python
class ChunkedPrefillScheduler:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        
    def schedule_prefill(self, sequence):
        prompt_len = len(sequence.prompt_tokens)
        
        if prompt_len <= self.chunk_size:
            # Single prefill
            return [(sequence, 0, prompt_len)]
        else:
            # Multiple chunks
            chunks = []
            for start in range(0, prompt_len, self.chunk_size):
                end = min(start + self.chunk_size, prompt_len)
                chunks.append((sequence, start, end))
            return chunks
```

## Performance Optimization

### 1. Request Reordering

Optimize memory access patterns:
```python
def reorder_for_cache_locality(sequences):
    # Group sequences with shared prefixes
    prefix_groups = defaultdict(list)
    
    for seq in sequences:
        prefix_hash = hash(tuple(seq.prompt_tokens[:100]))
        prefix_groups[prefix_hash].append(seq)
    
    # Flatten groups to maintain locality
    reordered = []
    for group in prefix_groups.values():
        reordered.extend(group)
        
    return reordered
```

### 2. Lookahead Scheduling

Predict future resource needs:
```python
class LookaheadScheduler:
    def __init__(self, lookahead_steps=5):
        self.lookahead_steps = lookahead_steps
        
    def schedule_with_lookahead(self):
        current_state = self.get_current_state()
        best_schedule = None
        best_score = -inf
        
        # Try different scheduling decisions
        for schedule in self.generate_candidates():
            score = self.simulate_future(schedule, self.lookahead_steps)
            if score > best_score:
                best_score = score
                best_schedule = schedule
                
        return best_schedule
```

### 3. Hardware-Aware Scheduling

Consider GPU characteristics:
```python
class HardwareAwareScheduler:
    def __init__(self, gpu_type):
        # Different GPUs have different optimal batch sizes
        self.optimal_sizes = {
            'A100': {'prefill': 4, 'decode': 64},
            'V100': {'prefill': 2, 'decode': 32},
            'T4': {'prefill': 1, 'decode': 16}
        }
        self.gpu_config = self.optimal_sizes.get(gpu_type)
```

## Monitoring and Metrics

### Key Metrics to Track

```python
class SchedulerMetrics:
    def __init__(self):
        self.requests_scheduled = 0
        self.requests_preempted = 0
        self.total_wait_time = 0
        self.gpu_utilization = []
        self.memory_utilization = []
        self.batch_sizes = []
        
    def record_scheduling_decision(self, decision):
        self.requests_scheduled += len(decision.scheduled)
        self.requests_preempted += len(decision.preempted)
        self.batch_sizes.append(len(decision.scheduled))
        
    def get_average_wait_time(self):
        return self.total_wait_time / self.requests_scheduled
```

### Debugging Scheduling Issues

Common problems and solutions:

1. **Low GPU Utilization**:
   - Increase batch size limits
   - Reduce scheduling overhead
   - Enable continuous batching

2. **High Latency Variance**:
   - Implement fairness mechanisms
   - Set maximum wait times
   - Use priority scheduling

3. **Memory Fragmentation**:
   - Improve preemption strategy
   - Better memory-aware scheduling
   - Adjust block sizes

## Best Practices

### 1. Start Simple
Begin with FIFO and gradually add complexity:
```python
# Start here
scheduler = FIFOScheduler()

# Then add batching
scheduler = BatchingScheduler(max_batch_size=8)

# Then continuous batching
scheduler = ContinuousBatchingScheduler()

# Finally, add advanced features
scheduler = PriorityMemoryAwareScheduler()
```

### 2. Profile Your Workload
Understand your request patterns:
```python
def analyze_workload(requests):
    prompt_lengths = [len(r.prompt_tokens) for r in requests]
    output_lengths = [r.num_output_tokens for r in requests]
    
    print(f"Average prompt length: {np.mean(prompt_lengths)}")
    print(f"P95 prompt length: {np.percentile(prompt_lengths, 95)}")
    print(f"Average output length: {np.mean(output_lengths)}")
```

### 3. Tune for Your SLA
Different applications need different strategies:
- **Chat**: Optimize for low TTFT
- **Batch processing**: Optimize for throughput
- **Real-time**: Optimize for consistent latency

### 4. Monitor Production Behavior
```python
def monitor_scheduler_health():
    metrics = scheduler.get_metrics()
    
    # Alert on concerning patterns
    if metrics.average_wait_time > SLA_THRESHOLD:
        alert("High wait times detected")
        
    if metrics.preemption_rate > 0.1:
        alert("Excessive preemption")
```

## Conclusion

Effective scheduling is crucial for LLM inference performance. Key principles:

1. **Continuous batching** is essential for high utilization
2. **Memory awareness** prevents OOM and enables smart preemption
3. **Phase-specific** optimization improves both prefill and decode
4. **Fairness mechanisms** prevent starvation
5. **Monitoring** helps identify bottlenecks

The scheduler ties together all components of an inference engine. In the next section, we'll dive deep into the model runner that executes the scheduled requests.