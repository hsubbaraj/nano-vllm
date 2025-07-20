# LLMEngine: The Orchestration Heart

## Overview

The LLMEngine (`nanovllm/engine/llm_engine.py`) is the central coordinator that orchestrates all components of the inference system. It manages the complete request lifecycle, coordinates multi-process execution, and provides performance monitoring.

## Core Responsibilities

1. **Process Management**: Spawn and coordinate tensor parallel worker processes
2. **Component Orchestration**: Initialize and coordinate scheduler, model runner, tokenizer
3. **Request Lifecycle**: Manage requests from input to output
4. **Performance Monitoring**: Track throughput and latency metrics
5. **Error Handling**: Graceful handling of failures and resource constraints

## Architecture

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # 1. Configuration setup
        self.config = Config(model, **kwargs)
        
        # 2. Multi-process coordination setup
        self.ps = []      # Worker processes
        self.events = []  # Synchronization events
        
        # 3. Component initialization
        self.model_runner = ModelRunner(config, rank=0, events=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.scheduler = Scheduler(config)
        
        # 4. Performance tracking
        self.stats = PerformanceStats()
```

## Multi-Process Architecture Deep Dive

### Process Spawning Strategy

```python
def __init__(self, model, **kwargs):
    config = Config(model, **kwargs)
    
    # Use spawn context for clean process separation
    ctx = mp.get_context("spawn")
    
    # Spawn worker processes for ranks 1, 2, ..., N-1
    for rank in range(1, config.tensor_parallel_size):
        # Each worker needs its own synchronization event
        event = ctx.Event()
        
        # Create worker process with config and rank
        process = ctx.Process(
            target=ModelRunner,
            args=(config, rank, event)
        )
        process.start()
        
        self.ps.append(process)
        self.events.append(event)
    
    # Main process handles rank 0
    self.model_runner = ModelRunner(config, rank=0, events=self.events)
```

### Why Multi-Process Instead of Multi-Threading?

**Benefits of Multi-Process**:
1. **No GIL**: Each process has its own Python interpreter
2. **Process Isolation**: Worker crash doesn't affect main process
3. **Memory Independence**: Each process has its own memory space
4. **Easier Debugging**: Can attach debugger to individual processes

**Trade-offs**:
1. **Higher Memory Usage**: Each process loads its own model partition
2. **IPC Overhead**: Communication through serialization
3. **Startup Cost**: Process spawning is slower than thread creation

### Inter-Process Coordination

```python
class ModelRunner:
    def call(self, method_name, *args, **kwargs):
        """Execute method across all tensor parallel processes"""
        
        if self.rank == 0:
            # Main process: signal all workers
            for event in self.events:
                event.set()  # Wake up worker process
            
            # Execute on main process
            result = getattr(self, method_name)(*args, **kwargs)
            
            # Workers execute in parallel (implicit synchronization)
            return result
        else:
            # Worker process: wait for signal then execute
            self.event.wait()  # Block until signaled
            self.event.clear()  # Reset for next call
            return getattr(self, method_name)(*args, **kwargs)
```

## Request Management System

### Request Addition Pipeline

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    """Add a new request to the processing pipeline"""
    
    # Step 1: Tokenization
    if isinstance(prompt, str):
        prompt_tokens = self.tokenizer.encode(prompt)
    else:
        prompt_tokens = prompt
    
    # Step 2: Sequence creation with unique ID
    seq_id = next(self.seq_id_counter)
    seq = Sequence(
        seq_id=seq_id,
        prompt_tokens=prompt_tokens,
        sampling_params=sampling_params
    )
    
    # Step 3: Add to scheduler
    try:
        self.scheduler.add(seq)
        self.stats.requests_added += 1
    except OutOfMemoryError:
        self.stats.requests_rejected += 1
        raise
```

### Execution Loop Architecture

The heart of the engine is the execution loop that processes requests:

```python
def generate(self, prompts, sampling_params):
    """Main generation loop"""
    
    # Phase 1: Add all requests
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    
    # Phase 2: Execute until all requests complete
    while self.scheduler.has_pending_requests():
        step_start_time = time.perf_counter()
        self.step()
        step_time = time.perf_counter() - step_start_time
        self.stats.update_step_time(step_time)
    
    # Phase 3: Collect and format outputs
    outputs = []
    while self.scheduler.has_completed_requests():
        completed_seq = self.scheduler.get_completed_request()
        output = self.format_output(completed_seq)
        outputs.append(output)
    
    return outputs

def step(self):
    """Single execution step"""
    # Step 1: Schedule next batch
    seqs, is_prefill = self.scheduler.schedule()
    if not seqs:
        return  # Nothing to process
    
    # Step 2: Execute model
    sampled_tokens = self.model_runner.call("run", seqs, is_prefill)
    
    # Step 3: Post-process results
    self.scheduler.postprocess(seqs, sampled_tokens, is_prefill)
    
    # Step 4: Update performance metrics
    self.update_performance_stats(seqs, is_prefill, sampled_tokens)
```

## Performance Monitoring System

### Comprehensive Metrics Collection

```python
class PerformanceStats:
    def __init__(self):
        # Throughput tracking
        self.prefill_throughput = ThroughputCalculator()
        self.decode_throughput = ThroughputCalculator()
        
        # Latency tracking
        self.request_latencies = []
        
        # Resource utilization
        self.memory_usage = MemoryTracker()
        self.gpu_utilization = GPUUtilizationTracker()
        
        # Request statistics
        self.requests_added = 0
        self.requests_completed = 0
        self.requests_rejected = 0
        
    def update_step_performance(self, seqs, is_prefill, step_time):
        """Update performance metrics after each step"""
        
        if is_prefill:
            # Prefill metrics: count all prompt tokens
            total_tokens = sum(len(seq.prompt_tokens) for seq in seqs)
            self.prefill_throughput.add_sample(total_tokens, step_time)
        else:
            # Decode metrics: one token per sequence
            total_tokens = len(seqs)
            self.decode_throughput.add_sample(total_tokens, step_time)
        
        # Update resource utilization
        self.memory_usage.update()
        self.gpu_utilization.update()
    
    def print_stats(self):
        """Print comprehensive performance statistics"""
        print("=== Performance Statistics ===")
        print(f"Requests: {self.requests_completed}/{self.requests_added} completed")
        print(f"Prefill throughput: {self.prefill_throughput.get_throughput():.2f} tok/s")
        print(f"Decode throughput: {self.decode_throughput.get_throughput():.2f} tok/s")
        print(f"Memory usage: {self.memory_usage.get_utilization():.1f}%")
        print(f"GPU utilization: {self.gpu_utilization.get_average():.1f}%")
```

### Throughput Calculation

```python
class ThroughputCalculator:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)  # (tokens, time) pairs
        
    def add_sample(self, num_tokens, time_elapsed):
        """Add a new throughput sample"""
        self.samples.append((num_tokens, time_elapsed))
    
    def get_throughput(self):
        """Calculate current throughput over the window"""
        if not self.samples:
            return 0.0
            
        total_tokens = sum(tokens for tokens, _ in self.samples)
        total_time = sum(time for _, time in self.samples)
        
        if total_time == 0:
            return 0.0
            
        return total_tokens / total_time
    
    def get_instantaneous_throughput(self):
        """Get throughput from most recent sample"""
        if not self.samples:
            return 0.0
            
        tokens, time = self.samples[-1]
        return tokens / time if time > 0 else 0.0
```

## Output Formatting System

### Structured Output Generation

```python
def format_output(self, seq: Sequence):
    """Convert completed sequence to user-friendly output"""
    
    # Decode completion tokens to text
    completion_text = self.tokenizer.decode(
        seq.completion_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Determine finish reason
    if seq.status == SequenceStatus.FINISHED_EOS:
        finish_reason = "stop"
    elif seq.status == SequenceStatus.FINISHED_LENGTH:
        finish_reason = "length"
    elif seq.status == SequenceStatus.FINISHED_ABORT:
        finish_reason = "abort"
    else:
        finish_reason = "unknown"
    
    # Calculate timing metrics
    metrics = {
        'time_to_first_token': seq.metrics.time_to_first_token(),
        'inter_token_latency': seq.metrics.average_inter_token_latency(),
        'total_generation_time': seq.metrics.total_generation_time(),
        'prefill_time': seq.metrics.prefill_time,
        'decode_time': seq.metrics.total_generation_time() - seq.metrics.prefill_time
    }
    
    return {
        'text': completion_text,
        'token_ids': seq.completion_tokens.copy(),
        'finish_reason': finish_reason,
        'usage': {
            'prompt_tokens': len(seq.prompt_tokens),
            'completion_tokens': len(seq.completion_tokens),
            'total_tokens': len(seq.prompt_tokens) + len(seq.completion_tokens)
        },
        'metrics': metrics
    }
```

## Error Handling and Recovery

### Graceful Process Management

```python
def exit(self):
    """Clean shutdown of all processes"""
    try:
        # Signal model runner to shutdown
        self.model_runner.call("exit")
        
        # Clean up model runner resources
        del self.model_runner
        
        # Wait for all worker processes to finish
        for process in self.ps:
            process.join(timeout=10.0)  # Wait max 10 seconds
            if process.is_alive():
                process.terminate()  # Force termination if needed
                process.join()
                
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        # Ensure cleanup happens
        atexit.register(self.exit)
```

### Memory Pressure Handling

```python
def handle_memory_pressure(self):
    """React to memory pressure situations"""
    
    # Get current memory usage
    free_blocks = self.scheduler.block_manager.get_num_free_blocks()
    total_blocks = self.scheduler.block_manager.get_total_blocks()
    utilization = 1.0 - (free_blocks / total_blocks)
    
    if utilization > 0.95:  # Critical memory pressure
        # Try aggressive preemption
        preempted = self.scheduler.preempt_sequences(target_free_blocks=total_blocks * 0.2)
        self.stats.sequences_preempted += preempted
        
        if free_blocks < total_blocks * 0.1:
            # Still critical - start rejecting new requests
            self.scheduler.set_rejection_mode(True)
    
    elif utilization > 0.8:  # Moderate memory pressure
        # Conservative preemption
        self.scheduler.preempt_sequences(target_free_blocks=total_blocks * 0.1)
    
    else:
        # Normal operation
        self.scheduler.set_rejection_mode(False)
```

## Configuration Management

The engine uses a centralized configuration system:

```python
@dataclass
class Config:
    # Model configuration
    model: str
    tensor_parallel_size: int = 1
    
    # Memory configuration
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    block_size: int = 256
    
    # Batching configuration
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    
    # Performance configuration
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    
    def __post_init__(self):
        # Validation
        assert 1 <= self.tensor_parallel_size <= 8
        assert 0.1 <= self.gpu_memory_utilization <= 1.0
        assert self.block_size % 256 == 0  # Hardware alignment
```

## Integration Points

### With Scheduler
```python
# Engine provides high-level orchestration
seqs, is_prefill = self.scheduler.schedule()
self.scheduler.postprocess(seqs, tokens, is_prefill)

# Scheduler provides detailed request management
class Scheduler:
    def schedule(self): ...      # Return next batch to process
    def postprocess(self): ...   # Update state after processing
    def has_pending_requests(self): ...
```

### With ModelRunner
```python
# Engine coordinates distributed execution
tokens = self.model_runner.call("run", seqs, is_prefill)

# ModelRunner handles actual model execution
class ModelRunner:
    def call(self, method, *args): ...  # Coordinate across processes
    def run(self, seqs, is_prefill): ... # Execute model
```

### With Components
```python
# Clean separation of concerns
self.tokenizer.encode(prompt)      # Text processing
self.scheduler.add(seq)            # Request management  
self.model_runner.call(...)        # Model execution
self.format_output(seq)            # Output processing
```

## Performance Optimizations

### Batching Strategy
- **Prefill Batching**: Pack variable-length sequences efficiently
- **Decode Batching**: Process all running sequences together
- **Memory-Aware**: Respect memory constraints when forming batches

### Process Communication
- **Minimal Serialization**: Only method calls are serialized
- **Event-Based Sync**: Efficient synchronization using multiprocessing.Event
- **Result Caching**: Avoid redundant communication

### Memory Management
- **Lazy Allocation**: Only allocate resources when needed
- **Reference Tracking**: Automatic cleanup through reference counting
- **Pressure Monitoring**: Proactive memory pressure handling

## Key Design Patterns

1. **Orchestrator Pattern**: Central coordinator managing multiple subsystems
2. **Process Pool Pattern**: Multi-process execution with coordination
3. **Pipeline Pattern**: Request flows through processing stages
4. **Observer Pattern**: Performance monitoring throughout the system
5. **Strategy Pattern**: Configurable behavior through dependency injection

## Common Issues and Solutions

### Issue: Process Hangs
**Cause**: Deadlock in inter-process communication
**Solution**: Timeout-based synchronization, proper event handling

### Issue: Memory Leaks  
**Cause**: Circular references between objects
**Solution**: Explicit cleanup, weak references, context managers

### Issue: Poor Performance
**Cause**: Small batch sizes, inefficient scheduling
**Solution**: Better batching policies, continuous batching

The LLMEngine demonstrates how to build a robust, high-performance orchestration layer that coordinates complex distributed systems while maintaining clean abstractions and comprehensive monitoring.