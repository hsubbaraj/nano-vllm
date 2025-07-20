# The Model Runner: Executing LLM Inference

## Overview

The Model Runner is the workhorse of an LLM inference engine, responsible for actually executing the model forward pass. It manages GPU execution, handles different inference phases (prefill vs decode), implements performance optimizations like CUDA graphs, and coordinates tensor parallelism across multiple GPUs.

This guide provides an in-depth look at how the Model Runner works, using nano-vLLM's implementation as a reference.

## Core Responsibilities

The Model Runner handles:

1. **Model Loading**: Loading weights and preparing the model for inference
2. **Input Preparation**: Converting sequences to model inputs
3. **Forward Pass Execution**: Running the actual model computation
4. **KV Cache Management**: Storing and retrieving cached keys/values
5. **Performance Optimization**: CUDA graphs, torch compilation, memory management
6. **Tensor Parallelism**: Coordinating execution across multiple GPUs

## Architecture Overview

```python
class ModelRunner:
    def __init__(self, model_path, tp_rank, tp_size, ...):
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # KV cache storage
        self.kv_cache = None
        self.block_manager = BlockManager()
        
        # Performance optimizations
        self.cuda_graphs = {}
        self.graph_runners = {}
        
        # Tensor parallel coordination
        self.tp_rank = tp_rank
        self.tp_size = tp_size
```

## Model Loading and Initialization

### Loading the Model

```python
def load_model(self, model_path):
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    
    # Initialize model architecture
    if config.model_type == "qwen2":
        self.model = Qwen3ForCausalLM(config, self.tp_rank, self.tp_size)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Load weights with tensor parallelism
    self.load_weights(model_path)
    
    # Move to GPU
    self.model = self.model.to(self.device)
    self.model.eval()
```

### Weight Loading with Tensor Parallelism

When using multiple GPUs, weights must be split correctly:

```python
def load_weights(self, model_path):
    # Each rank loads its portion of weights
    state_dict = {}
    
    for name, param in self.model.named_parameters():
        # Load full weight
        full_weight = load_tensor(model_path, name)
        
        # Split according to parallelism strategy
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            # Split attention weights by head
            split_weight = split_by_head(full_weight, self.tp_rank, self.tp_size)
        elif "gate_proj" in name or "up_proj" in name:
            # Split MLP input by column
            split_weight = split_column(full_weight, self.tp_rank, self.tp_size)
        elif "down_proj" in name:
            # Split MLP output by row
            split_weight = split_row(full_weight, self.tp_rank, self.tp_size)
        else:
            # Replicate other weights
            split_weight = full_weight
            
        state_dict[name] = split_weight
    
    self.model.load_state_dict(state_dict)
```

## Input Preparation

### Converting Sequences to Model Inputs

The Model Runner must prepare various inputs for the model:

```python
def prepare_inputs(self, sequences, slot_mapping):
    # Collect tokens from all sequences
    input_tokens = []
    positions = []
    
    for seq in sequences:
        if seq.is_prefill():
            # Prefill: process all prompt tokens
            tokens = seq.prompt_tokens
            pos = list(range(len(tokens)))
        else:
            # Decode: only the last generated token
            tokens = [seq.get_last_token()]
            pos = [seq.get_len() - 1]
            
        input_tokens.extend(tokens)
        positions.extend(pos)
    
    # Convert to tensors
    input_ids = torch.tensor(input_tokens, device=self.device)
    position_ids = torch.tensor(positions, device=self.device)
    
    # Prepare attention mask for prefill
    if any(seq.is_prefill() for seq in sequences):
        attn_mask = self.prepare_attention_mask(sequences)
    else:
        attn_mask = None
        
    return input_ids, position_ids, attn_mask
```

### Attention Mask Creation

For prefill phase with multiple sequences:

```python
def prepare_attention_mask(self, sequences):
    # Create block diagonal attention mask
    total_len = sum(seq.get_len() for seq in sequences)
    mask = torch.full((total_len, total_len), -float('inf'))
    
    offset = 0
    for seq in sequences:
        seq_len = seq.get_len()
        # Each sequence can only attend to itself
        mask[offset:offset+seq_len, offset:offset+seq_len] = 0
        offset += seq_len
        
    return mask.to(self.device)
```

## Forward Pass Execution

### The Main Execution Path

```python
def execute_model(self, sequences, sampling_params):
    # Prepare inputs
    input_ids, positions, attn_mask = self.prepare_inputs(sequences)
    
    # Determine execution mode
    is_prefill = any(seq.is_prefill() for seq in sequences)
    
    # Set up KV cache context
    kv_cache_context = self.setup_kv_cache_context(sequences)
    
    # Execute model
    if not is_prefill and self.use_cuda_graphs:
        # Use CUDA graphs for decode
        logits = self.run_cuda_graph(input_ids, positions, kv_cache_context)
    else:
        # Regular execution for prefill or when graphs not available
        with set_kv_cache_context(kv_cache_context):
            logits = self.model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=attn_mask
            )
    
    # Sample next tokens
    next_tokens = self.sample(logits, sampling_params)
    
    return next_tokens
```

### KV Cache Context Management

The model needs to know where to store/retrieve KV cache:

```python
def setup_kv_cache_context(self, sequences):
    context = KVCacheContext()
    
    # Slot mapping: token index -> physical cache slot
    context.slot_mapping = []
    for seq in sequences:
        for token_idx in range(seq.get_len()):
            block_idx = token_idx // self.block_size
            block_offset = token_idx % self.block_size
            
            if block_idx < len(seq.block_table):
                block_id = seq.block_table[block_idx]
                slot = block_id * self.block_size + block_offset
                context.slot_mapping.append(slot)
    
    # Block tables for decode phase
    context.block_tables = [seq.block_table for seq in sequences]
    
    # Sequence lengths
    context.sequence_lengths = [seq.get_len() for seq in sequences]
    
    return context
```

## CUDA Graph Optimization

### What are CUDA Graphs?

CUDA graphs capture a sequence of GPU operations and can replay them with minimal CPU overhead. This is particularly effective for the decode phase where the computation pattern is fixed.

### Capturing CUDA Graphs

```python
def capture_cuda_graphs(self):
    print("Capturing CUDA graphs...")
    
    # Capture graphs for different batch sizes
    for batch_size in [1, 2, 4, 8, 16, 32]:
        self._capture_graph_for_batch_size(batch_size)
        
def _capture_graph_for_batch_size(self, batch_size):
    # Prepare static inputs
    static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
    static_positions = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
    static_kv_context = self._create_static_kv_context(batch_size)
    
    # Warm up
    for _ in range(3):
        with set_kv_cache_context(static_kv_context):
            self.model(static_input_ids, static_positions)
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with set_kv_cache_context(static_kv_context):
            static_output = self.model(static_input_ids, static_positions)
    
    # Store graph and runner
    self.cuda_graphs[batch_size] = graph
    self.graph_runners[batch_size] = CUDAGraphRunner(
        graph=graph,
        input_ids=static_input_ids,
        positions=static_positions,
        output=static_output,
        kv_context=static_kv_context
    )
```

### Replaying CUDA Graphs

```python
def run_cuda_graph(self, input_ids, positions, kv_context):
    batch_size = input_ids.shape[0]
    
    if batch_size not in self.graph_runners:
        # Fall back to eager execution
        return self.model(input_ids, positions)
    
    runner = self.graph_runners[batch_size]
    
    # Copy inputs to static buffers
    runner.input_ids.copy_(input_ids)
    runner.positions.copy_(positions)
    runner.kv_context.update(kv_context)
    
    # Replay graph
    self.cuda_graphs[batch_size].replay()
    
    # Return output
    return runner.output.clone()
```

## Memory Management

### KV Cache Allocation

```python
def allocate_kv_cache(self):
    # Calculate available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    used_memory = torch.cuda.memory_allocated()
    
    # Reserve memory for model and activations
    model_memory = sum(p.nbytes for p in self.model.parameters())
    activation_memory = 2 * 1024**3  # 2GB reserve
    
    # Calculate KV cache size
    available = total_memory - used_memory - model_memory - activation_memory
    kv_cache_memory = int(0.9 * available)  # Use 90% of available
    
    # Allocate cache
    num_blocks = kv_cache_memory // self._get_block_size_bytes()
    
    self.kv_cache = torch.zeros(
        num_blocks,
        2,  # K and V
        self.num_layers,
        self.num_kv_heads,
        self.block_size,
        self.head_dim,
        dtype=self.dtype,
        device=self.device
    )
    
    print(f"Allocated {num_blocks} KV cache blocks ({kv_cache_memory / 1e9:.2f} GB)")
```

### Memory Pool Warmup

```python
def warmup(self):
    # Allocate maximum possible tensors to reserve memory
    max_batch_size = 256
    max_seq_len = 4096
    
    # Temporary allocations
    warmup_tokens = torch.zeros(max_batch_size, max_seq_len, dtype=torch.long, device=self.device)
    warmup_positions = torch.zeros_like(warmup_tokens)
    
    # Run model once to allocate all internal buffers
    with torch.no_grad():
        self.model(warmup_tokens[:1, :1], warmup_positions[:1, :1])
    
    # Clear warmup tensors
    del warmup_tokens, warmup_positions
    torch.cuda.empty_cache()
    
    # Now allocate KV cache with remaining memory
    self.allocate_kv_cache()
```

## Sampling Implementation

### Basic Sampling

```python
def sample(self, logits, sampling_params):
    next_tokens = []
    
    for i, params in enumerate(sampling_params):
        if params.temperature == 0:
            # Greedy sampling
            token = torch.argmax(logits[i])
        else:
            # Temperature sampling
            probs = torch.softmax(logits[i] / params.temperature, dim=-1)
            token = torch.multinomial(probs, 1)[0]
            
        next_tokens.append(token.item())
    
    return next_tokens
```

### Advanced Sampling Techniques

```python
def sample_with_top_p(self, logits, temperature, top_p):
    # Apply temperature
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff
    cutoff_index = torch.searchsorted(cumsum_probs, top_p).item() + 1
    
    # Zero out tokens beyond cutoff
    probs_filtered = probs.clone()
    probs_filtered[sorted_indices[cutoff_index:]] = 0
    
    # Renormalize and sample
    probs_filtered = probs_filtered / probs_filtered.sum()
    token = torch.multinomial(probs_filtered, 1)[0]
    
    return token
```

## Tensor Parallelism Coordination

### Multi-Process Execution

```python
def run_mp(rank, world_size, model_path, queues):
    # Initialize process
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model runner
    runner = ModelRunner(
        model_path=model_path,
        tp_rank=rank,
        tp_size=world_size
    )
    
    # Main loop
    while True:
        # Get command from main process
        cmd = queues[rank].get()
        
        if cmd["type"] == "generate":
            # Only rank 0 receives full inputs
            if rank == 0:
                sequences = cmd["sequences"]
                sampling_params = cmd["sampling_params"]
            else:
                # Other ranks wait for broadcast
                sequences = None
                sampling_params = None
            
            # Coordinate execution
            output = runner.generate_step(sequences, sampling_params)
            
            # Only rank 0 returns results
            if rank == 0:
                queues[rank].put(output)
        
        elif cmd["type"] == "shutdown":
            break
```

### Synchronization Points

```python
def generate_step(self, sequences, sampling_params):
    # Broadcast inputs from rank 0
    if self.tp_rank == 0:
        input_data = self.prepare_broadcast_data(sequences, sampling_params)
    else:
        input_data = None
    
    input_data = broadcast_object(input_data, src=0)
    
    # All ranks execute model
    with torch.cuda.nvtx.range(f"model_forward_rank_{self.tp_rank}"):
        logits = self.execute_model_local(input_data)
    
    # Gather results at rank 0
    if self.tp_size > 1:
        all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
        dist.all_gather(all_logits, logits)
        
        if self.tp_rank == 0:
            # Combine results
            final_logits = self.combine_tp_results(all_logits)
            next_tokens = self.sample(final_logits, sampling_params)
            return next_tokens
    else:
        # Single GPU
        return self.sample(logits, sampling_params)
```

## Performance Profiling

### Built-in Profiling

```python
class ProfilingModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_data = defaultdict(list)
    
    def execute_model(self, sequences, sampling_params):
        # Time each component
        times = {}
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Input preparation
        start.record()
        inputs = self.prepare_inputs(sequences)
        end.record()
        torch.cuda.synchronize()
        times['input_prep'] = start.elapsed_time(end)
        
        # Model forward
        start.record()
        logits = super().execute_model(sequences, sampling_params)
        end.record()
        torch.cuda.synchronize()
        times['model_forward'] = start.elapsed_time(end)
        
        # Sampling
        start.record()
        tokens = self.sample(logits, sampling_params)
        end.record()
        torch.cuda.synchronize()
        times['sampling'] = start.elapsed_time(end)
        
        # Record
        for key, value in times.items():
            self.profile_data[key].append(value)
        
        return tokens
    
    def print_profile_summary(self):
        for key, values in self.profile_data.items():
            avg_time = np.mean(values)
            p99_time = np.percentile(values, 99)
            print(f"{key}: avg={avg_time:.2f}ms, p99={p99_time:.2f}ms")
```

## Error Handling and Recovery

### Handling OOM Errors

```python
def execute_with_fallback(self, sequences, sampling_params):
    try:
        return self.execute_model(sequences, sampling_params)
    except torch.cuda.OutOfMemoryError:
        # Clear cache and retry with smaller batch
        torch.cuda.empty_cache()
        
        if len(sequences) > 1:
            # Split batch in half
            mid = len(sequences) // 2
            results1 = self.execute_with_fallback(sequences[:mid], sampling_params[:mid])
            results2 = self.execute_with_fallback(sequences[mid:], sampling_params[mid:])
            return results1 + results2
        else:
            # Single sequence still OOM - need preemption
            raise RuntimeError("Single sequence exceeds memory capacity")
```

### Graceful Degradation

```python
def adaptive_execution(self, sequences, sampling_params):
    # Try optimal path first
    if self.cuda_graphs_available and not any(s.is_prefill() for s in sequences):
        try:
            return self.run_cuda_graph(sequences, sampling_params)
        except Exception as e:
            print(f"CUDA graph failed: {e}, falling back to eager")
            self.cuda_graphs_available = False
    
    # Try torch.compile
    if self.compiled_model is not None:
        try:
            return self.run_compiled(sequences, sampling_params)
        except Exception as e:
            print(f"Compiled model failed: {e}, falling back to eager")
            self.compiled_model = None
    
    # Fall back to eager execution
    return self.run_eager(sequences, sampling_params)
```

## Best Practices

### 1. Memory Management
- Pre-allocate all buffers during initialization
- Use memory pools to avoid allocation during inference
- Monitor memory usage and implement proper cleanup

### 2. Performance Optimization
- Use CUDA graphs for decode phase
- Batch operations whenever possible
- Profile regularly to identify bottlenecks

### 3. Error Handling
- Implement fallback mechanisms for all optimizations
- Handle OOM gracefully with batch splitting
- Log errors for debugging but don't crash

### 4. Testing
```python
def test_model_runner():
    runner = ModelRunner(model_path, tp_rank=0, tp_size=1)
    
    # Test single sequence
    seq = Sequence("Test prompt", max_tokens=10)
    tokens = runner.execute_model([seq], [SamplingParams()])
    assert len(tokens) == 1
    
    # Test batching
    seqs = [Sequence(f"Prompt {i}", max_tokens=10) for i in range(8)]
    tokens = runner.execute_model(seqs, [SamplingParams()] * 8)
    assert len(tokens) == 8
    
    # Test prefill vs decode
    # ... more tests
```

## Conclusion

The Model Runner is where computation meets optimization in an LLM inference engine. Key takeaways:

1. **Separation of concerns**: Model execution is isolated from scheduling and memory management
2. **Phase-aware optimization**: Different strategies for prefill vs decode
3. **Performance features**: CUDA graphs, torch compilation, and profiling are essential
4. **Robustness**: Fallback mechanisms ensure reliability
5. **Tensor parallelism**: Careful coordination enables multi-GPU scaling

Understanding the Model Runner is crucial for building high-performance inference engines. Next, we'll explore the Block Manager that handles the complex KV cache allocation the Model Runner depends on.