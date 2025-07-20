# CUDA Graphs: Eliminating Kernel Launch Overhead

## Overview

CUDA graphs capture and replay GPU workloads, eliminating CPU-GPU synchronization overhead. For LLM inference, especially during the decode phase where the computation pattern is fixed, CUDA graphs can provide 10-30% performance improvement by removing kernel launch latency.

## Understanding CUDA Graphs

### The Problem: Kernel Launch Overhead

In typical GPU execution:
```python
# Each operation launches a separate kernel
for _ in range(num_tokens):
    # CPU schedules kernel launch (~5-20μs each)
    hidden = layer_norm(hidden)        # Kernel launch overhead
    hidden = attention(hidden)         # Kernel launch overhead  
    hidden = residual_add(hidden)      # Kernel launch overhead
    hidden = mlp(hidden)              # Kernel launch overhead
    # ... many more kernels
```

For a model with hundreds of operations, this overhead becomes significant.

### The Solution: Graph Capture and Replay

CUDA graphs solve this by:
1. **Capture**: Record the entire workload once
2. **Instantiate**: Create an executable graph
3. **Replay**: Execute the graph with minimal overhead

```python
# Capture phase (once)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input)

# Replay phase (many times)
graph.replay()  # ~1μs total overhead
```

## Basic CUDA Graph Implementation

### Simple Graph Capture

```python
class CUDAGraphRunner:
    def __init__(self, model: nn.Module, batch_size: int, seq_len: int):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Allocate static tensors
        self.static_input = torch.zeros(
            batch_size, seq_len, model.hidden_size,
            device='cuda', dtype=torch.float16
        )
        self.static_output = None
        
        # Capture the graph
        self.graph = self._capture_graph()
    
    def _capture_graph(self):
        # Warmup to ensure kernels are compiled
        for _ in range(3):
            _ = self.model(self.static_input)
        
        # Synchronize before capture
        torch.cuda.synchronize()
        
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.static_output = self.model(self.static_input)
        
        return graph
    
    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Copy input to static buffer
        self.static_input.copy_(input_tensor)
        
        # Replay graph
        self.graph.replay()
        
        # Return output (copy if needed)
        return self.static_output.clone()
```

### Handling Variable Batch Sizes

```python
class MultiSizeCUDAGraphRunner:
    def __init__(self, model: nn.Module, max_batch_size: int):
        self.model = model
        self.graphs = {}
        self.runners = {}
        
        # Pre-capture graphs for common batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for bs in batch_sizes:
            if bs <= max_batch_size:
                self._capture_graph_for_batch_size(bs)
    
    def _capture_graph_for_batch_size(self, batch_size: int):
        print(f"Capturing CUDA graph for batch size {batch_size}")
        
        # Create static tensors
        static_input = torch.zeros(
            batch_size, 1, self.model.hidden_size,  # Decode: seq_len=1
            device='cuda', dtype=torch.float16
        )
        static_output = None
        
        # Warmup
        for _ in range(3):
            self.model(static_input)
        
        # Capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph):
            static_output = self.model(static_input)
        
        # Store graph and buffers
        self.graphs[batch_size] = graph
        self.runners[batch_size] = {
            'input': static_input,
            'output': static_output,
            'graph': graph
        }
    
    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        
        if batch_size in self.runners:
            # Use captured graph
            runner = self.runners[batch_size]
            runner['input'].copy_(input_tensor)
            runner['graph'].replay()
            return runner['output'].clone()
        else:
            # Fallback to eager execution
            return self.model(input_tensor)
```

## Advanced CUDA Graph Features

### Graph Capture with Dynamic Control Flow

```python
class ConditionalCUDAGraph:
    def __init__(self, model: nn.Module):
        self.model = model
        self.graphs = {}
        
    def capture_conditional_graphs(self):
        # Capture different execution paths
        for use_cache in [True, False]:
            for is_prefill in [True, False]:
                key = (use_cache, is_prefill)
                self.graphs[key] = self._capture_graph(use_cache, is_prefill)
    
    def _capture_graph(self, use_cache: bool, is_prefill: bool):
        # Set model mode
        self.model.set_mode(use_cache=use_cache, is_prefill=is_prefill)
        
        # Prepare appropriate input
        if is_prefill:
            input_shape = (1, 128, self.model.hidden_size)  # Longer sequence
        else:
            input_shape = (1, 1, self.model.hidden_size)    # Single token
        
        static_input = torch.zeros(*input_shape, device='cuda', dtype=torch.float16)
        
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(static_input)
        
        return {
            'graph': graph,
            'input': static_input,
            'output': static_output
        }
```

### Memory Pool Management

```python
class CUDAGraphMemoryPool:
    def __init__(self, size_mb: int = 1024):
        self.size = size_mb * 1024 * 1024
        self.pool = None
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        # Create a dedicated memory pool for graphs
        self.pool = torch.cuda.graph_pool_handle()
        
        # Pre-allocate memory
        dummy = torch.zeros(self.size // 4, device='cuda', dtype=torch.float32)
        del dummy
        torch.cuda.empty_cache()
    
    def capture_with_pool(self, func, *args, **kwargs):
        # Use dedicated pool for graph capture
        graph = torch.cuda.CUDAGraph(pool=self.pool)
        
        with torch.cuda.graph(graph):
            output = func(*args, **kwargs)
        
        return graph, output
```

### Stream-Aware Graph Capture

```python
class StreamAwareCUDAGraph:
    def __init__(self, model: nn.Module, num_streams: int = 2):
        self.model = model
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.graphs = []
        
    def capture_on_streams(self):
        for i, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                # Ensure stream ordering
                stream.wait_stream(torch.cuda.current_stream())
                
                # Capture graph on this stream
                graph = torch.cuda.CUDAGraph()
                static_input = torch.zeros(1, 1, self.model.hidden_size, device='cuda')
                
                with torch.cuda.graph(graph, stream=stream):
                    static_output = self.model(static_input)
                
                self.graphs.append({
                    'graph': graph,
                    'stream': stream,
                    'input': static_input,
                    'output': static_output
                })
    
    def run_parallel(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        
        for i, input_tensor in enumerate(inputs):
            graph_info = self.graphs[i % len(self.graphs)]
            
            # Copy input and launch on appropriate stream
            with torch.cuda.stream(graph_info['stream']):
                graph_info['input'].copy_(input_tensor)
                graph_info['graph'].replay()
                outputs.append(graph_info['output'].clone())
        
        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()
        
        return outputs
```

## Integration with LLM Inference

### Model Runner with CUDA Graphs

```python
class GraphOptimizedModelRunner:
    def __init__(self, model: nn.Module, config: InferenceConfig):
        self.model = model
        self.config = config
        
        # Graph runners for different scenarios
        self.decode_graphs = {}
        self.prefill_runner = None  # Prefill typically doesn't use graphs
        
        # Initialize graphs
        self._initialize_cuda_graphs()
    
    def _initialize_cuda_graphs(self):
        print("Initializing CUDA graphs...")
        
        # Capture decode graphs for different batch sizes
        for batch_size in self.config.graph_batch_sizes:
            self._capture_decode_graph(batch_size)
        
        print(f"Captured {len(self.decode_graphs)} decode graphs")
    
    def _capture_decode_graph(self, batch_size: int):
        # Prepare static tensors
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        static_input_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        static_position_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        static_kv_cache = self._create_static_kv_cache(batch_size)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    kv_cache=static_kv_cache,
                    use_cache=True
                )
        
        # Capture graph
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph):
            static_logits = self.model(
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                kv_cache=static_kv_cache,
                use_cache=True
            )
        
        self.decode_graphs[batch_size] = {
            'graph': graph,
            'input_ids': static_input_ids,
            'position_ids': static_position_ids,
            'kv_cache': static_kv_cache,
            'logits': static_logits
        }
    
    def run_decode(self, 
                   input_ids: torch.Tensor,
                   position_ids: torch.Tensor,
                   kv_cache: KVCache) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        
        if batch_size in self.decode_graphs:
            # Use CUDA graph
            return self._run_decode_graph(
                batch_size, input_ids, position_ids, kv_cache
            )
        else:
            # Fallback to eager execution
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_cache=kv_cache,
                use_cache=True
            )
    
    def _run_decode_graph(self, batch_size: int, input_ids: torch.Tensor,
                         position_ids: torch.Tensor, kv_cache: KVCache):
        graph_info = self.decode_graphs[batch_size]
        
        # Copy inputs to static buffers
        graph_info['input_ids'].copy_(input_ids)
        graph_info['position_ids'].copy_(position_ids)
        
        # Update KV cache pointers
        graph_info['kv_cache'].update_from(kv_cache)
        
        # Replay graph
        graph_info['graph'].replay()
        
        # Return logits
        return graph_info['logits']
```

### KV Cache Integration

```python
class GraphCompatibleKVCache:
    def __init__(self, num_blocks: int, num_layers: int, 
                 num_heads: int, head_dim: int, block_size: int):
        # Pre-allocate all memory
        self.cache_data = torch.zeros(
            num_blocks * block_size, num_layers, 2, num_heads, head_dim,
            dtype=torch.float16, device='cuda'
        )
        
        # Static tensors for graph capture
        self.static_k_cache = self.cache_data[:, :, 0]
        self.static_v_cache = self.cache_data[:, :, 1]
        
        # Dynamic metadata (not part of graph)
        self.slot_mapping = None
        self.block_tables = None
    
    def prepare_for_graph(self, slot_mapping: torch.Tensor, 
                         block_tables: List[torch.Tensor]):
        """Update metadata without breaking graph capture."""
        # These updates don't affect captured graph
        self.slot_mapping = slot_mapping
        self.block_tables = block_tables
    
    def get_cache_tensors(self, layer_idx: int):
        """Return cache tensors for graph capture."""
        return self.static_k_cache[:, layer_idx], self.static_v_cache[:, layer_idx]
```

### Attention Layer with Graph Support

```python
class GraphOptimizedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        # Standard layers
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Graph-friendly cache interface
        self.use_cuda_graph = False
        self.static_cache_refs = None
    
    def setup_for_graph(self, kv_cache: GraphCompatibleKVCache, layer_idx: int):
        """Setup static references for graph capture."""
        self.use_cuda_graph = True
        self.static_cache_refs = kv_cache.get_cache_tensors(layer_idx)
    
    def forward(self, hidden_states: torch.Tensor, 
                position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if self.use_cuda_graph and seq_len == 1:  # Decode phase
            # Use static cache references
            output = self._graph_compatible_attention(q, k, v, position_ids)
        else:
            # Regular attention
            output = self._standard_attention(q, k, v)
        
        # Reshape and project
        output = output.view(batch_size, seq_len, -1)
        return self.o_proj(output)
    
    def _graph_compatible_attention(self, q, k, v, position_ids):
        """Attention computation compatible with CUDA graphs."""
        # Store KV in static cache location
        cache_idx = position_ids[0, 0]  # Assuming single token decode
        
        k_cache, v_cache = self.static_cache_refs
        
        # Update cache at specific position
        k_cache[cache_idx] = k[0, 0]
        v_cache[cache_idx] = v[0, 0]
        
        # Compute attention with cached KV
        # This uses static memory locations, compatible with graphs
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache[:cache_idx+1],
            v_cache=v_cache[:cache_idx+1],
            causal=True
        )
```

## Optimization Techniques

### Graph Optimization Flags

```python
def optimize_graph_capture():
    # Enable graph optimization
    os.environ['CUDA_GRAPH_ENABLE_OPTIMIZATION'] = '1'
    
    # Reduce memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Use dedicated graph memory pool
    torch.cuda.set_graph_pool_handle(
        torch.cuda.graph_pool_handle()
    )
    
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### Minimizing Graph Overhead

```python
class OptimizedGraphRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.graph_cache = {}
        
        # Pre-allocate output buffers
        self.output_buffers = {}
        
        # Use pinned memory for faster copies
        self.pinned_inputs = {}
    
    def _prepare_pinned_memory(self, batch_size: int):
        if batch_size not in self.pinned_inputs:
            self.pinned_inputs[batch_size] = torch.zeros(
                batch_size, 1, self.model.hidden_size,
                dtype=torch.float16
            ).pin_memory()
    
    def run_with_minimal_overhead(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        
        # Async copy to pinned memory
        pinned = self.pinned_inputs[batch_size]
        pinned.copy_(input_tensor, non_blocking=True)
        
        # Get graph runner
        runner = self.graph_cache[batch_size]
        
        # Async copy to GPU
        runner['input'].copy_(pinned, non_blocking=True)
        
        # Replay graph
        runner['graph'].replay()
        
        # Return pre-allocated output (no copy needed)
        return runner['output']
```

### Profiling CUDA Graphs

```python
class GraphProfiler:
    def __init__(self):
        self.events = defaultdict(list)
    
    def profile_graph_performance(self, model: nn.Module, num_iterations: int = 1000):
        # Test different execution modes
        results = {}
        
        # Eager execution
        input_tensor = torch.randn(8, 1, model.hidden_size, device='cuda')
        
        # Warmup
        for _ in range(100):
            _ = model(input_tensor)
        
        # Profile eager
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        end.record()
        
        torch.cuda.synchronize()
        eager_time = start.elapsed_time(end)
        
        # Profile with CUDA graphs
        graph_runner = CUDAGraphRunner(model, batch_size=8, seq_len=1)
        
        start.record()
        for _ in range(num_iterations):
            _ = graph_runner.run(input_tensor)
        end.record()
        
        torch.cuda.synchronize()
        graph_time = start.elapsed_time(end)
        
        # Results
        results['eager_ms_per_iter'] = eager_time / num_iterations
        results['graph_ms_per_iter'] = graph_time / num_iterations
        results['speedup'] = eager_time / graph_time
        
        return results
```

## Error Handling and Fallback

### Robust Graph Execution

```python
class RobustGraphRunner:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.graphs = {}
        self.graph_failures = defaultdict(int)
        self.max_failures = 3
    
    def run(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        key = self._get_graph_key(inputs)
        
        # Check if we should use graphs
        if self.graph_failures[key] >= self.max_failures:
            return self._run_eager(inputs)
        
        try:
            if key not in self.graphs:
                self._capture_graph(key, inputs)
            
            return self._run_graph(key, inputs)
            
        except RuntimeError as e:
            # Log failure and fallback
            print(f"Graph execution failed: {e}")
            self.graph_failures[key] += 1
            
            # Clear failed graph
            if key in self.graphs:
                del self.graphs[key]
            
            # Fallback to eager
            return self._run_eager(inputs)
    
    def _get_graph_key(self, inputs: Dict[str, torch.Tensor]) -> Tuple:
        # Create key based on input shapes
        return tuple(
            (k, tuple(v.shape)) 
            for k, v in sorted(inputs.items())
        )
    
    def _capture_graph(self, key: Tuple, inputs: Dict[str, torch.Tensor]):
        # Validate inputs are suitable for graphs
        if not self._validate_for_graph_capture(inputs):
            raise RuntimeError("Inputs not suitable for graph capture")
        
        # Create static copies
        static_inputs = {
            k: v.clone() for k, v in inputs.items()
        }
        
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(**static_inputs)
        
        self.graphs[key] = {
            'graph': graph,
            'inputs': static_inputs,
            'output': static_output
        }
    
    def _validate_for_graph_capture(self, inputs: Dict[str, torch.Tensor]) -> bool:
        for k, v in inputs.items():
            # Check tensor properties
            if not v.is_cuda:
                return False
            if v.requires_grad:
                return False
            if not v.is_contiguous():
                return False
        return True
```

### Dynamic Shape Handling

```python
class DynamicShapeGraphRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.shape_to_graph = {}
        self.shape_stats = defaultdict(int)
        
    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        shape_key = tuple(input_tensor.shape)
        self.shape_stats[shape_key] += 1
        
        # Only create graphs for frequent shapes
        if self.shape_stats[shape_key] > 10:
            if shape_key not in self.shape_to_graph:
                self._capture_for_shape(shape_key)
            
            if shape_key in self.shape_to_graph:
                return self._run_graph(shape_key, input_tensor)
        
        # Eager execution for rare shapes
        return self.model(input_tensor)
    
    def cleanup_rare_graphs(self, threshold: int = 100):
        """Remove graphs for rarely used shapes."""
        total_calls = sum(self.shape_stats.values())
        
        for shape, count in list(self.shape_stats.items()):
            if count < threshold and shape in self.shape_to_graph:
                del self.shape_to_graph[shape]
                print(f"Removed graph for shape {shape} (used {count}/{total_calls} times)")
```

## Best Practices

### 1. Graph Capture Guidelines

```python
# DO: Capture graphs after model initialization and warmup
model = Model()
model.eval()
model = model.cuda()

# Warmup
for _ in range(3):
    model(dummy_input)
    
# NOW capture graphs
graph = capture_graph(model)

# DON'T: Capture graphs with training mode or gradient computation
model.train()  # Wrong!
with torch.enable_grad():  # Wrong!
    graph = capture_graph(model)
```

### 2. Memory Management

```python
class GraphMemoryManager:
    def __init__(self, reserved_mb: int = 2048):
        # Reserve memory for graphs
        self.reserved_memory = torch.cuda.memory_reserved()
        
        # Pre-allocate to avoid fragmentation
        dummy = torch.zeros(
            reserved_mb * 1024 * 1024 // 4,
            device='cuda',
            dtype=torch.float32
        )
        del dummy
        
        # Set memory fraction for graphs
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    def monitor_memory(self):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        print(f"Allocated: {allocated / 1e9:.2f} GB")
        print(f"Reserved: {reserved / 1e9:.2f} GB")
        print(f"Free: {(reserved - allocated) / 1e9:.2f} GB")
```

### 3. Testing Graph Correctness

```python
def test_graph_equivalence(model: nn.Module, input_shapes: List[Tuple]):
    """Verify graph outputs match eager execution."""
    model.eval()
    
    for shape in input_shapes:
        # Create test input
        test_input = torch.randn(*shape, device='cuda', dtype=torch.float16)
        
        # Eager execution
        with torch.no_grad():
            eager_output = model(test_input)
        
        # Graph execution  
        graph_runner = CUDAGraphRunner(model, *shape[:-1])
        graph_output = graph_runner.run(test_input)
        
        # Compare
        max_diff = (eager_output - graph_output).abs().max().item()
        assert max_diff < 1e-3, f"Graph output differs by {max_diff}"
        
        print(f"Shape {shape}: PASSED (max diff: {max_diff})")
```

## Production Deployment

### Complete CUDA Graph System

```python
class ProductionCUDAGraphSystem:
    def __init__(self, model: nn.Module, config: ProductionConfig):
        self.model = model
        self.config = config
        
        # Graph storage
        self.decode_graphs = {}
        self.graph_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
        # Memory management
        self.memory_pool = self._init_memory_pool()
        
        # Monitoring
        self.metrics = GraphMetrics()
        
        # Initialize graphs
        self._initialize_system()
    
    def _initialize_system(self):
        # Set optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Pre-capture common configurations
        for batch_size in self.config.common_batch_sizes:
            try:
                self._capture_decode_graph(batch_size)
                print(f"Captured graph for batch size {batch_size}")
            except Exception as e:
                print(f"Failed to capture graph for batch size {batch_size}: {e}")
    
    def execute(self, 
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                kv_cache: KVCache,
                temperature: float = 1.0) -> torch.Tensor:
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Decide execution strategy
        if seq_len == 1 and batch_size in self.decode_graphs:
            # Use CUDA graph for decode
            logits = self._execute_graph(
                batch_size, input_ids, position_ids, kv_cache
            )
            self.graph_stats[batch_size]['hits'] += 1
        else:
            # Eager execution for prefill or unsupported batch size
            logits = self._execute_eager(
                input_ids, position_ids, kv_cache
            )
            self.graph_stats[batch_size]['misses'] += 1
        
        # Sample tokens (outside of graph)
        next_tokens = self._sample(logits, temperature)
        
        # Update metrics
        self.metrics.update(batch_size, seq_len)
        
        return next_tokens
    
    def get_stats(self) -> Dict:
        total_hits = sum(s['hits'] for s in self.graph_stats.values())
        total_misses = sum(s['misses'] for s in self.graph_stats.values())
        
        return {
            'graph_hit_rate': total_hits / (total_hits + total_misses) if total_hits + total_misses > 0 else 0,
            'batch_size_stats': dict(self.graph_stats),
            'memory_usage_gb': torch.cuda.memory_allocated() / 1e9,
            'num_graphs_cached': len(self.decode_graphs)
        }
```

## Conclusion

CUDA graphs provide significant performance improvements for LLM inference:

1. **Overhead elimination**: Remove kernel launch latency (10-30% speedup)
2. **Predictable performance**: Consistent execution times
3. **Memory efficiency**: Pre-allocated buffers reduce fragmentation
4. **Scalability**: Essential for high-throughput serving

Key considerations:
- Best suited for decode phase with fixed shapes
- Requires careful memory management
- Need fallback for dynamic scenarios
- Regular profiling ensures optimization

Next, we'll explore implementing prefix caching for even greater efficiency gains.