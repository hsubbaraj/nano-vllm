# Implementing Tensor Parallelism for Multi-GPU Inference

## Overview

Tensor parallelism (TP) is a model parallelism technique that splits individual layers across multiple GPUs, enabling inference of models too large for a single GPU. This guide covers implementing tensor parallelism from scratch, including communication patterns, layer splitting strategies, and synchronization.

## Understanding Tensor Parallelism

### Key Concepts

1. **Layer Splitting**: Divide weight matrices across GPUs
2. **Communication**: Synchronize intermediate results
3. **Load Balancing**: Ensure equal work distribution
4. **Memory Efficiency**: Each GPU holds only a portion of weights

### Parallelism Strategies

```python
# Model Parallelism Types
class ParallelismType(Enum):
    DATA = "data"          # Split batch across GPUs
    TENSOR = "tensor"      # Split model layers across GPUs  
    PIPELINE = "pipeline"  # Split model stages across GPUs
    SEQUENCE = "sequence"  # Split sequence length across GPUs
```

## Communication Primitives

### Setting Up NCCL

```python
import torch.distributed as dist

def init_distributed(rank: int, world_size: int, master_addr: str = "localhost", 
                     master_port: str = "29500"):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Verify setup
    print(f"Rank {rank}/{world_size} initialized on GPU {torch.cuda.current_device()}")
```

### Core Communication Operations

```python
class CommunicationOps:
    @staticmethod
    def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce across all ranks."""
        dist.all_reduce(tensor, op=op)
        return tensor
    
    @staticmethod
    def all_gather(output_list: List[torch.Tensor], input_tensor: torch.Tensor):
        """Gather tensors from all ranks."""
        dist.all_gather(output_list, input_tensor)
    
    @staticmethod
    def scatter(output_tensor: torch.Tensor, input_list: List[torch.Tensor], src: int = 0):
        """Scatter tensors to all ranks."""
        dist.scatter(output_tensor, input_list, src=src)
    
    @staticmethod
    def broadcast(tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source rank."""
        dist.broadcast(tensor, src=src)
```

## Layer Parallelism Patterns

### Column Parallel Linear

Split output dimension across GPUs:

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, gather_output: bool = True,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Divide output dimension by tensor parallel size
        self.out_features_per_partition = out_features // tp_size
        
        # Initialize weight for this partition
        self.weight = nn.Parameter(torch.empty(
            self.out_features_per_partition,
            self.in_features,
            device=torch.cuda.current_device(),
            dtype=torch.float16
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.out_features_per_partition,
                device=torch.cuda.current_device(),
                dtype=torch.float16
            ))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize as if it were a single large layer
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input: [batch_size, seq_len, in_features]
        # Weight: [out_features_per_partition, in_features]
        
        # Local computation
        output = F.linear(input, self.weight, self.bias)
        
        # Gather output from all ranks if needed
        if self.gather_output and self.tp_size > 1:
            output_list = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)
        
        return output
```

### Row Parallel Linear

Split input dimension across GPUs:

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, input_is_parallel: bool = False,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Divide input dimension by tensor parallel size
        self.in_features_per_partition = in_features // tp_size
        
        # Initialize weight for this partition
        self.weight = nn.Parameter(torch.empty(
            self.out_features,
            self.in_features_per_partition,
            device=torch.cuda.current_device(),
            dtype=torch.float16
        ))
        
        if bias:
            # Bias is not partitioned in row parallel
            self.bias = nn.Parameter(torch.empty(
                self.out_features,
                device=torch.cuda.current_device(),
                dtype=torch.float16
            ))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Split input if not already parallel
        if not self.input_is_parallel and self.tp_size > 1:
            input_list = input.chunk(self.tp_size, dim=-1)
            input = input_list[self.tp_rank]
        
        # Local computation
        output = F.linear(input, self.weight, None)
        
        # All-reduce across tensor parallel ranks
        if self.tp_size > 1:
            dist.all_reduce(output)
        
        # Add bias after all-reduce
        if self.bias is not None:
            output = output + self.bias
        
        return output
```

## Attention Layer Parallelism

### Parallel Multi-Head Attention

```python
class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Split heads across tensor parallel ranks
        assert num_heads % tp_size == 0
        self.num_heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.hidden_size_per_partition = self.num_heads_per_partition * self.head_dim
        
        # QKV projection - column parallel
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=False,
            gather_output=False,
            tp_size=tp_size,
            tp_rank=tp_rank
        )
        
        # Output projection - row parallel
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            tp_size=tp_size,
            tp_rank=tp_rank
        )
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection (already partitioned)
        qkv = self.qkv_proj(hidden_states)
        
        # Reshape to separate Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads_per_partition, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Attention computation (local to each rank)
        attn_output = self.compute_attention(q, k, v, attention_mask)
        
        # Reshape for output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size_per_partition)
        
        # Output projection (includes all-reduce)
        output = self.o_proj(attn_output)
        
        return output
```

## MLP Layer Parallelism

### Parallel Feed-Forward Network

```python
class ParallelMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Gate and up projections - column parallel
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )
        
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )
        
        # Down projection - row parallel
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            input_is_parallel=True, tp_size=tp_size, tp_rank=tp_rank
        )
        
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Column parallel computations
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Element-wise operations work naturally with partitioned tensors
        intermediate = gate * up
        
        # Row parallel computation with all-reduce
        down = self.down_proj(intermediate)
        
        return down
```

## Embedding and Output Layers

### Parallel Embedding

```python
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Split vocabulary across ranks
        self.vocab_start_idx = vocab_size * tp_rank // tp_size
        self.vocab_end_idx = vocab_size * (tp_rank + 1) // tp_size
        self.vocab_size_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        # Local embedding table
        self.weight = nn.Parameter(torch.empty(
            self.vocab_size_per_partition,
            self.embedding_dim,
            device=torch.cuda.current_device(),
            dtype=torch.float16
        ))
        
        self.reset_parameters()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Mask input IDs for this partition
        masked_input = input_ids.clone()
        masked_input[(input_ids < self.vocab_start_idx) | 
                    (input_ids >= self.vocab_end_idx)] = 0
        
        # Shift to local indices
        local_input = masked_input - self.vocab_start_idx
        
        # Local embedding lookup
        output = F.embedding(local_input, self.weight)
        
        # Mask output for out-of-partition tokens
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        output = output * mask.unsqueeze(-1).to(output.dtype)
        
        # All-reduce to combine embeddings from all partitions
        if self.tp_size > 1:
            dist.all_reduce(output)
        
        return output
```

### Parallel LM Head

```python
class ParallelLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Split vocabulary across ranks
        self.vocab_start_idx = vocab_size * tp_rank // tp_size
        self.vocab_end_idx = vocab_size * (tp_rank + 1) // tp_size
        self.vocab_size_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        # Local output projection
        self.weight = nn.Parameter(torch.empty(
            self.vocab_size_per_partition,
            self.hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16
        ))
        
        self.reset_parameters()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Local computation
        local_logits = F.linear(hidden_states, self.weight, None)
        
        # Gather logits from all partitions
        if self.tp_size > 1:
            logits_list = [torch.empty_like(local_logits) for _ in range(self.tp_size)]
            dist.all_gather(logits_list, local_logits)
            logits = torch.cat(logits_list, dim=-1)
        else:
            logits = local_logits
        
        return logits
```

## Complete Parallel Model

### Assembling the Parallel Transformer

```python
class ParallelTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Parallel embedding
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ParallelTransformerLayer(config, tp_size, tp_rank)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output norm (replicated across ranks)
        self.norm = RMSNorm(config.hidden_size)
        
        # Parallel LM head
        self.lm_head = ParallelLMHead(
            config.hidden_size,
            config.vocab_size,
            tp_size=tp_size,
            tp_rank=tp_rank
        )
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # Embedding lookup with all-reduce
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output projection with gather
        logits = self.lm_head(hidden_states)
        
        return logits
```

## Weight Loading and Distribution

### Distributed Weight Loading

```python
class DistributedWeightLoader:
    def __init__(self, tp_size: int, tp_rank: int):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
    
    def load_and_distribute_weights(self, model: nn.Module, checkpoint_path: str):
        """Load weights and distribute across tensor parallel ranks."""
        # Only rank 0 loads the checkpoint
        if self.tp_rank == 0:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            state_dict = None
        
        # Distribute weights
        for name, param in model.named_parameters():
            if self.tp_rank == 0:
                full_param = state_dict[name]
                
                # Determine how to split this parameter
                if self.should_partition_parameter(name):
                    partitioned_params = self.partition_parameter(
                        full_param, name, self.tp_size
                    )
                else:
                    # Replicate across all ranks
                    partitioned_params = [full_param] * self.tp_size
            else:
                partitioned_params = None
            
            # Scatter the partitioned parameters
            local_param = torch.empty_like(param)
            if self.tp_rank == 0:
                dist.scatter(local_param, partitioned_params, src=0)
            else:
                dist.scatter(local_param, None, src=0)
            
            # Copy to model parameter
            param.data.copy_(local_param)
    
    def partition_parameter(self, param: torch.Tensor, name: str, 
                           tp_size: int) -> List[torch.Tensor]:
        """Partition a parameter based on its name and type."""
        if 'qkv_proj' in name or 'gate_proj' in name or 'up_proj' in name:
            # Column parallel - split along output dimension
            return param.chunk(tp_size, dim=0)
        elif 'o_proj' in name or 'down_proj' in name:
            # Row parallel - split along input dimension
            return param.chunk(tp_size, dim=1)
        elif 'embed_tokens' in name or 'lm_head' in name:
            # Vocabulary parallel - split along vocab dimension
            return param.chunk(tp_size, dim=0)
        else:
            # Replicate
            return [param] * tp_size
```

## Optimized Communication

### Overlapping Computation and Communication

```python
class OverlappedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.tp_size = tp_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Split computation into chunks
        chunk_size = input.shape[0] // 4  # Process in 4 chunks
        outputs = []
        
        # Stream for computation
        comp_stream = torch.cuda.Stream()
        
        for i in range(0, input.shape[0], chunk_size):
            chunk = input[i:i + chunk_size]
            
            # Compute on separate stream
            with torch.cuda.stream(comp_stream):
                output_chunk = F.linear(chunk, self.weight)
            
            # Start all-reduce while computing next chunk
            handle = dist.all_reduce(output_chunk, async_op=True)
            
            outputs.append((output_chunk, handle))
        
        # Wait for all operations to complete
        results = []
        for output_chunk, handle in outputs:
            handle.wait()
            results.append(output_chunk)
        
        return torch.cat(results, dim=0)
```

### Custom Communication Kernels

```python
@triton.jit
def efficient_all_reduce_kernel(
    input_ptr, output_ptr, size,
    BLOCK_SIZE: tl.constexpr
):
    """Custom kernel for efficient all-reduce."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < size
    
    # Load local data
    local_data = tl.load(input_ptr + offset, mask=mask)
    
    # Custom reduction logic here
    # (Implementation depends on hardware topology)
    
    # Store result
    tl.store(output_ptr + offset, local_data, mask=mask)
```

## Handling Edge Cases

### Uneven Splits

```python
class UnevenSplitHandler:
    @staticmethod
    def split_tensor(tensor: torch.Tensor, world_size: int, dim: int = 0):
        """Handle uneven tensor splits across ranks."""
        size = tensor.shape[dim]
        base_size = size // world_size
        remainder = size % world_size
        
        splits = []
        start = 0
        
        for rank in range(world_size):
            # Ranks < remainder get one extra element
            rank_size = base_size + (1 if rank < remainder else 0)
            end = start + rank_size
            
            # Extract slice
            indices = torch.arange(start, end, device=tensor.device)
            split = torch.index_select(tensor, dim, indices)
            splits.append(split)
            
            start = end
        
        return splits
```

### Dynamic Shapes

```python
class DynamicParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        
        # Store full weight but only use partition
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Handle variable sequence lengths
        batch_size, seq_len, hidden = input.shape
        
        # Pad sequence length to be divisible by tp_size
        pad_len = (self.tp_size - seq_len % self.tp_size) % self.tp_size
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, hidden, 
                                device=input.device, dtype=input.dtype)
            input = torch.cat([input, padding], dim=1)
        
        # Now safe to split
        output = self.parallel_forward(input)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :-pad_len]
        
        return output
```

## Performance Monitoring

### Profiling Distributed Execution

```python
class DistributedProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.timers = defaultdict(list)
    
    @contextmanager
    def timer(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.timers[name].append(elapsed)
    
    def profile_model_forward(self, model: nn.Module, input_ids: torch.Tensor):
        with self.timer("total_forward"):
            with self.timer("embedding"):
                hidden = model.embed_tokens(input_ids)
            
            for i, layer in enumerate(model.layers):
                with self.timer(f"layer_{i}"):
                    with self.timer(f"layer_{i}_attention"):
                        attn_out = layer.attention(hidden)
                    with self.timer(f"layer_{i}_mlp"):
                        hidden = layer.mlp(hidden)
            
            with self.timer("lm_head"):
                logits = model.lm_head(hidden)
        
        return logits
    
    def print_summary(self):
        print(f"\nRank {self.rank} Profile Summary:")
        for name, times in self.timers.items():
            avg_time = sum(times) / len(times)
            print(f"{name}: {avg_time*1000:.2f} ms")
```

## Testing Tensor Parallel Implementation

### Unit Tests

```python
def test_column_parallel_linear():
    torch.manual_seed(42)
    world_size = 4
    
    # Create reference linear layer
    reference = nn.Linear(1024, 4096, bias=False)
    reference_output = reference(torch.randn(2, 8, 1024))
    
    # Test column parallel version
    outputs = []
    for rank in range(world_size):
        parallel = ColumnParallelLinear(
            1024, 4096, bias=False, 
            tp_size=world_size, tp_rank=rank
        )
        
        # Copy appropriate weight slice
        start = rank * 1024
        end = (rank + 1) * 1024
        parallel.weight.data = reference.weight.data[start:end]
        
        output = parallel(torch.randn(2, 8, 1024))
        outputs.append(output)
    
    # Concatenate outputs
    combined = torch.cat(outputs, dim=-1)
    
    # Should match reference
    assert torch.allclose(combined, reference_output, rtol=1e-3)
```

### Integration Tests

```python
def test_parallel_model_equivalence():
    """Test that parallel model produces same output as single GPU."""
    config = ModelConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        vocab_size=50257
    )
    
    # Single GPU reference
    single_model = TransformerModel(config)
    
    # Multi-GPU parallel model
    world_size = 4
    parallel_models = []
    for rank in range(world_size):
        parallel_model = ParallelTransformerModel(
            config, tp_size=world_size, tp_rank=rank
        )
        parallel_models.append(parallel_model)
    
    # Load same weights
    state_dict = single_model.state_dict()
    loader = DistributedWeightLoader(world_size, 0)
    
    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    
    single_output = single_model(input_ids)
    
    # Simulate distributed forward
    # (In practice, this runs on separate processes)
    parallel_output = simulate_distributed_forward(
        parallel_models, input_ids, world_size
    )
    
    assert torch.allclose(single_output, parallel_output, rtol=1e-3)
```

## Best Practices

### 1. Minimize Communication

```python
# Bad: Multiple small all-reduces
for tensor in tensors:
    dist.all_reduce(tensor)

# Good: Single large all-reduce
combined = torch.cat(tensors)
dist.all_reduce(combined)
results = combined.split(sizes)
```

### 2. Use Correct Process Group

```python
# Create separate process groups for different parallelism types
tp_group = dist.new_group(ranks=tp_ranks)
pp_group = dist.new_group(ranks=pp_ranks)

# Use appropriate group for communication
dist.all_reduce(tensor, group=tp_group)
```

### 3. Handle Initialization Properly

```python
def init_parallel_model(rank: int, world_size: int):
    # Set random seeds consistently
    torch.manual_seed(42)
    
    # Initialize process group
    init_distributed(rank, world_size)
    
    # Create model after distributed init
    model = ParallelTransformerModel(config, world_size, rank)
    
    # Load weights after model creation
    if rank == 0:
        print("Loading and distributing weights...")
    loader = DistributedWeightLoader(world_size, rank)
    loader.load_and_distribute_weights(model, checkpoint_path)
    
    return model
```

## Conclusion

Implementing tensor parallelism requires careful attention to:

1. **Communication patterns**: Minimize and overlap with computation
2. **Memory efficiency**: Each rank holds only necessary weights
3. **Load balancing**: Ensure even distribution of work
4. **Correctness**: Verify equivalence with single-GPU execution
5. **Performance**: Profile and optimize communication bottlenecks

Tensor parallelism enables scaling to larger models and is essential for production inference systems. Next, we'll explore implementing Flash Attention for efficient attention computation.