# Parallelism Strategies in LLM Inference

## Why Parallelism Matters

Modern LLMs are massive:
- **Llama 70B**: 70 billion parameters ≈ 140GB in float16
- **GPT-4 scale**: Hundreds of billions of parameters
- **Single GPU Memory**: Typically 24-80GB

**Challenge**: Models often don't fit on a single GPU, and even if they do, single-GPU inference may be too slow for production workloads.

## Types of Parallelism

### 1. Data Parallelism
- **Concept**: Process different requests on different devices
- **Use Case**: Independent requests with identical model copies
- **Memory**: Each device holds a full model copy
- **Communication**: Minimal (no communication needed between requests)

```
GPU 1: Request A → Model Copy 1 → Response A
GPU 2: Request B → Model Copy 2 → Response B  
GPU 3: Request C → Model Copy 3 → Response C
```

**Limitations**: 
- Requires full model to fit on each GPU
- Doesn't help with large models that exceed single GPU memory
- Inefficient for variable-length sequences

### 2. Pipeline Parallelism  
- **Concept**: Split model layers across devices, pipeline execution
- **Use Case**: Models too large for single GPU, sequential processing
- **Memory**: Each device holds a subset of layers
- **Communication**: Activations between adjacent devices

```
Input → GPU 1 (Layers 1-8) → GPU 2 (Layers 9-16) → GPU 3 (Layers 17-24) → Output
```

**For Inference Challenges**:
- **Bubble Problem**: During decode phase, only one GPU active at a time
- **Sequential Dependency**: Can't parallelize single sequence processing
- **Load Balancing**: Different layers may have different computation requirements

### 3. Tensor Parallelism
- **Concept**: Split individual operations across devices
- **Use Case**: Large models, need to parallelize within single forward pass
- **Memory**: Each device holds a partition of each layer's parameters
- **Communication**: Frequent all-reduce operations

```
Single Matrix Multiplication: Y = X @ W
Split W across GPUs: W = [W₁, W₂, W₃, W₄]
Each GPU computes: Y₁ = X @ W₁, Y₂ = X @ W₂, etc.
```

## Tensor Parallelism Deep Dive

Tensor parallelism is the most common approach for LLM inference because it:
1. **Enables Large Models**: Split model across multiple GPUs
2. **Maintains Semantics**: Each forward pass uses full model capacity
3. **Scales Well**: Linear scaling with number of GPUs (with good interconnect)

### Parallelizing Linear Layers

#### Column Parallelism
Split output dimension across devices:

```python
# Original computation
Y = X @ W  # X: [batch, input_dim], W: [input_dim, output_dim]

# Column parallel version
W = [W₁, W₂, W₃, W₄]  # Split along output dimension
Y₁ = X @ W₁  # GPU 1: [batch, output_dim/4]
Y₂ = X @ W₂  # GPU 2: [batch, output_dim/4] 
Y₃ = X @ W₃  # GPU 3: [batch, output_dim/4]
Y₄ = X @ W₄  # GPU 4: [batch, output_dim/4]
Y = concat([Y₁, Y₂, Y₃, Y₄], dim=-1)  # [batch, output_dim]
```

**Key Property**: No communication required during computation!

#### Row Parallelism
Split input dimension across devices:

```python
# Original computation  
Y = X @ W  # X: [batch, input_dim], W: [input_dim, output_dim]

# Row parallel version
X = [X₁, X₂, X₃, X₄]  # Split input along feature dimension
W = [W₁; W₂; W₃; W₄]  # Split along input dimension (stacked vertically)
Y₁ = X₁ @ W₁  # GPU 1: [batch, output_dim]
Y₂ = X₂ @ W₂  # GPU 2: [batch, output_dim]
Y₃ = X₃ @ W₃  # GPU 3: [batch, output_dim]
Y₄ = X₄ @ W₄  # GPU 4: [batch, output_dim]
Y = Y₁ + Y₂ + Y₃ + Y₄  # Requires ALL-REDUCE communication!
```

### Parallelizing Attention

Multi-Head Attention is naturally parallelizable:

```python
class MultiHeadAttention:
    def __init__(self, d_model=2048, num_heads=32):
        # Split heads across GPUs (column parallel)
        heads_per_gpu = num_heads // world_size  # 32 ÷ 4 = 8 heads per GPU
        
        self.q_proj = ColumnParallelLinear(d_model, d_model)  # Split output
        self.k_proj = ColumnParallelLinear(d_model, d_model)  # Split output  
        self.v_proj = ColumnParallelLinear(d_model, d_model)  # Split output
        self.o_proj = RowParallelLinear(d_model, d_model)    # Split input, need all-reduce
```

**Attention Computation**:
```python
def forward(self, x):
    # Each GPU computes subset of attention heads
    q = self.q_proj(x)  # [batch, seq_len, d_model/world_size]
    k = self.k_proj(x)  # [batch, seq_len, d_model/world_size]
    v = self.v_proj(x)  # [batch, seq_len, d_model/world_size]
    
    # Attention computation (no communication needed)
    attn_out = flash_attention(q, k, v)  # [batch, seq_len, d_model/world_size]
    
    # Output projection (requires all-reduce)
    output = self.o_proj(attn_out)  # [batch, seq_len, d_model]
    return output
```

### Parallelizing MLP (Feed-Forward Networks)

Modern MLPs use SwiGLU activation, which can be efficiently parallelized:

```python
class SwiGLU_MLP:
    def __init__(self, d_model=2048, d_ff=8192):
        # Gate and Up projections can be fused and column-parallelized
        self.gate_up_proj = ColumnParallelLinear(d_model, 2 * d_ff)
        # Down projection is row-parallelized  
        self.down_proj = RowParallelLinear(d_ff, d_model)
        
    def forward(self, x):
        # Split into gate and up components
        gate_up = self.gate_up_proj(x)  # [batch, seq_len, 2*d_ff/world_size]
        gate, up = gate_up.chunk(2, dim=-1)  # Each: [batch, seq_len, d_ff/world_size]
        
        # SwiGLU activation (no communication)
        hidden = F.silu(gate) * up  # [batch, seq_len, d_ff/world_size]
        
        # Down projection (all-reduce happens inside)
        output = self.down_proj(hidden)  # [batch, seq_len, d_model]
        return output
```

## Communication Patterns

### All-Reduce Operation
The fundamental communication primitive in tensor parallelism:

```python
# Each GPU has partial result
GPU 1: [a₁, b₁, c₁, d₁]
GPU 2: [a₂, b₂, c₂, d₂]  
GPU 3: [a₃, b₃, c₃, d₃]
GPU 4: [a₄, b₄, c₄, d₄]

# After all-reduce, each GPU has the sum
All GPUs: [a₁+a₂+a₃+a₄, b₁+b₂+b₃+b₄, c₁+c₂+c₃+c₄, d₁+d₂+d₃+d₄]
```

### Communication Volume
For a model with parameters distributed across `P` devices:

**Per Layer Communication**:
- **Attention**: 1 all-reduce of size `[batch_size, seq_len, d_model]`
- **MLP**: 1 all-reduce of size `[batch_size, seq_len, d_model]`

**Total Communication per Token**:
- **Prefill**: `2 × num_layers × batch_size × seq_len × d_model × sizeof(dtype)`
- **Decode**: `2 × num_layers × batch_size × d_model × sizeof(dtype)`

### Communication Optimization
```python
# Overlap computation with communication
def optimized_linear_forward(x, weight):
    # Start computation
    local_output = torch.mm(x, weight)
    
    # Asynchronous all-reduce (overlaps with next layer computation)
    handle = dist.all_reduce(local_output, async_op=True)
    
    # Do other work while communication happens
    # ... 
    
    # Wait for communication to complete
    handle.wait()
    return local_output
```

## Tensor Parallelism in Practice: nano-vLLM Implementation

nano-vLLM implements tensor parallelism through specialized linear layers:

### QKVParallelLinear
```python
class QKVParallelLinear(nn.Module):
    """Fused Q, K, V projections with column parallelism"""
    def __init__(self, hidden_size, head_dim, total_num_heads, total_num_kv_heads):
        tp_size = dist.get_world_size()
        
        # Split heads across tensor parallel ranks
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        
        # Weight shapes include all Q, K, V weights
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv_heads * head_dim
        self.weight = nn.Parameter(torch.empty(hidden_size, q_size + 2 * kv_size))
        
    def forward(self, x):
        output = torch.mm(x.view(-1, x.size(-1)), self.weight)
        # Split into Q, K, V portions
        q_size = self.num_heads * self.head_dim  
        kv_size = self.num_kv_heads * self.head_dim
        q = output[:, :q_size]
        k = output[:, q_size:q_size + kv_size] 
        v = output[:, q_size + kv_size:]
        return q, k, v
```

### Process Coordination
nano-vLLM uses multiprocessing for tensor parallelism:

```python
# In llm_engine.py
def __init__(self, model, **kwargs):
    config = Config(model, **kwargs)
    
    # Spawn worker processes for ranks 1+
    ctx = mp.get_context("spawn")
    for rank in range(1, config.tensor_parallel_size):
        process = ctx.Process(target=ModelRunner, args=(config, rank))
        process.start()
    
    # Main process handles rank 0
    self.model_runner = ModelRunner(config, rank=0)
```

### Distributed Execution
```python
class ModelRunner:
    def call(self, method, *args):
        if self.rank == 0:
            # Broadcast method call to all ranks
            for event in self.events:
                event.set()  # Signal worker processes
                
        # Execute method on all ranks
        return getattr(self, method)(*args)
```

## Scaling Considerations

### Network Bandwidth Requirements
For efficient tensor parallelism, you need high-bandwidth interconnects:

**NVLink/NVSwitch**: 
- Bandwidth: 600+ GB/s bidirectional between GPUs
- Latency: ~1-2 μs
- **Ideal for**: Intra-node tensor parallelism

**InfiniBand**:
- Bandwidth: 200-400 GB/s 
- Latency: ~1-5 μs
- **Good for**: Inter-node tensor parallelism with careful optimization

**Ethernet**:
- Bandwidth: 25-100 GB/s
- Latency: ~10-50 μs  
- **Challenging for**: Tensor parallelism (too much overhead)

### Scaling Efficiency
Tensor parallelism efficiency depends on:

1. **Compute-to-Communication Ratio**: Larger models/batches have better ratios
2. **Interconnect Quality**: High bandwidth, low latency improves efficiency  
3. **Communication Overlap**: Overlapping compute with communication
4. **Load Balancing**: Ensuring equal work distribution

**Example Scaling**:
```
Model: Llama 70B, Batch Size: 32, Sequence Length: 2048

1 GPU:  Not possible (model doesn't fit)
2 GPUs: ~85% efficiency (high communication overhead)
4 GPUs: ~90% efficiency (better compute-to-communication ratio)
8 GPUs: ~95% efficiency (optimal for this model size)
16 GPUs: ~80% efficiency (communication overhead dominates)
```

## Hybrid Parallelism Strategies

Real deployments often combine multiple parallelism types:

### Tensor Parallel + Data Parallel
```python
# 8 GPUs total: 2x tensor parallel, 4x data parallel
# Group 1: GPUs 0-1 (tensor parallel group 1)
# Group 2: GPUs 2-3 (tensor parallel group 2)  
# Group 3: GPUs 4-5 (tensor parallel group 3)
# Group 4: GPUs 6-7 (tensor parallel group 4)

# Each tensor parallel group processes different requests
```

### Pipeline + Tensor Parallel
```python
# 16 GPUs: 4 pipeline stages × 4 tensor parallel per stage
# Stage 1: Layers 1-8  on GPUs 0-3  (tensor parallel)
# Stage 2: Layers 9-16 on GPUs 4-7  (tensor parallel)  
# Stage 3: Layers 17-24 on GPUs 8-11 (tensor parallel)
# Stage 4: Layers 25-32 on GPUs 12-15 (tensor parallel)
```

## Key Takeaways

1. **Tensor Parallelism is King**: Most effective for LLM inference due to model sizes
2. **Communication is Critical**: High-bandwidth interconnects are essential
3. **Natural Parallelization**: Attention and MLP layers parallelize naturally
4. **All-Reduce Overhead**: Communication cost scales with number of devices
5. **Implementation Complexity**: Requires careful coordination across processes
6. **Scaling Sweet Spots**: Optimal parallelism degree depends on model size and hardware

Understanding parallelism strategies is essential for scaling beyond single-GPU deployments. The next section will explore the broader landscape of optimization techniques available for LLM inference.