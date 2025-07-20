# Multi-GPU Scaling: Distributed Inference Across Multiple Nodes

## Overview

As LLMs grow beyond single-GPU capacity, efficient multi-GPU and multi-node scaling becomes critical. This guide covers advanced techniques for distributed inference including pipeline parallelism, sequence parallelism, and hybrid parallelism strategies for maximum throughput.

## Parallelism Strategies Overview

### Types of Parallelism

1. **Data Parallelism (DP)**: Split batch across GPUs
2. **Tensor Parallelism (TP)**: Split layers across GPUs
3. **Pipeline Parallelism (PP)**: Split model layers sequentially
4. **Sequence Parallelism (SP)**: Split sequence dimension
5. **Expert Parallelism (EP)**: For MoE models

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.distributed as dist

@dataclass
class ParallelismConfig:
    """Configuration for different parallelism strategies."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False
    
    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
    
    def get_rank_info(self, global_rank: int) -> dict:
        """Get parallelism group info for a rank."""
        tp_rank = global_rank % self.tensor_parallel_size
        pp_rank = (global_rank // self.tensor_parallel_size) % self.pipeline_parallel_size
        dp_rank = global_rank // (self.tensor_parallel_size * self.pipeline_parallel_size)
        
        return {
            'global_rank': global_rank,
            'tp_rank': tp_rank,
            'pp_rank': pp_rank,
            'dp_rank': dp_rank,
            'tp_group': self._get_tp_group(dp_rank, pp_rank),
            'pp_group': self._get_pp_group(dp_rank, tp_rank),
            'dp_group': self._get_dp_group(tp_rank, pp_rank)
        }
```

## Pipeline Parallelism

### Basic Pipeline Parallel Implementation

```python
class PipelineParallelModel(nn.Module):
    """Model with pipeline parallelism."""
    
    def __init__(self, config: ModelConfig, pp_rank: int, pp_size: int):
        super().__init__()
        self.config = config
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        
        # Divide layers across pipeline stages
        layers_per_stage = config.num_layers // pp_size
        start_layer = pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        # Create local layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(start_layer, end_layer)
        ])
        
        # Embeddings only on first stage
        if pp_rank == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None
            
        # LM head only on last stage
        if pp_rank == pp_size - 1:
            self.ln_f = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        else:
            self.ln_f = None
            self.lm_head = None
    
    def forward(self, hidden_states: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First stage: embeddings
        if self.pp_rank == 0:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert hidden_states is not None
        
        # Apply local transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Last stage: final norm and LM head
        if self.pp_rank == self.pp_size - 1:
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            return hidden_states
```

### GPipe-style Pipeline Schedule

```python
class GPipePipelineEngine:
    """GPipe-style pipeline with gradient accumulation."""
    
    def __init__(self, model: PipelineParallelModel, pp_group: dist.ProcessGroup):
        self.model = model
        self.pp_group = pp_group
        self.pp_size = dist.get_world_size(pp_group)
        self.pp_rank = dist.get_rank(pp_group)
        
    def forward_backward(self, microbatches: List[torch.Tensor], 
                        grad_accumulation_steps: int) -> List[torch.Tensor]:
        """Execute pipeline forward/backward with microbatches."""
        outputs = []
        
        # Forward pass for all microbatches
        activations = []
        for mb in microbatches:
            act = self.forward_step(mb)
            activations.append(act)
            
        # Backward pass in reverse order
        gradients = []
        for act in reversed(activations):
            grad = self.backward_step(act)
            gradients.append(grad)
            
        return outputs
    
    def forward_step(self, microbatch: torch.Tensor) -> torch.Tensor:
        """Single forward step with communication."""
        # Receive from previous stage
        if self.pp_rank > 0:
            input_tensor = self.recv_from_prev()
        else:
            input_tensor = microbatch
        
        # Local forward
        output_tensor = self.model(hidden_states=input_tensor)
        
        # Send to next stage
        if self.pp_rank < self.pp_size - 1:
            self.send_to_next(output_tensor)
            
        return output_tensor
    
    def send_to_next(self, tensor: torch.Tensor):
        """Send tensor to next pipeline stage."""
        dist.send(tensor.contiguous(), dst=self.pp_rank + 1, group=self.pp_group)
    
    def recv_from_prev(self) -> torch.Tensor:
        """Receive tensor from previous pipeline stage."""
        # Need to know tensor shape
        tensor_shape = self.get_activation_shape()
        tensor = torch.empty(tensor_shape, device='cuda')
        dist.recv(tensor, src=self.pp_rank - 1, group=self.pp_group)
        return tensor
```

### Interleaved Pipeline Schedule

```python
class InterleavedPipelineSchedule:
    """Interleaved 1F1B pipeline schedule for better efficiency."""
    
    def __init__(self, num_microbatches: int, pp_size: int, pp_rank: int):
        self.num_microbatches = num_microbatches
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        
    def get_schedule(self) -> List[Tuple[str, int]]:
        """Get forward/backward schedule for this rank."""
        schedule = []
        
        # Warmup phase: forward passes
        num_warmup = min(self.pp_size - self.pp_rank - 1, self.num_microbatches)
        for i in range(num_warmup):
            schedule.append(('forward', i))
        
        # 1F1B phase
        for i in range(self.num_microbatches - num_warmup):
            schedule.append(('forward', num_warmup + i))
            schedule.append(('backward', i))
        
        # Cooldown phase: backward passes
        for i in range(num_warmup):
            schedule.append(('backward', self.num_microbatches - num_warmup + i))
        
        return schedule
    
    def execute_schedule(self, microbatches: List[torch.Tensor]):
        """Execute the interleaved schedule."""
        schedule = self.get_schedule()
        
        forward_cache = {}
        outputs = []
        
        for action, mb_idx in schedule:
            if action == 'forward':
                output = self.forward_step(microbatches[mb_idx])
                forward_cache[mb_idx] = output
                if self.pp_rank == self.pp_size - 1:
                    outputs.append(output)
            else:  # backward
                grad = self.backward_step(forward_cache[mb_idx])
                del forward_cache[mb_idx]  # Free memory
        
        return outputs
```

## Sequence Parallelism

### Basic Sequence Parallel Implementation

```python
class SequenceParallelAttention(nn.Module):
    """Attention with sequence parallelism."""
    
    def __init__(self, config: ModelConfig, sp_group: dist.ProcessGroup):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.sp_rank = dist.get_rank(sp_group)
        
        # QKV projections (not partitioned)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Split sequence across GPUs
        seq_len_per_gpu = seq_len // self.sp_size
        start_idx = self.sp_rank * seq_len_per_gpu
        end_idx = start_idx + seq_len_per_gpu
        
        # Local sequence chunk
        local_hidden = hidden_states[:, start_idx:end_idx, :]
        
        # Compute QKV for local chunk
        q_local = self.q_proj(local_hidden)
        k_local = self.k_proj(local_hidden)
        v_local = self.v_proj(local_hidden)
        
        # All-gather K and V across sequence dimension
        k_all = self.all_gather_sequence(k_local)
        v_all = self.all_gather_sequence(v_local)
        
        # Reshape for attention
        q_local = q_local.view(batch_size, seq_len_per_gpu, self.num_heads, self.head_dim)
        k_all = k_all.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_all = v_all.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention (each GPU computes for its Q chunk)
        attn_output = self.compute_attention(q_local, k_all, v_all)
        
        # Output projection
        attn_output = attn_output.reshape(batch_size, seq_len_per_gpu, hidden_size)
        output = self.o_proj(attn_output)
        
        # All-gather output
        output = self.all_gather_sequence(output)
        
        return output
    
    def all_gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather across sequence dimension."""
        world_size = dist.get_world_size(self.sp_group)
        
        # Prepare output tensor
        batch_size, local_seq_len, hidden_size = tensor.shape
        output_shape = (batch_size, local_seq_len * world_size, hidden_size)
        output = torch.empty(output_shape, device=tensor.device, dtype=tensor.dtype)
        
        # All-gather
        dist.all_gather_into_tensor(output, tensor, group=self.sp_group)
        
        return output
```

### Ulysses Sequence Parallelism

```python
class UlyssesAttention(nn.Module):
    """Ulysses-style sequence parallelism with optimized communication."""
    
    def __init__(self, config: ModelConfig, sp_group: dist.ProcessGroup):
        super().__init__()
        self.config = config
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        
        # Ring attention setup
        self.ring_comm = RingCommunication(sp_group)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        local_seq_len = seq_len // self.sp_size
        
        # Compute local QKV
        q, k, v = self.compute_qkv(hidden_states)
        
        # Ring attention: each GPU processes different KV blocks in rounds
        attn_output = self.ring_attention(q, k, v)
        
        return attn_output
    
    def ring_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Ring attention to avoid all-gather communication."""
        batch_size, local_seq_len, num_heads, head_dim = q.shape
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Current KV block (starts with local)
        k_block = k.clone()
        v_block = v.clone()
        
        # Ring passes
        for step in range(self.sp_size):
            # Compute attention with current KV block
            block_attn = self.compute_block_attention(
                q, k_block, v_block, 
                block_idx=step,
                causal=True
            )
            output += block_attn
            
            # Ring communication: send to next, receive from prev
            if step < self.sp_size - 1:
                k_block = self.ring_comm.send_recv(k_block)
                v_block = self.ring_comm.send_recv(v_block)
        
        return output
    
    def compute_block_attention(self, q, k, v, block_idx, causal=True):
        """Compute attention for a specific KV block."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if causal:
            # Apply causal mask considering block position
            mask = self.get_causal_mask_for_block(q.shape[1], k.shape[1], block_idx)
            scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output
```

## Hybrid Parallelism

### 3D Parallelism Implementation

```python
class HybridParallelModel(nn.Module):
    """Model with TP + PP + DP parallelism."""
    
    def __init__(self, config: ModelConfig, parallel_config: ParallelismConfig):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        
        # Get rank info
        rank_info = parallel_config.get_rank_info(dist.get_rank())
        self.tp_rank = rank_info['tp_rank']
        self.pp_rank = rank_info['pp_rank']
        self.dp_rank = rank_info['dp_rank']
        
        # Create process groups
        self.tp_group = dist.new_group(rank_info['tp_group'])
        self.pp_group = dist.new_group(rank_info['pp_group'])
        self.dp_group = dist.new_group(rank_info['dp_group'])
        
        # Build model with hybrid parallelism
        self.build_hybrid_model()
    
    def build_hybrid_model(self):
        """Build model layers with appropriate parallelism."""
        # Determine local layers (pipeline parallelism)
        layers_per_stage = self.config.num_layers // self.parallel_config.pipeline_parallel_size
        start_layer = self.pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        # Embeddings (tensor parallel on first pipeline stage)
        if self.pp_rank == 0:
            self.embed_tokens = ParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                tp_group=self.tp_group
            )
        
        # Transformer layers (tensor parallel within each layer)
        self.layers = nn.ModuleList()
        for i in range(start_layer, end_layer):
            layer = HybridTransformerLayer(
                self.config,
                layer_idx=i,
                tp_group=self.tp_group,
                sp_enabled=self.parallel_config.sequence_parallel
            )
            self.layers.append(layer)
        
        # Output layers (last pipeline stage)
        if self.pp_rank == self.parallel_config.pipeline_parallel_size - 1:
            self.ln_f = nn.LayerNorm(self.config.hidden_size)
            self.lm_head = ParallelLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                tp_group=self.tp_group
            )
    
    def forward(self, input_ids: Optional[torch.Tensor] = None,
                hidden_states: Optional[torch.Tensor] = None):
        # Pipeline parallel forward
        if self.pp_rank == 0:
            hidden_states = self.embed_tokens(input_ids)
        
        # Apply local layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Pipeline communication
        if self.pp_rank < self.parallel_config.pipeline_parallel_size - 1:
            self.send_to_next_stage(hidden_states)
            return None
        else:
            # Final processing
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            
            # Reduce across data parallel replicas if needed
            if self.parallel_config.data_parallel_size > 1:
                dist.all_reduce(logits, group=self.dp_group)
                logits = logits / self.parallel_config.data_parallel_size
            
            return logits
```

### Communication Optimization

```python
class OptimizedCommunication:
    """Optimized communication patterns for hybrid parallelism."""
    
    def __init__(self, parallel_config: ParallelismConfig):
        self.parallel_config = parallel_config
        self.setup_communication_groups()
        
    def setup_communication_groups(self):
        """Setup optimized communication groups."""
        # Create hierarchical groups for efficient communication
        self.local_tp_groups = self._create_local_groups('tp')
        self.local_pp_groups = self._create_local_groups('pp')
        
        # NCCL optimizations
        self._optimize_nccl_params()
    
    def _optimize_nccl_params(self):
        """Optimize NCCL parameters for multi-node."""
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Always use tree algorithm
        os.environ['NCCL_IB_DISABLE'] = '0'  # Enable InfiniBand
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Network interface
        
    def overlap_communication(self, compute_fn, comm_fn):
        """Overlap computation and communication."""
        # Create CUDA streams
        compute_stream = torch.cuda.Stream()
        comm_stream = torch.cuda.Stream()
        
        # Start communication on separate stream
        with torch.cuda.stream(comm_stream):
            comm_handle = comm_fn(async_op=True)
        
        # Perform computation on default stream
        with torch.cuda.stream(compute_stream):
            compute_result = compute_fn()
        
        # Wait for communication
        comm_handle.wait()
        
        return compute_result
```

## Multi-Node Scaling

### Multi-Node Setup

```python
class MultiNodeInferenceCluster:
    """Manage multi-node inference cluster."""
    
    def __init__(self, head_node: str, num_nodes: int, gpus_per_node: int):
        self.head_node = head_node
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.world_size = num_nodes * gpus_per_node
        
    def launch_distributed(self, script_path: str, args: List[str]):
        """Launch distributed inference across nodes."""
        # Use torchrun for multi-node launch
        cmd = [
            'torchrun',
            '--nnodes', str(self.num_nodes),
            '--nproc_per_node', str(self.gpus_per_node),
            '--rdzv_endpoint', f'{self.head_node}:29500',
            '--rdzv_backend', 'c10d',
            script_path
        ] + args
        
        subprocess.run(cmd)
    
    def setup_network_optimization(self):
        """Optimize network for multi-node communication."""
        # Enable GPUDirect RDMA
        os.environ['NCCL_NET_GDR_LEVEL'] = '5'
        
        # Set optimal buffer sizes
        os.environ['NCCL_BUFFSIZE'] = str(32 * 1024 * 1024)  # 32MB
        
        # Enable IB adaptive routing
        os.environ['NCCL_IB_ADAPTIVE_ROUTING'] = '1'
```

### Fault Tolerance

```python
class FaultTolerantInference:
    """Fault-tolerant multi-node inference."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.heartbeat_interval = 10.0  # seconds
        self.failure_handlers = {}
        
    def setup_fault_tolerance(self):
        """Setup fault tolerance mechanisms."""
        # Enable elastic training
        from torch.distributed.elastic.multiprocessing import Std
        
        # Setup heartbeat monitoring
        self.start_heartbeat_monitor()
        
        # Register failure handlers
        self.register_failure_handler('gpu_failure', self.handle_gpu_failure)
        self.register_failure_handler('node_failure', self.handle_node_failure)
        
    def checkpoint_state(self, model, optimizer, iteration):
        """Checkpoint model state for recovery."""
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'iteration': iteration,
            'parallel_config': self.get_parallel_config()
        }
        
        # Async checkpoint to avoid blocking
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_iter_{iteration}.pt'
        )
        
        # Use async I/O
        thread = threading.Thread(
            target=lambda: torch.save(checkpoint, checkpoint_path)
        )
        thread.start()
        
    def handle_gpu_failure(self, failed_rank: int):
        """Handle single GPU failure."""
        print(f"GPU failure detected on rank {failed_rank}")
        
        # Exclude failed rank from process group
        new_group = self.create_group_excluding_rank(failed_rank)
        
        # Redistribute work
        self.redistribute_work(new_group)
        
    def handle_node_failure(self, failed_node: int):
        """Handle complete node failure."""
        print(f"Node {failed_node} failure detected")
        
        # Reconfigure parallelism
        self.reconfigure_parallelism_without_node(failed_node)
        
        # Load from checkpoint
        self.load_latest_checkpoint()
```

## Performance Optimization

### Communication-Computation Overlap

```python
class PipelinedExecution:
    """Overlap communication and computation in pipeline."""
    
    def __init__(self, model: nn.Module, parallel_config: ParallelismConfig):
        self.model = model
        self.parallel_config = parallel_config
        
        # Create CUDA events for synchronization
        self.comp_events = [torch.cuda.Event() for _ in range(2)]
        self.comm_events = [torch.cuda.Event() for _ in range(2)]
        
    def execute_with_overlap(self, microbatches: List[torch.Tensor]):
        """Execute with computation-communication overlap."""
        outputs = []
        
        # Double buffering for overlap
        compute_stream = torch.cuda.Stream()
        comm_stream = torch.cuda.Stream()
        
        for i, mb in enumerate(microbatches):
            # Computation on compute stream
            with torch.cuda.stream(compute_stream):
                if i > 0:
                    # Wait for previous communication
                    compute_stream.wait_event(self.comm_events[(i-1) % 2])
                
                output = self.model(mb)
                self.comp_events[i % 2].record(compute_stream)
            
            # Communication on comm stream
            if i < len(microbatches) - 1:
                with torch.cuda.stream(comm_stream):
                    # Wait for computation to finish
                    comm_stream.wait_event(self.comp_events[i % 2])
                    
                    # Send to next stage
                    self.send_activation(output)
                    self.comm_events[i % 2].record(comm_stream)
            
            outputs.append(output)
        
        # Synchronize at the end
        torch.cuda.synchronize()
        
        return outputs
```

### Memory Optimization for Large Models

```python
class MemoryOptimizedParallelism:
    """Memory optimizations for multi-GPU inference."""
    
    def __init__(self, model: nn.Module, parallel_config: ParallelismConfig):
        self.model = model
        self.parallel_config = parallel_config
        
    def setup_memory_optimizations(self):
        """Setup various memory optimizations."""
        # CPU offloading for inactive layers
        self.setup_cpu_offload()
        
        # Activation checkpointing
        self.enable_selective_checkpointing()
        
        # Memory pool optimization
        self.optimize_memory_pools()
        
    def setup_cpu_offload(self):
        """Offload inactive pipeline stages to CPU."""
        if self.parallel_config.pipeline_parallel_size > 1:
            # Offload non-active stages
            for name, param in self.model.named_parameters():
                if not self.is_active_stage(name):
                    param.data = param.data.cpu()
                    param.grad = None
    
    def enable_selective_checkpointing(self):
        """Enable checkpointing for memory-intensive layers."""
        for layer in self.model.layers:
            if self.should_checkpoint_layer(layer):
                layer.enable_gradient_checkpointing()
    
    def optimize_memory_pools(self):
        """Optimize CUDA memory pools for multi-GPU."""
        # Set memory fraction per GPU
        memory_fraction = 0.9 / self.parallel_config.tensor_parallel_size
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable memory pool for each device
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
```

### Dynamic Load Balancing

```python
class DynamicLoadBalancer:
    """Dynamic load balancing for heterogeneous clusters."""
    
    def __init__(self, cluster_config: Dict):
        self.cluster_config = cluster_config
        self.performance_history = defaultdict(list)
        
    def profile_nodes(self) -> Dict[int, float]:
        """Profile performance of each node."""
        node_performance = {}
        
        for node_id in range(self.cluster_config['num_nodes']):
            # Run benchmark
            latency = self.run_node_benchmark(node_id)
            node_performance[node_id] = 1.0 / latency  # Throughput
            
        return node_performance
    
    def rebalance_work(self, current_assignment: Dict) -> Dict:
        """Rebalance work based on node performance."""
        node_performance = self.profile_nodes()
        
        # Calculate new assignment proportional to performance
        total_performance = sum(node_performance.values())
        new_assignment = {}
        
        for node_id, perf in node_performance.items():
            work_fraction = perf / total_performance
            new_assignment[node_id] = self.calculate_work_assignment(work_fraction)
        
        return new_assignment
    
    def calculate_work_assignment(self, work_fraction: float) -> Dict:
        """Calculate specific work assignment for a node."""
        return {
            'batch_size': int(self.cluster_config['total_batch_size'] * work_fraction),
            'num_pipeline_stages': self.get_optimal_pipeline_stages(work_fraction),
            'tensor_parallel_size': self.get_optimal_tp_size(work_fraction)
        }
```

## Monitoring and Profiling

### Distributed Performance Monitoring

```python
class DistributedPerformanceMonitor:
    """Monitor performance across all GPUs."""
    
    def __init__(self, parallel_config: ParallelismConfig):
        self.parallel_config = parallel_config
        self.metrics = defaultdict(list)
        self.profiler = None
        
    def start_profiling(self):
        """Start distributed profiling."""
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=6),
            on_trace_ready=self.process_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.start()
    
    def collect_distributed_metrics(self) -> Dict:
        """Collect metrics from all ranks."""
        local_metrics = {
            'throughput': self.calculate_local_throughput(),
            'memory_usage': torch.cuda.memory_allocated(),
            'communication_time': self.get_communication_time(),
            'computation_time': self.get_computation_time()
        }
        
        # Gather from all ranks
        all_metrics = [None] * dist.get_world_size()
        dist.all_gather_object(all_metrics, local_metrics)
        
        return self.aggregate_metrics(all_metrics)
    
    def visualize_performance(self, metrics: Dict):
        """Visualize distributed performance."""
        import matplotlib.pyplot as plt
        
        # Communication matrix
        comm_matrix = self.build_communication_matrix(metrics)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(comm_matrix, cmap='hot')
        plt.title('Communication Heatmap')
        plt.colorbar()
        
        # Throughput by rank
        plt.subplot(2, 2, 2)
        ranks = list(range(len(metrics['throughput_by_rank'])))
        plt.bar(ranks, metrics['throughput_by_rank'])
        plt.title('Throughput by Rank')
        plt.xlabel('Rank')
        plt.ylabel('Tokens/sec')
        
        # Memory usage
        plt.subplot(2, 2, 3)
        plt.plot(metrics['memory_timeline'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Step')
        plt.ylabel('Memory (GB)')
        
        # Load imbalance
        plt.subplot(2, 2, 4)
        self.plot_load_imbalance(metrics)
        plt.title('Load Imbalance')
        
        plt.tight_layout()
        plt.savefig('distributed_performance.png')
```

### Communication Profiling

```python
class CommunicationProfiler:
    """Profile communication patterns in distributed inference."""
    
    def __init__(self):
        self.comm_events = defaultdict(list)
        
    def profile_all_reduce(self, tensor_size: int, group: dist.ProcessGroup):
        """Profile all-reduce operation."""
        tensor = torch.randn(tensor_size, device='cuda')
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, group=group)
        
        # Profile
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            dist.all_reduce(tensor, group=group)
            
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bandwidth = (tensor_size * 4 * 100 * (dist.get_world_size(group) - 1)) / elapsed / 1e9
        
        return {
            'operation': 'all_reduce',
            'tensor_size': tensor_size,
            'time_ms': elapsed * 1000 / 100,
            'bandwidth_gbps': bandwidth
        }
    
    def analyze_communication_pattern(self, model: nn.Module) -> Dict:
        """Analyze communication pattern of a model."""
        patterns = {
            'all_reduce_ops': 0,
            'all_gather_ops': 0,
            'p2p_ops': 0,
            'total_volume': 0
        }
        
        # Hook into distributed operations
        def comm_hook(state, bucket):
            patterns['all_reduce_ops'] += 1
            patterns['total_volume'] += bucket.buffer().numel() * 4  # FP32
            return bucket.buffer()
        
        # Register hooks
        model.register_comm_hook(state=None, hook=comm_hook)
        
        return patterns
```

## Best Practices

### 1. Choosing Parallelism Strategy

```python
def select_parallelism_strategy(model_size: int, 
                              num_gpus: int,
                              gpu_memory: int,
                              sequence_length: int) -> ParallelismConfig:
    """Select optimal parallelism strategy."""
    
    # Estimate memory requirements
    memory_per_param = 2  # FP16
    activation_memory = estimate_activation_memory(model_size, sequence_length)
    total_memory_needed = model_size * memory_per_param + activation_memory
    
    # Determine minimum TP size for memory
    min_tp_size = math.ceil(total_memory_needed / (gpu_memory * 0.9))
    
    # Balance between TP and PP
    if num_gpus <= 8:
        # Single node: prefer TP
        tp_size = min(num_gpus, min_tp_size)
        pp_size = num_gpus // tp_size
    else:
        # Multi-node: balance TP and PP
        tp_size = min(8, min_tp_size)  # Keep TP within node
        pp_size = num_gpus // tp_size
    
    # Enable sequence parallelism for very long sequences
    use_sp = sequence_length > 8192
    
    return ParallelismConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        sequence_parallel=use_sp
    )
```

### 2. Network Topology Awareness

```python
class TopologyAwareParallelism:
    """Parallelism assignment aware of network topology."""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        
    def assign_ranks(self, parallel_config: ParallelismConfig) -> Dict:
        """Assign ranks considering network topology."""
        assignments = {}
        
        # Keep TP group within same node (NVLink)
        nodes = self.topology.get_nodes()
        for node in nodes:
            gpus_in_node = self.topology.get_gpus_in_node(node)
            
            # Assign TP ranks within node
            for i in range(0, len(gpus_in_node), parallel_config.tensor_parallel_size):
                tp_group = gpus_in_node[i:i + parallel_config.tensor_parallel_size]
                for local_rank, gpu in enumerate(tp_group):
                    assignments[gpu] = {
                        'node': node,
                        'tp_rank': local_rank,
                        'pp_rank': i // parallel_config.tensor_parallel_size
                    }
        
        return assignments
```

### 3. Debugging Distributed Issues

```python
class DistributedDebugger:
    """Tools for debugging distributed inference issues."""
    
    def __init__(self):
        self.debug_mode = os.environ.get('DISTRIBUTED_DEBUG', '0') == '1'
        
    def check_tensor_consistency(self, tensor: torch.Tensor, name: str):
        """Check if tensor is consistent across ranks."""
        if not self.debug_mode:
            return
            
        # Gather tensors from all ranks
        world_size = dist.get_world_size()
        tensors = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors, tensor)
        
        # Check consistency
        reference = tensors[0]
        for rank, t in enumerate(tensors[1:], 1):
            if not torch.allclose(reference, t, rtol=1e-5):
                diff = (reference - t).abs().max()
                print(f"Inconsistency in {name} between rank 0 and {rank}: max_diff={diff}")
    
    def trace_communication(self):
        """Trace all communication operations."""
        original_all_reduce = dist.all_reduce
        
        def traced_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            size = tensor.numel() * tensor.element_size()
            print(f"[Rank {dist.get_rank()}] all_reduce: {size / 1e6:.2f} MB")
            return original_all_reduce(tensor, op, group, async_op)
        
        dist.all_reduce = traced_all_reduce
```

## Conclusion

Multi-GPU scaling is essential for large model inference:

1. **Pipeline Parallelism**: Splits model depth-wise, good for memory constraints
2. **Tensor Parallelism**: Splits layers, best within high-bandwidth connections
3. **Sequence Parallelism**: Handles long sequences efficiently
4. **Hybrid Approaches**: Combine strategies for optimal performance
5. **Fault Tolerance**: Critical for multi-node deployments

Key insights:
- Keep TP within nodes (NVLink/NVSwitch)
- Use PP across nodes to minimize communication
- Profile and monitor continuously
- Plan for failures in multi-node setups
- Overlap communication with computation

Modern LLM serving requires sophisticated parallelism strategies to achieve good performance at scale.