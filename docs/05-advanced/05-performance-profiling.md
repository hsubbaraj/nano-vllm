# Performance Profiling and Optimization for LLM Inference

## Overview

Performance profiling is critical for optimizing LLM inference engines. This guide covers comprehensive profiling techniques, bottleneck identification, and optimization strategies to achieve maximum throughput and minimal latency in production deployments.

## Key Performance Metrics

### Primary Metrics

```python
@dataclass
class InferenceMetrics:
    """Core metrics for LLM inference performance."""
    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float
    
    # Latency metrics  
    time_to_first_token_p50: float  # TTFT
    time_to_first_token_p99: float
    inter_token_latency_p50: float  # ITL
    inter_token_latency_p99: float
    end_to_end_latency_p50: float
    
    # Utilization metrics
    gpu_utilization: float
    gpu_memory_utilization: float
    mfu: float  # Model FLOPs Utilization
    mbu: float  # Model Bandwidth Utilization
    
    # Efficiency metrics
    batch_size_avg: float
    kv_cache_hit_rate: float
    preemption_rate: float
```

## Profiling Tools and Techniques

### GPU Profiling with PyTorch Profiler

```python
import torch.profiler
from contextlib import contextmanager
import json

class InferenceProfiler:
    """Comprehensive profiler for LLM inference."""
    
    def __init__(self, warmup_steps: int = 3, active_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.profile_data = {}
        
    @contextmanager
    def profile_inference(self, name: str, trace_path: str = None):
        """Profile inference with detailed GPU metrics."""
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=1
        )
        
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path) if trace_path else None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            yield prof
            
        # Extract key metrics
        self.profile_data[name] = self._extract_metrics(prof)
    
    def _extract_metrics(self, prof) -> dict:
        """Extract key metrics from profiler."""
        # Get kernel statistics
        kernel_stats = {}
        for evt in prof.key_averages():
            if evt.is_cuda:
                kernel_stats[evt.key] = {
                    'cuda_time_ms': evt.cuda_time_total / 1000,
                    'cpu_time_ms': evt.cpu_time_total / 1000,
                    'count': evt.count,
                    'flops': evt.flops,
                    'cuda_memory_mb': evt.cuda_memory_usage / (1024 * 1024) if evt.cuda_memory_usage else 0
                }
        
        # Sort by CUDA time
        top_kernels = sorted(
            kernel_stats.items(), 
            key=lambda x: x[1]['cuda_time_ms'], 
            reverse=True
        )[:10]
        
        return {
            'total_cuda_time_ms': sum(k[1]['cuda_time_ms'] for k in top_kernels),
            'top_kernels': dict(top_kernels),
            'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3)
        }
```

### NVIDIA Nsight Systems Integration

```python
import subprocess
import os
from typing import Optional

class NsightProfiler:
    """Wrapper for NVIDIA Nsight Systems profiling."""
    
    def __init__(self, output_dir: str = "./profiles"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def profile_command(self, 
                       command: str,
                       profile_name: str,
                       duration_sec: Optional[int] = None) -> str:
        """Profile a command with nsys."""
        output_file = os.path.join(self.output_dir, f"{profile_name}.nsys-rep")
        
        nsys_cmd = [
            "nsys", "profile",
            "--output", output_file,
            "--force-overwrite", "true",
            "--capture-range", "cudaProfilerApi",
            "--cuda-memory-usage", "true",
            "--gpu-metrics-device", "all"
        ]
        
        if duration_sec:
            nsys_cmd.extend(["--duration", str(duration_sec)])
            
        nsys_cmd.extend(command.split())
        
        # Run profiling
        result = subprocess.run(nsys_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Generate readable report
            self._generate_report(output_file, profile_name)
            return output_file
        else:
            raise RuntimeError(f"Nsight profiling failed: {result.stderr}")
    
    def _generate_report(self, nsys_file: str, profile_name: str):
        """Generate human-readable report from nsys file."""
        report_types = ["cuda_gpu_trace", "cuda_api_sum", "gpu_kern_sum"]
        
        for report_type in report_types:
            output_file = os.path.join(
                self.output_dir, 
                f"{profile_name}_{report_type}.txt"
            )
            
            cmd = [
                "nsys", "stats",
                "--report", report_type,
                "--format", "table",
                "--output", output_file,
                nsys_file
            ]
            
            subprocess.run(cmd, check=True)
```

### Custom Profiling Hooks

```python
import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np

class DetailedProfiler:
    """Fine-grained profiler with custom hooks."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.active_timers = {}
        
    @contextmanager
    def timer(self, name: str):
        """Time a code block."""
        start_time = time.perf_counter()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        
        cuda_start.record()
        self.active_timers[name] = (start_time, cuda_start, cuda_end)
        
        try:
            yield
        finally:
            cuda_end.record()
            torch.cuda.synchronize()
            
            cpu_time = time.perf_counter() - start_time
            cuda_time = cuda_start.elapsed_time(cuda_end) / 1000  # Convert to seconds
            
            self.timings[f"{name}_cpu"].append(cpu_time)
            self.timings[f"{name}_cuda"].append(cuda_time)
            
            del self.active_timers[name]
    
    def count(self, name: str, value: int = 1):
        """Count occurrences."""
        self.counters[name] += value
    
    def get_summary(self) -> dict:
        """Get profiling summary."""
        summary = {}
        
        # Timing statistics
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'p50': np.percentile(times, 50),
                    'p90': np.percentile(times, 90),
                    'p99': np.percentile(times, 99),
                    'count': len(times),
                    'total': np.sum(times)
                }
        
        # Counters
        summary['counters'] = dict(self.counters)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n=== Profiling Summary ===")
        print("\nTimings (seconds):")
        for name, stats in sorted(summary.items()):
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"\n{name}:")
                print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  P50/P90/P99: {stats['p50']:.4f}/{stats['p90']:.4f}/{stats['p99']:.4f}")
                print(f"  Total: {stats['total']:.4f} ({stats['count']} calls)")
```

## Bottleneck Analysis

### Identifying Performance Bottlenecks

```python
class BottleneckAnalyzer:
    """Analyze and identify performance bottlenecks."""
    
    def __init__(self, profiler: DetailedProfiler):
        self.profiler = profiler
        
    def analyze(self) -> dict:
        """Comprehensive bottleneck analysis."""
        summary = self.profiler.get_summary()
        bottlenecks = {}
        
        # 1. Memory Bandwidth Analysis
        bottlenecks['memory_bandwidth'] = self._analyze_memory_bandwidth(summary)
        
        # 2. Compute Utilization
        bottlenecks['compute_utilization'] = self._analyze_compute_utilization(summary)
        
        # 3. Kernel Launch Overhead
        bottlenecks['kernel_overhead'] = self._analyze_kernel_overhead(summary)
        
        # 4. Communication Overhead (for multi-GPU)
        bottlenecks['communication'] = self._analyze_communication_overhead(summary)
        
        # 5. CPU-GPU Synchronization
        bottlenecks['synchronization'] = self._analyze_synchronization(summary)
        
        return bottlenecks
    
    def _analyze_memory_bandwidth(self, summary: dict) -> dict:
        """Analyze memory bandwidth utilization."""
        # Calculate achieved bandwidth
        attention_time = summary.get('attention_cuda', {}).get('total', 0)
        attention_bytes = self.profiler.counters.get('attention_bytes', 0)
        
        if attention_time > 0:
            achieved_bandwidth_gbps = (attention_bytes / attention_time) / 1e9
            theoretical_bandwidth_gbps = 1555  # A100 HBM bandwidth
            utilization = achieved_bandwidth_gbps / theoretical_bandwidth_gbps
            
            return {
                'achieved_gbps': achieved_bandwidth_gbps,
                'theoretical_gbps': theoretical_bandwidth_gbps,
                'utilization': utilization,
                'is_bottleneck': utilization < 0.5
            }
        
        return {'is_bottleneck': False}
    
    def _analyze_compute_utilization(self, summary: dict) -> dict:
        """Analyze compute (FLOP) utilization."""
        # Model FLOPs calculation
        model_flops = self.profiler.counters.get('model_flops', 0)
        total_time = summary.get('forward_pass_cuda', {}).get('total', 0)
        
        if total_time > 0:
            achieved_tflops = (model_flops / total_time) / 1e12
            theoretical_tflops = 312  # A100 FP16 tensor core peak
            mfu = achieved_tflops / theoretical_tflops
            
            return {
                'achieved_tflops': achieved_tflops,
                'theoretical_tflops': theoretical_tflops,
                'mfu': mfu,
                'is_bottleneck': mfu < 0.3
            }
        
        return {'is_bottleneck': False}
```

### Profiling Model Components

```python
class ComponentProfiler:
    """Profile individual model components."""
    
    def __init__(self, model, profiler: DetailedProfiler):
        self.model = model
        self.profiler = profiler
        
    def profile_forward_pass(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        """Profile each component of forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        with self.profiler.timer("embedding"):
            hidden_states = self.model.embed_tokens(input_ids)
            self.profiler.count("embedding_bytes", 
                              hidden_states.numel() * hidden_states.element_size())
        
        # Transformer layers
        for i, layer in enumerate(self.model.layers):
            with self.profiler.timer(f"layer_{i}"):
                # Attention
                with self.profiler.timer(f"layer_{i}_attention"):
                    attn_output = layer.self_attn(
                        hidden_states,
                        position_ids=position_ids
                    )
                    
                    # Calculate memory access
                    kv_cache_access = 2 * batch_size * seq_len * layer.hidden_size * 2
                    self.profiler.count("attention_bytes", kv_cache_access)
                
                # MLP
                with self.profiler.timer(f"layer_{i}_mlp"):
                    mlp_output = layer.mlp(hidden_states)
                    
                    # Calculate FLOPs
                    mlp_flops = 2 * batch_size * seq_len * layer.hidden_size * layer.intermediate_size * 2
                    self.profiler.count("model_flops", mlp_flops)
        
        # Output projection
        with self.profiler.timer("output_projection"):
            logits = self.model.lm_head(hidden_states)
        
        return logits
```

## Optimization Strategies

### Memory Optimization

```python
class MemoryOptimizer:
    """Optimize memory usage and access patterns."""
    
    def __init__(self, model_config):
        self.config = model_config
        
    def optimize_kv_cache_layout(self, 
                                batch_size: int,
                                max_seq_len: int) -> dict:
        """Optimize KV cache memory layout."""
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        num_layers = self.config.num_hidden_layers
        
        # Option 1: Standard layout [batch, heads, seq, head_dim]
        standard_layout = {
            'shape': (batch_size, num_heads, max_seq_len, head_dim),
            'stride': (num_heads * max_seq_len * head_dim, 
                      max_seq_len * head_dim,
                      head_dim,
                      1),
            'memory_access_pattern': 'strided'
        }
        
        # Option 2: Transposed layout [batch, seq, heads, head_dim]
        transposed_layout = {
            'shape': (batch_size, max_seq_len, num_heads, head_dim),
            'stride': (max_seq_len * num_heads * head_dim,
                      num_heads * head_dim,
                      head_dim,
                      1),
            'memory_access_pattern': 'sequential'
        }
        
        # Option 3: Paged layout for dynamic allocation
        page_size = 16  # tokens per page
        num_pages = (max_seq_len + page_size - 1) // page_size
        paged_layout = {
            'shape': (batch_size, num_heads, num_pages, page_size, head_dim),
            'page_size': page_size,
            'memory_access_pattern': 'paged'
        }
        
        # Benchmark and select best layout
        best_layout = self._benchmark_layouts([
            ('standard', standard_layout),
            ('transposed', transposed_layout),
            ('paged', paged_layout)
        ])
        
        return best_layout
    
    def optimize_attention_kernel(self, seq_len: int) -> str:
        """Select optimal attention kernel based on sequence length."""
        if seq_len <= 512:
            return "flash_attention"  # Best for short sequences
        elif seq_len <= 2048:
            return "flash_attention_2"  # Optimized for medium sequences
        else:
            return "block_sparse_attention"  # For very long sequences
```

### Kernel Fusion

```python
class KernelFusionOptimizer:
    """Fuse operations to reduce kernel launches."""
    
    def __init__(self):
        self.fused_kernels = {}
        
    def create_fused_attention_kernel(self):
        """Create fused attention kernel with RMSNorm and residual."""
        import triton
        import triton.language as tl
        
        @triton.jit
        def fused_attention_residual_rmsnorm_kernel(
            hidden_states_ptr,
            attention_output_ptr,
            weight_ptr,
            output_ptr,
            hidden_size,
            eps,
            BLOCK_SIZE: tl.constexpr
        ):
            # Get program ID
            pid = tl.program_id(0)
            
            # Load data
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hidden_size
            
            # Load tensors
            hidden = tl.load(hidden_states_ptr + offsets, mask=mask)
            attn_out = tl.load(attention_output_ptr + offsets, mask=mask)
            weight = tl.load(weight_ptr + offsets, mask=mask)
            
            # Residual connection
            residual = hidden + attn_out
            
            # RMSNorm
            var = tl.sum(residual * residual, axis=0) / hidden_size
            rstd = 1 / tl.sqrt(var + eps)
            norm = residual * rstd
            
            # Apply weight
            output = norm * weight
            
            # Store result
            tl.store(output_ptr + offsets, output, mask=mask)
        
        return fused_attention_residual_rmsnorm_kernel
    
    def fuse_mlp_operations(self):
        """Fuse MLP operations: Linear -> Activation -> Linear."""
        @torch.jit.script
        def fused_mlp(x: torch.Tensor, 
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      w3: torch.Tensor) -> torch.Tensor:
            # Fuse gate and up projections with SiLU activation
            gate = F.silu(F.linear(x, w1))
            up = F.linear(x, w3)
            intermediate = gate * up
            
            # Down projection
            output = F.linear(intermediate, w2)
            return output
        
        return fused_mlp
```

### Dynamic Optimization

```python
class DynamicOptimizer:
    """Runtime optimization based on workload characteristics."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_config = {}
        
    def adapt_to_workload(self, metrics: InferenceMetrics):
        """Dynamically adapt optimizations based on metrics."""
        optimizations = []
        
        # 1. Batch size optimization
        if metrics.gpu_utilization < 0.7:
            # Increase batch size
            new_batch_size = min(
                int(self.current_config.get('batch_size', 8) * 1.5),
                256
            )
            optimizations.append(('batch_size', new_batch_size))
        
        # 2. Memory allocation strategy
        if metrics.preemption_rate > 0.1:
            # Switch to more aggressive memory reservation
            optimizations.append(('memory_strategy', 'oversubscribe'))
        
        # 3. Kernel selection
        if metrics.inter_token_latency_p99 > 50:  # ms
            # Use faster kernels at cost of memory
            optimizations.append(('attention_kernel', 'flash_attention_v2'))
        
        # 4. Precision optimization
        if metrics.mfu < 0.3 and self.current_config.get('dtype') == 'float16':
            # Try INT8 quantization for compute-bound workloads
            optimizations.append(('quantization', 'int8'))
        
        return optimizations
    
    def profile_guided_optimization(self, 
                                  profiling_data: dict,
                                  optimization_space: dict) -> dict:
        """Use profiling data to guide optimizations."""
        best_config = self.current_config.copy()
        best_score = float('-inf')
        
        # Grid search over optimization space
        for batch_size in optimization_space['batch_sizes']:
            for block_size in optimization_space['block_sizes']:
                for kernel in optimization_space['kernels']:
                    config = {
                        'batch_size': batch_size,
                        'block_size': block_size,
                        'attention_kernel': kernel
                    }
                    
                    # Estimate performance
                    score = self._estimate_performance(config, profiling_data)
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
        
        return best_config
```

## Production Monitoring

### Real-time Performance Dashboard

```python
import asyncio
from datetime import datetime
from typing import Dict, List
import psutil
import pynvml

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history = []
        self.alerts = []
        
        # Initialize NVML for GPU monitoring
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        
    async def start_monitoring(self):
        """Start continuous monitoring."""
        while True:
            metrics = await self.collect_metrics()
            self.metrics_history.append(metrics)
            
            # Check for anomalies
            self.check_anomalies(metrics)
            
            # Keep history bounded
            if len(self.metrics_history) > 3600:  # 1 hour at 1s intervals
                self.metrics_history = self.metrics_history[-3600:]
            
            await asyncio.sleep(self.update_interval)
    
    async def collect_metrics(self) -> dict:
        """Collect current system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {},
            'memory': {},
            'gpu': []
        }
        
        # CPU metrics
        metrics['cpu']['utilization'] = psutil.cpu_percent(interval=0.1)
        metrics['cpu']['frequency'] = psutil.cpu_freq().current
        
        # Memory metrics
        mem = psutil.virtual_memory()
        metrics['memory']['total_gb'] = mem.total / (1024**3)
        metrics['memory']['used_gb'] = mem.used / (1024**3)
        metrics['memory']['percent'] = mem.percent
        
        # GPU metrics
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            gpu_metrics = {
                'index': i,
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                'memory_used_gb': pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3),
                'memory_total_gb': pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3),
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                'power_watts': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            }
            
            metrics['gpu'].append(gpu_metrics)
        
        return metrics
    
    def check_anomalies(self, metrics: dict):
        """Check for performance anomalies."""
        # High CPU utilization
        if metrics['cpu']['utilization'] > 90:
            self.alerts.append({
                'type': 'high_cpu',
                'value': metrics['cpu']['utilization'],
                'timestamp': metrics['timestamp']
            })
        
        # GPU thermal throttling
        for gpu in metrics['gpu']:
            if gpu['temperature'] > 80:
                self.alerts.append({
                    'type': 'gpu_thermal',
                    'gpu_index': gpu['index'],
                    'temperature': gpu['temperature'],
                    'timestamp': metrics['timestamp']
                })
        
        # Memory pressure
        if metrics['memory']['percent'] > 95:
            self.alerts.append({
                'type': 'memory_pressure',
                'percent': metrics['memory']['percent'],
                'timestamp': metrics['timestamp']
            })
```

### Performance Regression Detection

```python
class RegressionDetector:
    """Detect performance regressions between versions."""
    
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline = baseline_metrics
        self.thresholds = {
            'tokens_per_second': 0.95,  # 5% regression threshold
            'time_to_first_token_p50': 1.1,  # 10% regression
            'memory_usage': 1.05  # 5% increase
        }
        
    def check_regression(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check for performance regressions."""
        regressions = []
        
        for metric, threshold in self.thresholds.items():
            if metric not in current_metrics or metric not in self.baseline:
                continue
                
            baseline_value = self.baseline[metric]
            current_value = current_metrics[metric]
            
            # For throughput metrics, lower is worse
            if 'per_second' in metric:
                if current_value < baseline_value * threshold:
                    regression_pct = (1 - current_value / baseline_value) * 100
                    regressions.append(
                        f"{metric}: {regression_pct:.1f}% regression "
                        f"({baseline_value:.2f} -> {current_value:.2f})"
                    )
            # For latency metrics, higher is worse
            else:
                if current_value > baseline_value * threshold:
                    regression_pct = (current_value / baseline_value - 1) * 100
                    regressions.append(
                        f"{metric}: {regression_pct:.1f}% regression "
                        f"({baseline_value:.2f} -> {current_value:.2f})"
                    )
        
        return regressions
```

## Optimization Case Studies

### Case Study 1: Attention Optimization

```python
def optimize_attention_for_workload(seq_lengths: List[int]) -> dict:
    """Optimize attention based on sequence length distribution."""
    
    # Analyze sequence length distribution
    avg_len = np.mean(seq_lengths)
    p95_len = np.percentile(seq_lengths, 95)
    max_len = max(seq_lengths)
    
    optimization_config = {}
    
    # Short sequences: Use standard attention
    if p95_len <= 512:
        optimization_config['kernel'] = 'standard_attention'
        optimization_config['block_size'] = 512
        
    # Medium sequences: Use Flash Attention
    elif p95_len <= 2048:
        optimization_config['kernel'] = 'flash_attention'
        optimization_config['block_size'] = 1024
        
    # Long sequences: Use Flash Attention 2 with sliding window
    elif p95_len <= 8192:
        optimization_config['kernel'] = 'flash_attention_2'
        optimization_config['block_size'] = 2048
        optimization_config['window_size'] = 4096
        
    # Very long sequences: Use sparse attention
    else:
        optimization_config['kernel'] = 'block_sparse_attention'
        optimization_config['block_size'] = 64
        optimization_config['sparsity_pattern'] = 'fixed'
    
    return optimization_config
```

### Case Study 2: Batch Size Tuning

```python
def auto_tune_batch_size(model, test_prompts: List[str]) -> int:
    """Automatically find optimal batch size."""
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for batch_size in batch_sizes:
        try:
            # Test with current batch size
            throughput = benchmark_batch_size(model, test_prompts, batch_size)
            
            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'efficiency': throughput / batch_size  # Tokens per sequence
            })
            
        except torch.cuda.OutOfMemoryError:
            # Batch size too large
            break
    
    # Find batch size with best throughput
    best_result = max(results, key=lambda x: x['throughput'])
    
    # Check if efficiency drops significantly at this batch size
    max_efficiency = max(r['efficiency'] for r in results)
    if best_result['efficiency'] < 0.8 * max_efficiency:
        # Use smaller batch size for better latency
        for result in reversed(results):
            if result['efficiency'] >= 0.9 * max_efficiency:
                return result['batch_size']
    
    return best_result['batch_size']
```

## Best Practices

### 1. Continuous Profiling

```python
class ContinuousProfiler:
    """Continuous profiling in production."""
    
    def __init__(self, sample_rate: float = 0.01):
        self.sample_rate = sample_rate
        self.profile_storage = ProfileStorage()
        
    def should_profile(self) -> bool:
        """Decide whether to profile this request."""
        return random.random() < self.sample_rate
    
    async def profile_request(self, request_id: str, func, *args, **kwargs):
        """Profile a request if sampled."""
        if self.should_profile():
            with self.profile_inference(request_id):
                result = await func(*args, **kwargs)
            
            # Store profile for analysis
            self.profile_storage.store(request_id, self.get_profile_data())
        else:
            result = await func(*args, **kwargs)
        
        return result
```

### 2. A/B Testing Optimizations

```python
class OptimizationABTest:
    """A/B test optimization changes."""
    
    def __init__(self, control_config: dict, test_config: dict):
        self.control = control_config
        self.test = test_config
        self.results = {'control': [], 'test': []}
        
    def route_request(self, request_id: str) -> str:
        """Route request to control or test."""
        # Use consistent hashing for sticky routing
        group = 'test' if hash(request_id) % 2 == 0 else 'control'
        return group
    
    def analyze_results(self) -> dict:
        """Analyze A/B test results."""
        control_metrics = self._calculate_metrics(self.results['control'])
        test_metrics = self._calculate_metrics(self.results['test'])
        
        # Statistical significance test
        improvement = {}
        for metric in control_metrics:
            control_val = control_metrics[metric]
            test_val = test_metrics[metric]
            
            improvement[metric] = {
                'control': control_val,
                'test': test_val,
                'improvement_pct': ((test_val - control_val) / control_val) * 100,
                'significant': self._is_significant(
                    self.results['control'],
                    self.results['test'],
                    metric
                )
            }
        
        return improvement
```

## Conclusion

Effective performance profiling and optimization requires:

1. **Comprehensive Monitoring**: Track all relevant metrics continuously
2. **Bottleneck Identification**: Use profiling tools to find actual bottlenecks
3. **Targeted Optimization**: Focus on the most impactful optimizations
4. **Continuous Validation**: Monitor for regressions and validate improvements
5. **Workload Adaptation**: Dynamically adjust to changing workloads

Key insights:
- Profile first, optimize second - avoid premature optimization
- Memory bandwidth is often the primary bottleneck in LLM inference
- Kernel fusion and attention optimization provide the biggest gains
- Continuous profiling in production catches real-world issues
- A/B testing validates optimization impact

With systematic profiling and optimization, you can achieve 2-5x performance improvements in production LLM serving.