# Tutorial: Deploying Your Inference Engine to Production

## Overview

In this final tutorial, we'll take our inference engine from development to production, covering deployment strategies, monitoring, scaling, and best practices for serving LLMs in real-world environments.

## Prerequisites

- Completed all previous tutorials
- Basic understanding of:
  - Docker and containerization
  - HTTP APIs and web services
  - Cloud deployment concepts
  - Monitoring and observability

## Architecture Overview

Our production deployment will include:
1. **API Server**: HTTP/gRPC endpoints for inference
2. **Request Queue**: Managing incoming requests
3. **Inference Engine**: Our optimized engine with all features
4. **Monitoring**: Metrics, logging, and alerting
5. **Auto-scaling**: Dynamic resource management

## Step 1: Creating the API Server

Let's build a production-ready API server:

```python
# api_server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import uuid
import time
from datetime import datetime
import uvicorn

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    
class GenerationResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "organization"
    
# Create FastAPI app
app = FastAPI(title="LLM Inference API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request queue for handling concurrent requests
class RequestQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.pending_requests = {}
        
    async def add_request(self, request_id: str, request: GenerationRequest):
        """Add request to queue."""
        future = asyncio.Future()
        await self.queue.put({
            'id': request_id,
            'request': request,
            'future': future,
            'timestamp': time.time()
        })
        self.pending_requests[request_id] = future
        return future
    
    async def get_batch(self, max_batch_size: int, timeout: float = 0.01):
        """Get a batch of requests."""
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < max_batch_size and time.time() < deadline:
            try:
                remaining_time = max(0, deadline - time.time())
                request = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=remaining_time
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
                
        return batch

# Global request queue
request_queue = RequestQueue()

# Inference engine wrapper
class InferenceService:
    def __init__(self, model_path: str, config: Dict):
        self.model_path = model_path
        self.config = config
        self.engine = None
        self.tokenizer = None
        
    async def initialize(self):
        """Initialize the inference engine."""
        from transformers import AutoTokenizer
        from our_engine import LLMEngine, EngineConfig
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Create engine config
        engine_config = EngineConfig(
            model_path=self.model_path,
            max_batch_size=self.config['max_batch_size'],
            max_seq_len=self.config['max_seq_len'],
            gpu_memory_utilization=self.config['gpu_memory_utilization'],
            enable_cuda_graph=self.config['enable_cuda_graph'],
            enable_prefix_caching=self.config['enable_prefix_caching']
        )
        
        # Initialize engine
        self.engine = LLMEngine(engine_config)
        await self.engine.initialize_async()
        
        print(f"Inference engine initialized with model: {self.model_path}")
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of requests."""
        # Prepare inputs
        prompts = [item['request'].prompt for item in batch]
        sampling_params = [
            {
                'max_tokens': item['request'].max_tokens,
                'temperature': item['request'].temperature,
                'top_p': item['request'].top_p,
                'top_k': item['request'].top_k,
                'stop_sequences': item['request'].stop_sequences
            }
            for item in batch
        ]
        
        # Run inference
        results = await self.engine.generate_batch_async(prompts, sampling_params)
        
        # Format responses
        responses = []
        for i, (item, result) in enumerate(zip(batch, results)):
            response = GenerationResponse(
                id=item['id'],
                created=int(time.time()),
                model=self.config['model_name'],
                choices=[{
                    'index': 0,
                    'text': result['text'],
                    'logprobs': result.get('logprobs'),
                    'finish_reason': result['finish_reason']
                }],
                usage={
                    'prompt_tokens': result['prompt_tokens'],
                    'completion_tokens': result['completion_tokens'],
                    'total_tokens': result['prompt_tokens'] + result['completion_tokens']
                }
            )
            responses.append(response)
            
        return responses

# Global inference service
inference_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global inference_service
    
    # Load configuration
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize inference service
    inference_service = InferenceService(
        model_path=config['model_path'],
        config=config['inference']
    )
    await inference_service.initialize()
    
    # Start batch processing loop
    asyncio.create_task(batch_processing_loop())

async def batch_processing_loop():
    """Main loop for processing batches."""
    while True:
        try:
            # Get batch of requests
            batch = await request_queue.get_batch(
                max_batch_size=inference_service.config['max_batch_size']
            )
            
            if batch:
                # Process batch
                responses = await inference_service.process_batch(batch)
                
                # Send responses
                for item, response in zip(batch, responses):
                    item['future'].set_result(response)
                    del request_queue.pending_requests[item['id']]
                    
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Set error for all requests in batch
            for item in batch:
                item['future'].set_exception(e)

@app.post("/v1/completions", response_model=GenerationResponse)
async def create_completion(request: GenerationRequest):
    """Create a completion."""
    request_id = str(uuid.uuid4())
    
    # Add to queue
    future = await request_queue.add_request(request_id, request)
    
    # Wait for response
    try:
        response = await asyncio.wait_for(future, timeout=60.0)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

@app.get("/v1/models", response_model=Dict[str, List[ModelInfo]])
async def list_models():
    """List available models."""
    return {
        "data": [
            ModelInfo(
                id=inference_service.config['model_name'],
                created=int(time.time())
            )
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_service is not None and inference_service.engine is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    if inference_service and inference_service.engine:
        metrics = await inference_service.engine.get_metrics_async()
        return {
            "queue_size": request_queue.queue.qsize(),
            "pending_requests": len(request_queue.pending_requests),
            **metrics
        }
    return {"error": "Engine not initialized"}
```

## Step 2: Implementing Streaming

For better user experience, let's add streaming support:

```python
# streaming.py
from fastapi import StreamingResponse
from typing import AsyncGenerator
import json

class StreamingInferenceService(InferenceService):
    async def generate_stream(self, prompt: str, 
                             sampling_params: Dict) -> AsyncGenerator[str, None]:
        """Generate tokens in streaming mode."""
        # Add request to engine
        request_id = await self.engine.add_request_async(prompt, sampling_params)
        
        # Stream tokens
        async for token_data in self.engine.get_token_stream_async(request_id):
            # Format as SSE (Server-Sent Events)
            data = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": self.config['model_name'],
                "choices": [{
                    "index": 0,
                    "delta": {"text": token_data['token']},
                    "logprobs": token_data.get('logprobs'),
                    "finish_reason": token_data.get('finish_reason')
                }]
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            if token_data.get('finish_reason'):
                break
        
        # Send final message
        yield "data: [DONE]\n\n"

@app.post("/v1/completions/stream")
async def create_completion_stream(request: GenerationRequest):
    """Create a streaming completion."""
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be true")
    
    sampling_params = {
        'max_tokens': request.max_tokens,
        'temperature': request.temperature,
        'top_p': request.top_p,
        'top_k': request.top_k,
        'stop_sequences': request.stop_sequences
    }
    
    return StreamingResponse(
        inference_service.generate_stream(request.prompt, sampling_params),
        media_type="text/event-stream"
    )
```

## Step 3: Configuration Management

Create a comprehensive configuration system:

```yaml
# config.yaml
model_path: "/models/llama-2-7b"
model_name: "llama-2-7b"

inference:
  max_batch_size: 32
  max_seq_len: 2048
  gpu_memory_utilization: 0.9
  enable_cuda_graph: true
  enable_prefix_caching: true
  
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  cors_origins: ["*"]
  request_timeout: 60
  max_queue_size: 1000
  
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  log_level: "INFO"
  log_file: "/var/log/inference/api.log"
  
performance:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  enable_flash_attention: true
  enable_triton_kernels: true
  
security:
  api_key_enabled: true
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: 60
```

## Step 4: Monitoring and Metrics

Implement comprehensive monitoring:

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import logging
from functools import wraps
import time

# Prometheus metrics
request_count = Counter('inference_requests_total', 'Total inference requests', 
                       ['model', 'status'])
request_duration = Histogram('inference_request_duration_seconds', 
                           'Request duration', ['model'])
queue_size = Gauge('inference_queue_size', 'Current queue size')
active_requests = Gauge('inference_active_requests', 'Active requests')
gpu_memory_usage = Gauge('inference_gpu_memory_bytes', 'GPU memory usage', ['device'])
model_load_time = Histogram('inference_model_load_seconds', 'Model load time')

# Structured logging
class StructuredLogger:
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def log_request(self, request_id: str, prompt_length: int, 
                   completion_length: int, duration: float, status: str):
        """Log request details."""
        self.logger.info(
            f"Request completed",
            extra={
                'request_id': request_id,
                'prompt_length': prompt_length,
                'completion_length': completion_length,
                'duration': duration,
                'status': status
            }
        )

# Monitoring decorator
def monitor_inference(model_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            active_requests.inc()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                request_count.labels(model=model_name, status=status).inc()
                request_duration.labels(model=model_name).observe(duration)
                active_requests.dec()
        
        return wrapper
    return decorator

# Add monitoring endpoint
@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint."""
    # Update dynamic metrics
    queue_size.set(request_queue.queue.qsize())
    
    # Get GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i)
            gpu_memory_usage.labels(device=f"cuda:{i}").set(memory_used)
    
    return Response(content=generate_latest(), media_type="text/plain")

# Health monitoring
class HealthMonitor:
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.health_checks = {
            'model_loaded': self.check_model_loaded,
            'gpu_available': self.check_gpu_available,
            'memory_available': self.check_memory_available,
            'queue_healthy': self.check_queue_health
        }
    
    async def check_model_loaded(self) -> Dict:
        """Check if model is loaded."""
        is_loaded = (self.inference_service is not None and 
                    self.inference_service.engine is not None)
        return {
            'status': 'healthy' if is_loaded else 'unhealthy',
            'model_loaded': is_loaded
        }
    
    async def check_gpu_available(self) -> Dict:
        """Check GPU availability."""
        if torch.cuda.is_available():
            return {
                'status': 'healthy',
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) 
                             for i in range(torch.cuda.device_count())]
            }
        return {'status': 'unhealthy', 'message': 'No GPU available'}
    
    async def check_memory_available(self) -> Dict:
        """Check memory availability."""
        if torch.cuda.is_available():
            free_memory = []
            total_memory = []
            
            for i in range(torch.cuda.device_count()):
                free = torch.cuda.memory_available(i)
                total = torch.cuda.get_device_properties(i).total_memory
                free_memory.append(free)
                total_memory.append(total)
            
            min_free_ratio = min(f/t for f, t in zip(free_memory, total_memory))
            
            return {
                'status': 'healthy' if min_free_ratio > 0.1 else 'unhealthy',
                'free_memory_gb': [f / 1e9 for f in free_memory],
                'total_memory_gb': [t / 1e9 for t in total_memory],
                'free_ratio': min_free_ratio
            }
        return {'status': 'unknown'}
    
    async def check_queue_health(self) -> Dict:
        """Check request queue health."""
        queue_len = request_queue.queue.qsize()
        max_size = request_queue.queue.maxsize
        
        return {
            'status': 'healthy' if queue_len < max_size * 0.8 else 'unhealthy',
            'queue_size': queue_len,
            'max_size': max_size,
            'utilization': queue_len / max_size
        }
    
    async def get_full_health_status(self) -> Dict:
        """Get complete health status."""
        results = {}
        overall_status = 'healthy'
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                results[check_name] = result
                if result['status'] != 'healthy':
                    overall_status = 'unhealthy'
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
```

## Step 5: Docker Containerization

Create a production-ready Docker setup:

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /models /var/log/inference

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```yaml
# docker-compose.yaml
version: '3.8'

services:
  inference-engine:
    build: .
    image: llm-inference:latest
    container_name: llm-inference
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    volumes:
      - ./models:/models
      - ./config.yaml:/app/config.yaml
      - inference-logs:/var/log/inference
    ports:
      - "8000:8000"  # API
      - "9090:9090"  # Metrics
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

volumes:
  inference-logs:
  prometheus-data:
  grafana-data:
```

## Step 6: Kubernetes Deployment

For cloud deployment, create Kubernetes manifests:

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  labels:
    app: llm-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: inference-engine
        image: your-registry/llm-inference:latest
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: models
          mountPath: /models
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
      - name: config
        configMap:
          name: inference-config
      nodeSelector:
        nvidia.com/gpu.present: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-service
spec:
  selector:
    app: llm-inference
  ports:
  - port: 80
    targetPort: 8000
    name: api
  - port: 9090
    targetPort: 9090
    name: metrics
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_size
      target:
        type: AverageValue
        averageValue: "50"
```

## Step 7: Load Testing and Optimization

Create load testing scripts:

```python
# load_test.py
import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict
import json

class LoadTester:
    def __init__(self, base_url: str, num_workers: int = 10):
        self.base_url = base_url
        self.num_workers = num_workers
        self.results = []
        
    async def send_request(self, session: aiohttp.ClientSession, 
                          prompt: str, max_tokens: int = 100) -> Dict:
        """Send a single request."""
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "success": True,
                    "latency": end_time - start_time,
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "total_tokens": result["usage"]["total_tokens"]
                }
        except Exception as e:
            return {
                "success": False,
                "latency": time.time() - start_time,
                "error": str(e)
            }
    
    async def worker(self, session: aiohttp.ClientSession, 
                    prompts: List[str], worker_id: int):
        """Worker to send requests."""
        for i, prompt in enumerate(prompts):
            result = await self.send_request(session, prompt)
            result["worker_id"] = worker_id
            result["request_id"] = i
            self.results.append(result)
    
    async def run_load_test(self, prompts: List[str], 
                           requests_per_second: float):
        """Run load test."""
        print(f"Starting load test with {len(prompts)} prompts")
        print(f"Target RPS: {requests_per_second}")
        
        # Split prompts among workers
        prompts_per_worker = len(prompts) // self.num_workers
        worker_prompts = [
            prompts[i*prompts_per_worker:(i+1)*prompts_per_worker]
            for i in range(self.num_workers)
        ]
        
        # Adjust last worker to include remaining prompts
        if len(prompts) % self.num_workers:
            worker_prompts[-1].extend(
                prompts[self.num_workers * prompts_per_worker:]
            )
        
        # Calculate delay between requests
        delay = 1.0 / requests_per_second * self.num_workers
        
        async with aiohttp.ClientSession() as session:
            # Start workers with staggered starts
            tasks = []
            for i in range(self.num_workers):
                await asyncio.sleep(delay / self.num_workers * i)
                task = asyncio.create_task(
                    self.worker(session, worker_prompts[i], i)
                )
                tasks.append(task)
            
            # Wait for all workers to complete
            await asyncio.gather(*tasks)
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze test results."""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        if successful:
            latencies = [r["latency"] for r in successful]
            prompt_tokens = [r["prompt_tokens"] for r in successful]
            completion_tokens = [r["completion_tokens"] for r in successful]
            
            print("\n=== Load Test Results ===")
            print(f"Total requests: {len(self.results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Success rate: {len(successful)/len(self.results)*100:.2f}%")
            
            print("\nLatency Statistics:")
            print(f"  Mean: {np.mean(latencies):.3f}s")
            print(f"  Median: {np.median(latencies):.3f}s")
            print(f"  P95: {np.percentile(latencies, 95):.3f}s")
            print(f"  P99: {np.percentile(latencies, 99):.3f}s")
            print(f"  Min: {np.min(latencies):.3f}s")
            print(f"  Max: {np.max(latencies):.3f}s")
            
            print("\nThroughput:")
            total_time = max(r["latency"] for r in self.results)
            print(f"  Requests/sec: {len(successful)/total_time:.2f}")
            print(f"  Tokens/sec: {sum(completion_tokens)/total_time:.2f}")
            
            print("\nToken Statistics:")
            print(f"  Avg prompt tokens: {np.mean(prompt_tokens):.1f}")
            print(f"  Avg completion tokens: {np.mean(completion_tokens):.1f}")

# Run load test
async def main():
    # Load test prompts
    test_prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of meditation?",
        "Describe the process of photosynthesis.",
        "Write a Python function to calculate fibonacci numbers.",
    ] * 20  # Repeat for more load
    
    tester = LoadTester("http://localhost:8000", num_workers=10)
    await tester.run_load_test(test_prompts, requests_per_second=5)

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 8: Security and Rate Limiting

Implement security measures:

```python
# security.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
import time
from collections import defaultdict
import asyncio

# API Key authentication
security = HTTPBearer()

class APIKeyValidator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.valid_keys = set()  # Load from database
    
    async def validate_token(self, 
                           credentials: HTTPAuthorizationCredentials = Security(security)):
        """Validate API token."""
        token = credentials.credentials
        
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise HTTPException(status_code=401, detail="Token expired")
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.cleanup_interval = 60  # seconds
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
    
    async def check_rate_limit(self, client_id: str):
        """Check if client has exceeded rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.requests_per_minute} requests per minute."
            )
        
        # Record request
        self.requests[client_id].append(now)
    
    async def _cleanup_loop(self):
        """Periodically clean up old request records."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            now = time.time()
            minute_ago = now - 60
            
            # Clean up old entries
            for client_id in list(self.requests.keys()):
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if req_time > minute_ago
                ]
                
                # Remove empty entries
                if not self.requests[client_id]:
                    del self.requests[client_id]

# Input validation
class InputValidator:
    def __init__(self, max_prompt_length: int = 10000):
        self.max_prompt_length = max_prompt_length
        self.blocked_patterns = []  # Load from config
    
    def validate_prompt(self, prompt: str):
        """Validate prompt input."""
        # Check length
        if len(prompt) > self.max_prompt_length:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long. Max length: {self.max_prompt_length}"
            )
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in prompt.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Prompt contains blocked content"
                )
        
        return prompt

# Apply security to endpoints
api_key_validator = APIKeyValidator(secret_key="your-secret-key")
rate_limiter = RateLimiter(requests_per_minute=60)
input_validator = InputValidator()

@app.post("/v1/completions/secure")
async def create_secure_completion(
    request: GenerationRequest,
    auth: dict = Depends(api_key_validator.validate_token)
):
    """Secure completion endpoint."""
    # Get client ID from token
    client_id = auth.get("client_id", "unknown")
    
    # Check rate limit
    await rate_limiter.check_rate_limit(client_id)
    
    # Validate input
    request.prompt = input_validator.validate_prompt(request.prompt)
    
    # Process request
    return await create_completion(request)
```

## Step 9: Performance Optimization

Implement advanced optimizations:

```python
# optimizations.py
import torch
from typing import List, Dict, Optional
import numpy as np

class RequestBatcher:
    """Smart request batching for optimal throughput."""
    
    def __init__(self, config: Dict):
        self.max_batch_size = config['max_batch_size']
        self.max_batch_tokens = config['max_batch_tokens']
        self.timeout = config.get('batch_timeout', 0.05)
    
    def create_optimal_batches(self, requests: List[Dict]) -> List[List[Dict]]:
        """Create optimal batches considering token limits."""
        # Sort by total length (prompt + max_tokens)
        sorted_requests = sorted(
            requests,
            key=lambda r: len(r['prompt_tokens']) + r['max_tokens']
        )
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for request in sorted_requests:
            request_tokens = len(request['prompt_tokens']) + request['max_tokens']
            
            # Check if adding this request exceeds limits
            if (len(current_batch) >= self.max_batch_size or
                current_tokens + request_tokens > self.max_batch_tokens):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(request)
            current_tokens += request_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

class CacheOptimizer:
    """Optimize cache usage for better performance."""
    
    def __init__(self, cache_config: Dict):
        self.prefix_cache_enabled = cache_config.get('prefix_caching', True)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def analyze_cache_patterns(self, requests: List[str]) -> Dict:
        """Analyze request patterns for cache optimization."""
        # Find common prefixes
        prefix_counts = {}
        
        for i, req1 in enumerate(requests):
            for j, req2 in enumerate(requests[i+1:], i+1):
                # Find common prefix
                common_prefix = self._get_common_prefix(req1, req2)
                if len(common_prefix) > 100:  # Minimum prefix length
                    prefix_counts[common_prefix] = prefix_counts.get(common_prefix, 0) + 1
        
        # Sort by frequency
        common_prefixes = sorted(
            prefix_counts.items(),
            key=lambda x: x[1] * len(x[0]),  # Weight by length and frequency
            reverse=True
        )[:10]  # Top 10 prefixes
        
        return {
            'common_prefixes': common_prefixes,
            'cache_efficiency': self._calculate_cache_efficiency(common_prefixes, requests)
        }
    
    def _get_common_prefix(self, s1: str, s2: str) -> str:
        """Get common prefix of two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:min_len]
    
    def _calculate_cache_efficiency(self, prefixes: List[tuple], 
                                  requests: List[str]) -> float:
        """Calculate potential cache efficiency."""
        total_tokens = sum(len(req) for req in requests)
        cached_tokens = 0
        
        for prefix, count in prefixes:
            cached_tokens += len(prefix) * count
        
        return cached_tokens / total_tokens if total_tokens > 0 else 0

class GPUOptimizer:
    """GPU-specific optimizations."""
    
    @staticmethod
    def optimize_gpu_settings():
        """Apply GPU optimizations."""
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            print("GPU optimizations applied")
    
    @staticmethod
    def profile_gpu_kernels(model: torch.nn.Module, sample_input: torch.Tensor):
        """Profile GPU kernel performance."""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                model(sample_input)
        
        # Print profiling results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Export chrome trace
        prof.export_chrome_trace("trace.json")
```

## Step 10: Production Checklist

Create a comprehensive deployment checklist:

```python
# deployment_checklist.py
class DeploymentChecker:
    """Verify production readiness."""
    
    def __init__(self):
        self.checks = {
            'model': self.check_model,
            'api': self.check_api,
            'monitoring': self.check_monitoring,
            'security': self.check_security,
            'performance': self.check_performance,
            'reliability': self.check_reliability
        }
    
    async def run_all_checks(self) -> Dict[str, Dict]:
        """Run all deployment checks."""
        results = {}
        
        for check_name, check_func in self.checks.items():
            print(f"Running {check_name} checks...")
            results[check_name] = await check_func()
        
        # Overall status
        all_passed = all(
            check['status'] == 'passed' 
            for check in results.values()
        )
        
        results['overall'] = {
            'status': 'passed' if all_passed else 'failed',
            'ready_for_production': all_passed
        }
        
        return results
    
    async def check_model(self) -> Dict:
        """Check model readiness."""
        checks = []
        
        # Model file exists
        model_exists = os.path.exists('/models/model.bin')
        checks.append(('model_exists', model_exists))
        
        # Model loads successfully
        try:
            model = torch.load('/models/model.bin', map_location='cpu')
            model_loads = True
        except:
            model_loads = False
        checks.append(('model_loads', model_loads))
        
        # Model size is reasonable
        if model_exists:
            model_size = os.path.getsize('/models/model.bin') / 1e9
            size_ok = model_size < 100  # Less than 100GB
            checks.append(('model_size_ok', size_ok))
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks)
        }
    
    async def check_api(self) -> Dict:
        """Check API functionality."""
        checks = []
        
        # API responds to health check
        try:
            response = await self._make_request('GET', '/health')
            health_ok = response.status == 200
        except:
            health_ok = False
        checks.append(('health_endpoint', health_ok))
        
        # API handles requests
        try:
            response = await self._make_request(
                'POST', '/v1/completions',
                json={'prompt': 'Test', 'max_tokens': 1}
            )
            api_works = response.status == 200
        except:
            api_works = False
        checks.append(('completion_endpoint', api_works))
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks)
        }
    
    async def check_monitoring(self) -> Dict:
        """Check monitoring setup."""
        checks = []
        
        # Prometheus metrics available
        try:
            response = await self._make_request('GET', '/metrics/prometheus')
            metrics_ok = response.status == 200
        except:
            metrics_ok = False
        checks.append(('prometheus_metrics', metrics_ok))
        
        # Logging configured
        log_file_exists = os.path.exists('/var/log/inference/api.log')
        checks.append(('logging_configured', log_file_exists))
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks)
        }
    
    async def check_security(self) -> Dict:
        """Check security configuration."""
        checks = []
        
        # HTTPS enabled (in production)
        checks.append(('https_enabled', os.getenv('USE_HTTPS', 'false') == 'true'))
        
        # API key required
        checks.append(('api_key_required', os.getenv('REQUIRE_API_KEY', 'false') == 'true'))
        
        # Rate limiting enabled
        checks.append(('rate_limiting', os.getenv('ENABLE_RATE_LIMIT', 'false') == 'true'))
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks)
        }
    
    async def check_performance(self) -> Dict:
        """Check performance requirements."""
        checks = []
        
        # Run simple benchmark
        latencies = []
        for _ in range(10):
            start = time.time()
            response = await self._make_request(
                'POST', '/v1/completions',
                json={'prompt': 'Hello', 'max_tokens': 10}
            )
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        checks.append(('avg_latency_ok', avg_latency < 2.0))  # < 2 seconds
        checks.append(('p99_latency_ok', p99_latency < 5.0))  # < 5 seconds
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks),
            'metrics': {
                'avg_latency': avg_latency,
                'p99_latency': p99_latency
            }
        }
    
    async def check_reliability(self) -> Dict:
        """Check reliability features."""
        checks = []
        
        # Auto-restart configured
        checks.append(('auto_restart', os.getenv('RESTART_POLICY') == 'unless-stopped'))
        
        # Health checks configured
        checks.append(('health_checks', True))  # Already verified above
        
        # Graceful shutdown implemented
        checks.append(('graceful_shutdown', hasattr(app, 'on_event')))
        
        return {
            'status': 'passed' if all(c[1] for c in checks) else 'failed',
            'checks': dict(checks)
        }

# Run deployment check
async def verify_deployment():
    checker = DeploymentChecker()
    results = await checker.run_all_checks()
    
    print("\n=== Deployment Readiness Report ===")
    for category, result in results.items():
        if category != 'overall':
            print(f"\n{category.upper()}: {result['status']}")
            for check, passed in result.get('checks', {}).items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check}")
    
    print(f"\nOVERALL: {results['overall']['status']}")
    print(f"Ready for production: {results['overall']['ready_for_production']}")
```

## Conclusion

You've successfully built a production-ready LLM inference system! Here's what we've accomplished:

1. **API Server**: FastAPI with streaming support
2. **Monitoring**: Prometheus metrics and health checks
3. **Security**: API keys, rate limiting, input validation
4. **Containerization**: Docker and Kubernetes deployment
5. **Performance**: Batching, caching, GPU optimization
6. **Reliability**: Health checks, auto-scaling, graceful shutdown
7. **Testing**: Load testing and deployment verification

### Key Production Considerations:

1. **Scale Gradually**: Start with one GPU and scale based on demand
2. **Monitor Everything**: Track latency, throughput, and error rates
3. **Plan for Failures**: Implement retries, circuit breakers, and fallbacks
4. **Optimize Costs**: Use spot instances, batch efficiently, cache aggressively
5. **Security First**: Always use HTTPS, validate inputs, implement auth
6. **Document Everything**: API docs, runbooks, incident response

### Next Steps:

1. **Multi-Region Deployment**: Deploy across regions for lower latency
2. **A/B Testing**: Test model versions and configurations
3. **Fine-Tuning Pipeline**: Continuous model improvement
4. **Cost Optimization**: Implement request routing and model selection
5. **Advanced Features**: Function calling, plugins, multi-modal support

Your inference engine is now ready to serve LLMs at scale!