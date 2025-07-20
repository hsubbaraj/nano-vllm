# Module 1: Building a Minimal Inference Engine

## 🎯 Module Overview

In this module, we'll build a minimal but functional LLM inference engine from scratch. This will help you understand the core concepts of autoregressive text generation without the complexity of optimizations.

**Time Required**: 1-2 hours

## 📋 Learning Objectives

By the end of this module, you will:
- ✅ Understand autoregressive text generation
- ✅ Build a basic inference engine (~100 lines)
- ✅ Implement different sampling strategies
- ✅ Measure baseline performance
- ✅ Identify optimization opportunities

## 🏗️ What We're Building

```
Input: "Hello, how are"
   ↓
[Tokenizer] → [101, 234, 567]
   ↓
[Model Forward Pass]
   ↓
[Sample Next Token] → [432]
   ↓
[Decode] → "you"
   ↓
Output: "Hello, how are you"
```

## 📚 Core Concepts

### Autoregressive Generation

LLMs generate text one token at a time, using previously generated tokens as context:

```python
# Pseudocode for autoregressive generation
tokens = tokenize(prompt)
for i in range(max_new_tokens):
    logits = model(tokens)
    next_token = sample(logits[-1])  # Use last position
    tokens.append(next_token)
    if next_token == eos_token:
        break
return decode(tokens)
```

### The Generation Loop

1. **Tokenization**: Convert text to token IDs
2. **Forward Pass**: Run model to get logits
3. **Sampling**: Select next token from probability distribution
4. **Decoding**: Convert tokens back to text
5. **Repeat**: Until stop condition is met

## 💻 Implementation

Let's build our minimal engine step by step.

### Step 1: Basic Structure

Create `minimal_engine.py`:

```python
#!/usr/bin/env python3
"""
Minimal LLM Inference Engine
A simple implementation to understand core concepts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Dict
import time

class MinimalInferenceEngine:
    """A minimal LLM inference engine for educational purposes."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: HuggingFace model name or local path
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing engine on {self.device}")
        
        # Load tokenizer
        print(f"📚 Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"🤖 Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"✅ Engine ready! Model has {self.count_parameters()}M parameters")
    
    def count_parameters(self) -> float:
        """Count model parameters in millions."""
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def generate(self, 
                prompt: str, 
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                do_sample: bool = True,
                stream: bool = False) -> str:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            stream: Whether to yield tokens as they're generated
            
        Returns:
            Generated text completion
        """
        # Record start time
        start_time = time.time()
        
        # Step 1: Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        input_length = input_ids.shape[1]
        
        print(f"\n📝 Prompt: '{prompt}'")
        print(f"🔢 Input tokens: {input_length}")
        
        # Storage for generated tokens
        generated_tokens = []
        
        # Step 2: Autoregressive generation loop
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model outputs
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply sampling
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                    
                    # Sample from distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Add to generated tokens
                generated_tokens.append(next_token.item())
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stream output if requested
                if stream:
                    token_text = self.tokenizer.decode([next_token.item()])
                    print(token_text, end='', flush=True)
        
        # Step 3: Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / generation_time
        
        if not stream:
            print(f"💬 Generated: '{generated_text}'")
        else:
            print()  # New line after streaming
        
        print(f"\n📊 Performance Metrics:")
        print(f"  - Generated tokens: {len(generated_tokens)}")
        print(f"  - Generation time: {generation_time:.2f}s")
        print(f"  - Tokens/second: {tokens_per_second:.2f}")
        
        return generated_text
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Filter logits to keep only top k tokens.
        
        Args:
            logits: Raw model logits
            top_k: Number of top tokens to keep
            
        Returns:
            Filtered logits
        """
        if top_k == 0:
            return logits
        
        # Find top-k values and indices
        values, indices = torch.topk(logits, top_k)
        
        # Create a mask of -inf
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        # Fill in the top-k values
        filtered_logits[indices] = logits[indices]
        
        return filtered_logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.
        
        Args:
            logits: Raw model logits
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Get indices to remove in original order
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        
        # Set filtered positions to -inf
        logits[indices_to_remove] = float('-inf')
        
        return logits
```

### Step 2: Testing Different Sampling Strategies

Create `test_sampling.py`:

```python
#!/usr/bin/env python3
"""
Test different sampling strategies
"""

from minimal_engine import MinimalInferenceEngine

def test_sampling_strategies():
    """Compare different sampling strategies."""
    
    # Initialize engine with a small model
    engine = MinimalInferenceEngine("gpt2")
    
    prompt = "The future of artificial intelligence is"
    
    print("🧪 Testing Sampling Strategies\n")
    print(f"Prompt: '{prompt}'\n")
    print("=" * 60)
    
    # Test 1: Greedy Decoding
    print("\n1️⃣ GREEDY DECODING (deterministic)")
    print("-" * 40)
    result = engine.generate(
        prompt,
        max_new_tokens=30,
        do_sample=False
    )
    
    # Test 2: Temperature Sampling
    print("\n2️⃣ TEMPERATURE SAMPLING")
    temperatures = [0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature = {temp}")
        print("-" * 40)
        result = engine.generate(
            prompt,
            max_new_tokens=30,
            temperature=temp,
            do_sample=True
        )
    
    # Test 3: Top-k Sampling
    print("\n3️⃣ TOP-K SAMPLING")
    k_values = [10, 50, 100]
    
    for k in k_values:
        print(f"\n🎯 Top-k = {k}")
        print("-" * 40)
        result = engine.generate(
            prompt,
            max_new_tokens=30,
            top_k=k,
            temperature=0.8,
            do_sample=True
        )
    
    # Test 4: Top-p (Nucleus) Sampling
    print("\n4️⃣ TOP-P (NUCLEUS) SAMPLING")
    p_values = [0.5, 0.9, 0.95]
    
    for p in p_values:
        print(f"\n🎲 Top-p = {p}")
        print("-" * 40)
        result = engine.generate(
            prompt,
            max_new_tokens=30,
            top_p=p,
            top_k=0,  # Disable top-k
            temperature=0.8,
            do_sample=True
        )

if __name__ == "__main__":
    test_sampling_strategies()
```

### Step 3: Performance Analysis

Create `benchmark.py`:

```python
#!/usr/bin/env python3
"""
Benchmark the minimal inference engine
"""

import time
import torch
import numpy as np
from minimal_engine import MinimalInferenceEngine
from typing import List, Dict

def benchmark_engine(engine: MinimalInferenceEngine, 
                    prompts: List[str],
                    max_new_tokens: int = 50) -> Dict:
    """
    Benchmark engine performance on multiple prompts.
    
    Returns:
        Dictionary with performance metrics
    """
    results = []
    
    print("🏃 Running benchmark...\n")
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        # Warm up
        if i == 0:
            print("Warming up...")
            _ = engine.generate(prompt, max_new_tokens=5, do_sample=False)
        
        # Measure generation
        start_time = time.time()
        
        # Tokenize separately to measure pure generation time
        input_ids = engine.tokenizer.encode(prompt, return_tensors='pt').to(engine.device)
        input_tokens = input_ids.shape[1]
        
        # Generate
        output = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True
        )
        
        end_time = time.time()
        
        # Calculate metrics
        output_tokens = len(engine.tokenizer.encode(output))
        new_tokens = output_tokens - input_tokens
        total_time = end_time - start_time
        tokens_per_second = new_tokens / total_time if total_time > 0 else 0
        
        results.append({
            'prompt_length': input_tokens,
            'generated_tokens': new_tokens,
            'total_time': total_time,
            'tokens_per_second': tokens_per_second
        })
        
        print(f"  ⚡ {tokens_per_second:.2f} tokens/sec\n")
    
    # Aggregate results
    avg_tokens_per_second = np.mean([r['tokens_per_second'] for r in results])
    avg_time_per_token = 1.0 / avg_tokens_per_second if avg_tokens_per_second > 0 else float('inf')
    total_tokens = sum(r['generated_tokens'] for r in results)
    total_time = sum(r['total_time'] for r in results)
    
    print("\n📊 Benchmark Results")
    print("=" * 40)
    print(f"Total prompts: {len(prompts)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/second: {avg_tokens_per_second:.2f}")
    print(f"Average ms/token: {avg_time_per_token * 1000:.2f}")
    
    # Memory usage
    if engine.device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n💾 GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    return {
        'total_prompts': len(prompts),
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_tokens_per_second': avg_tokens_per_second,
        'avg_ms_per_token': avg_time_per_token * 1000,
        'results': results
    }

def analyze_bottlenecks():
    """Analyze performance bottlenecks in the engine."""
    print("\n🔍 Analyzing Performance Bottlenecks\n")
    
    engine = MinimalInferenceEngine("gpt2")
    prompt = "The quick brown fox jumps over the lazy dog. " * 5
    
    # Profile different sequence lengths
    sequence_lengths = [10, 50, 100, 200]
    results = []
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        start_time = time.time()
        _ = engine.generate(
            prompt,
            max_new_tokens=seq_len,
            do_sample=False  # Deterministic for consistency
        )
        total_time = time.time() - start_time
        
        results.append({
            'sequence_length': seq_len,
            'total_time': total_time,
            'time_per_token': total_time / seq_len
        })
    
    # Analyze scaling
    print("\n📈 Performance Scaling:")
    print("Seq Length | Total Time | Time/Token | Slowdown")
    print("-" * 50)
    
    base_time_per_token = results[0]['time_per_token']
    
    for r in results:
        slowdown = r['time_per_token'] / base_time_per_token
        print(f"{r['sequence_length']:10} | {r['total_time']:10.2f}s | {r['time_per_token']:10.3f}s | {slowdown:8.2f}x")
    
    print("\n⚠️  Notice: Time per token increases with sequence length!")
    print("This is because we recompute attention for all previous tokens.")
    print("This is the key bottleneck that KV caching will solve.")

def main():
    """Run all benchmarks."""
    
    # Test prompts of varying complexity
    test_prompts = [
        "Hello world",
        "The meaning of life is",
        "In a galaxy far, far away",
        "Once upon a time in a land of dragons and magic",
        "The quick brown fox jumps over the lazy dog",
        "Explain quantum computing in simple terms:",
        "def fibonacci(n):\n    '''Calculate the nth Fibonacci number'''",
        "The weather today is",
    ]
    
    # Initialize engine
    print("🚀 Initializing engine for benchmarking...\n")
    engine = MinimalInferenceEngine("gpt2")
    
    # Run main benchmark
    benchmark_results = benchmark_engine(engine, test_prompts, max_new_tokens=50)
    
    # Analyze bottlenecks
    analyze_bottlenecks()
    
    # Compare with HuggingFace baseline
    print("\n🤝 Comparing with HuggingFace generate()...")
    
    from transformers import pipeline
    hf_generator = pipeline('text-generation', model='gpt2', device=0 if torch.cuda.is_available() else -1)
    
    start_time = time.time()
    for prompt in test_prompts[:3]:  # Test subset
        _ = hf_generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.8)
    hf_time = time.time() - start_time
    
    print(f"HuggingFace time: {hf_time:.2f}s")
    print(f"Our engine time: {sum(r['total_time'] for r in benchmark_results['results'][:3]):.2f}s")
    print("\nNote: HuggingFace includes optimizations we'll implement in later modules!")

if __name__ == "__main__":
    main()
```

## 🔍 Understanding the Performance

### What's Happening Under the Hood

For each token generation:

1. **Forward Pass**: O(n²) attention computation where n is sequence length
2. **Sampling**: O(vocab_size) for computing probabilities
3. **Memory**: Store all intermediate activations

### Performance Characteristics

With our minimal engine on a typical GPU (RTX 3090):

| Sequence Length | Tokens/sec | Time/token |
|----------------|------------|------------|
| 10 tokens      | ~25        | 40ms       |
| 50 tokens      | ~20        | 50ms       |
| 100 tokens     | ~15        | 67ms       |
| 200 tokens     | ~10        | 100ms      |

### Key Bottlenecks

1. **Redundant Computation**: Recomputing attention for all previous tokens
2. **Memory Transfers**: Moving data between CPU and GPU
3. **No Batching**: Processing one sequence at a time
4. **Python Overhead**: Loop in Python instead of optimized kernels

## 🎯 Exercises

### Exercise 1: Add Streaming Output

Modify the engine to show tokens as they're generated:

```python
def generate_streaming(self, prompt: str, **kwargs):
    """Generate text with real-time streaming output."""
    # Hint: Yield tokens as they're generated
    # Update the main generate() method to support this
    pass
```

### Exercise 2: Implement Beam Search

Add beam search as an alternative to sampling:

```python
def beam_search(self, prompt: str, num_beams: int = 4, **kwargs):
    """Generate using beam search for better quality."""
    # Maintain top-k hypotheses
    # Expand each hypothesis and keep best scoring
    pass
```

### Exercise 3: Profile Memory Usage

Track memory allocation during generation:

```python
def generate_with_profiling(self, prompt: str, **kwargs):
    """Generate text while profiling memory usage."""
    # Use torch.cuda.memory_allocated() before/after each step
    # Plot memory usage over time
    pass
```

### Exercise 4: Add Stop Sequences

Implement early stopping when certain sequences are generated:

```python
def generate_with_stops(self, prompt: str, stop_sequences: List[str], **kwargs):
    """Stop generation when hitting stop sequences."""
    # Check generated text for stop sequences
    # Handle partial matches at token boundaries
    pass
```

## 📊 Performance Comparison

Here's how our minimal engine compares to optimized systems:

| System | Implementation | Tokens/sec | Relative Speed |
|--------|---------------|------------|----------------|
| Ours (Minimal) | Pure Python loop | ~20 | 1x (baseline) |
| HuggingFace | Some optimizations | ~50 | 2.5x |
| vLLM | Full optimizations | ~1400 | 70x |

The 70x gap shows the impact of optimizations we'll implement!

## 🎓 Key Takeaways

1. **Simplicity**: LLM inference can be implemented in ~100 lines
2. **Autoregressive**: Generate one token at a time using previous context
3. **Sampling Matters**: Different strategies produce different outputs
4. **Performance Gap**: Huge optimization potential (70x possible)
5. **Bottlenecks**: Redundant computation and lack of batching

## 🐛 Common Issues

### "CUDA out of memory"
- Use a smaller model (e.g., "gpt2" instead of "gpt2-medium")
- Reduce max_new_tokens
- Clear cache: `torch.cuda.empty_cache()`

### "Slow performance on CPU"
- This is expected - GPU acceleration is important
- Use shorter sequences for testing
- Consider using cloud GPUs

### "Different output each time"
- This is normal with sampling
- Use `do_sample=False` for deterministic output
- Set a fixed seed: `torch.manual_seed(42)`

## ⏭️ Next Steps

Our minimal engine works but has major inefficiencies:

1. **No batching**: Can't process multiple requests efficiently
2. **No KV cache**: Recomputes everything for each token
3. **No optimization**: Missing CUDA kernels, compilation, etc.

In [Module 2: Adding Batching](../module-02-batching/README.md), we'll implement dynamic batching for 3-4x speedup!

---

**Congratulations!** 🎉 You've built your first LLM inference engine. While simple, it demonstrates all the core concepts that production systems build upon.