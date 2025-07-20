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


def main():
    """Example usage of the minimal engine."""
    
    # Initialize engine
    engine = MinimalInferenceEngine("gpt2")
    
    # Test different prompts
    test_prompts = [
        "The meaning of life is",
        "Once upon a time",
        "def hello_world():",
    ]
    
    print("="*60)
    print("🧪 Testing Minimal Inference Engine")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        result = engine.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True
        )
        print(f"{'='*60}\n")
    
    # Test streaming
    print("\n🌊 Testing streaming output:")
    print("-"*40)
    engine.generate(
        "The future of AI is",
        max_new_tokens=50,
        temperature=0.8,
        stream=True
    )


if __name__ == "__main__":
    main()