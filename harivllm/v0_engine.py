import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import time

class V0Engine:
    def __init__(self, model_path: str, device: str = "cuda"):

        if not torch.cuda.is_available():
            raise Exception("CUDA is required to run this engine. Please check your GPU configuration.")
        
        self.device = device

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
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"✅ Engine ready! Model has {self.count_parameters()}M parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def generate(self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.6, 
        top_p: float = 0.95, 
        top_k: int = 40, 
        do_sample: bool = True,
        stream: bool = False
    ):

        start_time = time.time()
        # Tokenize the prompt
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        input_ids = self.tokenizer.encode([text], return_tensors="pt").to(self.device)

        input_length = input_ids.shape[1]

        # Storage for generated tokens
        generated_tokens = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits

                next_token_logits = logits[0, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if do_sample:

                    if top_k > 0:
                        next_token_logits = self.top_k_filter(next_token_logits, top_k)
                    
                    if top_p < 1.0:
                        next_token_logits = self.top_p_filter(next_token_logits, top_p)
                    
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

                else:
                    next_token = torch.argmax(next_token_logits)
                
                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)

                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                if stream:
                    token_text = self.tokenizer.decode([next_token.item()])
                    print(token_text, end="", flush=True)
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        
        values, indices = torch.topk(logits, top_k)

        # Copy of logits with all values set to -inf to match the shape of logits
        filtered_logits = torch.full_like(logits, -float('inf'))

        # Fill in only the top k values
        filtered_logits[indices] = logits[indices]
        
        return filtered_logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Filter logits such that cumulative probability of the top p tokens is less than or equal to top_p.
        
        Args:
            logits: Raw model logits
            top_p: Probability threshold for top tokens
            
        Returns:
            Filtered logits
        """

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Takes the cumulative sum of the softmaxed logits, should look like [1, 0.98, 0.95, etc]
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Find the cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Get indices to remove in original order
        indices_to_remove = sorted_indices[sorted_indices_to_remove]

        # Set filtered positions to -inf
        logits[indices_to_remove] = float('-inf')
        
        return logits





