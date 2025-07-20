#!/usr/bin/env python3
"""
Test Model Loading and Inference
Verifies that we can load and run a small language model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def test_model_loading():
    print("🤖 Testing model loading and inference...\n")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    
    print(f"📥 Loading model: {model_name}")
    print("This may take a few minutes on first run...\n")
    
    start_time = time.time()
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"\n✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"🖥️  Running on: {device.upper()}")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params / 1e6:.1f}M")
        
        # Test inference
        test_prompts = [
            "Hello, how are you?",
            "What is the meaning of life?",
            "Tell me a joke about programming."
        ]
        
        print("\n🧪 Running inference tests...\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}")
            print(f"Prompt: {prompt}")
            
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            input_length = inputs.shape[1]
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"Response: {generated_text}")
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            
            # Calculate metrics
            output_length = outputs.shape[1]
            new_tokens = output_length - input_length
            tokens_per_second = new_tokens / generation_time
            
            print(f"📊 Metrics:")
            print(f"   - Input tokens: {input_length}")
            print(f"   - Generated tokens: {new_tokens}")
            print(f"   - Tokens/second: {tokens_per_second:.2f}")
            print("-" * 50 + "\n")
        
        # Memory usage (if GPU)
        if device == "cuda":
            print("💾 GPU Memory Usage:")
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   - Allocated: {allocated:.2f} GB")
            print(f"   - Reserved: {reserved:.2f} GB")
        
        print("\n✅ All tests passed! Your environment is ready for the tutorial.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection (for model download)")
        print("2. Ensure you have enough disk space")
        print("3. Try a smaller model like 'gpt2'")
        print("4. Check the error message above for specific issues")
        
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()

def test_batch_inference():
    """Test batch inference capabilities"""
    print("\n🔬 Testing batch inference...\n")
    
    try:
        model_name = "gpt2"  # Even smaller model for batch testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Multiple prompts
        prompts = [
            "The future of AI is",
            "In the year 2050,",
            "The most important invention"
        ]
        
        print(f"Testing batch size: {len(prompts)}")
        
        # Tokenize all prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        # Generate for all prompts at once
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True
            )
        batch_time = time.time() - start_time
        
        # Decode all outputs
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        print(f"\n✅ Batch inference successful!")
        print(f"⏱️  Total time: {batch_time:.2f}s")
        print(f"📊 Time per prompt: {batch_time/len(prompts):.2f}s")
        
        for prompt, response in zip(prompts, responses):
            print(f"\nPrompt: {prompt}")
            print(f"Completion: {response[len(prompt):]}")
            
    except Exception as e:
        print(f"⚠️  Batch inference test failed: {e}")
        print("This is optional - you can continue with the tutorial")

if __name__ == "__main__":
    test_model_loading()
    
    # Optional: test batch inference
    print("\n" + "="*60)
    response = input("\nTest batch inference? (y/n): ")
    if response.lower() == 'y':
        test_batch_inference()