# test_inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def test_model_loading():
    print("🤖 Testing model loading and inference...\n")
    
    # Use the regular FP16 model instead of FP8 to avoid segfault
    model_name = "Qwen/Qwen3-0.6B"
    
    print(f"Loading model: {model_name}")
    start_time = time.time()

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    start_time = time.time()
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    
    generation_time = time.time() - start_time
    
    # Decode
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

    print(f"\n⏱️  Generation time: {generation_time:.2f} seconds")
    print(f"📊 Tokens generated: {len(output_ids)}")
    print(f"⚡ Tokens/second: {len(output_ids) / generation_time:.2f}")

if __name__ == "__main__":
    test_model_loading()