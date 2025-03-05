import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

def test_adapter(repo_id):
    print("Loading base model and tokenizer...")
    base_model = "microsoft/phi-2"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("\nLoading your LoRA adapter...")
    # Load the adapter
    model = PeftModel.from_pretrained(model, repo_id)
    
    # Test prompts
    test_prompts = [
        "Write a Python function to calculate the factorial of a number",
        "Explain what is machine learning in simple terms",
        "Write a professional email to schedule a team meeting"
    ]
    
    print("\nTesting with sample prompts...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"\nResponse (generated in {time.time() - start_time:.2f}s):")
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    # Replace with your actual repository ID
    repo_id = "pradeep6kumar2024/phi2-qlora-assistant"  # Update this
    test_adapter(repo_id) 