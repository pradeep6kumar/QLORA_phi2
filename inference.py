import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import format_prompt, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Phi-2 model")
    parser.add_argument("--model_dir", type=str, default="./output", help="Directory containing the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Base model name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--sample_prompts", action="store_true", help="Run with sample prompts")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapters from {args.model_dir}")
    model = PeftModel.from_pretrained(model, args.model_dir)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, args):
    """Generate a response for the given prompt."""
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    response = generated_text.split("Assistant:")[-1].strip()
    
    return response

def interactive_mode(model, tokenizer, args):
    """Run the model in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if the user wants to exit
        if user_input.lower() == "exit":
            break
        
        # Generate and print the response
        response = generate_response(model, tokenizer, user_input, args)
        print(f"\nAssistant: {response}")

def sample_prompts_mode(model, tokenizer, args):
    """Run the model with sample prompts."""
    sample_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "How do I improve my programming skills?",
        "What are the ethical concerns with AI development?",
    ]
    
    print("\n=== Sample Prompts Mode ===")
    
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt, args)
        print(f"Response: {response}")
        print("-" * 50)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Run in the specified mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.sample_prompts:
        sample_prompts_mode(model, tokenizer, args)
    else:
        print("Please specify a mode: --interactive or --sample_prompts")

if __name__ == "__main__":
    main() 