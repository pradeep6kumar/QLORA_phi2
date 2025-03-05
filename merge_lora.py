import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Base model name")
    parser.add_argument("--lora_model", type=str, default="./output", help="Directory containing the LoRA weights")
    parser.add_argument("--output_dir", type=str, default="./merged_model", help="Output directory for the merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA model: {args.lora_model}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    print("Merging weights...")
    merged_model = lora_model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Model merging complete!")

if __name__ == "__main__":
    main() 