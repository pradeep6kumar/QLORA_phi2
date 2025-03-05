import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import format_prompt, set_seed
from datasets import load_dataset, Dataset
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import platform
import gc
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Phi-2 model")
    parser.add_argument("--model_dir", type=str, default="./output", help="Directory containing the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Base model name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--windows_mode", action="store_true", help="Enable Windows-specific optimizations")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for Windows")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading tokenizer from {args.base_model}")
    max_retries = 3
    retry_delay = 5  # seconds
    
    # Try loading tokenizer with retries
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model, 
                trust_remote_code=True,
                use_fast=not args.windows_mode  # Disable fast tokenizer on Windows
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error loading tokenizer (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to load tokenizer after {max_retries} attempts. Trying with different parameters...")
                tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model, 
                    trust_remote_code=True,
                    use_fast=False,
                    local_files_only=False
                )
    
    print(f"Loading base model from {args.base_model}")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Windows-specific device map
    if args.windows_mode and args.cpu_offload:
        device_map = {
            "": "cpu",  # Default placement
            "model.embed_tokens": "cuda:0",
            "model.layers.0": "cuda:0",
            "model.layers.1": "cuda:0",
            "lm_head": "cuda:0",
        }
        print("Using CPU offloading for Windows")
    else:
        device_map = "auto"
    
    # Try loading model with retries
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=args.windows_mode,  # Enable for Windows
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error loading model (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to load model after {max_retries} attempts. Trying with different parameters...")
                # Try with more conservative settings
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    offload_folder="offload_folder"
                )
    
    print(f"Loading LoRA adapters from {args.model_dir}")
    try:
        model = PeftModel.from_pretrained(model, args.model_dir)
    except Exception as e:
        print(f"Error loading LoRA adapters: {e}")
        print("Continuing with base model only...")
    
    return model, tokenizer

def load_validation_data(data_dir, num_samples=None):
    """Load validation data."""
    val_path = os.path.join(data_dir, "val.csv")
    
    # Try with different encodings for Windows
    try:
        val_df = pd.read_csv(val_path, encoding='utf-8')
    except Exception as e:
        print(f"Error loading validation data with utf-8 encoding: {e}")
        print("Trying with different encoding...")
        val_df = pd.read_csv(val_path, encoding='latin1')
    
    if num_samples and num_samples < len(val_df):
        val_df = val_df.sample(num_samples, random_state=42)
    
    return val_df

def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate a response for the given prompt."""
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=False,  # Use greedy decoding for evaluation
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    response = generated_text.split("Assistant:")[-1].strip()
    
    return response

def compute_metrics(predictions, references):
    """Compute evaluation metrics."""
    # Initialize ROUGE scorer
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Initialize metrics
    rouge_scores = {rouge_type: [] for rouge_type in rouge_types}
    bleu_scores = []
    
    # Compute metrics for each example
    for pred, ref in zip(predictions, references):
        # ROUGE scores
        rouge_score = scorer.score(ref, pred)
        for rouge_type in rouge_types:
            rouge_scores[rouge_type].append(rouge_score[rouge_type].fmeasure)
        
        # BLEU score
        smoothing = SmoothingFunction().method1
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        
        # Handle empty predictions or references
        if len(pred_tokens) == 0:
            pred_tokens = [""]
        if len(ref_tokens) == 0:
            ref_tokens = [""]
            
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    # Compute average scores
    metrics = {
        "rouge1": np.mean(rouge_scores["rouge1"]),
        "rouge2": np.mean(rouge_scores["rouge2"]),
        "rougeL": np.mean(rouge_scores["rougeL"]),
        "bleu": np.mean(bleu_scores),
    }
    
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Print system info
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load validation data
    try:
        val_df = load_validation_data(args.data_dir, args.num_samples)
        print(f"Loaded {len(val_df)} validation examples")
    except FileNotFoundError:
        print(f"Validation file not found at {os.path.join(args.data_dir, 'val.csv')}")
        print("Creating a small sample dataset for testing...")
        # Create a small sample dataset for testing
        val_df = pd.DataFrame({
            "prompt": ["What is machine learning?", "Explain quantum computing"],
            "response": ["Machine learning is a field of AI...", "Quantum computing uses quantum bits..."]
        })
    
    # Generate predictions
    predictions = []
    references = []
    
    print("Generating predictions...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        prompt = row["prompt"]
        reference = row["response"]
        
        prediction = generate_response(model, tokenizer, prompt, args.max_length)
        
        predictions.append(prediction)
        references.append(reference)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    
    # Save predictions and metrics
    results_dir = os.path.join(args.model_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        "prompt": val_df["prompt"].tolist(),
        "reference": references,
        "prediction": predictions
    })
    results_df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 