import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import set_seed, format_prompt
import pandas as pd
from tqdm import tqdm
import json
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare OpenAssistant dataset for Phi-2 fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Model name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of data to use for training")
    parser.add_argument("--lang", type=str, default="en", help="Language filter for dataset")
    parser.add_argument("--quality_threshold", type=float, default=0.5, help="Quality threshold for filtering")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--min_response_length", type=int, default=10, help="Minimum response length in tokens")
    parser.add_argument("--max_response_length", type=int, default=1024, help="Maximum response length in tokens")
    parser.add_argument("--save_jsonl", action="store_true", help="Save data in JSONL format as well")
    parser.add_argument("--skip_quality_filter", action="store_true", help="Skip quality filtering")
    parser.add_argument("--truncate_long_sequences", action="store_true", help="Truncate sequences longer than max_length instead of filtering them out")
    parser.add_argument("--max_processing_time", type=int, default=120, help="Maximum time in minutes to spend processing the dataset")
    return parser.parse_args()

def filter_and_process_oasst(dataset, tokenizer, args):
    """Filter and process the OpenAssistant dataset."""
    # Filter by language
    print(f"Filtering by language: {args.lang}")
    filtered_data = dataset.filter(lambda x: x["lang"] == args.lang)
    
    # Filter by quality (if available and not skipped)
    if not args.skip_quality_filter and "labels" in filtered_data.column_names:
        print(f"Filtering by quality threshold: {args.quality_threshold}")
        
        def quality_filter(example):
            # Handle None values for labels
            if example.get("labels") is None:
                return False
                
            # Check if quality exists in labels
            if isinstance(example["labels"], dict) and "quality" in example["labels"]:
                quality_value = example["labels"].get("quality", {}).get("value", 0)
                return quality_value >= args.quality_threshold
            return True
        
        filtered_data = filtered_data.filter(quality_filter)
    else:
        if args.skip_quality_filter:
            print("Skipping quality filtering as requested")
        else:
            print("No 'labels' field found for quality filtering")
    
    # Extract prompt-response pairs
    processed_data = []
    skipped_long_sequences = 0
    truncated_sequences = 0
    
    # Set a time limit for processing
    import time
    start_time = time.time()
    max_time_seconds = args.max_processing_time * 60  # Convert minutes to seconds
    
    # Limit the number of items to process if max_samples is specified
    items_to_process = filtered_data
    if args.max_samples and args.max_samples < len(filtered_data):
        # Take a random sample instead of just the first N items
        import random
        random_indices = random.sample(range(len(filtered_data)), args.max_samples)
        items_to_process = [filtered_data[i] for i in sorted(random_indices)]
        print(f"Randomly sampled {args.max_samples} items from {len(filtered_data)} total items")
    
    for item in tqdm(items_to_process, desc="Processing dataset"):
        # Check if we've exceeded the time limit
        if time.time() - start_time > max_time_seconds:
            print(f"\nReached maximum processing time of {args.max_processing_time} minutes. Stopping early.")
            break
            
        # Extract the message tree
        if "message_tree_id" in item and "text" in item and "role" in item:
            # Simple extraction for prompt-response pairs
            if item["role"] == "assistant" and item.get("parent_id") is not None:
                # Find the parent message (prompt)
                parent = next((x for x in filtered_data if x["message_id"] == item["parent_id"]), None)
                if parent and parent["role"] == "prompter":
                    try:
                        # Check response length
                        response_tokens = len(tokenizer.encode(item["text"]))
                        if response_tokens < args.min_response_length:
                            continue
                        if args.max_response_length and response_tokens > args.max_response_length:
                            continue
                        
                        # Check combined length
                        combined_text = format_prompt(parent["text"], item["text"])
                        combined_tokens = len(tokenizer.encode(combined_text))
                        
                        # Handle sequences longer than max_length
                        if combined_tokens > args.max_length:
                            if args.truncate_long_sequences:
                                # Truncate by reducing the response length
                                truncated_sequences += 1
                                # We'll still include this item, but note that it will be truncated during training
                            else:
                                skipped_long_sequences += 1
                                continue
                        
                        processed_data.append({
                            "prompt": parent["text"],
                            "response": item["text"],
                            "message_id": item["message_id"],
                            "parent_id": item["parent_id"],
                            "prompt_tokens": len(tokenizer.encode(parent["text"])),
                            "response_tokens": response_tokens,
                            "total_tokens": combined_tokens,
                            "needs_truncation": combined_tokens > args.max_length
                        })
                    except Exception as e:
                        # Skip items that cause tokenization errors
                        print(f"Error processing item: {e}")
                        continue
    
    print(f"Skipped {skipped_long_sequences} items due to sequence length > {args.max_length}")
    if args.truncate_long_sequences:
        print(f"Marked {truncated_sequences} items for truncation during training")
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Print token statistics
    print("\nToken Statistics (before any truncation):")
    print(f"Average prompt tokens: {df['prompt_tokens'].mean():.2f}")
    print(f"Average response tokens: {df['response_tokens'].mean():.2f}")
    print(f"Average total tokens: {df['total_tokens'].mean():.2f}")
    print(f"Max total tokens: {df['total_tokens'].max()}")
    print(f"Items needing truncation: {df['needs_truncation'].sum()} ({df['needs_truncation'].mean()*100:.2f}%)")
    
    return df

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {args.model_name}")
    try:
        # Add timeout and use_auth_token for Windows
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            use_fast=True,
            timeout=120  # Increase timeout for Windows
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying again with different parameters...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            use_fast=False  # Try without fast tokenizer on Windows
        )
    
    print("Loading OpenAssistant dataset...")
    try:
        # Add cache_dir for Windows to avoid path length issues
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset("OpenAssistant/oasst1", cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying again with different parameters...")
        dataset = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)
    
    print(f"Processing dataset...")
    processed_df = filter_and_process_oasst(
        dataset["train"], 
        tokenizer,
        args
    )
    
    print(f"Processed dataset size: {len(processed_df)}")
    
    # Check if we have enough data
    if len(processed_df) < 100:
        print("Warning: Very small dataset. Consider adjusting filters.")
    
    # Split into train and validation
    train_size = int(len(processed_df) * args.train_size)
    train_df = processed_df.iloc[:train_size]
    val_df = processed_df.iloc[train_size:]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Save to disk
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    val_csv_path = os.path.join(args.output_dir, "val.csv")
    
    # Use encoding parameter for Windows
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    
    # Save in JSONL format if requested
    if args.save_jsonl:
        train_jsonl_path = os.path.join(args.output_dir, "train.jsonl")
        val_jsonl_path = os.path.join(args.output_dir, "val.jsonl")
        
        with open(train_jsonl_path, "w", encoding="utf-8") as f:
            for _, row in train_df.iterrows():
                f.write(json.dumps({
                    "prompt": row["prompt"],
                    "response": row["response"]
                }, ensure_ascii=False) + "\n")
        
        with open(val_jsonl_path, "w", encoding="utf-8") as f:
            for _, row in val_df.iterrows():
                f.write(json.dumps({
                    "prompt": row["prompt"],
                    "response": row["response"]
                }, ensure_ascii=False) + "\n")
    
    # Print token statistics
    print("\nFinal Dataset Token Statistics:")
    print(f"Average prompt tokens: {processed_df['prompt_tokens'].mean():.2f}")
    print(f"Average response tokens: {processed_df['response_tokens'].mean():.2f}")
    print(f"Average total tokens: {processed_df['total_tokens'].mean():.2f}")
    print(f"Max total tokens: {processed_df['total_tokens'].max()}")
    
    print(f"Dataset preparation complete. Files saved to {args.output_dir}")
    print(f"Operating system: {platform.system()} {platform.release()}")

if __name__ == "__main__":
    main() 