import os
import argparse
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
import platform
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed as transformers_set_seed,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
import bitsandbytes as bnb
from utils import (
    set_seed,
    get_device,
    prepare_training_inputs,
    save_model_checkpoint,
    print_trainable_parameters,
    format_prompt,
)
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 with QLoRA on OpenAssistant dataset")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Model name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer to use")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    
    # QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4 or 8)")
    parser.add_argument("--target_modules", type=str, default="all", 
                        help="Target modules for LoRA. Options: 'all', 'attention', 'mlp', or comma-separated list")
    
    # Windows-specific arguments
    parser.add_argument("--windows_mode", action="store_true", help="Enable Windows-specific optimizations")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for Windows")
    
    return parser.parse_args()

def load_datasets(data_dir):
    """Load datasets from CSV files."""
    train_csv_path = os.path.join(data_dir, "train.csv")
    val_csv_path = os.path.join(data_dir, "val.csv")
    
    # Use encoding parameter for Windows
    train_df = pd.read_csv(train_csv_path, encoding='utf-8')
    val_df = pd.read_csv(val_csv_path, encoding='utf-8')
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def prepare_model_and_tokenizer(args):
    """Prepare the model and tokenizer with QLoRA configuration."""
    print(f"Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying again with different parameters...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            use_fast=False  # Try without fast tokenizer on Windows
        )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {args.model_name}")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # normalized float 4
    )
    
    # Windows-specific device map
    if args.windows_mode and args.cpu_offload:
        device_map = {
            "": "cpu",  # Default placement
            "model.embed_tokens": "cuda:0",
            "model.layers.0": "cuda:0",
            "model.layers.1": "cuda:0",
            "model.layers.2": "cuda:0",
            "model.layers.3": "cuda:0",
            "lm_head": "cuda:0",
        }
        print("Using CPU offloading for Windows")
    else:
        device_map = "auto"
    
    # Load model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying again with different parameters...")
        # Try with less memory-intensive settings
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Determine target modules based on the argument
    if args.target_modules == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif args.target_modules == "attention":
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    elif args.target_modules == "mlp":
        target_modules = ["fc1", "fc2"]
    else:
        target_modules = [module.strip() for module in args.target_modules.split(",")]
    
    print(f"Targeting modules: {target_modules}")
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        modules_to_save=["lm_head"],  # Save the LM head for better generation
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    return model, tokenizer

def preprocess_datasets(train_dataset, val_dataset, tokenizer, max_length):
    """Preprocess datasets for training."""
    print("Preprocessing training dataset...")
    
    # Define preprocessing function
    def preprocess_function(examples):
        return prepare_training_inputs(examples, tokenizer, max_length)
    
    # Apply preprocessing to datasets
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        remove_columns=val_dataset.column_names,
    )
    
    return train_dataset, val_dataset

def main():
    args = parse_args()
    set_seed(args.seed)
    transformers_set_seed(args.seed)
    
    # Print system info
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(args.data_dir)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args)
    
    # Preprocess datasets
    train_dataset, val_dataset = preprocess_datasets(
        train_dataset, val_dataset, tokenizer, args.max_length
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",  # Set to "wandb" if you want to use Weights & Biases
        # Windows-specific settings
        dataloader_num_workers=0 if args.windows_mode else 4,  # Avoid multiprocessing issues on Windows
        dataloader_pin_memory=not args.windows_mode,  # Disable pin memory on Windows
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 