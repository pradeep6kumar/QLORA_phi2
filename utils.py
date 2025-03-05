import torch
from transformers import AutoTokenizer
import random
import numpy as np
import os
from typing import Dict, List, Union, Any

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def format_prompt(prompt: str, response: str = None) -> str:
    """Format prompt for Phi-2 model.
    
    Args:
        prompt: The user prompt
        response: Optional model response for training examples
    
    Returns:
        Formatted prompt string
    """
    if response:
        return f"Human: {prompt}\n\nAssistant: {response}"
    else:
        return f"Human: {prompt}\n\nAssistant:"

def prepare_training_inputs(examples: Dict[str, List], tokenizer, max_length: int):
    """Prepare inputs for training from dataset examples.
    
    Args:
        examples: Dictionary of examples from the dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Processed inputs for the model
    """
    # Format prompts and responses
    formatted_texts = []
    truncated_count = 0
    
    for i, (prompt, response) in enumerate(zip(examples["prompt"], examples["response"])):
        formatted_text = format_prompt(prompt, response)
        
        # Check if this will be truncated
        tokens = tokenizer(formatted_text, return_tensors="pt")["input_ids"].shape[1]
        if tokens > max_length:
            truncated_count += 1
            
            # Try to preserve the prompt and truncate the response if needed
            human_part = format_prompt(prompt).strip()
            human_tokens = tokenizer(human_part, return_tensors="pt")["input_ids"].shape[1]
            
            # If the human part alone is too long, we'll have to truncate it
            if human_tokens >= max_length - 50:  # Leave some room for the response
                # Just let the tokenizer handle truncation
                pass
            else:
                # Calculate how many tokens we can use for the response
                available_tokens = max_length - human_tokens
                
                # Truncate the response to fit
                response_tokens = tokenizer(response, return_tensors="pt")["input_ids"][0]
                if len(response_tokens) > available_tokens:
                    # Take the first available_tokens tokens of the response
                    truncated_response = tokenizer.decode(response_tokens[:available_tokens], skip_special_tokens=True)
                    formatted_text = format_prompt(prompt, truncated_response)
        
        formatted_texts.append(formatted_text)
    
    if truncated_count > 0:
        print(f"Warning: {truncated_count} sequences were longer than {max_length} tokens and were truncated.")
    
    # Tokenize the texts
    tokenized_inputs = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Prepare the labels (same as input_ids, but with -100 for non-response tokens)
    labels = tokenized_inputs["input_ids"].clone()
    
    # For each example, find where the Assistant part starts and mask the Human part
    for i, text in enumerate(formatted_texts):
        human_part = format_prompt(examples["prompt"][i]).strip()
        human_tokens = len(tokenizer(human_part, return_tensors="pt")["input_ids"][0])
        # Set labels for non-response tokens to -100 (ignored in loss calculation)
        labels[i, :human_tokens] = -100
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels
    }

def save_model_checkpoint(model, tokenizer, output_dir: str, step: int = None):
    """Save model checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save to
        step: Optional step number to include in the directory name
    """
    if step is not None:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    else:
        save_dir = output_dir
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model state
    model.save_pretrained(save_dir)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model saved to {save_dir}")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    ) 