import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model and tokenizer
base_model_name = "microsoft/phi-2"
adapter_path = "./output"  # Path to your trained LoRA adapter

def load_model():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

def generate_response(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response using the fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=4),
        gr.Slider(minimum=64, maximum=1024, value=512, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P"),
    ],
    outputs=gr.Textbox(label="Generated Response", lines=8),
    title="Phi-2 QLoRA Fine-tuned Assistant",
    description="Enter a prompt to generate a response using the fine-tuned Phi-2 model.",
    examples=[
        ["Write a Python function to calculate the factorial of a number"],
        ["Explain the concept of machine learning in simple terms"],
        ["Write a professional email requesting a meeting with a client"],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True) 