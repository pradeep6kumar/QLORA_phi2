---
title: Phi-2 QLoRA Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# Phi-2 QLoRA Fine-tuned Assistant (CPU-Optimized)

This is a lightweight CPU-optimized version of Microsoft's Phi-2 model fine-tuned using QLoRA (Quantized Low-Rank Adaptation) technique. The model has been optimized to run efficiently on CPU environments while still providing helpful responses for coding, explanations, and writing tasks.

## Model Description

- **Base Model**: Microsoft Phi-2
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Optimization**: CPU-optimized with reduced parameters
- **Primary Use Cases**: Code generation, technical explanations, and professional writing

## Usage Tips

### For Code Generation (Temperature: 0.3-0.5)
```python
# Example prompt:
"Write a Python function to calculate factorial"
```

### For Technical Explanations (Temperature: 0.4-0.5)
```text
# Example prompt:
"Explain machine learning simply"
```

### For Professional Writing (Temperature: 0.4-0.6)
```text
# Example prompt:
"Write a short email to schedule a meeting"
```

## Parameters Guide (CPU-Optimized)

- **Maximum Length**: 64-256 (default: 192)
  - Keep this low (128-192) for faster responses on CPU
  - Higher values will significantly slow down generation

- **Temperature**: 0.1-0.7 (default: 0.4)
  - 0.3-0.4: Best for code generation
  - 0.4-0.5: Best for explanations
  - 0.5-0.6: Best for creative writing

- **Top P**: 0.5-0.9 (default: 0.8)
  - Controls diversity of word choices
  - Lower values = more focused responses

## Performance Notes

This is a CPU-optimized version with the following considerations:
- Responses will be shorter than the GPU version
- Generation takes longer on CPU (be patient)
- Memory usage is optimized for CPU environments
- Best for shorter, focused prompts

## Model Links

- **Model Card**: [pradeep6kumar2024/phi2-qlora-assistant](https://huggingface.co/pradeep6kumar2024/phi2-qlora-assistant)
- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

## License

This demo is released under the MIT License.

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and adapter (CPU optimized)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(
    base_model, 
    "pradeep6kumar2024/phi2-qlora-assistant",
    torch_dtype=torch.float32,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Generate text (CPU optimized)
prompt = "Write a Python function to calculate factorial"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=256,
    temperature=0.4,
    top_p=0.8,
    num_beams=1  # Greedy decoding for CPU
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Example Outputs

1. **Coding Task**:
   ```python
   def factorial(n):
       if n == 0 or n == 1:
           return 1
       return n * factorial(n-1)
   ```

2. **Technical Explanation**:
   "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. It works by analyzing patterns in data and making predictions based on those patterns."

3. **Professional Writing**:
   "Subject: Team Meeting Request
   
   Hi Team,
   
   I'd like to schedule a meeting next week to discuss our current project. Please let me know your availability.
   
   Thanks,
   [Your Name]"

## Limitations

- CPU version generates shorter responses than GPU version
- Generation is slower on CPU environments
- Works best with clear, concise prompts
- Memory constraints may limit very complex generations

## Acknowledgments

- Microsoft for the Phi-2 base model
- Hugging Face for the transformers library and hosting
- The QLoRA paper authors for the fine-tuning technique 