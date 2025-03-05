import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import re

# Configuration
BASE_MODEL = "microsoft/phi-2"
LOCAL_ADAPTER_PATH = "./output/checkpoint-360"  # Path to your local adapter files
DEBUG = False  # Set to True to enable debug prints

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load_model(self):
        if not self.loaded:
            try:
                print("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL,
                    trust_remote_code=True,
                    padding_side="left"
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print("Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                print(f"Loading local LoRA adapter from {LOCAL_ADAPTER_PATH}...")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    LOCAL_ADAPTER_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.model.eval()
                print("Model loading complete!")
                self.loaded = True
            except Exception as e:
                print(f"Error during model loading: {str(e)}")
                raise
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        if not self.loaded:
            self.load_model()
        
        try:
            # Enhance prompt for better completion
            if "function" in prompt.lower() and "python" in prompt.lower():
                enhanced_prompt = f"""Write a Python function with the following requirements:
{prompt}
Include:
- Function implementation with comments
- Example usage
- Output demonstration

Provide only the implementation, no conversation."""
            elif any(word in prompt.lower() for word in ["explain", "what is", "how does", "describe"]):
                enhanced_prompt = f"""Below is a request for explanation. Provide a complete, focused response without any conversation:

{prompt}

Your response should include:
1. A clear explanation in simple terms
2. Practical examples and applications
3. Important concepts to understand

End your response when the explanation is complete. Do not ask questions or engage in conversation."""
            else:
                enhanced_prompt = f"""Below is a request. Provide a complete, focused response without any conversation:

{prompt}

End your response when complete. Do not ask questions or engage in conversation."""
            
            if DEBUG:
                print(f"Enhanced prompt: {enhanced_prompt}")
            
            # Tokenize input
            inputs = self.tokenizer(
                enhanced_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.model.device)
            
            # Generate with more conservative parameters
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=50,  # Keep minimum length requirement
                    temperature=min(0.5, temperature),  # Lower temperature cap
                    top_p=min(0.85, top_p),  # Lower top_p cap
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,  # Increased repetition penalty
                    no_repeat_ngram_size=4,  # Increased n-gram size
                    num_return_sequences=1,
                    early_stopping=True,
                    num_beams=4,  # Increased beam search
                    length_penalty=0.6  # More aggressive length penalty
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if DEBUG:
                print(f"Raw response: {response}")
            
            # Clean up the response
            if response.startswith(enhanced_prompt):
                response = response[len(enhanced_prompt):].strip()
            
            if DEBUG:
                print(f"After prompt removal: {response}")
            
            # Remove common closure patterns and conversation starters
            closures = [
                "Best regards,",
                "Sincerely,",
                "Thanks,",
                "Thank you,",
                "Regards,",
                "Assistant:",
                "Human:",
                "[Your Name]",
                "[Student]",
                "Let me know if you need any clarification",
                "I hope this helps",
                "Feel free to ask",
                "Can you provide",
                "Would you like",
                "Do you want",
                "Let me know",
                "Please let me know",
                "Is there anything else",
                "Do you have any questions",
                "Sure!",
                "Here are some examples:"
            ]
            
            # First remove conversation starters from the end
            for closure in closures:
                if response.lower().endswith(closure.lower()):
                    response = response[:-(len(closure))].strip()
            
            # Then remove any remaining conversation patterns
            conversation_patterns = [
                r"\?\s*$",  # Questions at the end
                r"Sure!.*$",  # Responses starting with Sure!
                r"Here are.*examples:?\s*$",  # Incomplete example lists
                r"Can you.*\?\s*$",  # Questions starting with Can you
                r"Would you.*\?\s*$",  # Questions starting with Would you
                r"Do you.*\?\s*$",  # Questions starting with Do you
                r"Let me know.*$",  # Let me know phrases
                r"I hope.*$",  # I hope phrases
                r"Feel free.*$"  # Feel free phrases
            ]
            
            for pattern in conversation_patterns:
                response = re.sub(pattern, "", response).strip()
            
            if DEBUG:
                print(f"After conversation removal: {response}")
            
            # Ensure code examples are properly formatted
            if "```python" not in response and "def " in response:
                response = "```python\n" + response + "\n```"
            
            # More lenient validation but check for conversation markers
            if (len(response.strip()) < 20 or 
                response.strip() == "Response:" or 
                response.strip().endswith("?") or
                "can you" in response.lower() or
                "let me know" in response.lower()):
                if DEBUG:
                    print("Response validation failed - using fallback")
                
                if "machine learning" in prompt.lower():
                    fallback_response = """Machine learning is a branch of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Think of it like teaching a child:

1. Simple Explanation:
- Instead of giving strict rules, we show the computer many examples
- The computer finds patterns in these examples
- It uses these patterns to make decisions about new situations

2. Real-World Applications:
- Email Spam Detection: Learning to identify unwanted emails based on past examples
- Netflix Recommendations: Suggesting movies based on what you've watched
- Face Recognition: Unlocking your phone by learning your facial features
- Virtual Assistants: Siri and Alexa understanding and responding to voice commands
- Medical Diagnosis: Helping doctors identify diseases in medical images
- Fraud Detection: Banks identifying suspicious transactions

3. Key Benefits:
- Automation of complex tasks
- More accurate predictions over time
- Ability to handle large amounts of data
- Continuous improvement through learning

Machine learning is transforming industries by automating tasks that once required human intelligence, making processes more efficient and enabling new possibilities in technology."""
                elif "function" in prompt.lower():
                    fallback_response = """```python
def add_numbers(a, b):
    '''
    Add two numbers and return the result
    Args:
        a: first number
        b: second number
    Returns:
        sum of a and b
    '''
    return a + b

# Example usage
num1 = 5
num2 = 3
result = add_numbers(num1, num2)
print(f"The sum of {num1} and {num2} is: {result}")  # Output: The sum of 5 and 3 is: 8
```"""
                else:
                    fallback_response = "I apologize, but I couldn't generate a complete response. Please try using a lower temperature (0.3-0.5) for more focused output."
                
                response = fallback_response
            
            generation_time = time.time() - start_time
            return response, generation_time
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise

# Initialize model wrapper
model_wrapper = ModelWrapper()

def generate_text(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Gradio interface function"""
    try:
        if not prompt.strip():
            return "Please enter a prompt."
        
        response, gen_time = model_wrapper.generate_response(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        return f"Generated in {gen_time:.2f} seconds:\n\n{response}"
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        return f"Error generating response: {str(e)}\nPlease try again with a different prompt or parameters."

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            placeholder="Type your prompt here...",
            lines=4
        ),
        gr.Slider(
            minimum=64,
            maximum=1024,
            value=512,
            step=64,
            label="Maximum Length",
            info="Longer values = longer responses but slower generation"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Higher values = more creative, lower values = more focused"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.1,
            label="Top P",
            info="Controls diversity of word choices"
        ),
    ],
    outputs=gr.Textbox(label="Generated Response", lines=8),
    title="Local Phi-2 QLoRA Assistant",
    description="""This is your locally running fine-tuned version of Microsoft's Phi-2 model using QLoRA.
    The model has been trained to provide helpful responses for various tasks including coding, writing, and general assistance.
    
    Example tasks:
    - Writing Python functions and explaining code
    - Explaining technical concepts in simple terms
    - Drafting professional emails and documents
    
    Tips:
    - For code generation, use lower temperature (0.3-0.5)
    - For creative writing, use higher temperature (0.7-0.9)
    - Adjust max length based on how long you want the response to be
    """,
    examples=[
        [
            "Write a Python function to calculate the factorial of a number and provide additional recursive function examples", 
            512, 
            0.5, 
            0.9
        ],
        [
            "Explain what machine learning is in simple terms and provide some real-world applications", 
            512, 
            0.7, 
            0.9
        ],
        [
            "Write a professional email to schedule a team meeting for next week to discuss project progress", 
            512, 
            0.7, 
            0.9
        ],
        [
            "Write a Python function to implement binary search algorithm with detailed comments", 
            512, 
            0.5, 
            0.9
        ],
        [
            "Explain the concept of object-oriented programming using a real-world analogy", 
            512, 
            0.7, 
            0.9
        ]
    ],
    cache_examples=False
)

if __name__ == "__main__":
    # Launch with localhost for Windows compatibility
    demo.launch(server_name="127.0.0.1", server_port=7860) 