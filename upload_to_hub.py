from huggingface_hub import HfApi
import os
import shutil

# Your Hugging Face username and model name
USERNAME = "pradeep6kumar2024"  # Your actual username
MODEL_NAME = "phi2-qlora-assistant"  # Replace with your desired model name
REPO_ID = f"{USERNAME}/{MODEL_NAME}"

# Paths
LOCAL_MODEL_PATH = "./output"
CHECKPOINT_PATH = os.path.join(LOCAL_MODEL_PATH, "checkpoint-360")
TEMP_DIR = "./temp_upload"

# Essential LoRA files that should be present (now including safetensors)
ESSENTIAL_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",  # Changed from .bin to .safetensors
]

def prepare_files():
    """Prepare files for upload"""
    print("Preparing files for upload...")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # First try to copy from main output directory
    print("Copying files from output directory...")
    files_copied = []
    for file in os.listdir(LOCAL_MODEL_PATH):
        # Skip checkpoint directory
        if file == "checkpoint-360":
            continue
        # Copy adapter files, configs, and tokenizer files
        if any(file.startswith(prefix) for prefix in ["adapter_", "config", "special_tokens_map", "tokenizer", "vocab"]):
            src_file = os.path.join(LOCAL_MODEL_PATH, file)
            dst_file = os.path.join(TEMP_DIR, file)
            shutil.copy2(src_file, dst_file)
            files_copied.append(file)
            print(f"Copied {file}")
    
    # Verify essential files
    missing_files = []
    for file in ESSENTIAL_FILES:
        if file not in files_copied:
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing essential files in output directory: {', '.join(missing_files)}"
        )
    
    # Copy additional files from root directory
    files_to_copy = ['README.md', 'requirements.txt', 'gradio_app.py']
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, TEMP_DIR)
            print(f"Copied {file}")

def upload_to_hub():
    """Upload the LoRA adapter to Hugging Face Hub"""
    api = HfApi()
    
    print(f"Uploading LoRA adapter files to {REPO_ID}...")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print("\nError creating repository. Please check:")
        print("1. You have logged in using 'huggingface-cli login'")
        print("2. Your username is correct")
        print("3. You have write permissions for this namespace")
        print(f"\nError details: {str(e)}")
        raise
    
    # Upload all files from temp directory
    api.upload_folder(
        folder_path=TEMP_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    print("Upload completed!")

def cleanup():
    """Clean up temporary files"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    try:
        print("Starting LoRA adapter upload process...")
        prepare_files()
        upload_to_hub()
    finally:
        cleanup()
    
    print(f"\nYour LoRA adapter is now available at: https://huggingface.co/{REPO_ID}")
    print("\nTo use your LoRA adapter, others will need to:")
    print("1. Load the base Phi-2 model")
    print("2. Load your LoRA adapter using PeftModel.from_pretrained()")
    print("\nTo create a Gradio Space:")
    print("1. Go to huggingface.co/spaces")
    print("2. Click 'Create new Space'")
    print("3. Choose 'Gradio' as the SDK")
    print("4. Link it to your model repository")
    print("5. Copy the contents of gradio_app.py to the Space") 