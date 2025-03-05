# Fine-tuning Phi-2 on Windows with QLoRA

This guide provides specific instructions for fine-tuning the Microsoft Phi-2 model on Windows using QLoRA (Quantized Low-Rank Adaptation) on the OpenAssistant dataset.

## Windows-Specific Setup

### Prerequisites

1. **Python Environment**: We recommend using Python 3.10 for Windows compatibility
2. **CUDA Toolkit**: Install CUDA 11.8 or 12.1 (compatible with PyTorch)
3. **Visual C++ Build Tools**: Required for some Python packages

### Installation Steps

1. Install Visual C++ Build Tools:
   - Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++" during installation

2. Install CUDA Toolkit:
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation instructions

3. Create a virtual environment:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install PyTorch with CUDA support:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Running the Pipeline on Windows

We've provided several Windows batch files to make it easier to run the pipeline:

### Full Pipeline

To run the entire pipeline in one go:

```cmd
run_pipeline.bat
```

This script will:
1. Install required dependencies
2. Prepare the OpenAssistant dataset
3. Fine-tune Phi-2 with QLoRA
4. Evaluate the model
5. Merge the LoRA weights with the base model
6. Run inference with sample prompts

### Individual Steps

For better control and troubleshooting, we've also provided batch files for each step:

1. **Data Preparation Only**:
   ```cmd
   prepare_data.bat
   ```

2. **Training Only**:
   ```cmd
   train_model.bat
   ```

3. **Evaluation Only**:
   ```cmd
   evaluate_model.bat
   ```

4. **Inference Only**:
   ```cmd
   run_inference.bat
   ```

## Troubleshooting Windows-Specific Issues

### Memory Issues

If you encounter CUDA out of memory errors:

1. Reduce batch size:
   ```cmd
   python train.py --batch_size 2 --gradient_accumulation_steps 8 --windows_mode
   ```

2. Enable CPU offloading (slower but uses less GPU memory):
   ```cmd
   python train.py --windows_mode --cpu_offload
   ```

### Path Length Issues

Windows has a 260-character path length limit. If you encounter path length errors:

1. Enable long paths in Windows:
   - Run PowerShell as Administrator
   - Execute: `Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1`
   - Restart your computer

2. Use shorter paths for cache and output directories:
   ```cmd
   python train.py --output_dir C:\ML\output
   ```

### Tokenizer Issues

If you encounter tokenizer errors:

1. Try using the non-fast tokenizer:
   ```cmd
   python train.py --windows_mode
   ```

2. Clear the Hugging Face cache:
   ```cmd
   rmdir /s /q %USERPROFILE%\.cache\huggingface
   ```

### Dataset Preparation Issues

If you encounter issues with the dataset preparation:

1. Skip quality filtering:
   ```cmd
   python data_preparation.py --skip_quality_filter
   ```

2. Limit the number of samples:
   ```cmd
   python data_preparation.py --max_samples 1000 --skip_quality_filter
   ```

## Performance Optimization for Windows

- **Disable Windows Defender Real-time Protection** temporarily during training
- **Close other GPU-intensive applications**
- **Use a dedicated GPU** if possible
- **Ensure proper cooling** as training can be intensive

## Additional Windows Tips

- Use Windows Terminal instead of CMD for better output formatting
- Consider using WSL2 (Windows Subsystem for Linux) for a more Linux-like environment
- If using VSCode, install the Python extension for better debugging 

## Detailed Batch File Documentation

This section provides detailed information about each batch file in the project and how to use them effectively.

### prepare_data.bat

This batch file prepares the OpenAssistant dataset for fine-tuning.

**What it does:**
1. Installs required dependencies from requirements.txt
2. Creates a data directory if it doesn't exist
3. Runs data_preparation.py with specific parameters:
   - Sets output directory to ./data
   - Sets maximum sequence length to 2048 tokens
   - Filters for English language content
   - Sets quality threshold to 0.6
   - Sets minimum response length to 20 tokens
   - Sets maximum response length to 1024 tokens
   - Saves data in JSONL format
   - Skips quality filtering (for faster processing)
   - Truncates sequences that are too long
   - Limits processing time to 60 seconds per sample
   - Limits the dataset to 5000 samples

**When to use it:**
- Use this as your first step in the pipeline
- Run this before training to prepare your dataset
- If you want to customize the dataset, edit the parameters in this file

**Example usage:**
```cmd
prepare_data.bat
```

### train_model.bat

This batch file handles the QLoRA fine-tuning of the Phi-2 model.

**What it does:**
1. Installs required dependencies from requirements.txt
2. Creates an output directory if it doesn't exist
3. Runs train.py with specific parameters:
   - Uses microsoft/phi-2 as the base model
   - Loads data from ./data directory
   - Saves the fine-tuned model to ./output
   - Sets maximum sequence length to 2048 tokens
   - Uses a batch size of 4
   - Accumulates gradients over 4 steps (effective batch size of 16)
   - Sets learning rate to 2e-4
   - Trains for 3 epochs
   - Uses LoRA with rank 16 and alpha 32
   - Quantizes the model to 4-bit precision
   - Uses 3% of steps for warmup
   - Enables gradient checkpointing for memory efficiency
   - Uses 8-bit AdamW optimizer with paging
   - Uses cosine learning rate scheduler
   - Applies LoRA to all compatible modules
   - Enables Windows-specific optimizations

**When to use it:**
- Run this after preparing your data
- If you have limited GPU memory, consider reducing batch_size
- For longer training, increase num_train_epochs

**Example usage:**
```cmd
train_model.bat
```

**Customization options:**
- For faster training but potentially lower quality: reduce num_train_epochs to 1
- For better results but longer training: increase num_train_epochs to 5
- For memory issues: reduce batch_size to 2 and increase gradient_accumulation_steps to 8

### evaluate_model.bat

This batch file evaluates the performance of your fine-tuned model.

**What it does:**
1. Installs required dependencies from requirements.txt
2. Runs evaluate.py with specific parameters:
   - Loads the model from ./output directory
   - Uses test data from ./data directory
   - Evaluates on 100 random samples
   - Enables Windows-specific optimizations

**When to use it:**
- Run this after training to assess model performance
- Use before deploying your model to production
- Run multiple times with different checkpoints to compare performance

**Example usage:**
```cmd
evaluate_model.bat
```

**Output:**
- Evaluation metrics will be saved in output/evaluation_results
- Look for metrics like accuracy, ROUGE scores, and other NLP evaluation metrics

### run_inference.bat

This batch file allows you to test your fine-tuned model with sample prompts.

**What it does:**
1. Installs required dependencies from requirements.txt
2. Runs inference.py with the --sample_prompts flag
3. Loads the model from ./output directory
4. Generates responses for predefined sample prompts

**When to use it:**
- Use after training to test your model with real prompts
- Use for quick qualitative assessment of model capabilities
- Use to demonstrate your model to others

**Example usage:**
```cmd
run_inference.bat
```

**Customization:**
- To test with your own prompts, you can modify the sample_prompts list in inference.py
- For interactive mode, you can modify inference.py to accept user input

### run_pipeline.bat

This batch file runs the entire fine-tuning pipeline from start to finish.

**What it does:**
1. Installs required dependencies from requirements.txt
2. Creates necessary directories (data, output, merged_model)
3. Prepares the dataset (same as prepare_data.bat)
4. Trains the model (same as train_model.bat)
5. Evaluates the model (same as evaluate_model.bat)
6. Merges LoRA weights with the base model
7. Runs inference with sample prompts (same as run_inference.bat)

**When to use it:**
- Use for a complete end-to-end run of the pipeline
- Ideal for first-time users who want to see the entire process
- Good for automated runs or scheduled training jobs

**Example usage:**
```cmd
run_pipeline.bat
```

**Note:**
- This is a time-intensive process that may take several hours depending on your hardware
- Make sure you have sufficient disk space (at least 20GB) before running
- The merged model will be saved in the ./merged_model directory

## Customizing the Batch Files

You can customize any of these batch files by editing them in a text editor. Here are some common customizations:

### Changing the Model

To use a different base model, modify the --model_name parameter in train_model.bat:

```cmd
--model_name microsoft/phi-1_5
```

### Changing Training Parameters

To adjust training parameters, modify the corresponding values in train_model.bat:

```cmd
--batch_size 2 ^
--gradient_accumulation_steps 8 ^
--learning_rate 1e-4 ^
--num_train_epochs 5
```

### Changing Dataset Size

To use more or fewer samples, modify the --max_samples parameter in prepare_data.bat:

```cmd
--max_samples 10000
```

### Adding Custom Evaluation

To add custom evaluation metrics, you'll need to modify evaluate.py and then run evaluate_model.bat as usual.

## Troubleshooting Batch File Issues

### Batch File Exits Immediately

If a batch file closes immediately:
1. Run it from Command Prompt to see error messages
2. Check that Python is in your PATH
3. Verify that all required files exist in the correct locations

### Out of Memory Errors

If you encounter CUDA out of memory errors:
1. Reduce batch_size in train_model.bat
2. Increase gradient_accumulation_steps
3. Add --cpu_offload to the train.py command

### Long Training Times

If training takes too long:
1. Reduce --num_train_epochs
2. Reduce --max_samples in prepare_data.bat
3. Use a more powerful GPU if available

## Advanced Usage

### Resuming Training

To resume training from a checkpoint, add the following to train_model.bat:

```cmd
--resume_from_checkpoint .\output\checkpoint-1000
```

### Exporting for Deployment

After training, you can export your model for deployment by using merge_lora.py:

```cmd
python merge_lora.py ^
    --base_model microsoft/phi-2 ^
    --lora_model .\output ^
    --output_dir .\deployment_model ^
    --push_to_hub
```

This will create a merged model that can be used without the LoRA adapter 