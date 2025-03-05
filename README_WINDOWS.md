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