@echo off
setlocal enabledelayedexpansion

:: Set up environment
echo Setting up environment...
pip install -r requirements.txt

:: Create directories
if not exist data mkdir data
if not exist output mkdir output
if not exist merged_model mkdir merged_model

:: Prepare the dataset
echo Preparing the dataset...
python data_preparation.py ^
    --output_dir .\data ^
    --max_length 2048 ^
    --lang en ^
    --quality_threshold 0.6 ^
    --min_response_length 20 ^
    --max_response_length 1024 ^
    --save_jsonl ^
    --skip_quality_filter ^
    --truncate_long_sequences ^
    --max_processing_time 60 ^
    --max_samples 5000

:: Train the model
echo Training the model...
python train.py ^
    --model_name microsoft/phi-2 ^
    --data_dir .\data ^
    --output_dir .\output ^
    --max_length 2048 ^
    --batch_size 4 ^
    --gradient_accumulation_steps 4 ^
    --learning_rate 2e-4 ^
    --num_train_epochs 3 ^
    --lora_r 16 ^
    --lora_alpha 32 ^
    --bits 4 ^
    --warmup_ratio 0.03 ^
    --gradient_checkpointing ^
    --optim paged_adamw_8bit ^
    --lr_scheduler_type cosine ^
    --target_modules all ^
    --windows_mode

:: Evaluate the model
echo Evaluating the model...
python evaluate.py ^
    --model_dir .\output ^
    --data_dir .\data ^
    --num_samples 100 ^
    --windows_mode

:: Merge LoRA weights with base model
echo Merging LoRA weights with base model...
python merge_lora.py ^
    --base_model microsoft/phi-2 ^
    --lora_model .\output ^
    --output_dir .\merged_model

:: Run inference with sample prompts
echo Running inference with sample prompts...
python inference.py ^
    --model_dir .\output ^
    --sample_prompts

echo Pipeline completed successfully!

:: Add a pause at the end to keep the window open
pause 