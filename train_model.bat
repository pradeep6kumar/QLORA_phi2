@echo off
setlocal enabledelayedexpansion

:: Set up environment
echo Setting up environment...
pip install -r requirements.txt

:: Create output directory
if not exist output mkdir output

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

echo Training completed!
echo Check the output directory for the trained model.

:: Add a pause at the end to keep the window open
pause 