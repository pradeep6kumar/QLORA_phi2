@echo off
setlocal enabledelayedexpansion

:: Set up environment
echo Setting up environment...
pip install -r requirements.txt

:: Run inference
echo Running inference...
python inference.py ^
    --model_dir .\output ^
    --sample_prompts

echo Inference completed!

:: Add a pause at the end to keep the window open
pause 