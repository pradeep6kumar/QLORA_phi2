@echo off
setlocal enabledelayedexpansion

:: Set up environment
echo Setting up environment...
pip install -r requirements.txt

:: Evaluate the model
echo Evaluating the model...
python evaluate.py ^
    --model_dir .\output ^
    --data_dir .\data ^
    --num_samples 100 ^
    --windows_mode

echo Evaluation completed!
echo Check the output/evaluation_results directory for the evaluation results.

:: Add a pause at the end to keep the window open
pause 