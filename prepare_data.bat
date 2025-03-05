@echo off
setlocal enabledelayedexpansion

:: Set up environment
echo Setting up environment...
pip install -r requirements.txt

:: Create directories
if not exist data mkdir data

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

echo Dataset preparation completed!
echo Check the data directory for the generated files.

:: Add a pause at the end to keep the window open
pause 