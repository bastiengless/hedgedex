#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generate a timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Define the output and error log file names with timestamp
#SBATCH --output=logs/training_output_${timestamp}.log
#SBATCH --error=logs/training_error_${timestamp}.log

# Load necessary modules
module load python
module load cuda/11.4
module load cudnn/8.2.4
module load pytorch/1.10

# Activate the virtual environment
source hedgedexv2.4/bin/activate

# Execute the Python script with the timestamp as an argument
python hedgedexv2.4/scripts/train.py --timestamp ${timestamp}
