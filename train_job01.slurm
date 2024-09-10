#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384M
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<bastien.gless@ensta-paris.fr>
#SBATCH --mail-type=ALL

# Load necessary modules
module load python
module load cuda
module load cudnn

# Activate the virtual environment
source ENV/bin/activate

# Generate a timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")

# Set output and error file paths using the generated timestamp
output_file="hedgedexv2.4/output/logs/output_${timestamp}.out"
error_file="hedgedexv2.4/output/logs/error_${timestamp}.err"

# Execute the Python script with the timestamp as an argument
python hedgedexv2.4/scripts/train.py --timestamp ${timestamp} --epochs 20 --bs 16 --dssize 1024 --lr 0.001 > ${output_file} 2> ${error_file}
