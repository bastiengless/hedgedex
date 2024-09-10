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

cp -r /home/bgless/scratch/data/dataset04.zip $SLURM_TMPDIR/
cd $SLURM_TMPDIR
unzip dataset04.zip -x "__MACOSX/*" -d $SLURM_TMPDIR/
cd /home/bgless/projects/def-mlecuyer/bgless/hedgedex/

# Generate a timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")

# Set output and error file paths using the generated timestamp
output_file="hedgedexv2.4/output/logs/output_${timestamp}.out"

# Execute the Python script with the timestamp as an argument
python hedgedexv2.4/scripts/train.py --timestamp ${timestamp} --epochs 20 --bs 8 --dssize 1024 --lr 0.01 --data_dir $SLURM_TMPDIR/dataset04/ &> ${output_file}
