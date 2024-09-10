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

cp -r /home/bgless/projects/def-mlecuyer/bgless/hedgedex/data/dataset05/* $SLURM_TMPDIR/
cd $SLURM_TMPDIR
cd /home/bgless/projects/def-mlecuyer/bgless/hedgedex/

# Unzip all zip files into the target directory
for zip_file in $SLURM_TMPDIR/dataset05-*.zip; do
    if [ -f "$zip_file" ]; then
        unzip "$zip_file" -d $SLURM_TMPDIR/
    else
        echo "No zip files found."
    fi
done

cd /home/bgless/projects/def-mlecuyer/bgless/hedgedex/

# Generate a timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")

# Set output and error file paths using the generated timestamp
output_file="hedgedexv2.5/output/logs/output_${timestamp}.out"

# Execute the Python script with the timestamp as an argument
python hedgedexv2.5/scripts/train.py --timestamp ${timestamp} --epochs 40 --bs 8 --dssize 10000 --lr 0.0001 --data_dir $SLURM_TMPDIR/dataset05/ &> ${output_file}
