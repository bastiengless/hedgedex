#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384M
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<bastien.gless@ensta-paris.fr>
#SBATCH --mail-type=ALL

##################################################
# Job for Unet-time training within hedgedexv5.1 #
##################################################


# Load necessary modules
module load python
module load cuda
module load cudnn

# Activate the virtual environment
source ENV/bin/activate

cp -r /home/bgless/projects/def-mlecuyer/bgless/hedgedex/data/dataset06/* $SLURM_TMPDIR/
cd $SLURM_TMPDIR
cd /home/bgless/projects/def-mlecuyer/bgless/hedgedex/

# Unzip all zip files into the target directory
for zip_file in $SLURM_TMPDIR/dataset06-*.zip; do
    if [ -f "$zip_file" ]; then
        unzip "$zip_file" -d $SLURM_TMPDIR/
    else
        echo "No zip files found."
    fi
done

file_count=$(ls -1 $SLURM_TMPDIR/dataset06/ | wc -l)
echo "Number of files in dataset06 directory: $file_count"


cd /home/bgless/projects/def-mlecuyer/bgless/hedgedex/


# Generate a timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")
lr=$1
bs=$2
e=$3
b=$4
l=$5
p=$6


# Set output and error file paths using the generated timestamp
output_file="hedgedexv5.1/output/logs/log_${SLURM_JOB_ID}_bs${bs}_lr${lr}_epochs${e}_bands${b}_${l}_pretrained${p}.out"

# Execute the Python script with the timestamp as an argument
python hedgedexv5.1/scripts/train.py --timestamp ${timestamp} --epochs $e --bs $bs --dssize 10000 --lr $lr --loss $l --bands $b --data_dir $SLURM_TMPDIR/dataset06/ --job_id $SLURM_JOB_ID --pretrained $p &> ${output_file}
echo "Job is running on node: $SLURM_JOB_NODELIST" >> ${output_file}

