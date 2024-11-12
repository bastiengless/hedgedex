#!/bin/bash

##########################################################################
# Grid search for hedgedexv4.1, with HRNet+OCR + Random Crop             #
# NOT FINISHED                                                           #
##########################################################################

# Define the path to data
SRC_DATA_DIR="/home/bgless/projects/def-mlecuyer/bgless/hedgedex/data/dataset05/*"
DST_DATA_DIR="$SLURM_TMPDIR/"

# Define hyperparameter grid
learning_rates=("0.0002" "0.002" "0.02")
batch_sizes=("4" "8" "16")
epochs=("200")
bands=("3" "5" "9")
losses=("jaccard_loss")


# Loop through all combinations of hyperparameters
for lr in "${learning_rates[@]}"
do
    for bs in "${batch_sizes[@]}"
    do
        for e in "${epochs[@]}"
        do
            for b in "${bands[@]}"
            do
                for l in "${losses[@]}"
                do
                    # Submit a job for each combination
                    sbatch train_job08.slurm $lr $bs $e $b $l 0
                done
            done
        done
    done
done
