#!/bin/bash

##########################################################################
# Grid search for hedgedexv3.0, with HRNet                               #
# NOT FINISHED                                                           #
##########################################################################

# Define the path to data
SRC_DATA_DIR="/home/bgless/projects/def-mlecuyer/bgless/hedgedex/data/dataset05/*"
DST_DATA_DIR="$SLURM_TMPDIR/"

# Define hyperparameter grid
learning_rates=("0.0002" "0.0010" "0.0050")
batch_sizes=("8" "16")
epochs=("100")
bands=("5" "9")
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
                    sbatch train_job04.slurm $lr $bs $e $b $l
                done
            done
        done
    done
done
