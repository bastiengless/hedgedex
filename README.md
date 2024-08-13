# HedgeDEX

## Overview

This project revolves around the HedgeDEX, hedgerow detector.
It is based on a UNet that takes as input multi-band 256*256 tif files
and outputs same size probability maps showing which pixels are likely
to contain hedgerows.

## Directory Structure
- `data/`: Contains TIF files for training and validation.
- `scripts/`: Python scripts for model architecture and training.
- `results/`: Checkpoints and logs from training.
- `job_scripts/`: SLURM job scripts for running the training on Compute Canada.

## How to Run
1. Load necessary modules:
    ```bash
    module load python/3.9
    module load cuda/11.4
    module load pytorch/1.9
    ```
2. Submit the training job:
    ```bash
    sbatch job_scripts/train_unet.sh
    ```

## Contact
For questions or issues, please contact Bastien at bastien.gless@ensta-paris.fr
