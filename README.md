# HedgeDEX

## Overview

This project revolves around the HedgeDEX, hedgerow detector.
It is based on a segmentation network that takes as input multi-band 256*256 tif files
and outputs same size probability maps showing which pixels are likely
to contain hedgerows.
Versions 2.4 to 4.0 are trained on Sentinel-2 data and Copernicus Woody Vegetation Mask (WVM) labels.

## Directory Structure
- `data/`: Contains TIF files for training and validation.
- `hedgedexvX.XX/scripts/`: Python scripts for model architecture and training.
- `hedgedexvX.XX/output/`: Checkpoints and logs from training.

## Versions
hedgedexv2.4 : UNet with dataset04  
hedgedexv2.5 : UNet with dataset05  
hedgedexv3.0 : HRNet with dataset05  
hedgedexv4.0 : HRNet+OCR with dataset05  
hedgedexv5.0 : UNet_time with dataset06


## How to Run

Locally, in the root directory of the repo :
`python hedgedexv5.0/scripts/train.py --timestamp TESTTEST --epochs 10 --bs 8 --dssize 10000 --lr 0.0001 --loss jaccard_loss --bands 5 --data_dir $SLURM_TMPDIR/dataset06/ --job_id 0000 --pretrained 0`
On ComputeCanada, in the root directory of the repo :
Use the `train_jobXX.slurm` files and then :
`sbatch train_job07.slurm 0.0001 8 10 5 jaccard_loss 0`

## Contact
For questions or issues, please contact Bastien at bastien.gless@ensta-paris.fr
