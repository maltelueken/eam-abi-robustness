#!/bin/bash
#SBATCH --job-name=train_npe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=01:00:00

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_1/train_npe.py epochs=100
