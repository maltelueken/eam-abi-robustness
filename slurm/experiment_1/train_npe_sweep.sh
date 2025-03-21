#!/bin/bash
#SBATCH --job-name=train_npe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=24:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/train_npe_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/train_npe_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_1/train_npe.py sweeper=optuna --multirun
