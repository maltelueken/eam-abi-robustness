#!/bin/bash
#SBATCH --job-name=predict_npe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=01:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/predict_npe_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/predict_npe_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_2/predict_npe.py \
    experiment=experiment_2 \
    model=rdm_simple_meta_lower
