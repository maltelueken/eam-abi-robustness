#!/bin/bash
#SBATCH --job-name=check_robustness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=12:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/check_robustness_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/check_robustness_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_2/check_robustness.py \
    experiment=experiment_2 \
    model=rdm_simple_meta
