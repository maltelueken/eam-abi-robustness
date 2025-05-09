#!/bin/bash
#SBATCH --job-name=check_summary_mmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=6:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/check_summary_mmd_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/check_summary_mmd_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_4/check_summary_mmd_meta.py experiment=experiment_4 model=rdm_simple_meta
