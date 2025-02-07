#!/bin/bash
#SBATCH --job-name=fit_mcmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=fat_rome
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/fit_mcmc_%j.out
#SBATCH --error=slurm/logs/fit_mcmc_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_1/fit_mcmc.py
