#!/bin/bash
#SBATCH --job-name=rdm_simp_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_simp_gpu_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_simp_gpu_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

experiment=experiment_4
model=rdm_simple

# One GPU job vmaps chains *and* every subject in a file in a single call
# (fit_mcmc_gpu.py, rdm_jax.fit_mcmc_gpu_batch), replacing the old per-subject
# SLURM-array + pmap-on-fake-CPU-devices approach.
param_1=("FF1.txt" "FV1.txt" "FN1.txt")

for f in ${param_1[@]}
do
    python $experiment/fit_mcmc_gpu.py experiment=$experiment model=$model +slurm_filename=$f
done
