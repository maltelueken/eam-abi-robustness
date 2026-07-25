#!/bin/bash
#SBATCH --job-name=rdm_lowr_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_lowr_gpu_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_lowr_gpu_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

experiment=experiment_4
model=rdm_simple_lower

# See fit_mcmc_gpu_simple.sh for the vmap-over-subjects rationale.
param_1=("FF1.txt" "FV1.txt" "FN1.txt")

for f in ${param_1[@]}
do
    python $experiment/fit_mcmc_gpu.py experiment=$experiment model=$model +slurm_filename=$f
done
