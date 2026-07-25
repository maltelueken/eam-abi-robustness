#!/bin/bash
#SBATCH --job-name=rdm_lowr_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_lowr_gpu_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_lowr_gpu_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

experiment=experiment_2
model=rdm_simple_lower

# See fit_mcmc_gpu_simple.sh for the vmap-over-datasets rationale. Keep param_1/
# param_2 in sync with conf/experiment/experiment_2.yaml's meta_param_1/meta_param_2.
param_1=(0.5 1.0 1.5 2.0 2.5)
param_2=(0.05 0.1 0.15 0.2 0.25)

for p1 in ${param_1[@]}
do
    for p2 in ${param_2[@]}
    do
        python $experiment/fit_mcmc_gpu.py experiment=$experiment model=$model +slurm_p1=$p1 +slurm_p2=$p2
    done
done
