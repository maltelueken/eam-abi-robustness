#!/bin/bash
#SBATCH --job-name=fit_mcmc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_gpu_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_gpu_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

# One GPU job vmaps chains *and* all test_batch_size datasets for a given num_obs
# value in a single call (experiment_1/fit_mcmc_gpu.py, rdm_jax.fit_mcmc_gpu_batch),
# replacing the old per-dataset SLURM-array + pmap-on-fake-CPU-devices approach
# (fit_mcmc_slurm.py + fit_mcmc_sweep.sh). Sequential loop over num_obs, not an
# array job -- each iteration reuses the same GPU.
for t in {50..1200..50}
do
    python experiment_1/fit_mcmc_gpu.py +slurm_num_obs=$t
done
