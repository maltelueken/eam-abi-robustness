#!/bin/bash
#SBATCH --job-name=fit_mcmc
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat_rome
#SBATCH --time=12:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

for t in {50..1200..50}
do
    for i in {0..99}
    do
        srun --exclusive --ntasks=1 python experiment_1/fit_mcmc_slurm.py +slurm_num_obs=$t +slurm_idx=$i &
    done
    wait
    python experiment_1/collect_mcmc.py +slurm_num_obs=$t
    rm -r outputs/experiment_1/rdm_simple/mcmc_samples/$t
done
