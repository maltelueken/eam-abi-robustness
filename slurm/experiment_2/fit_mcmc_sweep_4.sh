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

param_1=(4.0)
param_2=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

for p1 in ${param_1[@]}
do
    for p2 in ${param_2[@]}
    do
        for i in {0..99}
        do
            srun --exclusive --ntasks=1 python experiment_2/fit_mcmc_slurm.py experiment=experiment_2 +slurm_p1=$p1 +slurm_p2=$p2 +slurm_idx=$i &
        done
        wait
        python experiment_2/collect_mcmc.py experiment=experiment_2 +slurm_p1=$p1 +slurm_p2=$p2
        rm -r outputs/experiment_2/rdm_simple/mcmc_samples/${p1}_${p2}
    done
done
