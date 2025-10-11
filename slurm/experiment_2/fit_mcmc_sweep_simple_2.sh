#!/bin/bash
#SBATCH --job-name=rdm_simp
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat_rome
#SBATCH --time=12:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_simp_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_simp_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

param_1=(1.25 1.5 1.75)
param_2=(0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25)

experiment=experiment_2
model=rdm_simple

for p1 in ${param_1[@]}
do
    for p2 in ${param_2[@]}
    do
        for i in {0..99}
        do
            srun --exclusive --ntasks=1 python $experiment/fit_mcmc_slurm.py experiment=$experiment model=$model +slurm_p1=$p1 +slurm_p2=$p2 +slurm_idx=$i &
        done
        wait
        python $experiment/collect_mcmc.py experiment=$experiment model=$model +slurm_p1=$p1 +slurm_p2=$p2
        rm -r outputs/$experiment/$model/mcmc_samples/${p1}_${p2}
    done
done
