#!/bin/bash
#SBATCH --job-name=fit_mcmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

param_1=("FF1.txt" "FV1.txt" "FN1.txt")

for f in ${param_1[@]}
do
    for i in {0..124}
    do
        srun --exclusive --ntasks=1 python experiment_4/fit_mcmc_meta_slurm.py experiment=experiment_4 model=rdm_simple_meta_lower +slurm_filename=$f +slurm_idx=$i &
    done
    wait
    python experiment_4/collect_mcmc.py experiment=experiment_4 +slurm_filename=$f
done
