#!/bin/bash
#SBATCH --job-name=rdm_mlwr
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_mlwr_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/fit_mcmc_rdm_mlwr_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

experiment=experiment_4

model=rdm_simple_meta_lower

param_1=("FF1.txt" "FV1.txt" "FN1.txt")

for f in ${param_1[@]}
do
    for i in {0..124}
    do
        srun --exclusive --ntasks=1 python $experiment/fit_mcmc_slurm.py experiment=$experiment model=$model +slurm_filename=$f +slurm_idx=$i &
    done
    wait
    python $experiment/collect_mcmc.py experiment=$experiment model=$model +slurm_filename=$f
done
