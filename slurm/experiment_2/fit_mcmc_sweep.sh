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

steps=10

step_p1=$(echo "scale=6; (4 - 0.5) / ($steps - 1)" | bc)
step_p2=$(echo "scale=6; (0.5 - 0.05) / ($steps - 1)" | bc)

for i in $(seq 0 $(($steps - 1)))
do
    p1=$(echo "scale=2; 0.5 + ($i * $step_p1)" | bc)
    for j in $(seq 0 $(($steps - 1)))
    do
        p2=$(echo "scale=2; 0.05 + ($j * $step_p2)" | bc)
        for i in {0..99}
        do
            srun --exclusive --ntasks=1 python experiment_2/fit_mcmc_slurm.py experiment=experiment_2 +slurm_p1=$p1 +slurm_p2=$p2 +slurm_idx=$i &
        done
        wait
        python experiment_2/collect_mcmc.py experiment=experiment_2 +slurm_p1=$p1 +slurm_p2=$p2
        rm -r outputs/experiment_2/rdm_simple/mcmc_samples/${p1}_${p2}
    done
done
