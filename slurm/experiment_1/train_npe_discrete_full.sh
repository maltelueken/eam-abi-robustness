#!/bin/bash
#SBATCH --job-name=npe_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=08:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/train_npe_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/train_npe_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_1/train_npe.py \
    experiment=experiment_1 \
    model=rdm_simple_discrete_full \
    inference_mlp_depth=5 \
    inference_mlp_width=6 \
    approximator.inference_network.use_optimal_transport=false \
    approximator.summary_network.summary_dim=27 \
    embed_depth=3 \
    embed_width=5 \
    mlp_depth=3 \
    mlp_width=7 \
    approximator.summary_network.num_seeds=4
