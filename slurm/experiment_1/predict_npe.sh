#!/bin/bash
#SBATCH --job-name=predict_npe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=01:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/predict_npe_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/predict_npe_%j.err

cd /projects/0/prjs1372/eam-abi-robustness

module load 2023

source bin/activate

python experiment_1/predict_npe.py \
    --multirun \
    sweeper=basic \
    inference_mlp_depth=5 \
    inference_mlp_width=8 \
    approximator.inference_network.use_optimal_transport=true \
    approximator.summary_network.summary_dim=19 \
    embed_depth=1 \
    embed_width=4 \
    mlp_depth=2 \
    mlp_width=5 \
    approximator.summary_network.num_seeds=3
