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

python experiment_1/predict_npe.py \
    --multirun \
    embed_depth=3 \
    embed_width=7 \
    inference_mlp_width=9 \
    mlp_depth=1 \
    mlp_width=8 \
    sweeper=basic \
    workflow.inference_network.subnet_kwargs.depth=9 \
    workflow.inference_network.use_optimal_transport=True \
    workflow.summary_network.num_seeds=2 \
    workflow.summary_network.summary_dim=14
