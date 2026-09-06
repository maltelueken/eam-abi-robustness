#!/bin/bash
#
# Submit one pipeline stage for one experiment/model combination.
#
#   sbatch slurm/submit.sh <stage> <experiment> <model> [hydra overrides...]
#
# e.g.  sbatch slurm/submit.sh train_npe        experiment_1 rdm_simple
#       sbatch slurm/submit.sh fit_mcmc_cpu     experiment_2 rdm_simple_meta
#       sbatch slurm/submit.sh check_robustness experiment_4 rdm_simple
#
# Resources are selected per stage below rather than per script, so adding a model or an
# experiment does not mean adding another batch file. Every stage runs the same
# `scripts/<stage>.py`; the test cases it loops over come from `conf/test_case/`, so the
# parameter grids are no longer mirrored in bash.
#
# Every stage trains and reads an ensemble of NPEs, because that is `conf/config.yaml`'s default
# approximator -- nothing has to be passed here for it. The Optuna architecture sweep is the one
# thing that wants a single network, and it has its own script: `slurm/sweep.sh`.
#
# The directives below are the GPU stages' defaults. `fit_mcmc_cpu` wants the opposite -- no GPU
# and one core per MCMC chain -- and `slurm/submit_all.sh` overrides them on the sbatch command
# line, where they win over what is written here. Submitting that stage by hand means doing the
# same:
#
#   sbatch --gpus=0 --cpus-per-task=4 --partition=genoa \
#          --export=ALL,MCMC_NUM_CPU_DEVICES=4 \
#          slurm/submit.sh fit_mcmc_cpu experiment_1 rdm_simple
#
# `MCMC_NUM_CPU_DEVICES` has to match `mcmc_sampling_fun.num_chains`: it is how many devices
# JAX splits the host CPU into, and one chain runs on each. See src/cpu_devices.py.
#
#SBATCH --job-name=eam_abi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${EAM_ABI_ROOT:-/projects/0/prjs1372/eam-abi-robustness}"

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <stage> <experiment> <model> [hydra overrides...]" >&2
    exit 2
fi

stage=$1
experiment=$2
model=$3
shift 3

cd "$PROJECT_ROOT"

mkdir -p slurm/logs

module load 2023
source bin/activate

echo "[$(date -Is)] ${stage} | ${experiment} | ${model} | extra: $*"

python "scripts/${stage}.py" \
    experiment="${experiment}" \
    model="${model}" \
    "$@"
