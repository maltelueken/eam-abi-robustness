#!/bin/bash
#
# Submit every row of slurm/jobs.tsv, optionally filtered.
#
#   ./slurm/submit_all.sh                      # everything
#   ./slurm/submit_all.sh train_npe            # one stage
#   ./slurm/submit_all.sh '' experiment_2      # one experiment
#   DRY_RUN=1 ./slurm/submit_all.sh            # print without submitting
#
# Every stage but one runs on a GPU, and takes `slurm/submit.sh`'s own SBATCH directives.
# `fit_mcmc_cpu` runs its MCMC chains one per CPU core instead, so it is submitted with no GPU,
# one core per chain, and `MCMC_NUM_CPU_DEVICES` set to that count -- which is what
# `src/cpu_devices.py` reads to decide how many devices to split the host CPU into. Passed on
# the sbatch command line, where they override the directives in the batch file.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage_filter=${1:-}
experiment_filter=${2:-}
model_filter=${3:-}

# Chains per dataset, hence cores. Keep in step with `mcmc_sampling_fun.num_chains` in
# conf/experiment/experiment_1.yaml; `mcmc.chain_devices` fails loudly if they disagree.
mcmc_chains=${MCMC_NUM_CPU_DEVICES:-4}
cpu_partition=${CPU_PARTITION:-genoa}

while IFS=$'\t' read -r stage experiment model; do
    [[ -z "${stage// }" || "$stage" == \#* ]] && continue
    [[ -n "$stage_filter" && "$stage" != "$stage_filter" ]] && continue
    [[ -n "$experiment_filter" && "$experiment" != "$experiment_filter" ]] && continue
    [[ -n "$model_filter" && "$model" != "$model_filter" ]] && continue

    opts=(--job-name="${stage}_${model}")
    if [ "$stage" = "fit_mcmc_cpu" ]; then
        opts+=(
            --gpus=0
            --partition="$cpu_partition"
            --cpus-per-task="$mcmc_chains"
            --export="ALL,MCMC_NUM_CPU_DEVICES=${mcmc_chains}"
        )
    fi

    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch ${opts[*]} $here/submit.sh $stage $experiment $model"
    else
        sbatch "${opts[@]}" "$here/submit.sh" "$stage" "$experiment" "$model"
    fi
done < "$here/jobs.tsv"
