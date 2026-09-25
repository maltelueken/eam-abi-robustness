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
# `fit_mcmc_cpu` spreads its MCMC fits -- one per chain per dataset -- over CPU cores instead, so
# it is submitted with no GPU and `MCMC_CPUS` cores; `src/cpu_devices.py` turns every core of the
# allocation into a JAX device. Passed on the sbatch command line, where they override the
# directives in the batch file.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage_filter=${1:-}
experiment_filter=${2:-}
model_filter=${3:-}

# Cores per MCMC job. A full genoa node by default: a simulated case is 4 chains x 100 datasets
# = 400 independent fits, so every core up to that count shortens the job.
mcmc_cpus=${MCMC_CPUS:-192}
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
            --cpus-per-task="$mcmc_cpus"
        )
    fi

    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch ${opts[*]} $here/submit.sh $stage $experiment $model"
    else
        sbatch "${opts[@]}" "$here/submit.sh" "$stage" "$experiment" "$model"
    fi
done < "$here/jobs.tsv"
