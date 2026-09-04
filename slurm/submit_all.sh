#!/bin/bash
#
# Submit every row of slurm/jobs.tsv, optionally filtered.
#
#   ./slurm/submit_all.sh                      # everything
#   ./slurm/submit_all.sh train_npe            # one stage
#   ./slurm/submit_all.sh '' experiment_2      # one experiment
#   DRY_RUN=1 ./slurm/submit_all.sh            # print without submitting
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage_filter=${1:-}
experiment_filter=${2:-}
model_filter=${3:-}

while IFS=$'\t' read -r stage experiment model; do
    [[ -z "${stage// }" || "$stage" == \#* ]] && continue
    [[ -n "$stage_filter" && "$stage" != "$stage_filter" ]] && continue
    [[ -n "$experiment_filter" && "$experiment" != "$experiment_filter" ]] && continue
    [[ -n "$model_filter" && "$model" != "$model_filter" ]] && continue

    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch --job-name=${stage}_${model} $here/submit.sh $stage $experiment $model"
    else
        sbatch --job-name="${stage}_${model}" "$here/submit.sh" "$stage" "$experiment" "$model"
    fi
done < "$here/jobs.tsv"
