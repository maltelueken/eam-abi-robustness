#!/bin/bash
#
# Download the results of every (experiment, model) in slurm/jobs.tsv from Snellius, optionally
# filtered. Run it locally, from anywhere in the repository.
#
#   ./slurm/download_all.sh                    # everything
#   ./slurm/download_all.sh experiment_2       # one experiment
#   ./slurm/download_all.sh '' rdm_simple_meta # one model, in every experiment it appears in
#   DRY_RUN=1 ./slurm/download_all.sh          # list what would be transferred
#
# Only the small outputs are fetched -- the result CSVs the `check_*`, `prior_*` and timing stages
# write into each run directory, which is what `visualization/` reads -- never the test data,
# MCMC fits, NPE samples or checkpoints. The (experiment, model) pairs come from jobs.tsv rather
# than being listed here, so a row added there is downloaded without editing this file.
#
# Everything goes through one rsync call, so there is one SSH connection (one password/2FA
# prompt) rather than one per model, a directory a stage has not produced yet is skipped rather
# than failing the run, and files already up to date are not transferred again.
#
# With no filter, the architecture sweep's `multirun/{trials,timing}_<family>.csv` are fetched
# too; `visualization/appendix_optimization.R` plots them.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"

experiment_filter=${1:-}
model_filter=${2:-}

remote_host=${REMOTE_HOST:-snellius}
remote_root=${REMOTE_ROOT:-/projects/prjs1372/eam-abi-robustness}
# `hydra.run.dir` is outputs/<experiment>/<model>/<inference-method>/<override_dirname>; the
# rows of jobs.tsv pass no overrides, so the last segment is empty.
inference_method=${INFERENCE_METHOD:-flow_matching}

# Subdirectories of a run directory worth downloading, and single files next to them.
result_dirs=(metrics robustness timing posterior_predictive)
result_files=(prior_pushforward.png)

filters=()
declare -A seen=()

while IFS=$'\t' read -r stage experiment model; do
    [[ -z "${stage// }" || "$stage" == \#* ]] && continue
    [[ -n "$experiment_filter" && "$experiment" != "$experiment_filter" ]] && continue
    [[ -n "$model_filter" && "$model" != "$model_filter" ]] && continue

    run="/$experiment/$model/$inference_method"
    [[ -n "${seen[$run]:-}" ]] && continue
    seen[$run]=1

    # rsync only descends into a directory an include rule matches, so each parent is listed.
    filters+=(--include="/$experiment/" --include="/$experiment/$model/" --include="$run/")
    for dir in "${result_dirs[@]}"; do
        filters+=(--include="$run/$dir/***")
    done
    for file in "${result_files[@]}"; do
        filters+=(--include="$run/$file")
    done
done < "$here/jobs.tsv"

if [ "${#seen[@]}" -eq 0 ]; then
    echo "no rows of jobs.tsv match experiment='$experiment_filter' model='$model_filter'" >&2
    exit 1
fi

opts=(--archive --verbose --human-readable --prune-empty-dirs)
[ -n "${DRY_RUN:-}" ] && opts+=(--dry-run)

echo "Downloading ${#seen[@]} run directories from $remote_host:$remote_root"

mkdir -p "$repo/outputs"
rsync "${opts[@]}" "${filters[@]}" --exclude='*' \
    "$remote_host:$remote_root/outputs/" "$repo/outputs/"

if [ -z "$experiment_filter" ] && [ -z "$model_filter" ]; then
    mkdir -p "$repo/multirun"
    rsync "${opts[@]}" --include='trials_*.csv' --include='timing_*.csv' --exclude='*' \
        "$remote_host:$remote_root/multirun/" "$repo/multirun/"
fi
