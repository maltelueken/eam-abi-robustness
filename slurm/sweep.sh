#!/bin/bash
#
# Run the Optuna architecture sweep for one model family -- one job, `n_trials` trainings back
# to back, followed by the selection step that writes the winner back into the config.
#
#   sbatch slurm/sweep.sh                                        # experiment_2 / rdm_simple_meta
#   sbatch slurm/sweep.sh experiment_2 lba_simple_meta
#   sbatch slurm/sweep.sh experiment_2 rdm_simple_meta hydra.sweeper.n_trials=40
#
# There is one sweep per family, not one per (experiment, model): the RDM and the LBA are
# different forward models and deserve their own architecture, while the variants within a family
# differ only in a prior, a design or a data band. `conf/sweeper/optuna.yaml` therefore keys the
# study and the database on `architecture_family` -- so `rdm_simple` and `rdm_sat` write into the
# same RDM study, and `slurm/sweep_all.sh` submits exactly two jobs.
#
# The (experiment, model) the sweep is *hosted* on is therefore invisible to the study name, but
# it is not arbitrary: it is the family's full-varying arm under experiment_2, because that arm's
# task contains every other arm's. slurm/sweep_all.sh carries the argument; the header
# `scripts/select_architecture.py` writes into `conf/architecture/<family>.yaml` records which
# host a given architecture was actually selected on, since nothing else does.
#
# What the sweep must *not* be scored on is anything out of the swept model's own training
# distribution -- a shifted `drift_slope_loc`, or a `num_obs` outside the design simulator's
# grid. Both are what studies 1-3 measure, and an architecture selected on them would be
# selected on the studies' own outcome. `conf/experiment/experiment_1.yaml`'s `diag_num_obs`
# stays inside `random_num_obs_discrete`'s grid for that reason, and the meta simulator draws
# `drift_slope_loc` per batch element, so a diagnostic batch already averages over the full
# training range without scoring a single point outside it.
#
# The sweep tunes a *single* network, not an ensemble: `conf/config.yaml` defaults to the
# ensemble because that is what the experiments train, but searching an architecture five
# members at a time would cost five times as much per trial to select the same architecture --
# the members share it. So this is the one place that passes
# `approximator=continuous_approximator`.
#
# Nothing has to be copied by hand afterwards. `scripts/select_architecture.py` runs as the last
# step of this job and overwrites `conf/architecture/<family>.yaml`, which every model of that
# family composes -- so `slurm/submit.sh` trains the ensemble at the selected architecture with
# no further flags. Commit that file: it is the record of what the paper's networks were trained
# at, and it outlives the sweep database.
#
# Every submission sweeps into a *fresh* database. Optuna keys a study by name, and the name in
# `conf/sweeper/optuna.yaml` is derived only from the family and the network choices -- so an
# existing database would be resumed, silently merging trials scored under whatever code was
# checked out last time into one Pareto front with the ones scored now. Network defaults already
# moved once under the BayesFlow 2.0.14 upgrade; trials from either side of a change like that do
# not describe the same search space.
#
# The consequence is that a job killed at the walltime restarts from trial zero rather than
# resuming, so size `--time` and `hydra.sweeper.n_trials` to fit each other. The limit below is a
# guess at the partition's maximum; `sbatch --time=<hh:mm:ss> slurm/sweep.sh ...` overrides it.
#
#SBATCH --job-name=eam_abi_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=120:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${EAM_ABI_ROOT:-/projects/0/prjs1372/eam-abi-robustness}"

experiment=${1:-experiment_2}
model=${2:-rdm_simple_meta}

# Drop whichever of the two positionals were actually given, leaving "$@" as the overrides.
# Written arithmetically because `[ $# -ge 1 ] && shift` returns non-zero with no arguments,
# which `set -e` would take as a failure.
shift $(( $# < 2 ? $# : 2 ))

# The family the model belongs to, which is what the study and the database are named after.
# Every model file is `<family>_...`, and `conf/model/<model>.yaml` selects the matching
# `conf/architecture/<family>.yaml`; tests/test_config.py pins the two to agree, so this prefix
# is a safe way to name the database before Python has been started.
family=${model%%_*}

cd "$PROJECT_ROOT"

# `multirun/` too: the Optuna storage URL points a sqlite file at it, and sqlite will not create
# the directory -- on a fresh checkout the sweep dies with "unable to open database file" before
# the first trial starts.
mkdir -p slurm/logs multirun

# Start from an empty database (see the note at the top). The old one is moved aside rather than
# deleted: a completed sweep costs days of GPU time, and its trials are what
# `visualization/appendix_optimization.R` plots.
db=multirun/train_npe_trials_${family}.db

if [ -e "$db" ]; then
    archived="${db}.$(date +%Y%m%dT%H%M%S).bak"
    mv "$db" "$archived"
    echo "[$(date -Is)] archived previous sweep database to ${archived}"
fi

# Sidecar journal files outlive a moved database and sqlite would reattach them to the new one.
rm -f "${db}-wal" "${db}-shm" "${db}-journal"

module purge
module load 2025

# As in slurm/submit.sh: the environment is uv's, resolved from `uv.lock` at `uv run` rather
# than activated, and `--frozen` keeps a sweep from re-resolving it mid-flight. That matters
# more here than anywhere else -- a Pareto front is only comparable across trials scored
# against the same dependency versions, which is the same reason the database is started
# empty above.
export PATH="${PATH}:${HOME}/.local/bin"

# Take the whole MIG slice rather than the 75% the CUDA client preallocates by default. The
# summary network's attention scores are `(batch, 4 heads, num_obs, num_obs)` and materialize:
# at `train_npe`'s batch of 64 and the 1000 of `random_num_obs_discrete`'s grid that is ~3.7 GiB
# per `summary_embed_depth` block once the backward pass holds them, so the top of the search
# space needs ~17 GiB against a 20 GiB slice. The default 75% leaves 15 and the trial dies --
# and because a failed job aborts the whole `--multirun`, one such trial costs the sweep, not
# just itself. `summary_embed_depth` is capped for the same reason; see
# `conf/summary-method/set_transformer.yaml`. Harmless on a larger GPU.
export XLA_CLIENT_MEM_FRACTION=0.95

echo "[$(date -Is)] sweep | ${family} | ${experiment} | ${model} | extra: $*"

uv run --frozen python scripts/train_npe.py \
    --multirun \
    sweeper=optuna \
    approximator=continuous_approximator \
    experiment="${experiment}" \
    model="${model}" \
    "$@"

# Route the winner into conf/architecture/<family>.yaml, where every model of the family reads
# it. Given the same overrides as the sweep, since they can change the study the trials landed in.
echo "[$(date -Is)] selecting architecture | ${family}"

uv run --frozen python scripts/select_architecture.py "${model}" --experiment "${experiment}" "$@"

git --no-pager diff --stat -- "conf/architecture/${family}.yaml" || true
