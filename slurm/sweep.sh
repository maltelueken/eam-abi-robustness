#!/bin/bash
#
# Run the Optuna architecture sweep for one model family -- one job, `n_trials` trainings back
# to back, followed by the selection step that writes the winner back into the config.
#
#   sbatch slurm/sweep.sh                                        # experiment_1 / rdm_simple
#   sbatch slurm/sweep.sh experiment_1 lba_simple
#   sbatch slurm/sweep.sh experiment_1 rdm_simple hydra.sweeper.n_trials=40
#
# There is one sweep per family, not one per (experiment, model): the RDM and the LBA are
# different forward models and deserve their own architecture, while the variants within a family
# differ only in a prior, a design or a data band. `conf/sweeper/optuna.yaml` therefore keys the
# study and the database on `architecture_family` -- so `rdm_simple` and `rdm_sat` write into the
# same RDM study, and `slurm/sweep_all.sh` submits exactly two jobs.
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
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${EAM_ABI_ROOT:-/projects/0/prjs1372/eam-abi-robustness}"

experiment=${1:-experiment_1}
model=${2:-rdm_simple}

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

module load 2023
source bin/activate

echo "[$(date -Is)] sweep | ${family} | ${experiment} | ${model} | extra: $*"

python scripts/train_npe.py \
    --multirun \
    sweeper=optuna \
    approximator=continuous_approximator \
    experiment="${experiment}" \
    model="${model}" \
    "$@"

# Route the winner into conf/architecture/<family>.yaml, where every model of the family reads
# it. Given the same overrides as the sweep, since they can change the study the trials landed in.
echo "[$(date -Is)] selecting architecture | ${family}"

python scripts/select_architecture.py "${model}" --experiment "${experiment}" "$@"

git --no-pager diff --stat -- "conf/architecture/${family}.yaml" || true
