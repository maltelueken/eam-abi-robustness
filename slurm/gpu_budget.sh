#!/bin/bash
#
# Check that the architecture sweep's search space fits the GPUs it is trained on, and that it
# fits without XLA trading speed for memory.
#
#   sbatch slurm/gpu_budget.sh single                                     # gpu_mig, one network
#   sbatch --partition=gpu_a100 --cpus-per-task=18 slurm/gpu_budget.sh ensemble
#
#   PRESETS="current max" sbatch slurm/gpu_budget.sh single               # a subset
#   MODEL=lba_simple_meta sbatch slurm/gpu_budget.sh single               # the other family
#   sbatch slurm/gpu_budget.sh single --steps 100                         # flags for every run
#
# The two modes are the two constraints:
#
# * `single` -- a sweep trial (one network trained, then sampled for `train_npe`'s diagnostics)
#   has to run on the 20 GB MIG slice that slurm/sweep.sh is submitted to;
# * `ensemble` -- whatever the sweep selects is later trained as a five-member ensemble by
#   slurm/submit.sh on a 40 GB A100, and sampled by `predict_npe`. The trial counts here go up
#   to 1200 rather than 1000, because study 1's `discrete_upper` and `discrete_full` arms train
#   at those counts and its test sweep reaches them.
#
# Each mode runs scripts/gpu_budget.py over a list of architectures (`preset_args` below), one
# process per architecture and phase, so a failure costs only that run and each run's peak
# memory is its own. The presets run from the configured architecture up to the top corner of
# the search space, which the script reads from the sweeper config rather than from here.
# Memory grows with every swept knob, so the top corner is the worst case: if it passes, the
# whole space does. The intermediate presets show where the boundary lies when it does not.
#
# How to read the result (printed at the end of the job, kept in $out/results.csv):
#
# * `status`: `oom` means the architecture does not fit this GPU at that trial count. In the
#   sweep, that aborts the whole `--multirun`.
# * `peak_gib` against `limit_gib`: how much of XLA's pool a run needed. `capacity_gib -
#   pool_gib` is all the CUDA driver had left over, which is where CUDA graphs are instantiated.
#   A graph that could not be instantiated is what killed a sweep trial before.
# * `plan` rows: the largest train step compiled with and without XLA's memory limit. Equal
#   plans and a step-time ratio within a few percent of 1 (about what 50 timed steps resolve)
#   mean memory cost nothing. A smaller limited plan, or a ratio above that, means the compiler
#   worked around memory, which is a slowdown. An `oom` on the unlimited row alone means the step
#   only fits because of that.
# * the warnings table: XLA reports rematerialization, and autotuning starved of scratch space
#   ("performance gains if more memory were available"), on stderr only. Both are slowdowns
#   that do not fail anything.
#
# Both modes run under slurm/sweep.sh's memory settings (below). slurm/submit.sh sets neither,
# so an ensemble job currently gets the CUDA client's default instead: 75% of the device,
# preallocated. Read an ensemble's `peak_gib` against `0.75 * capacity_gib` to see whether
# submit.sh as it stands would fit.
#
#SBATCH --job-name=eam_abi_gpu_budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/eam-abi-robustness/slurm/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${EAM_ABI_ROOT:-/projects/0/prjs1372/eam-abi-robustness}"

mode=${1:-single}
shift $(( $# < 1 ? $# : 1 ))

experiment=${EXPERIMENT:-experiment_2}
model=${MODEL:-rdm_simple_meta}

case "$mode" in
    single)
        approximator=continuous_approximator
        expected_partition=gpu_mig
        # The sweep host's design grid (`random_num_obs_discrete`).
        train_num_obs="100,250,500,1000"
        # The sweep never runs predict_npe, so the single network is checked on what a trial
        # does and nothing more.
        phases="train plan"
        presets=${PRESETS:-"current max_depth2 max_width7 max"}
        ;;
    ensemble)
        approximator=ensemble_approximator
        expected_partition=gpu_a100
        train_num_obs="100,250,500,1000,1200"
        phases="train plan predict"
        presets=${PRESETS:-"current_depth1 current max_depth1 max_depth2 max"}
        ;;
    *)
        echo "usage: $0 single|ensemble [gpu_budget.py flags...]" >&2
        exit 2
        ;;
esac

# The architectures to try. `current` is conf/architecture/<family>.yaml as it stands, which
# is what slurm/submit.sh trains today. `max` is the top corner of the sweep's search space,
# and `max_*` back off one knob from it: the set transformer's depth costs the most, since
# each block materializes a `(batch, 4 heads, num_obs, num_obs)` attention score tensor
# whatever its width, and the width costs the next most.
preset_args() {
    case "$1" in
        current) echo "" ;;
        current_depth1) echo "summary_embed_depth=1" ;;
        max) echo "--corner max" ;;
        max_depth1) echo "--corner max summary_embed_depth=1" ;;
        max_depth2) echo "--corner max summary_embed_depth=2" ;;
        max_width7) echo "--corner max summary_embed_width=7" ;;
        *)
            echo "unknown preset '$1'" >&2
            return 1
            ;;
    esac
}

cd "$PROJECT_ROOT"

out="slurm/logs/gpu_budget/${SLURM_JOB_ID:-local}_${mode}_${model}"
mkdir -p "$out"

module purge
module load 2025

export PATH="${PATH}:${HOME}/.local/bin"

# Exactly slurm/sweep.sh's settings -- see there for why each is needed. A ceiling of 95% of
# the device, grown on demand rather than reserved up front.
export XLA_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ "${SLURM_JOB_PARTITION:-}" != "$expected_partition" ]; then
    echo "WARNING: mode '${mode}' is meant for ${expected_partition}, but this job runs on" \
        "'${SLURM_JOB_PARTITION:-?}'. The results describe this GPU, not that one." >&2
fi

echo "[$(date -Is)] gpu budget | ${mode} | ${experiment} | ${model} | presets: ${presets} | extra: $*"
nvidia-smi -L || true

printf "preset\tphase\texit\twarnings\tlog\n" > "$out/warnings.tsv"

for preset in $presets; do
    read -r -a architecture <<< "$(preset_args "$preset")"

    for phase in $phases; do
        log="$out/${preset}_${phase}.log"

        echo "[$(date -Is)] ${preset} | ${phase}"

        # A failed run is a result here, not a reason to stop: record it and go on.
        set +e
        uv run --frozen python scripts/gpu_budget.py "$phase" \
            --out "$out/results.csv" \
            --label "$preset" \
            --approximator "$approximator" \
            --experiment "$experiment" \
            --model "$model" \
            --train-num-obs "$train_num_obs" \
            "${architecture[@]}" \
            "$@" \
            > "$log" 2>&1
        status=$?
        set -e

        warnings=()
        grep -q "rematerialization" "$log" && warnings+=("rematerialization")
        grep -q "performance gains if more memory" "$log" && warnings+=("allocator-starved")
        grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$log" && warnings+=("driver-oom")

        printf "%s\t%s\t%s\t%s\t%s\n" "$preset" "$phase" "$status" "${warnings[*]:-none}" "$log" \
            >> "$out/warnings.tsv"

        grep -E "^phase=" "$log" || tail -n 5 "$log"
    done
done

echo

# Absent only if every run died before measuring anything -- e.g. JAX found no GPU.
if [ -f "$out/results.csv" ]; then
    echo "== device"
    cut -d, -f6 "$out/results.csv" | sed -n 2p

    echo
    echo "== results (${out}/results.csv)"
    cut -d, -f1,7-9,13-21 "$out/results.csv" | column -s, -t
else
    echo "== no results: every run failed before measuring; see the logs below"
fi

echo
echo "== XLA warnings and exit codes"
column -t -s $'\t' "$out/warnings.tsv"
