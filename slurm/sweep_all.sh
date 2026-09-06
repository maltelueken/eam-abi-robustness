#!/bin/bash
#
# Submit the architecture sweep for every model family -- two jobs, one per forward model.
#
#   ./slurm/sweep_all.sh                       # both families
#   ./slurm/sweep_all.sh rdm                   # one of them
#   DRY_RUN=1 ./slurm/sweep_all.sh             # print without submitting
#
# The variants within a family (priors, designs, data bands) share the family's architecture, so
# there is nothing here per row of `slurm/jobs.tsv` -- the sweep is the one stage that is not a
# row in it. Each family is swept on its plainest model under experiment_1; see slurm/sweep.sh.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

family_filter=${1:-}

for family in rdm lba; do
    [[ -n "$family_filter" && "$family" != "$family_filter" ]] && continue

    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch --job-name=sweep_${family} $here/sweep.sh experiment_1 ${family}_simple"
    else
        sbatch --job-name="sweep_${family}" "$here/sweep.sh" experiment_1 "${family}_simple"
    fi
done
