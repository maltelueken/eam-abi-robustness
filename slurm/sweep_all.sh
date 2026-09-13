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
# row in it. See slurm/sweep.sh.
#
# Each family is swept on its **full-varying** arm, `<family>_simple_meta` under experiment_2,
# rather than on the plainest fixed-prior model. One architecture serves every arm of a family,
# and the fixed-prior task is a strict sub-problem of the varying one -- same likelihood, same
# parameters, a narrower prior to cover. So the two ways of getting this wrong are not
# symmetric: selecting on a fixed-prior arm risks under-capacity on the varying arms, which
# would surface as degradation read off as study 2's result rather than as an artifact of the
# architecture, while selecting on the varying arm risks over-capacity on the fixed ones, which
# costs training time. It is also the conservative direction for the claim -- if the varying arm
# still loses at an architecture chosen for it, that is not the architecture's doing.
#
# experiment_2 carries a second advantage the sweep wants for its own sake: `diag_batch_size` is
# 1000 there against experiment_1's 100, so the three objectives `train_npe` returns are
# estimated on ten times the datasets. `sweeper.select_best_trial` picks the Pareto-front member
# nearest the ideal point, which is exactly where estimation noise moves the answer. It also
# makes each trial's diagnostic sampling ten times as long -- check the `diagnostics` rows of one
# trial's `timing/train_npe.csv` against slurm/sweep.sh's `--time` before committing 20 trials.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

family_filter=${1:-}

for family in rdm lba; do
    [[ -n "$family_filter" && "$family" != "$family_filter" ]] && continue

    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch --job-name=sweep_${family} $here/sweep.sh experiment_2 ${family}_simple_meta"
    else
        sbatch --job-name="sweep_${family}" "$here/sweep.sh" experiment_2 "${family}_simple_meta"
    fi
done
