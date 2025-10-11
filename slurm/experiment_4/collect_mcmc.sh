#!/bin/bash

experiment=experiment_4

for model in rdm_simple rdm_simple_meta rdm_simple_meta_lower rdm_simple_meta_upper; do
    for filename in FF1.txt FN1.txt FV1.txt; do
        python experiment_4/collect_mcmc.py experiment=$experiment model=$model +slurm_filename=$filename
    done
done