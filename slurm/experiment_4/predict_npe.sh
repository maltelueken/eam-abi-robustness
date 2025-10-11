#!/bin/bash

experiment=experiment_4

for model in rdm_simple_meta rdm_simple_meta_lower rdm_simple_meta_upper; do
    python experiment_4/predict_npe_meta.py experiment=$experiment model=$model
done

for model in rdm_simple rdm_simple_meta_no_params rdm_simple_meta_lower_no_params rdm_simple_meta_upper_no_params; do
    python experiment_4/predict_npe_simple.py experiment=$experiment model=$model
done
