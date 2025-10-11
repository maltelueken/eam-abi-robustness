#!/bin/bash

experiment=experiment_2

for model in "rdm_simple" "rdm_simple_lower" "rdm_simple_upper" "rdm_simple_meta" "rdm_simple_meta_lower" "rdm_simple_meta_upper"; do
    python $experiment/generate_test_data.py experiment=$experiment model=$model seed=2025
done
