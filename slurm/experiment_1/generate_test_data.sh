#!/bin/bash

experiment=experiment_1

for model in "rdm_simple" "rdm_simple_discrete_lower" "rdm_simple_discrete_upper" "rdm_simple_discrete_full"; do
    python $experiment/generate_test_data.py experiment=$experiment model=$model seed=2025
done