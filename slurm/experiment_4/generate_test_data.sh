#!/bin/bash

experiment=experiment_4

for model in "rdm_simple" "rdm_simple_lower" "rdm_simple_upper" "rdm_simple_meta" "rdm_simple_meta_lower" "rdm_simple_meta_upper"; do
    test_dir=outputs/$experiment/$model/test_data/
    mkdir -p $test_dir
    cp -r data_lerche2020/* $test_dir
done
