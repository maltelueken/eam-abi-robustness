#!/bin/bash

experiment=experiment_1

for model in "rdm_simple" "rdm_simple_discrete_lower" "rdm_simple_discrete_upper" "rdm_simple_discrete_full"; do
    mkdir -p outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/metrics/ \
        outputs/$experiment/$model/flow_matching/
done

experiment=experiment_2

for model in "rdm_simple" "rdm_simple_lower" "rdm_simple_upper" "rdm_simple_meta" "rdm_simple_meta_lower" "rdm_simple_meta_upper"; do
    mkdir -p outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/metrics/ \
        outputs/$experiment/$model/flow_matching/
done

experiment=experiment_4

for model in "rdm_simple" "rdm_simple_lower" "rdm_simple_upper" "rdm_simple_meta" "rdm_simple_meta_lower" "rdm_simple_meta_upper"; do
    mkdir -p outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/metrics/ \
        outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/posterior_predictive/ \
        outputs/$experiment/$model/flow_matching/
done




