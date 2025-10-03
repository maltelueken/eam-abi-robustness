#!/bin/bash

experiment=experiment_4

for model in rdm_simple rdm_simple_meta rdm_simple_meta_lower rdm_simple_meta_upper rdm_simple_meta_no_params rdm_simple_meta_lower_no_params rdm_simple_meta_upper_no_params; do
    mkdir -p outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/metrics/ \
        outputs/$experiment/$model/flow_matching/
    scp -r snellius:/projects/prjs1372/eam-abi-robustness/outputs/$experiment/$model/flow_matching/posterior_predictive/ \
        outputs/$experiment/$model/flow_matching/
done




