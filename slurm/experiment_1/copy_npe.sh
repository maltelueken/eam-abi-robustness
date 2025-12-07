#!/bin/bash

experiment=experiment_1
model=rdm_simple
inference_mlp_depth=5
inference_mlp_width=6
use_optimal_transport=False
summary_dim=27
embed_depth=3
embed_width=5
mlp_depth=3
mlp_width=7
num_seeds=4

for model in "rdm_simple" "rdm_simple_discrete_lower" "rdm_simple_discrete_upper" "rdm_simple_discrete_full"; do
    cp -r multirun/$experiment/$model/flow_matching/\
        approximator.inference_network.use_optimal_transport\=$use_optimal_transport/\
        approximator.summary_network.num_seeds\=$num_seeds/\
        approximator.summary_network.summary_dim\=$summary_dim/\
        embed_depth\=$embed_depth/\
        embed_width\=$embed_width/\
        inference_mlp_depth\=$inference_mlp_depth/\
        inference_mlp_width\=$inference_mlp_width/\
        mlp_depth\=$mlp_depth/\
        mlp_width\=$mlp_width/\
        sweeper\=optuna/* \
        outputs/$experiment/$model/flow_matching/
done
