#!/bin/bash

sbatch slurm/experiment_2/fit_mcmc_sweep_meta_lower_1.sh
sbatch slurm/experiment_2/fit_mcmc_sweep_meta_lower_2.sh
sbatch slurm/experiment_2/fit_mcmc_sweep_meta_lower_3.sh

