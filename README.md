Model training and analysis pipeline for assessing the generalization of amortized Bayesian inference for
evidence accumulation models.

This code supplements the paper "Generalization in neural posterior estimation: Case studies with the racing
diffusion model" (LINK).

## Overview

- `conf`: This directory contains Hydra configuration files. See https://hydra.cc/docs/intro/ for details.
- `scripts`: One Python script per pipeline stage, shared by all four studies. Which study a run
belongs to is a configuration choice (`experiment=experiment_1|experiment_2|experiment_3|experiment_4`),
not a different script. See `scripts/README.md`.
- `multirun`: Contains results from the hyperparameter optimization. See Fig Share (LINK).
- `outputs`: Contains all other results and simulated data. See Fig Share (LINK).
- `slurm`: Slurm submission scripts (`submit.sh` plus the job table in `jobs.tsv`).
- `src`: Python modules with the model implementations and utility functions used in the scripts.
- `tests`: Unit tests.
- `visualization`: R code for creating the figures in the paper.

The paper reports four studies: three simulation studies and an empirical one.

| study | what varies between training and test | test cases |
|---|---|---|
| `experiment_1` | the number of observations per dataset | a sweep of `num_obs` |
| `experiment_2` | the location of the prior on the drift slope | a sweep of `drift_slope_loc` |
| `experiment_3` | the prior on the speed-vs-accuracy threshold difference | a sweep of `threshold_diff_scale` |
| `experiment_4` | (empirical data) | one case per subject file |

Study 2 sweeps `drift_slope_loc` over nine points from 0.7 to 3.9 in steps of 0.4, holding every
other prior hyperparameter fixed. The third point is the 1.5 the models train on, so the sweep
contains the one cell where the test prior and the training prior agree. The range reaches 4.0 because that is where the empirical data are: across the
371 participant-datasets of study 4, the MCMC posterior medians for the drift slope have a median of
2.84 and a maximum of 3.70, and 80% of them fall outside the training prior's central 95% interval.
The study used to cross this axis with the threshold prior's scale; that second axis was dropped
because the same posteriors put every empirical threshold *inside* a fixed `Gamma(8, 0.15)`, so
randomizing it amortized the networks over a direction the empirical study never exercises.

Study 3 simulates a speed-vs-accuracy manipulation: every dataset contains a speed-instructed and an
accuracy-instructed block of trials, differing only in response threshold. The test cases shift the
prior on that threshold *difference* away from the prior the approximator was trained on. Its models
are `conf/model/rdm_sat*.yaml` for the racing diffusion model and `conf/model/lba_sat*.yaml` for the
linear ballistic accumulator.

## Running the pipeline

```console
python scripts/train_npe.py         experiment=experiment_1 model=rdm_simple
python scripts/generate_test_data.py experiment=experiment_1 model=rdm_simple
python scripts/predict_npe.py       experiment=experiment_1 model=rdm_simple
python scripts/fit_mcmc_gpu.py      experiment=experiment_1 model=rdm_simple
python scripts/check_robustness.py  experiment=experiment_1 model=rdm_simple
python scripts/prior_distance.py    experiment=experiment_2 model=rdm_simple
```

Simulated data and posterior samples are stored as NetCDF (`.nc`). Artifacts produced by earlier
versions of the pipeline are HDF5; convert them with `scripts/convert_hdf5_to_netcdf.py`.

## Installation

While this repository contains mainly scripts, it can also be installed as a Python package for easy usage.

To install the package `eam_abi_robustness` from this GitHub repository, do:

```console
git clone git@github.com:maltelueken/eam_abi_robustness.git
cd eam_abi_robustness
python -m pip install .
```

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
