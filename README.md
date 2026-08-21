Model training and analysis pipeline for assessing the generalization of amortized Bayesian inference for
evidence accumulation models.

This code supplements the paper "Generalization in neural posterior estimation: Case studies with the racing
diffusion model" (LINK).

## Overview

- `conf`: This directory contains Hydra configuration files. See https://hydra.cc/docs/intro/ for details.
- `scripts`: One Python script per pipeline stage, shared by all three studies. Which study a run
belongs to is a configuration choice (`experiment=experiment_1|experiment_2|experiment_4`), not a
different script. See `scripts/README.md`.
- `multirun`: Contains results from the hyperparameter optimization. See Fig Share (LINK).
- `outputs`: Contains all other results and simulated data. See Fig Share (LINK).
- `slurm`: Slurm submission scripts (`submit.sh` plus the job table in `jobs.tsv`).
- `src`: Python modules with the model implementations and utility functions used in the scripts.
- `tests`: Unit tests.
- `visualization`: R code for creating the figures in the paper.

The paper reports three studies: two simulation studies (`experiment_1`, `experiment_2`) and an
empirical study (`experiment_4`). Originally, we had a third simulation study planned but chose not
to conduct it because the results from the first two were already clear. Hence, `experiment_3` is
missing.

## Running the pipeline

```console
python scripts/train_npe.py         experiment=experiment_1 model=rdm_simple
python scripts/generate_test_data.py experiment=experiment_1 model=rdm_simple
python scripts/predict_npe.py       experiment=experiment_1 model=rdm_simple
python scripts/fit_mcmc_gpu.py      experiment=experiment_1 model=rdm_simple
python scripts/check_robustness.py  experiment=experiment_1 model=rdm_simple
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
