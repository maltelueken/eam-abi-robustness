Model training and analysis pipeline for assessing the generalization of amortized Bayesian inference for
evidence accumulation models.

This code supplements the paper "Generalization in neural posterior estimation: Case studies with the racing
diffusion model" (LINK).

## Overview

- `conf`: This directory contains Hydra configuration files. See https://hydra.cc/docs/intro/ for details.
- `experiment_1`: Python scripts for the first simulation study.
- `experiment_2`: Python scripts for the second simulation study.
- `experiment_4`: Python scripts for the empirical study. Originally, we had a third simulation study planned but chose to not
conduct it because the results from the first two were already clear. Hence, `experiment_3` is missing.
- `multirun`: Contains results from the hyperparameter optimization. See Fig Share (LINK).
- `outputs`: Contains all other results and simulated data. See Fig Share (LINK).
- `slurm`: Slurm scripts for training the NPEs, running MCMC, and analyzing results.
- `src`: Python modules with utility functions used in the scripts.
- `tests`: Some unit tests.
- `visualization`: R code for creating the figures in the paper.

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
