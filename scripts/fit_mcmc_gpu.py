"""Fit ground-truth MCMC posteriors on the GPU, one job per test case.

`mcmc.fit_mcmc_gpu_batch` vmaps BlackJAX NUTS over chains *and* over every dataset in the
case at once. The log-density, its parameter vector and the unconstraining transform all
come from the `mcmc` config group, so this script has no model-specific branching.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_mcmc_param_names
from mcmc import save_mcmc_posterior
from pipeline import load_case_data, select_cases, setup

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    """Fit every dataset of the selected case(s) and save the posterior samples."""
    artifacts, cases, _ = setup(cfg)

    make_logdensity_fn = instantiate(cfg["mcmc_model_fun"])
    to_unconstrained = instantiate(cfg["mcmc_to_unconstrained"])
    fit_fun = instantiate(cfg["mcmc_sampling_fun"])

    for case in select_cases(cases, cfg["case"]):
        data_x = load_case_data(case, artifacts)["x"]

        logger.info(
            "Fitting MCMC for case %s: %s datasets with %s trials each",
            case.key,
            data_x.shape[0],
            data_x.shape[1],
        )

        positions, infos = fit_fun(
            data=data_x,
            make_logdensity_fn=make_logdensity_fn,
            to_unconstrained=to_unconstrained,
        )
        positions.block_until_ready()

        logger.info("Divergence rate: %s", float(np.mean(infos.is_divergent)))

        path = artifacts.mcmc_samples(case)
        artifacts.ensure(path)
        logger.info("Saving MCMC samples to %s", path)
        save_mcmc_posterior(path, positions, get_mcmc_param_names(cfg))


if __name__ == "__main__":
    fit_mcmc()
