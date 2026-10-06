"""Posterior predictive response-time quantiles and accuracy, for NPE and MCMC."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from metrics import calc_posterior_predictive
from pipeline import iter_comparable_cases, load_case_data, num_obs_per_dataset, setup
from utils import load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_posterior_predictive(cfg: DictConfig):
    """Write `posterior_predictive/ppd.csv`, merging the NPE and MCMC predictions."""
    _, simulator = load_approximator(cfg)

    artifacts, cases, param_names = setup(cfg)

    num_samples = cfg["test_num_posterior_predictive_samples"]

    frames = []

    for case, posterior_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):
        logger.info("Computing posterior predictives for case %s", case.key)

        data = load_case_data(case, artifacts)
        data_x = data["x"][is_converged]
        num_obs = num_obs_per_dataset(data, is_converged)

        computed = {}

        for method, posterior in (("npe", posterior_npe), ("mcmc", posterior_mcmc)):
            # Thin to the requested number of predictive draws; simulating one dataset per
            # draw per subject is the expensive part here.
            stride = max(posterior.shape[1] // num_samples, 1)
            draws = posterior[:, ::stride, :][:, :num_samples, :]

            computed[method] = calc_posterior_predictive(
                data_x, draws, num_obs, simulator, num_samples, param_names,
            )

        merged = pd.merge(
            computed["npe"],
            computed["mcmc"],
            on=["id", "sample", "acc_true", "quantile", "rt_true"],
            suffixes=["_npe", "_mcmc"],
        )

        for label, value in case.labels.items():
            merged[label] = value

        frames.append(merged)

    path = artifacts.csv("posterior_predictive", "ppd")
    artifacts.ensure(path)
    logger.info("Writing %s", path)
    pd.concat(frames).to_csv(path, index=False)


if __name__ == "__main__":
    check_posterior_predictive()
