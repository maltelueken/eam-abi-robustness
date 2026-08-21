"""Posterior median and 95% credible interval per dataset, for NPE and MCMC alike."""

import logging

import hydra
from omegaconf import DictConfig

from metrics import posterior_summary, true_values
from pipeline import iter_comparable_cases, load_true_params, setup
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_stats(cfg: DictConfig):
    """Write `metrics/summary_stats.csv` in long format."""
    artifacts, cases, param_names = setup(cfg)

    records = []

    for case, posterior_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):
        logger.info("Summarizing posteriors for case %s", case.key)

        records += list(posterior_summary(posterior_npe, param_names, "npe", case.labels))
        records += list(posterior_summary(posterior_mcmc, param_names, "mcmc", case.labels))

        targets = load_true_params(case, artifacts, param_names, is_converged)

        if targets is not None:
            records += list(true_values(targets, param_names, case.labels))

    path = artifacts.csv("metrics", "summary_stats")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    check_summary_stats()
