"""Maximum mean discrepancy between the NPE and MCMC posteriors, per dataset."""

import logging

import bayesflow as bf
import hydra
from omegaconf import DictConfig

from pipeline import accuracy, iter_comparable_cases, load_case_data, rt_range, setup
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_robustness(cfg: DictConfig):
    """Write `robustness/mmd.csv`, one row per dataset per case."""
    artifacts, cases, param_names = setup(cfg)

    records = []

    for case, posterior_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):
        logger.info("Computing MMD for case %s", case.key)

        # Per-dataset properties of the generated data, carried alongside the mismatch: the
        # quantity study 5 selects on, and the pair study 6 selects on.
        data_x = load_case_data(case, artifacts)["x"]
        rates = accuracy(data_x, is_converged)
        lowest_rt, highest_rt = rt_range(data_x, is_converged)

        for index, (mcmc_draws, npe_draws) in enumerate(zip(posterior_mcmc, posterior_npe, strict=True)):
            records.append(
                {
                    **case.labels,
                    "dataset": index,
                    "accuracy": float(rates[index]),
                    "rt_min": float(lowest_rt[index]),
                    "rt_max": float(highest_rt[index]),
                    "mmd": float(bf.metrics.functional.maximum_mean_discrepancy(mcmc_draws, npe_draws)),
                },
            )

    path = artifacts.csv("robustness", "mmd")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    check_robustness()
