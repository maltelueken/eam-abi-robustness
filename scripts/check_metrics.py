"""Parameter-recovery metrics for NPE and MCMC against the simulating parameters.

Only meaningful for the simulation studies: the empirical study has no ground truth, so
cases without it are skipped rather than scored against a stand-in.
"""

import logging

import hydra
from omegaconf import DictConfig

from metrics import recovery_metrics
from pipeline import iter_comparable_cases, load_true_params, setup
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_metrics(cfg: DictConfig):
    """Write `metrics/metrics.csv` in long format."""
    artifacts, cases, param_names = setup(cfg)

    records = []

    for case, posterior_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):

        targets = load_true_params(case, artifacts, param_names, is_converged)

        if targets is None:
            logger.info("Case %s has no ground truth; skipping recovery metrics.", case.key)
            continue

        logger.info("Computing recovery metrics for case %s", case.key)
        records += list(recovery_metrics(posterior_npe, targets, param_names, "npe", case.labels))
        records += list(recovery_metrics(posterior_mcmc, targets, param_names, "mcmc", case.labels))

    if not records:
        logger.warning(
            "No case had ground-truth parameters, so there are no recovery metrics to write. "
            "This script only applies to the simulation studies.",
        )
        return

    path = artifacts.csv("metrics", "metrics")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    check_metrics()
