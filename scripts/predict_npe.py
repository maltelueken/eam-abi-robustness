"""Sample the trained NPE's posterior for every test case."""

import logging

import hydra
from omegaconf import DictConfig

from data import save_posterior, stack_posterior_dict
from pipeline import load_case_data, setup
from utils import load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def predict_npe(cfg: DictConfig):
    """Run posterior sampling per case and save the draws as NetCDF."""
    approximator, _ = load_approximator(cfg)

    artifacts, cases, param_names = setup(cfg)

    for case in cases:
        logger.info("Loading test data for case %s", case.key)
        conditions = load_case_data(case, artifacts)

        posterior_samples = approximator.sample(
            conditions=conditions,
            num_samples=cfg["test_num_posterior_samples"],
        )

        path = artifacts.npe_samples(case)
        artifacts.ensure(path)
        logger.info("Saving NPE samples to %s", path)
        save_posterior(path, stack_posterior_dict(posterior_samples, param_names), param_names)


if __name__ == "__main__":
    predict_npe()
