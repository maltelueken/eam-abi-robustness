"""Simulate the held-out test data for every test case."""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import save_dataset
from pipeline import setup

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    """Sample `test_batch_size` datasets per case and save them as NetCDF."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    artifacts, cases, _ = setup(cfg)

    for case in cases:
        if not case.is_simulated:
            logger.info("Case %s uses empirical data (%s); nothing to simulate.", case.key, case.source_path)
            continue

        logger.info("Creating test data for case %s", case.key)

        forward_dict = simulator.sample(cfg["test_batch_size"], **case.sim_kwargs)

        path = artifacts.test_data(case)
        artifacts.ensure(path)
        logger.info("Saving test data to %s", path)
        save_dataset(path, forward_dict)


if __name__ == "__main__":
    generate_test_data()
