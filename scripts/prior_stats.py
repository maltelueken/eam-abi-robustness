"""Dump prior draws, so the figures can show the prior's range alongside the posteriors.

`visualization/experiment_2.R` and `experiment_4.R` read this table; the script that used
to produce it was lost, leaving those figures unreproducible from the repository alone.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_param_names
from pipeline import setup
from simulation import sample_prior
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prior_stats(cfg: DictConfig):
    """Write `metrics/prior_stats.csv` in long format."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    artifacts, _, _ = setup(cfg)
    param_names = get_param_names(cfg)

    prior_dict = sample_prior(simulator, cfg["prior_stats_num_draws"])

    records = [
        {"draw": index, "param": param, "value": float(value)}
        for param in param_names
        for index, value in enumerate(np.reshape(prior_dict[param], -1))
    ]

    path = artifacts.csv("metrics", "prior_stats")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    prior_stats()
