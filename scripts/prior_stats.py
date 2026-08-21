"""Dump prior draws, so the figures can show the prior's range alongside the posteriors.

`visualization/experiment_2.R` and `experiment_4.R` read this table; the script that used
to produce it was lost, leaving those figures unreproducible from the repository alone.

Draws come from the full simulator rather than the prior alone, so the table reflects what a
model is actually trained on: study 5's approximators see the prior *restricted* to an accuracy
band, and the accuracy rows below are what those figures shade. Simulating a dataset per draw
used to be prohibitive here; since the simulators were vectorized it costs about a second.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_param_names
from pipeline import accuracy, setup
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prior_stats(cfg: DictConfig):
    """Write `metrics/prior_stats.csv` in long format."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    artifacts, _, _ = setup(cfg)
    param_names = get_param_names(cfg)

    prior_dict = simulator.sample(cfg["prior_stats_num_draws"], num_obs=np.array(cfg["eval_num_obs"]))

    records = [
        {"draw": index, "param": param, "value": float(value)}
        for param in param_names
        for index, value in enumerate(np.reshape(prior_dict[param], -1))
    ]

    # One row per draw for the accuracy its parameters produce, so the figures can shade the
    # range of performance a model was trained on, not only the range of each parameter.
    records += [
        {"draw": index, "param": "accuracy", "value": float(value)}
        for index, value in enumerate(accuracy(prior_dict["x"]))
    ]

    path = artifacts.csv("metrics", "prior_stats")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    prior_stats()
