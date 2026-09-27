"""Dump prior draws, so the figures can show the prior's range alongside the posteriors.

`visualization/experiment_2.R` and `experiment_4.R` read this table; the script that used
to produce it was lost, leaving those figures unreproducible from the repository alone.

Draws come from the full simulator rather than the prior alone, so the table reflects what a
model is actually trained on: study 5's approximators see the prior *restricted* to an accuracy
band and study 6's to a response-time window, and the `accuracy` / `rt_min` / `rt_max` rows below
are what those figures shade. Simulating a dataset per draw used to be prohibitive here; since
the simulators were vectorized it costs about a second.

A narrow band makes this expensive in a way the rejection sampler cannot absorb: at study 6's
tightest window roughly one draw in twenty is kept, so `prior_stats_num_draws` accepted datasets
need more simulation than `max_rounds` rounds allow, and the run aborts rather than quietly
returning a short table. Those conditions' training windows are constants in the yaml, so the
figures shade them directly instead.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_param_names
from pipeline import accuracy, rt_range, setup
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

    # One row per draw for the accuracy and the response-time bounds its parameters produce,
    # so the figures can shade the range of *data* a model was trained on, not only the range of
    # each parameter. These are what studies 5 and 6 respectively select on.
    lowest_rt, highest_rt = rt_range(prior_dict["x"])

    records += [
        {"draw": index, "param": name, "value": float(value)}
        for name, values in (
            ("accuracy", accuracy(prior_dict["x"])),
            ("rt_min", lowest_rt),
            ("rt_max", highest_rt),
        )
        for index, value in enumerate(values)
    ]

    path = artifacts.csv("metrics", "prior_stats")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    prior_stats()
