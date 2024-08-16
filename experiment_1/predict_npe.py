import os
import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def generate_test_data(cfg: DictConfig):
    trainer = instantiate(cfg["trainer"])
    # Disable context generator because we set num_obs manually
    trainer.generative_model.simulator.context_gen = None

    prior_means = instantiate(cfg["prior_stats"]["prior_means"])
    prior_stds = instantiate(cfg["prior_stats"]["prior_stds"])

    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs(["npe_samples"])

    for t in sample_sizes:
        logger.info("Loading test data for sample size %s", t)

        data = load_hdf5(os.path.join("test_data", f"test_data_sample_size_{t}.hdf5"))

        forward_dict = trainer.configurator(data)

        posterior_samples = trainer.amortizer.sample(
            forward_dict, cfg["test_num_posterior_samples"]
        )
        posterior_samples = posterior_samples * prior_stds + prior_means

        logger.info("Generated posterior samples with shape %s", posterior_samples.shape)

        save_hdf5(os.path.join("npe_samples", f"posterior_samples_sample_size_{t}.hdf5"), {"samples": posterior_samples})


if __name__ == "__main__":
    generate_test_data()
