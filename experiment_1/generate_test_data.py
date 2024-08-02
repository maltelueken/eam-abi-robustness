import os
import logging
import pickle

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def generate_test_data(cfg: DictConfig):
    trainer = instantiate(cfg["trainer"])
    # Disable context generator because we set num_obs manually
    trainer.generative_model.simulator.context_gen = None

    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs(["test_data"])

    for t in sample_sizes:
        logger.info("Creating test data set for sample size %s", t)
        data = trainer.generative_model(
            cfg["test_batch_size"], **{"sim_args": {"num_obs": t}}
        )
        data["sim_non_batchable_context"] = t

        np.savez_compressed(os.path.join("test_data", f"test_data_sample_size_{t}.npz"), data)

        logger.info("Shape of generated data set: %s", data["sim_data"].shape)


if __name__ == "__main__":
    generate_test_data()
